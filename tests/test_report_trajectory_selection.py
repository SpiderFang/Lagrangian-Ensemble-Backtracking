"""F03 代表軌跡固定容量八層 selector 的契約與生命週期測試。

本檔只建立公尺、秒與世界協調時間（UTC）奈秒組成的小型 synthetic 軌跡，驗證每站
``DJF/MAM/JJA/SON × spring_proxy/neap_proxy`` 配額、SHA-256 top-K、event／失敗排除、
重複與碰撞閘門，以及 finalize 後的 nested immutable snapshot。這些案例只證明工程選樣
可重建，不代表真實 OCM／NWW3 科學成果、絕對來源機率或因果歸因。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import numpy as np
import pytest

import lagrangian_backtracking.report_trajectory_selection as selection_module
from lagrangian_backtracking.engine import Observation, ParticleResult
from lagrangian_backtracking.models import ParticleState, ParticleStatus
from lagrangian_backtracking.report_spec import REPORT_SPEC_SCHEMA_VERSION, ReportSpec
from lagrangian_backtracking.report_trajectory_identity import (
    CORE_SEASONS,
    CORE_TIDE_CLASSES,
    REPRESENTATIVE_SELECTION_POLICY,
    representative_priority_digest,
)
from lagrangian_backtracking.report_trajectory_selection import (
    RepresentativeSelection,
    RepresentativeTrajectorySelector,
)

_HASH = "a" * 64
_OTHER_HASH = "b" * 64
_STRATA = tuple(
    (season, tide_class)
    for season in CORE_SEASONS
    for tide_class in CORE_TIDE_CLASSES
)


def _report_spec(*, representative_count: int = 8, selection_seed: int = 17) -> ReportSpec:
    """建立只供 selector 測試使用的 exact immutable ``ReportSpec``。

    每站代表軌跡數由測試明示；其餘 renderer、分位數、公尺帶寬及 SHA-256 欄位使用合法
    的最小固定值，避免 selector 測試依賴檔案 I/O 或 aggregate release。
    """

    return ReportSpec(
        schema_version=REPORT_SPEC_SCHEMA_VERSION,
        run_id="trajectory-selection-test",
        aggregate_spec_canonical_sha256=_HASH,
        primary_kde_bandwidth_m=100,
        minimum_kde_raw_count=1,
        low_sample_min_member_count=1,
        vertical_depth_bin_edges_m=(0.0, 5.0),
        representative_trajectory_count_per_site=representative_count,
        representative_selection_policy=REPRESENTATIVE_SELECTION_POLICY,
        representative_selection_seed=selection_seed,
        travel_age_quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        pathway_first_passage_quantiles=(0.25, 0.5, 0.75),
        figure_formats=("png", "svg", "pdf"),
        raster_dpi=300,
        renderer_style_version="academic_zh_tw_v1",
        language="zh-TW",
        source_sha256=_HASH,
        canonical_sha256=_OTHER_HASH,
    )


def _result(
    site_id: str,
    label: str,
    *,
    member_id: int = 0,
    status: ParticleStatus = ParticleStatus.MAX_AGE,
    observation_count: int = 2,
) -> ParticleResult:
    """建立 identity 唯一、終點一致且可選擇觀測筆數的 synthetic 粒子結果。

    兩筆正常觀測依逆向時間使用 age 遞增、UTC 遞減順序；一筆觀測模式只供驗證代表
    trajectory 至少需形成一段線段。DATA_GAP／NUMERICAL_FAILURE 測試可使用空觀測，因為
    這些成員應在建立代表物件前依分母政策排除。
    """

    particle_id = f"particle-{site_id}-{label}"
    observations: list[Observation]
    if observation_count <= 0:
        observations = []
    elif observation_count == 1:
        observations = [
            Observation(particle_id, 10, 1.0, 1.0, 2.0, -3.0, status),
        ]
    else:
        observations = [
            Observation(particle_id, 20, 0.0, 0.0, 0.0, -1.0, ParticleStatus.ACTIVE),
            Observation(particle_id, 10, 1.0, 1.0, 2.0, -3.0, status),
        ]
    return ParticleResult(
        final_state=ParticleState(
            particle_id=particle_id,
            scenario_id=f"scenario-{site_id}-{label}",
            member_id=member_id,
            study_site_id=site_id,
            analysis_region_id=f"region-{site_id}",
            receptor_id=f"receptor-{site_id}",
            x_m=1.0,
            y_m=2.0,
            z_m=-3.0,
            time_utc_ns=10,
            age_seconds=1.0,
            status=status,
        ),
        observations=observations,
        events=[],
        step_count=1,
        minimum_clamp_count=0,
    )


def _identity_kwargs(
    label: str,
    *,
    season: str,
    tide_class: str,
) -> dict[str, str]:
    """建立一筆有效核心候選的 report strata identity 關鍵字。"""

    return {
        "material_id": f"material-{label}",
        "arrival_time_id": f"arrival-{label}",
        "season": season,
        "tide_class": tide_class,
    }


def _add_all_strata(
    selector: RepresentativeTrajectorySelector,
    site_id: str,
    *,
    per_stratum: int = 1,
    prefix: str = "core",
) -> None:
    """為指定站點八層各加入固定數量、identity 唯一的有效軌跡。"""

    for season, tide_class in _STRATA:
        for index in range(per_stratum):
            label = f"{prefix}-{season}-{tide_class}-{index}"
            selector.add(
                _result(site_id, label, member_id=index),
                **_identity_kwargs(label, season=season, tide_class=tide_class),
            )


def _digest_for(
    spec: ReportSpec,
    result: ParticleResult,
    identity: dict[str, str],
) -> str:
    """用 production helper 計算測試候選的預期 priority digest。"""

    state = result.final_state
    return representative_priority_digest(
        spec.representative_selection_seed,
        state.particle_id,
        state.scenario_id,
        state.member_id,
        state.study_site_id,
        state.analysis_region_id,
        state.receptor_id,
        identity["material_id"],
        identity["arrival_time_id"],
        identity["season"],
        identity["tide_class"],
    )


def test_eight_strata_receive_equal_quota_and_sorted_digests() -> None:
    """K=8 時每站八層各保留一筆，輸出層順序與 eligible count 都完整。"""

    selector = RepresentativeTrajectorySelector(_report_spec(), ["site-a"])
    _add_all_strata(selector, "site-a")

    selection = selector.finalize()
    assert selection.site_ids == ("site-a",)
    assert selection.capacity_per_stratum == 1
    assert selection.selection_policy == REPRESENTATIVE_SELECTION_POLICY
    assert selection.selection_seed == 17
    assert tuple(selection.selected["site-a"]) == _STRATA
    assert selector.retained_count == 8
    for key in _STRATA:
        records = selection.selected["site-a"][key]
        expected_label = f"core-{key[0]}-{key[1]}-0"
        assert len(records) == 1
        assert (records[0].season, records[0].tide_class) == key
        assert records[0].material_id == f"material-{expected_label}"
        assert records[0].arrival_time_id == f"arrival-{expected_label}"
        assert isinstance(records[0].observations, tuple)
        assert selection.eligible_count_by_site_stratum["site-a"][key] == 1


def test_two_sites_are_sorted_and_selected_independently() -> None:
    """兩站即使以反序輸入，仍各自擁有獨立八層 reservoir 與 canonical site order。"""

    selector = RepresentativeTrajectorySelector(_report_spec(), ["site-b", "site-a"])
    assert selector.site_ids == ("site-a", "site-b")
    _add_all_strata(selector, "site-a", prefix="a")
    _add_all_strata(selector, "site-b", prefix="b")

    selection = selector.finalize()
    assert selection.site_ids == ("site-a", "site-b")
    assert selector.retained_count == 16
    for site_id in selection.site_ids:
        assert set(selection.selected[site_id]) == set(_STRATA)
        assert all(
            record.study_site_id == site_id
            for records in selection.selected[site_id].values()
            for record in records
        )


def test_top_k_retains_smallest_digests_in_strict_order() -> None:
    """每層容量為二時，只保留全部 eligible 候選中最小兩個 SHA-256。"""

    spec = _report_spec(representative_count=16)
    selector = RepresentativeTrajectorySelector(spec, ["site-a"])
    target = ("DJF", "spring_proxy")
    target_candidates: list[tuple[str, str]] = []

    for season, tide_class in _STRATA:
        count = 30 if (season, tide_class) == target else 2
        for index in range(count):
            label = f"topk-{season}-{tide_class}-{index}"
            result = _result("site-a", label, member_id=index)
            identity = _identity_kwargs(label, season=season, tide_class=tide_class)
            digest = _digest_for(spec, result, identity)
            selector.add(result, **identity)
            if (season, tide_class) == target:
                target_candidates.append((digest, result.final_state.particle_id))

    selection = selector.finalize()
    selected = selection.selected["site-a"][target]
    expected = sorted(target_candidates)[:2]
    assert tuple(record.priority_digest for record in selected) == tuple(item[0] for item in expected)
    assert tuple(record.particle_id for record in selected) == tuple(item[1] for item in expected)
    assert selection.eligible_count_by_site_stratum["site-a"][target] == 30
    assert selector.retained_count == 16


def test_event_and_invalid_members_are_counted_separately_and_active_raises() -> None:
    """失敗計 invalid，核心 season 加 event tide 才計 event，ACTIVE 不進入計數。"""

    selector = RepresentativeTrajectorySelector(_report_spec(), ["site-a"])
    selector.add(
        _result("site-a", "gap", status=ParticleStatus.DATA_GAP, observation_count=0),
        material_id="",
        arrival_time_id="",
        season="event",
        tide_class="event",
    )
    selector.add(
        _result(
            "site-a",
            "numerical",
            status=ParticleStatus.NUMERICAL_FAILURE,
            observation_count=0,
        ),
        material_id="",
        arrival_time_id="",
        season="event",
        tide_class="event",
    )
    selector.add(
        _result("site-a", "event-tide"),
        material_id="material-event",
        arrival_time_id="arrival-event-tide",
        season="DJF",
        tide_class="event",
    )
    with pytest.raises(ValueError):
        selector.add(
            _result("site-a", "unknown-season"),
            material_id="material-event",
            arrival_time_id="arrival-unknown-season",
            season="event",
            tide_class="spring_proxy",
        )
    with pytest.raises(ValueError):
        selector.add(
            _result("site-a", "unknown-event-season"),
            material_id="material-event",
            arrival_time_id="arrival-unknown-event-season",
            season="event",
            tide_class="event",
        )
    with pytest.raises(ValueError):
        selector.add(
            _result("site-a", "unknown-tide"),
            material_id="material-event",
            arrival_time_id="arrival-unknown-tide",
            season="DJF",
            tide_class="events",
        )
    with pytest.raises(ValueError):
        selector.add(
            _result("site-a", "active", status=ParticleStatus.ACTIVE),
            material_id="material-active",
            arrival_time_id="arrival-active",
            season="DJF",
            tide_class="spring_proxy",
        )

    _add_all_strata(selector, "site-a")
    selection = selector.finalize()
    assert selection.excluded_invalid_member_count_by_site == {"site-a": 2}
    assert selection.excluded_event_arrival_count_by_site == {"site-a": 1}
    assert sum(selection.eligible_count_by_site_stratum["site-a"].values()) == 8


def test_core_candidate_requires_at_least_two_exact_observations() -> None:
    """有效核心候選若沒有可畫線段的兩筆 observation，必須在計數前失敗。"""

    selector = RepresentativeTrajectorySelector(_report_spec(), ["site-a"])
    with pytest.raises(ValueError):
        selector.add(
            _result("site-a", "short", observation_count=1),
            **_identity_kwargs("short", season="DJF", tide_class="spring_proxy"),
        )
    assert selector.retained_count == 0


def test_finalize_fails_closed_for_underfilled_stratum_and_can_retry() -> None:
    """任一核心層不足時不建立結果；補足後同一 selector 仍可成功 finalize。"""

    selector = RepresentativeTrajectorySelector(_report_spec(), ["site-a"])
    missing = _STRATA[-1]
    for season, tide_class in _STRATA[:-1]:
        label = f"partial-{season}-{tide_class}"
        selector.add(
            _result("site-a", label),
            **_identity_kwargs(label, season=season, tide_class=tide_class),
        )
    with pytest.raises(ValueError):
        selector.finalize()

    label = "partial-final"
    selector.add(
        _result("site-a", label),
        **_identity_kwargs(label, season=missing[0], tide_class=missing[1]),
    )
    assert selector.finalize().capacity_per_stratum == 1


def test_duplicate_retained_trajectory_fails_without_incrementing_eligible_count() -> None:
    """完全相同且仍在 retained bucket 的候選不可重複加入或重複計數。"""

    selector = RepresentativeTrajectorySelector(_report_spec(), ["site-a"])
    result = _result("site-a", "duplicate")
    identity = _identity_kwargs("duplicate", season="DJF", tide_class="spring_proxy")
    selector.add(result, **identity)
    with pytest.raises(ValueError):
        selector.add(result, **identity)

    for season, tide_class in _STRATA[1:]:
        label = f"duplicate-fill-{season}-{tide_class}"
        selector.add(
            _result("site-a", label),
            **_identity_kwargs(label, season=season, tide_class=tide_class),
        )
    selection = selector.finalize()
    assert selection.eligible_count_by_site_stratum["site-a"][("DJF", "spring_proxy")] == 1


def test_retained_digest_collision_with_different_identity_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """相同 digest 若對應不同完整 identity，selector 必須 fail closed。"""

    monkeypatch.setattr(
        selection_module,
        "representative_priority_digest",
        lambda *args: "c" * 64,
    )
    selector = RepresentativeTrajectorySelector(_report_spec(), ["site-a"])
    selector.add(
        _result("site-a", "collision-a"),
        **_identity_kwargs("collision-a", season="DJF", tide_class="spring_proxy"),
    )
    with pytest.raises(ValueError):
        selector.add(
            _result("site-a", "collision-b"),
            **_identity_kwargs("collision-b", season="DJF", tide_class="spring_proxy"),
        )


def test_finalize_is_idempotent_and_add_after_success_is_rejected() -> None:
    """成功 finalize 後重複呼叫回同物件，selector 的串流生命週期也同步封閉。"""

    selector = RepresentativeTrajectorySelector(_report_spec(), ["site-a"])
    _add_all_strata(selector, "site-a")
    first = selector.finalize()
    second = selector.finalize()
    assert second is first
    with pytest.raises(RuntimeError):
        selector.add(
            _result("site-a", "late"),
            **_identity_kwargs("late", season="DJF", tide_class="spring_proxy"),
        )


def test_selection_nested_mappings_are_defensive_and_immutable() -> None:
    """輸入 dict/list 後續改動與結果端賦值都不能污染 selection snapshot。"""

    selector = RepresentativeTrajectorySelector(_report_spec(), ["site-a"])
    _add_all_strata(selector, "site-a")
    source = selector.finalize()

    raw_selected = {
        "site-a": {key: list(source.selected["site-a"][key]) for key in _STRATA}
    }
    raw_eligible = {
        "site-a": dict(source.eligible_count_by_site_stratum["site-a"])
    }
    raw_event = {"site-a": 0}
    raw_invalid = {"site-a": 0}
    snapshot = RepresentativeSelection(
        site_ids=["site-a"],  # type: ignore[arg-type]
        capacity_per_stratum=1,
        selection_policy=source.selection_policy,
        selection_seed=source.selection_seed,
        selected=raw_selected,
        eligible_count_by_site_stratum=raw_eligible,
        excluded_event_arrival_count_by_site=raw_event,
        excluded_invalid_member_count_by_site=raw_invalid,
    )

    raw_selected["site-a"][_STRATA[0]].clear()
    raw_eligible["site-a"][_STRATA[0]] = 999
    raw_event["site-a"] = 999
    assert len(snapshot.selected["site-a"][_STRATA[0]]) == 1
    assert snapshot.eligible_count_by_site_stratum["site-a"][_STRATA[0]] == 1
    assert snapshot.excluded_event_arrival_count_by_site["site-a"] == 0
    assert isinstance(snapshot.selected, MappingProxyType)
    assert isinstance(snapshot.selected["site-a"], MappingProxyType)
    assert isinstance(snapshot.eligible_count_by_site_stratum["site-a"], MappingProxyType)
    assert isinstance(snapshot.selected["site-a"][_STRATA[0]], tuple)

    with pytest.raises(TypeError):
        snapshot.selected["site-a"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.selected["site-a"][_STRATA[0]] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.excluded_invalid_member_count_by_site["site-a"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        snapshot.capacity_per_stratum = 2  # type: ignore[misc]


def test_selection_recomputes_digest_from_immutable_provenance() -> None:
    """replace 竄改 policy、seed 或 record digest 時，快照必須自行重算並拒絕。"""

    selector = RepresentativeTrajectorySelector(_report_spec(), ["site-a"])
    _add_all_strata(selector, "site-a")
    selection = selector.finalize()

    with pytest.raises(ValueError):
        replace(selection, selection_policy="other-policy")
    with pytest.raises(TypeError):
        replace(selection, selection_policy=np.str_(REPRESENTATIVE_SELECTION_POLICY))
    with pytest.raises(ValueError):
        replace(selection, selection_seed=selection.selection_seed + 1)
    with pytest.raises(TypeError):
        replace(selection, selection_seed=np.int64(selection.selection_seed))

    first_key = _STRATA[0]
    tampered_selected = {
        site_id: {key: list(records) for key, records in strata.items()}
        for site_id, strata in selection.selected.items()
    }
    original = tampered_selected["site-a"][first_key][0]
    replacement_prefix = "0" if original.priority_digest[0] != "0" else "1"
    tampered_selected["site-a"][first_key][0] = replace(
        original,
        priority_digest=replacement_prefix + original.priority_digest[1:],
    )
    with pytest.raises(ValueError):
        replace(selection, selected=tampered_selected)


@pytest.mark.parametrize(
    "field,value,expected_exception",
    [
        ("representative_trajectory_count_per_site", 7, ValueError),
        ("representative_trajectory_count_per_site", 10, ValueError),
        ("representative_trajectory_count_per_site", True, TypeError),
        ("representative_trajectory_count_per_site", np.int64(8), TypeError),
        ("representative_selection_policy", "other-policy", ValueError),
        ("representative_selection_policy", np.str_(REPRESENTATIVE_SELECTION_POLICY), TypeError),
        ("representative_selection_seed", True, TypeError),
        ("representative_selection_seed", 2**128, ValueError),
    ],
)
def test_selector_revalidates_mutated_report_spec_contract(
    field: str,
    value: object,
    expected_exception: type[Exception],
) -> None:
    """selector 不信任被低階手段竄改的 exact ReportSpec scalar。"""

    spec = _report_spec()
    object.__setattr__(spec, field, value)
    with pytest.raises(expected_exception):
        RepresentativeTrajectorySelector(spec, ["site-a"])


@pytest.mark.parametrize(
    "site_ids,expected_exception",
    [
        ([], ValueError),
        ("site-a", TypeError),
        (["site-a", "site-a"], ValueError),
        ([""], ValueError),
        ([" site-a"], ValueError),
        ([np.str_("site-a")], TypeError),
    ],
)
def test_selector_site_id_gates(site_ids: object, expected_exception: type[Exception]) -> None:
    """site_ids 必須是非空、原生、唯一且非字串容器。"""

    with pytest.raises(expected_exception):
        RepresentativeTrajectorySelector(_report_spec(), site_ids)


def test_selector_requires_exact_report_spec_and_registered_result_site() -> None:
    """constructor 拒絕非 exact spec，add 也拒絕未登錄站點與非 ParticleResult。"""

    with pytest.raises(TypeError):
        RepresentativeTrajectorySelector(object(), ["site-a"])  # type: ignore[arg-type]

    selector = RepresentativeTrajectorySelector(_report_spec(), ["site-a"])
    with pytest.raises(ValueError):
        selector.add(
            _result("site-b", "unknown"),
            **_identity_kwargs("unknown", season="DJF", tide_class="spring_proxy"),
        )
    with pytest.raises(TypeError):
        selector.add(  # type: ignore[arg-type]
            object(),
            material_id="material",
            arrival_time_id="arrival",
            season="DJF",
            tide_class="spring_proxy",
        )


@pytest.mark.parametrize(
    "overrides,expected_exception",
    [
        ({"material_id": ""}, ValueError),
        ({"arrival_time_id": np.str_("arrival")}, TypeError),
        ({"season": None}, TypeError),
        ({"tide_class": " tide"}, ValueError),
    ],
)
def test_add_core_identity_type_gates(
    overrides: dict[str, object],
    expected_exception: type[Exception],
) -> None:
    """有效成員在 event 判定前仍須提供四個原生非空 strata identity 欄位。"""

    selector = RepresentativeTrajectorySelector(_report_spec(), ["site-a"])
    values: dict[str, object] = {
        "material_id": "material",
        "arrival_time_id": "arrival",
        "season": "DJF",
        "tide_class": "spring_proxy",
    }
    values.update(overrides)
    with pytest.raises(expected_exception):
        selector.add(_result("site-a", "type-gate"), **values)  # type: ignore[arg-type]


def test_retained_count_stays_bounded_when_candidate_n_grows() -> None:
    """同一層輸入大量候選時 retained 數維持容量一，不隨 N 線性增加。"""

    selector = RepresentativeTrajectorySelector(_report_spec(), ["site-a"])
    for index in range(250):
        label = f"bounded-{index}"
        selector.add(
            _result("site-a", label, member_id=index),
            **_identity_kwargs(label, season="DJF", tide_class="spring_proxy"),
        )
        assert selector.retained_count == 1
    assert selector.retained_count <= selector.capacity_per_stratum
