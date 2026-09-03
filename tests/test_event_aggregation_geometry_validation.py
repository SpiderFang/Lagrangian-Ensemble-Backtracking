"""事件聚合的幾何、邊界弧長與跨站 identity fail-closed 測試。

本檔直接重用同目錄 ``test_event_aggregation_core`` 的最小兩站 fixture，所有
位置與邊界弧長 ``s`` 都以公尺表示；網格索引採 ``(y_cell, x_cell)``，且矩形的
最大 x、y 邊界屬於閉區間。跨站事件的 ``study_site_id`` 仍是粒子來源站，
``related_study_site_id`` 才是被進入或離開的目標站；因此 repeated ENTER 只
能對同一粒子／來源／目標組合計一次，EXIT 不應增加 unique 進入數。這些測試
只驗證公開事件聚合入口的輸入契約，不代表正式海洋資料或絕對來源機率。
"""

from dataclasses import replace

import numpy as np
import pytest
from test_event_aggregation_core import (
    _ARRIVAL_TIME_UTC_NS,
    _SHARED_SEGMENT_ID,
    _aggregate,
    _boundary_event,
    _particle_result,
    _scenario,
)

from lagrangian_backtracking.event_aggregation import CrossSiteAggregateKey
from lagrangian_backtracking.models import EventType, ParticleStatus


def _result_with_semantic_event(
    scenario,
    event,
    *,
    particle_id: str = "geometry-particle",
    final_x_m: float = 0.25,
    final_y_m: float = 0.25,
):
    """把一筆待測 semantic event 接上合法 MAX_AGE terminal event。

    helper 沿用核心 fixture 的兩筆 observation：第一筆在 arrival UTC 與 0 秒，
    最後一筆在 20 秒回溯年齡。如此測試中的壞資料只會落在指定 event 欄位，
    不會因缺少 terminal event 或觀測不完整而失去幾何驗證的意義。
    """

    terminal_event = _boundary_event(
        scenario,
        particle_id,
        EventType.MAX_AGE,
        20.0,
        x_m=final_x_m,
        y_m=final_y_m,
    )
    return _particle_result(
        scenario,
        particle_id,
        ParticleStatus.MAX_AGE,
        final_age_seconds=20.0,
        final_x_m=final_x_m,
        final_y_m=final_y_m,
        events=(event, terminal_event),
    )


def _semantic_event(
    scenario,
    *,
    event_type: EventType = EventType.LOCAL_DOMAIN_FIRST_EXIT,
    particle_id: str = "geometry-particle",
):
    """建立落在 2×2 公尺閉網格與 2.5 公尺邊界段內的合法基準事件。"""

    return _boundary_event(
        scenario,
        particle_id,
        event_type,
        10.0,
        x_m=0.25,
        y_m=0.25,
        boundary_segment_id=_SHARED_SEGMENT_ID,
        boundary_s_m=1.0,
    )


@pytest.mark.parametrize(
    "event_changes",
    (
        pytest.param({"particle_id": "different-particle"}, id="particle-id"),
        pytest.param({"scenario_id": "different-scenario"}, id="scenario-id"),
        pytest.param({"member_id": 1}, id="member-id"),
        pytest.param({"study_site_id": "site-b"}, id="study-site-id"),
        pytest.param({"analysis_region_id": "different-region"}, id="region-id"),
        pytest.param({"receptor_id": "different-receptor"}, id="receptor-id"),
        pytest.param(
            {"time_utc_ns": _ARRIVAL_TIME_UTC_NS + 1},
            id="event-after-arrival",
        ),
        pytest.param(
            {"time_utc_ns": _ARRIVAL_TIME_UTC_NS - 20_000_000_001},
            id="event-before-final",
        ),
        pytest.param({"fraction": -1.0e-9}, id="fraction-below-zero"),
        pytest.param({"fraction": 1.0 + 1.0e-9}, id="fraction-above-one"),
    ),
)
def test_rejects_event_identity_time_and_fraction_mismatch(event_changes) -> None:
    """事件 identity、UTC 時間與交點比例越界時，公開入口應拒絕整筆結果。

    event UTC 必須落在 Scenario arrival time 與 final state time 的閉區間；
    ``fraction`` 是單一步驟由步首到步末的交點比例，合法範圍為 [0, 1]。
    identity 欄位則必須和同一粒子的 final state 完全一致，避免把不同站點、
    受體或成員的事件錯誤併入同一個條件式來源足跡。
    """

    scenario = _scenario()
    valid_event = _semantic_event(scenario)
    bad_event = replace(valid_event, **event_changes)
    result = _result_with_semantic_event(scenario, bad_event)

    with pytest.raises(ValueError):
        _aggregate((result,), (scenario,))


@pytest.mark.parametrize(
    "invalid_flag",
    (
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param(1, id="integer"),
        pytest.param("true", id="string"),
        pytest.param(None, id="none"),
    ),
)
def test_rejects_non_native_also_local_domain_first_exit(invalid_flag) -> None:
    """``also_local_domain_first_exit`` 必須是 Python 原生 bool，而非可相等值。

    FLOW_DOMAIN_OPEN_EXIT 在 local 與 flow domain 重合時可同時承擔兩種語意，
    但旗標不是數值或字串的寬鬆真值判斷；嚴格型別可避免跨站與 local 分母因
    ``1``、NumPy bool 或序列化字串被誤當成第二次 crossing。座標仍以公尺傳遞。
    """

    scenario = _scenario()
    valid_event = _semantic_event(
        scenario,
        event_type=EventType.FLOW_DOMAIN_OPEN_EXIT,
    )
    bad_event = replace(
        valid_event,
        attributes={"also_local_domain_first_exit": invalid_flag},
    )
    result = _result_with_semantic_event(scenario, bad_event)

    with pytest.raises(ValueError):
        _aggregate((result,), (scenario,))


@pytest.mark.parametrize(
    "event_type,event_changes",
    (
        pytest.param(
            EventType.LOCAL_DOMAIN_FIRST_EXIT,
            {"boundary_segment_id": "unknown-local-segment"},
            id="local-segment-not-in-spec",
        ),
        pytest.param(
            EventType.FLOW_DOMAIN_OPEN_EXIT,
            {"boundary_segment_id": "unknown-outer-segment"},
            id="outer-segment-not-in-spec",
        ),
        pytest.param(
            EventType.LOCAL_DOMAIN_FIRST_EXIT,
            {"boundary_s_m": -1.0e-9},
            id="boundary-s-below-zero-m",
        ),
        pytest.param(
            EventType.FLOW_DOMAIN_OPEN_EXIT,
            {"boundary_s_m": 2.5 + 1.0e-9},
            id="boundary-s-above-length-m",
        ),
        pytest.param(
            EventType.LOCAL_DOMAIN_FIRST_EXIT,
            {"x_m": -1.0e-9},
            id="semantic-x-below-grid-min-m",
        ),
        pytest.param(
            EventType.LOCAL_DOMAIN_FIRST_EXIT,
            {"x_m": 2.0 + 1.0e-9},
            id="semantic-x-above-grid-max-m",
        ),
        pytest.param(
            EventType.FLOW_DOMAIN_OPEN_EXIT,
            {"y_m": -1.0e-9},
            id="semantic-y-below-grid-min-m",
        ),
        pytest.param(
            EventType.FLOW_DOMAIN_OPEN_EXIT,
            {"y_m": 2.0 + 1.0e-9},
            id="semantic-y-above-grid-max-m",
        ),
    ),
)
def test_rejects_invalid_segment_arclength_and_semantic_grid_coordinates(
    event_type: EventType,
    event_changes,
) -> None:
    """local／outer 邊界段、弧長與 semantic event 座標都必須符合公尺制規格。

    local 與 outer 使用各自登錄於 ``AggregateSpec`` 的 segment ID；``boundary_s_m``
    是該段從 0 m 到 ``length_m`` 的弧長，兩端均可取但不可超出。semantic event
    的 x、y 必須落在包含最大邊界的閉矩形，這保留恰落在網格外框上的合法事件，
    同時拒絕無法對應 cell 的域外位置。
    """

    scenario = _scenario()
    valid_event = _semantic_event(scenario, event_type=event_type)
    bad_event = replace(valid_event, **event_changes)
    result = _result_with_semantic_event(scenario, bad_event)

    with pytest.raises(ValueError):
        _aggregate((result,), (scenario,))


@pytest.mark.parametrize(
    "related_site_id",
    (
        pytest.param("unknown-site", id="unknown-related-site"),
        pytest.param("site-a", id="related-site-equals-origin"),
        pytest.param("site-b", id="repeated-enter-and-exit"),
    ),
)
def test_validates_other_site_identity_and_counts_unique_enter_only(related_site_id) -> None:
    """跨站 related site 必須是另一個已登錄站點，且 repeated ENTER 只計一次。

    ``study_site_id`` 是粒子的 origin，``related_study_site_id`` 是目標站；未知
    目標或與 origin 相同都會使輸入失效。對合法的 site-b 目標，兩筆同一粒子的
    ENTER 與一筆 EXIT 仍只形成一個 ``site-a → site-b`` unique member，因為 EXIT
    是離開診斷而非新的進入；所有位置在此仍以公尺表示。
    """

    scenario = _scenario()
    base_event = _boundary_event(
        scenario,
        "cross-particle",
        EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER,
        5.0,
        x_m=0.25,
        y_m=0.25,
        related_study_site_id=related_site_id,
    )

    if related_site_id != "site-b":
        bad_event = replace(
            base_event,
            related_study_site_id=related_site_id,
        )
        result = _result_with_semantic_event(
            scenario,
            bad_event,
            particle_id="cross-particle",
        )
        with pytest.raises(ValueError):
            _aggregate((result,), (scenario,))
        return

    later_enter = replace(
        base_event,
        time_utc_ns=_ARRIVAL_TIME_UTC_NS - 10_000_000_000,
    )
    exit_event = replace(
        base_event,
        event_type=EventType.OTHER_SITE_LOCAL_DOMAIN_EXIT,
        time_utc_ns=_ARRIVAL_TIME_UTC_NS - 15_000_000_000,
    )
    result = _result_with_semantic_event(
        scenario,
        base_event,
        particle_id="cross-particle",
    )
    result = replace(
        result,
        events=[base_event, later_enter, exit_event, result.events[-1]],
    )
    aggregate = _aggregate((result,), (scenario,))

    assert aggregate.cross_site_unique_member_count[
        CrossSiteAggregateKey("site-a", "site-b")
    ] == 1
    assert aggregate.cross_site_unique_member_count[
        CrossSiteAggregateKey("site-b", "site-a")
    ] == 0
