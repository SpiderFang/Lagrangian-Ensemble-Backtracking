"""事件聚合公開 API 的 fail-closed 驗證測試。

本檔案只驗證 ``initialize_event_aggregate`` 與 ``aggregate_result_events`` 的輸入
資料契約，不測試內部 helper。fixture 使用公尺制座標、UTC 奈秒整數與回溯年齡秒數，
藉此確認粒子 identity、時間方向與終止事件在資料進入正式聚合前即被拒絕；任何
無法確定對應關係的資料都不能靜默進入條件式來源足跡統計。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_event_aggregation_core import (
    _AGE_BIN_EDGES_SECONDS,
    _aggregate_spec,
    _boundary_event,
    _particle_result,
    _scenario,
)

from lagrangian_backtracking.engine import ParticleResult
from lagrangian_backtracking.event_aggregation import (
    aggregate_result_events,
    initialize_event_aggregate,
)
from lagrangian_backtracking.models import EventType, ParticleStatus


def _valid_result(
    *,
    scenario_id: str = "scenario-a",
    particle_id: str = "particle-validation",
) -> tuple[object, ParticleResult]:
    """建立可通過公開聚合入口驗證的最小 MAX_AGE 結果。

    事件時間與最後 observation 都由同一個 arrival UTC 推導，讓測試變異只改動
    指定的壞資料欄位。``ParticleResult`` 的 identity 必須與 Scenario 完全對齊，
    終止狀態則必須由唯一的 ``MAX_AGE`` terminal event 支持。
    """

    scenario = _scenario(scenario_id=scenario_id)
    terminal_event = _boundary_event(
        scenario,
        particle_id,
        EventType.MAX_AGE,
        10.0,
        x_m=1.0,
        y_m=1.0,
    )
    result = _particle_result(
        scenario,
        particle_id,
        ParticleStatus.MAX_AGE,
        final_age_seconds=10.0,
        final_x_m=1.0,
        final_y_m=1.0,
        events=(terminal_event,),
    )
    return scenario, result


def _aggregate(
    results: tuple[ParticleResult, ...],
    scenarios_by_id: dict[str, object],
    *,
    age_edges: np.ndarray = _AGE_BIN_EDGES_SECONDS,
):
    """以固定的正式 age 軸呼叫公開事件聚合 API。"""

    return aggregate_result_events(
        results,
        scenarios_by_id=scenarios_by_id,
        spec=_aggregate_spec(),
        age_bin_edges_seconds=age_edges,
    )


@pytest.mark.parametrize("api", ["initialize", "aggregate"])
def test_public_apis_reject_age_edges_not_exactly_equal_to_spec(api: str) -> None:
    """兩個公開入口都必須拒絕與 spec 不完全相等的 age edges。

    age 軸的邊界以秒表示，且會同時決定所有旅行年齡直方圖的欄位位置；即使只是
    浮點微小差異，也可能讓同一事件落入不同 bin。初始化與實際聚合若接受不同
    格線，後續結果就無法比較，因此必須以 ``np.array_equal`` 採 fail-closed 行為。
    """

    scenario, result = _valid_result()
    mismatched_edges = np.array([0.0, 10.0, 20.000000000001], dtype=np.float64)

    with pytest.raises(ValueError, match="age_bin_edges_seconds"):
        if api == "initialize":
            initialize_event_aggregate(
                _aggregate_spec(),
                age_bin_edges_seconds=mismatched_edges,
            )
        else:
            _aggregate(
                (result,),
                {scenario.scenario_id: scenario},
                age_edges=mismatched_edges,
            )


@pytest.mark.parametrize("bad_case", ["mapping_key", "missing_result_scenario", "duplicate_particle"])
def test_aggregate_rejects_scenario_and_particle_identity_mismatch(bad_case: str) -> None:
    """聚合器必須拒絕 Scenario mapping 與 particle identity 的不一致。

    Scenario ID 與 particle ID 是跨輸入表的連接鍵；key/value 對不上、結果引用
    未登錄情境，或同一 call 重複 particle，都會使分母與事件歸屬不可稽核。這些
    情況不能以最近似的資料猜測，必須在正式計數前直接 fail-closed。
    """

    scenario, result = _valid_result()
    scenarios_by_id = {scenario.scenario_id: scenario}

    if bad_case == "mapping_key":
        scenarios_by_id = {"wrong-key": scenario}
        results = (result,)
    elif bad_case == "missing_result_scenario":
        bad_state = replace(result.final_state, scenario_id="scenario-not-registered")
        results = (replace(result, final_state=bad_state),)
    else:
        duplicate = replace(result, step_count=result.step_count + 1)
        results = (result, duplicate)

    with pytest.raises(ValueError):
        _aggregate(results, scenarios_by_id)


@pytest.mark.parametrize("bad_case", ["observation_particle", "age_order", "utc_order", "final_state"])
def test_aggregate_rejects_observation_identity_time_and_final_mismatch(bad_case: str) -> None:
    """聚合器必須拒絕 observation identity、age/UTC 順序及終點不一致。

    ``particle_id`` 必須在 final state、每筆 observation 與事件間一致；回溯 age
    以秒嚴格增加，而 UTC 奈秒必須嚴格倒退，兩者共同描述同一條時間軸。最後一筆
    observation 若不等於 final state，便無法確認事件與終點屬於同一粒子，因此所有
    identity 或時間矛盾都採 fail-closed，而不讓聚合器自行修正或排序資料。
    """

    scenario, result = _valid_result()
    observations = list(result.observations)

    if bad_case == "observation_particle":
        observations[1] = replace(observations[1], particle_id="other-particle")
        bad_result = replace(result, observations=observations)
    elif bad_case == "age_order":
        observations[1] = replace(observations[1], age_seconds=0.0)
        bad_result = replace(result, observations=observations)
    elif bad_case == "utc_order":
        observations[1] = replace(
            observations[1],
            time_utc_ns=observations[0].time_utc_ns,
        )
        bad_result = replace(result, observations=observations)
    else:
        bad_state = replace(result.final_state, age_seconds=11.0)
        bad_result = replace(result, final_state=bad_state)

    with pytest.raises(ValueError):
        _aggregate((bad_result,), {scenario.scenario_id: scenario})


@pytest.mark.parametrize("bad_case", ["active", "missing", "duplicate", "wrong_type"])
def test_aggregate_rejects_active_or_invalid_terminal_event_contract(bad_case: str) -> None:
    """聚合器必須拒絕 ACTIVE 終點與缺失、重複或錯誤類型 terminal event。

    terminal event 是 final status 的唯一可稽核停止原因；若缺失、重複或類型不符，
    就不能判斷結果是否完整，也不能把粒子計入任何正式 outcome。ACTIVE 更代表
    執行尚未完成，故公開聚合 API 必須在事件累加前直接停止並回報資料不合法。
    """

    scenario, result = _valid_result()

    if bad_case == "active":
        active_state = replace(result.final_state, status=ParticleStatus.ACTIVE)
        active_observation = replace(
            result.observations[-1],
            status=ParticleStatus.ACTIVE,
        )
        bad_result = replace(
            result,
            final_state=active_state,
            observations=[result.observations[0], active_observation],
        )
    elif bad_case == "missing":
        bad_result = replace(result, events=[])
    elif bad_case == "duplicate":
        bad_result = replace(result, events=[*result.events, result.events[0]])
    else:
        wrong_terminal = replace(
            result.events[0],
            event_type=EventType.SURFACE_CONTACT,
        )
        bad_result = replace(result, events=[wrong_terminal])

    with pytest.raises(ValueError):
        _aggregate((bad_result,), {scenario.scenario_id: scenario})
