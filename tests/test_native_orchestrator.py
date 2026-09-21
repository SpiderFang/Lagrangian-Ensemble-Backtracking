"""WP-7：``native_runner.orchestrator`` 的規劃、排程、平行執行與缺口安全驗收測試。

涵蓋（對應工作包驗收要求逐項）：

1. 工作切分完整無重複（``plan_native_work_units`` ＋ ``validate_batch_partition``）。
2. 排程把同一局部性群組留在同一 worker，且跨 worker 負載大致平衡。
3. 平行執行（``run_native_plan``）與直接序列呼叫 ``run_unit`` 的結果一致。
4. 續跑：第二次呼叫命中既有完成證據時完全跳過，不重新呼叫 ``run_unit``。
5. 失敗隔離：單一工作單元擲出例外不影響同批其餘單元，且被彙整進失敗清單。
6. 進度：``run_native_plan`` 執行期間會寫出可讀的進度快照。
7. 缺口安全的到達時刻：``check_arrival_gap_safety`` 找出重疊的到達時刻；
   ``rebuild_gap_safe_arrival_times`` 重用既有 ``select_arrival_times`` 重新選點，
   ``diff_stratum_coverage`` 明確回報分層涵蓋是否劣化。
8. 端到端（若有 ``LBT_BAYTRACE_ROOT``）：以真正的 ``run_backward_batch``＋合成網格、
   合成流場，用兩個 worker 平行跑一組工作單元，確認能拿到非平凡的原生驅動結果。

模組層級函式與常數（例如 ``_toy_run_unit``、``_E2E_*``）刻意放在頂層、不放進測試
函式內部：``run_native_plan`` 預設用 ``multiprocessing`` 的 ``spawn`` 啟動子程序，
子程序需要能夠重新 import 這個模組並取得同名物件，巢狀函式或閉包在 ``spawn`` 下
無法可靠序列化。
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

from lagrangian_backtracking.arrival_times import ArrivalTime
from lagrangian_backtracking.native_runner import orchestrator as orch
from lagrangian_backtracking.native_runner.forcing_bridge import ForcingGapInventoryReport
from lagrangian_backtracking.native_runner.particle_input import ParticleIdentityTable
from lagrangian_backtracking.ocm_reconstruction import GapSpec

_HOUR_NS = 3_600_000_000_000
_DAY_NS = 24 * _HOUR_NS


# ══════════════════════════════════════════════════════════════════════════
# 合成身分表：不需要任何真實流場或 BayTrace 原始碼，只測試規劃／排程／執行引擎
# ══════════════════════════════════════════════════════════════════════════


def _build_table(rows: list[dict]) -> ParticleIdentityTable:
    n = len(rows)
    return ParticleIdentityTable(
        particle_id=np.array([f"p{i:06d}" for i in range(n)], dtype="<U32"),
        scenario_id=np.array([f"sc{i:06d}" for i in range(n)], dtype="<U32"),
        member_id=np.array([row.get("member_id", 0) for row in rows], dtype=np.int32),
        study_site_id=np.array([row["study_site_id"] for row in rows], dtype="<U32"),
        analysis_region_id=np.array([row["analysis_region_id"] for row in rows], dtype="<U32"),
        receptor_id=np.array([row.get("receptor_id", f"r{i}") for i, row in enumerate(rows)], dtype="<U32"),
        material_id=np.array([row.get("material_id", "mat") for row in rows], dtype="<U32"),
        arrival_time_id=np.array([row.get("arrival_time_id", "arr") for row in rows], dtype="<U32"),
        settling_velocity_mps=np.array([row["settling_velocity_mps"] for row in rows], dtype=np.float64),
        arrival_time_utc_ns=np.array([row["arrival_time_utc_ns"] for row in rows], dtype=np.int64),
        seed_hex=np.array([format(i, "032x") for i in range(n)], dtype="<U32"),
        x0_m=np.array([row.get("x0_m", 0.0) for row in rows], dtype=np.float64),
        y0_m=np.array([row.get("y0_m", 0.0) for row in rows], dtype=np.float64),
        z0_m=np.array([row.get("z0_m", 0.0) for row in rows], dtype=np.float64),
        experiment_case_id="wp7_synthetic_v1",
        master_seed=20260921,
        random_stream_id=None,
    )


def _synthetic_plan_table() -> ParticleIdentityTable:
    """2 個流場區域 × 3 個到達時刻 × 2 個沉降速度、每組合 2 顆粒子。

    區域 ``A`` 有 ``site1``、``site2`` 兩個研究站共用同一份流場（比照貢寮／龜山島）；
    區域 ``B`` 只有 ``site3``。合計 (2+1) 站 × 3 到達 × 2 沉降 = 18 個工作單元、
    36 顆粒子，足以驗證跨站、跨區域、跨到達時刻與跨沉降速度的切分同時正確。
    """

    arrivals = [10 * _DAY_NS + i * _DAY_NS for i in range(3)]
    settlings = [-0.001, -0.002]
    rows: list[dict] = []
    for region, sites in (("A", ("site1", "site2")), ("B", ("site3",))):
        for site in sites:
            for arrival in arrivals:
                for settling in settlings:
                    for particle_index in range(2):
                        rows.append(
                            {
                                "study_site_id": site,
                                "analysis_region_id": region,
                                "arrival_time_utc_ns": arrival,
                                "settling_velocity_mps": settling,
                                "x0_m": float(particle_index),
                                "receptor_id": f"receptor{particle_index}",
                            }
                        )
    return _build_table(rows)


REGION_TO_FLOW_DOMAIN = {"A": "flowA_cache_v1", "B": "flowB_cache_v1"}


def test_plan_native_work_units_partitions_completely_without_overlap() -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)

    assert len(units) == 18
    assert sum(u.particle_count for u in units) == len(table)
    assert len({u.unit_id for u in units}) == len(units)

    covered = np.zeros(len(table), dtype=bool)
    for unit in units:
        assert not np.any(covered[unit.particle_row_indices]), "工作單元之間不應重疊"
        covered[unit.particle_row_indices] = True
        # 每個工作單元內的欄位必須完全同質，否則違反 RK4／同步時鐘正確性前提。
        assert np.all(table.analysis_region_id[unit.particle_row_indices] == unit.analysis_region_id)
        assert np.all(table.study_site_id[unit.particle_row_indices] == unit.study_site_id)
        assert np.all(table.arrival_time_utc_ns[unit.particle_row_indices] == unit.arrival_time_utc_ns)
        assert np.all(table.settling_velocity_mps[unit.particle_row_indices] == unit.settling_velocity_mps)
    assert np.all(covered), "工作單元聯集必須涵蓋全部粒子"

    # 區域 A 的兩站共用流場區域，但因為站別不同仍必須拆成不同工作單元。
    region_a_units = [u for u in units if u.analysis_region_id == "A"]
    assert {u.study_site_id for u in region_a_units} == {"site1", "site2"}
    assert all(u.flow_domain_id == "flowA_cache_v1" for u in region_a_units)
    assert all(u.flow_domain_id == "flowB_cache_v1" for u in units if u.analysis_region_id == "B")


def test_plan_native_work_units_defaults_flow_domain_to_region_when_unmapped() -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table)
    assert {u.flow_domain_id for u in units} == {"A", "B"}


# ══════════════════════════════════════════════════════════════════════════
# WP-9：plan_native_hour_major_work_units 把逐工作單元摺疊成每個流場區域一個
# ══════════════════════════════════════════════════════════════════════════


def test_plan_native_hour_major_work_units_groups_by_flow_domain() -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)

    domain_units, members = orch.plan_native_hour_major_work_units(units)

    # 兩個流場區域各摺成一個排程單元，且依 flow_domain_id 字串排序。
    assert [u.flow_domain_id for u in domain_units] == sorted({"flowA_cache_v1", "flowB_cache_v1"})
    assert len(domain_units) == 2
    assert len({u.unit_id for u in domain_units}) == 2
    assert all(u.unit_id.startswith("hourmajor::") for u in domain_units)

    for domain_unit in domain_units:
        member_units = members[domain_unit.unit_id]
        # members 對照表裡的每個成員都真的屬於這個流場區域。
        assert all(m.flow_domain_id == domain_unit.flow_domain_id for m in member_units)
        # 摺疊出來的粒子列位置聯集，必須跟底下全部成員單元的列位置聯集完全相同
        # （不遺漏、不重複、不多算）。
        expected_rows = np.unique(np.concatenate([m.particle_row_indices for m in member_units]))
        assert np.array_equal(np.sort(domain_unit.particle_row_indices), expected_rows)
        assert domain_unit.particle_count == sum(m.particle_count for m in member_units)
        # 成員清單依 unit_id 確定性排序。
        assert list(member_units) == sorted(member_units, key=lambda m: m.unit_id)

    flow_a_members = members[
        next(u.unit_id for u in domain_units if u.flow_domain_id == "flowA_cache_v1")
    ]
    flow_b_members = members[
        next(u.unit_id for u in domain_units if u.flow_domain_id == "flowB_cache_v1")
    ]
    assert len(flow_a_members) == 12  # 2 站 × 3 到達 × 2 沉降
    assert len(flow_b_members) == 6  # 1 站 × 3 到達 × 2 沉降
    # 整個表的粒子數在兩個流場區域的排程單元之間沒有遺漏或重複。
    assert sum(u.particle_count for u in domain_units) == len(table)


def test_plan_native_hour_major_work_units_input_hash_changes_when_any_member_changes() -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)
    domain_units_before, _ = orch.plan_native_hour_major_work_units(units)
    hash_before = {u.flow_domain_id: u.input_hash for u in domain_units_before}

    # 第 0 列粒子屬於區域 A（見 ``_synthetic_plan_table`` 先建區域 A 再建區域 B），
    # 只動它的起點，模擬「只有流場區域 A 底下某個成員單元的輸入變了」。``content_
    # sha256`` 是凍結資料類別在建構當下（``__post_init__``）算好的，事後直接改
    # ``table.x0_m`` 陣列內容不會讓它重新計算，因此必須用 ``dataclasses.replace``
    # 重新走一次建構流程，而不是原地修改陣列。
    assert table.analysis_region_id[0] == "A"
    modified_x0 = table.x0_m.copy()
    modified_x0[0] += 1.0
    table2 = dataclasses.replace(table, x0_m=modified_x0)
    assert table2.content_sha256 != table.content_sha256
    units2 = orch.plan_native_work_units(table2, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)
    domain_units_after, _ = orch.plan_native_hour_major_work_units(units2)
    hash_after = {u.flow_domain_id: u.input_hash for u in domain_units_after}

    # 受影響流場區域（A）的摺疊雜湊必須改變，才能讓 run_native_plan 的續跑判斷
    # 正確地判定「這個流場區域的輸入變了，必須重算」，而不是誤判為已完成。注意
    # ``_unit_input_hash`` 本來就把整張身分表的 ``content_sha256`` 也混進每個
    # 單元的雜湊（見其實作），所以這裡連沒受影響的流場區域 B 雜湊也會一起變，
    # 這是既有 ``plan_native_work_units`` 的既有行為，不是本函式另外引入的差異，
    # 因此本測試不對 B 的雜湊做「維持不變」的斷言。
    assert hash_before["flowA_cache_v1"] != hash_after["flowA_cache_v1"]


def test_plan_native_hour_major_work_units_rejects_empty_input() -> None:
    with pytest.raises(orch.NativeOrchestratorError):
        orch.plan_native_hour_major_work_units([])


def test_native_plan_file_round_trips_hour_major_members(tmp_path: Path) -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)
    domain_units, members = orch.plan_native_hour_major_work_units(units)
    summary = orch.summarize_native_work_plan(domain_units)

    plan_path = tmp_path / "plan.json"
    orch.write_native_plan_file(
        plan_path, units=domain_units, gap_checks=(), summary=summary, hour_major_members=members
    )

    read_units, payload = orch.read_native_plan_file(plan_path)
    assert [u.unit_id for u in read_units] == [u.unit_id for u in domain_units]

    read_members = orch.read_native_hour_major_members(payload)
    assert set(read_members) == set(members)
    for domain_unit_id, member_units in members.items():
        restored = read_members[domain_unit_id]
        assert [m.unit_id for m in restored] == [m.unit_id for m in member_units]
        for original, restored_unit in zip(member_units, restored, strict=True):
            assert original.flow_domain_id == restored_unit.flow_domain_id
            assert original.arrival_time_utc_ns == restored_unit.arrival_time_utc_ns
            assert original.settling_velocity_mps == restored_unit.settling_velocity_mps
            assert original.study_site_id == restored_unit.study_site_id
            assert np.array_equal(original.particle_row_indices, restored_unit.particle_row_indices)
            assert original.input_hash == restored_unit.input_hash


def test_native_plan_file_without_hour_major_members_reads_back_empty(tmp_path: Path) -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)
    summary = orch.summarize_native_work_plan(units)

    plan_path = tmp_path / "plan.json"
    orch.write_native_plan_file(plan_path, units=units, gap_checks=(), summary=summary)

    _, payload = orch.read_native_plan_file(plan_path)
    assert orch.read_native_hour_major_members(payload) == {}


def test_summarize_native_work_plan_matches_manual_counts() -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)
    summary = orch.summarize_native_work_plan(units)
    assert summary.unit_count == 18
    assert summary.particle_count_total == 36
    assert summary.min_particles_per_unit == summary.max_particles_per_unit == 2
    # 3 個到達時刻在區域 A、B 各自成立一個局部性群組（不因區域內兩站而重複計數）。
    assert summary.locality_group_count == 6
    assert summary.region_ids == ("A", "B")


def test_estimate_formal_native_work_plan_matches_50000_times_m_design() -> None:
    for members, expected_total in ((1, 50_000), (10, 500_000)):
        summary = orch.estimate_formal_native_work_plan(members)
        assert summary.unit_count == 2500
        assert summary.particle_count_total == expected_total
        assert summary.mean_particles_per_unit == 20 * members


def test_plan_worker_schedule_keeps_locality_groups_together_and_balances_load() -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)
    schedule = orch.plan_worker_schedule(units, worker_count=3)

    unit_by_id = {u.unit_id: u for u in units}
    assigned_ids: list[str] = []
    for worker_ids in schedule.values():
        assigned_ids.extend(worker_ids)
    assert sorted(assigned_ids) == sorted(u.unit_id for u in units)

    # 同一個局部性群組（區域＋到達時刻）內的所有工作單元必須分到同一個 worker。
    locality_to_worker: dict[tuple[str, int], int] = {}
    for worker_index, unit_ids in schedule.items():
        for unit_id in unit_ids:
            unit = unit_by_id[unit_id]
            key = unit.locality_key
            if key in locality_to_worker:
                assert locality_to_worker[key] == worker_index, "同一到達時刻的工作單元被拆到不同 worker"
            else:
                locality_to_worker[key] = worker_index

    loads = [sum(unit_by_id[uid].particle_count for uid in ids) for ids in schedule.values()]
    assert max(loads) - min(loads) <= 4, f"三個 worker 的負載差距過大：{loads}"


def test_plan_worker_schedule_orders_units_by_region_then_arrival() -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)
    schedule = orch.plan_worker_schedule(units, worker_count=1)
    unit_by_id = {u.unit_id: u for u in units}
    ordered_keys = [
        (unit_by_id[uid].flow_domain_id, unit_by_id[uid].arrival_time_utc_ns) for uid in schedule[0]
    ]
    assert ordered_keys == sorted(ordered_keys)


def test_estimate_worker_count_respects_memory_and_cpu_bounds() -> None:
    assert orch.estimate_worker_count(available_memory_bytes=None, per_worker_memory_bytes=1) == 1
    assert (
        orch.estimate_worker_count(
            available_memory_bytes=10 * 1024**3,
            per_worker_memory_bytes=2 * 1024**3,
            cpu_count=100,
        )
        == 5
    )
    assert (
        orch.estimate_worker_count(
            available_memory_bytes=10 * 1024**3,
            per_worker_memory_bytes=2 * 1024**3,
            cpu_count=3,
        )
        == 3
    )
    assert (
        orch.estimate_worker_count(
            available_memory_bytes=1024,
            per_worker_memory_bytes=2 * 1024**3,
            min_workers=1,
        )
        == 1
    )


# ══════════════════════════════════════════════════════════════════════════
# 執行引擎：平行 vs 序列一致、續跑、失敗隔離、進度
# ══════════════════════════════════════════════════════════════════════════


def _toy_build_resources(worker_id: int) -> dict:
    return {"worker_id": worker_id}


def _toy_run_unit(resources: dict, spec: orch.NativeWorkUnitSpec) -> dict:
    """純函式：回傳工作單元列位置的總和與筆數，不依賴執行順序或 worker 身分。"""

    indices = spec.particle_row_indices
    return {
        "particle_count": int(indices.size),
        "index_checksum": int(indices.sum()),
        "settling_velocity_mps": spec.settling_velocity_mps,
    }


def _toy_flaky_run_unit(resources: dict, spec: orch.NativeWorkUnitSpec) -> dict:
    if spec.study_site_id == "site2":
        raise ValueError("模擬單一工作單元計算失敗")
    return _toy_run_unit(resources, spec)


def _toy_must_not_run(resources: dict, spec: orch.NativeWorkUnitSpec) -> dict:
    # pragma: no cover - 續跑測試預期這個函式不會被呼叫。
    raise AssertionError(f"續跑應該完全跳過 {spec.unit_id}，不應該重新呼叫 run_unit")


def test_run_native_plan_matches_sequential_execution(tmp_path: Path) -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)

    expected = {unit.unit_id: _toy_run_unit({}, unit.to_spec()) for unit in units}

    summary = orch.run_native_plan(
        units,
        worker_count=3,
        build_worker_resources=_toy_build_resources,
        run_unit=_toy_run_unit,
        output_dir=tmp_path,
    )

    assert summary.total_units == 18
    assert summary.completed_unit_count == 18
    assert summary.failed_unit_count == 0
    assert summary.skipped_unit_count == 0
    assert summary.all_succeeded

    for unit in units:
        result_path = tmp_path / f"{unit.unit_id}.result.json"
        assert result_path.is_file()
        import json

        actual = json.loads(result_path.read_text(encoding="utf-8"))
        assert actual == expected[unit.unit_id]


def test_run_native_plan_resume_skips_completed_units(tmp_path: Path) -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)

    first = orch.run_native_plan(
        units,
        worker_count=2,
        build_worker_resources=_toy_build_resources,
        run_unit=_toy_run_unit,
        output_dir=tmp_path,
    )
    assert first.completed_unit_count == 18

    second = orch.run_native_plan(
        units,
        worker_count=2,
        build_worker_resources=_toy_build_resources,
        run_unit=_toy_must_not_run,
        output_dir=tmp_path,
    )
    assert second.completed_unit_count == 0
    assert second.skipped_unit_count == 18
    assert second.failed_unit_count == 0
    assert second.all_succeeded


def test_run_native_plan_resume_recomputes_when_input_hash_changes(tmp_path: Path) -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)
    orch.run_native_plan(
        units,
        worker_count=2,
        build_worker_resources=_toy_build_resources,
        run_unit=_toy_run_unit,
        output_dir=tmp_path,
    )

    # 重新建一份「內容不同」的身分表（座標平移），使 content_sha256、進而每個
    # 工作單元的 input_hash 全部改變；續跑必須偵測到輸入已變動，全部重新計算。
    table2 = _synthetic_plan_table()
    rebuilt_rows = []
    for i in range(len(table2)):
        rebuilt_rows.append(
            {
                "study_site_id": str(table2.study_site_id[i]),
                "analysis_region_id": str(table2.analysis_region_id[i]),
                "arrival_time_utc_ns": int(table2.arrival_time_utc_ns[i]),
                "settling_velocity_mps": float(table2.settling_velocity_mps[i]),
                "x0_m": float(table2.x0_m[i]) + 999.0,
            }
        )
    table2 = _build_table(rebuilt_rows)
    units2 = orch.plan_native_work_units(table2, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)
    assert {u.unit_id for u in units2} != {u.unit_id for u in units}, "unit_id 涵蓋輸入雜湊，改變後應不同"

    second = orch.run_native_plan(
        units2,
        worker_count=2,
        build_worker_resources=_toy_build_resources,
        run_unit=_toy_run_unit,
        output_dir=tmp_path,
    )
    assert second.completed_unit_count == 18
    assert second.skipped_unit_count == 0


def test_run_native_plan_isolates_single_unit_failure(tmp_path: Path) -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)

    summary = orch.run_native_plan(
        units,
        worker_count=2,
        build_worker_resources=_toy_build_resources,
        run_unit=_toy_flaky_run_unit,
        output_dir=tmp_path,
    )

    failing_units = [u for u in units if u.study_site_id == "site2"]
    ok_units = [u for u in units if u.study_site_id != "site2"]
    assert summary.failed_unit_count == len(failing_units)
    assert summary.completed_unit_count == len(ok_units)
    assert set(summary.failed_unit_ids) == {u.unit_id for u in failing_units}
    assert not summary.all_succeeded

    for unit in failing_units:
        marker = orch._load_existing_marker(tmp_path, unit.unit_id)
        assert marker is not None
        assert marker["status"] == "failed"
        assert marker["error"]  # 保留完整 traceback 字串，不吞掉錯誤
    for unit in ok_units:
        marker = orch._load_existing_marker(tmp_path, unit.unit_id)
        assert marker is not None and marker["status"] == "completed"


def test_run_native_plan_writes_progress_snapshot(tmp_path: Path) -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)
    orch.run_native_plan(
        units,
        worker_count=2,
        build_worker_resources=_toy_build_resources,
        run_unit=_toy_run_unit,
        output_dir=tmp_path,
    )
    snapshot = orch.read_progress_snapshot(tmp_path)
    assert snapshot is not None
    assert snapshot["total_units"] == 18
    assert snapshot["completed_units"] == 18
    assert snapshot["pending_units"] == 0


def test_run_native_plan_all_units_already_complete_returns_without_workers(tmp_path: Path) -> None:
    table = _synthetic_plan_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain=REGION_TO_FLOW_DOMAIN)
    orch.run_native_plan(
        units,
        worker_count=2,
        build_worker_resources=_toy_build_resources,
        run_unit=_toy_run_unit,
        output_dir=tmp_path,
    )
    summary = orch.run_native_plan(
        units,
        worker_count=2,
        build_worker_resources=_toy_must_not_run,  # 連 build_worker_resources 都不該被呼叫
        run_unit=_toy_must_not_run,
        output_dir=tmp_path,
    )
    assert summary.skipped_unit_count == 18
    assert summary.worker_reports == ()


# ══════════════════════════════════════════════════════════════════════════
# 缺口安全的到達時刻
# ══════════════════════════════════════════════════════════════════════════


def _gap_report(
    *, flow_id: str, expected_start: int, expected_end: int, gaps: list[tuple[int, int]]
) -> ForcingGapInventoryReport:
    """建立最小可用的缺口盤點報告；``gaps`` 是 ``(起始小時, 結束小時)`` 的 UTC 奈秒區間。"""

    gap_specs = []
    for index, (start, end) in enumerate(gaps):
        missing = np.arange(start, end + _HOUR_NS, _HOUR_NS, dtype=np.int64)
        gap_specs.append(
            GapSpec(
                gap_id=f"{flow_id}-gap-{index}",
                missing_times_ns=missing,
                left_time_ns=int(missing[0] - _HOUR_NS),
                right_time_ns=int(missing[-1] + _HOUR_NS),
            )
        )
    return ForcingGapInventoryReport(
        flow_id=flow_id,
        expected_start_utc_ns=expected_start,
        expected_end_utc_ns=expected_end,
        expected_hour_count=int((expected_end - expected_start) // _HOUR_NS) + 1,
        observed_hour_count=0,
        reconstructed_hour_count=0,
        unregistered_gaps=tuple(gap_specs),
    )


def test_check_arrival_gap_safety_finds_overlap_and_reports() -> None:
    max_backtrack_seconds = 5 * 86400.0
    safe_arrival = 20 * _DAY_NS
    unsafe_arrival = 30 * _DAY_NS  # 回溯窗涵蓋缺口
    table = _build_table(
        [
            {
                "study_site_id": "site1",
                "analysis_region_id": "A",
                "arrival_time_utc_ns": safe_arrival,
                "settling_velocity_mps": -0.001,
            },
            {
                "study_site_id": "site1",
                "analysis_region_id": "A",
                "arrival_time_utc_ns": unsafe_arrival,
                "settling_velocity_mps": -0.001,
            },
        ]
    )
    gap_report = _gap_report(
        flow_id="A",
        expected_start=0,
        expected_end=60 * _DAY_NS,
        gaps=[(29 * _DAY_NS, 29 * _DAY_NS + 3 * _HOUR_NS)],
    )
    checks = orch.check_arrival_gap_safety(
        table,
        gap_reports_by_region={"A": gap_report},
        max_backtrack_seconds=max_backtrack_seconds,
    )
    by_arrival = {c.arrival_time_utc_ns: c for c in checks}
    assert by_arrival[safe_arrival].is_gap_safe
    assert not by_arrival[unsafe_arrival].is_gap_safe
    assert by_arrival[unsafe_arrival].overlapping_gap_ids == ("A-gap-0",)
    assert by_arrival[unsafe_arrival].reason == "backward_window_overlaps_unregistered_gap"


def test_check_arrival_gap_safety_reports_missing_region_report() -> None:
    table = _build_table(
        [
            {
                "study_site_id": "site3",
                "analysis_region_id": "B",
                "arrival_time_utc_ns": 20 * _DAY_NS,
                "settling_velocity_mps": -0.001,
            }
        ]
    )
    checks = orch.check_arrival_gap_safety(table, gap_reports_by_region={})
    assert len(checks) == 1
    assert not checks[0].is_gap_safe
    assert checks[0].reason == "no_gap_report_for_region"


def _synthetic_arrival_candidate_series(days: int = 730, seed: int = 0) -> dict[str, np.ndarray]:
    """建立足夠長、具備潮位與波高變化的合成候選序列，供 ``select_arrival_times`` 使用。"""

    rng = np.random.default_rng(seed)
    hours = days * 24
    # 從 unix epoch（1970-01-01 00:00 UTC）起算、不多加一小時位移，確保
    # ``days`` 為 730（兩個非閏年）時序列剛好落在 1970、1971 兩個年份內，
    # 相容既有 ``select_arrival_times`` legacy 兩年份 policy 的硬性要求。
    time_utc_ns = np.arange(hours, dtype=np.int64) * _HOUR_NS
    t = np.arange(hours, dtype=np.float64)
    # 半日潮＋以「月」為週期調變振幅，製造有意義的 spring/neap 差異。
    semi_diurnal = np.sin(2 * np.pi * t / 12.42)
    spring_neap_envelope = 1.0 + 0.6 * np.sin(2 * np.pi * t / (24 * 14.77))
    elevation_m = semi_diurnal * spring_neap_envelope
    significant_wave_height_m = (
        0.5 + 0.3 * np.abs(np.sin(2 * np.pi * t / (24 * 5.0))) + 0.01 * rng.random(hours)
    )
    current_speed_mps = 0.2 + 0.15 * np.abs(semi_diurnal) + 0.01 * rng.random(hours)
    valid_forcing = np.ones(hours, dtype=bool)
    return {
        "time_utc_ns": time_utc_ns,
        "elevation_m": elevation_m,
        "significant_wave_height_m": significant_wave_height_m,
        "current_speed_mps": current_speed_mps,
        "valid_forcing": valid_forcing,
    }


def test_derive_backward_window_available_excludes_only_windows_touching_gap() -> None:
    series = _synthetic_arrival_candidate_series(days=100)
    gap_report = _gap_report(
        flow_id="A",
        expected_start=0,
        expected_end=int(series["time_utc_ns"][-1]),
        gaps=[(40 * _DAY_NS, 40 * _DAY_NS + 2 * _HOUR_NS)],
    )
    available = orch.derive_backward_window_available(
        series["time_utc_ns"], gap_report, max_backtrack_seconds=5 * 86400.0
    )
    # 到達時刻早於缺口 5 天以上、且回溯窗仍完全落在盤點涵蓋範圍內者，應保持可用；
    # 太早的到達時刻（回溯窗左端小於 expected_start）則正確地因為超出盤點範圍而
    # 不可用，不屬於本測試要驗證的「碰到缺口」情境。
    early = (series["time_utc_ns"] >= 6 * _DAY_NS) & (series["time_utc_ns"] <= 34 * _DAY_NS)
    assert np.any(early)
    assert np.all(available[early])
    # 到達時刻落在 [40 天, 45 天] 之間者，回溯窗會涵蓋缺口，必須被排除。
    touching = (series["time_utc_ns"] >= 40 * _DAY_NS) & (series["time_utc_ns"] <= 41 * _DAY_NS)
    assert np.any(touching)
    assert not np.any(available[touching])


def test_rebuild_gap_safe_arrival_times_reuses_selector_and_flags_degradation() -> None:
    series = _synthetic_arrival_candidate_series(days=730)
    no_gap_report = _gap_report(
        flow_id="A", expected_start=0, expected_end=int(series["time_utc_ns"][-1]), gaps=[]
    )
    inputs = orch.ArrivalReselectionInputs(
        time_utc_ns=series["time_utc_ns"],
        elevation_m=series["elevation_m"],
        significant_wave_height_m=series["significant_wave_height_m"],
        current_speed_mps=series["current_speed_mps"],
        valid_forcing=series["valid_forcing"],
        design_version="wp7_test_v1",
    )
    before = orch.rebuild_gap_safe_arrival_times(
        "site1", inputs, gap_report=no_gap_report, max_backtrack_seconds=5 * 86400.0
    )
    assert len(before) == 50
    assert all(isinstance(item, ArrivalTime) for item in before)

    # 挖掉季節中段一小段資料（模擬新發現的缺口），重新選點後應仍能重用既有邏輯
    # 成功選出 50 筆（候選序列仍遠大於最低門檻），但個別分層的實際選中時刻可能
    # 改變；用 diff 明確檢查分層計數是否劣化，而不是假裝「換一個時刻」沒有代價。
    moderate_gap_report = _gap_report(
        flow_id="A",
        expected_start=0,
        expected_end=int(series["time_utc_ns"][-1]),
        gaps=[(300 * _DAY_NS, 300 * _DAY_NS + 15 * 24 * _HOUR_NS)],
    )
    after = orch.rebuild_gap_safe_arrival_times(
        "site1", inputs, gap_report=moderate_gap_report, max_backtrack_seconds=5 * 86400.0
    )
    assert len(after) == 50
    diff = orch.diff_stratum_coverage(before, after)
    # 不斷言一定劣化或一定不劣化（取決於合成資料的分層分布與缺口位置），只確認
    # 函式本身能在真實資料上跑完、回傳型別正確，且「劣化」判定與計數彼此一致。
    assert isinstance(diff.is_degraded, bool)
    if diff.is_degraded:
        assert all(diff.after_counts.get(key, 0) == 0 for key in diff.lost_strata)


def test_diff_stratum_coverage_flags_stratum_that_disappears() -> None:
    """純函式單元測試：合成兩組到達時刻，直接驗證「分層消失」會被明確標記。"""

    def _arrival(season: str, tide_class: str, utc_ns: int) -> ArrivalTime:
        return ArrivalTime(
            arrival_time_id=f"arr-{utc_ns}",
            study_site_id="site1",
            time_utc_ns=utc_ns,
            year=1970,
            season=season,
            tide_class=tide_class,
            phase_or_event="fastest_rising",
            metadata={},
        )

    before = [
        _arrival("DJF", "spring_proxy", 1),
        _arrival("JJA", "neap_proxy", 2),
    ]
    after = [
        _arrival("DJF", "spring_proxy", 3),
        # JJA/neap_proxy 完全消失，改由另一個分層頂替（例如替換到別的季節）。
        _arrival("MAM", "neap_proxy", 4),
    ]
    diff = orch.diff_stratum_coverage(before, after)
    assert diff.is_degraded
    assert diff.lost_strata == (("JJA", "neap_proxy"),)
    assert diff.before_counts[("JJA", "neap_proxy")] == 1
    assert diff.after_counts.get(("JJA", "neap_proxy"), 0) == 0


def test_rebuild_gap_safe_arrival_times_raises_when_too_few_candidates_remain() -> None:
    series = _synthetic_arrival_candidate_series(days=40)
    inputs = orch.ArrivalReselectionInputs(
        time_utc_ns=series["time_utc_ns"],
        elevation_m=series["elevation_m"],
        significant_wave_height_m=series["significant_wave_height_m"],
        current_speed_mps=series["current_speed_mps"],
        valid_forcing=series["valid_forcing"],
        design_version="wp7_test_v1",
    )
    # 缺口涵蓋幾乎整段序列，只留下遠少於 50 筆的可用回溯窗；既有
    # ``select_arrival_times`` 應該照原行為直接失敗，本模組不吞掉這個錯誤。
    tiny_gap_report = _gap_report(
        flow_id="A", expected_start=0, expected_end=int(series["time_utc_ns"][-1]), gaps=[(1, 35 * _DAY_NS)]
    )
    with pytest.raises(ValueError):
        orch.rebuild_gap_safe_arrival_times(
            "site1", inputs, gap_report=tiny_gap_report, max_backtrack_seconds=1 * 86400.0
        )


# ══════════════════════════════════════════════════════════════════════════
# 端到端：真正的 run_backward_batch ＋合成網格／流場，兩個 worker 平行執行
# ══════════════════════════════════════════════════════════════════════════

_E2E_NX = 4
_E2E_NY = 4
_E2E_SPACING_M = 500.0
_E2E_NVRT = 2
_E2E_DP_M = 50.0


def _e2e_node_index(i: int, j: int) -> int:
    return j * _E2E_NX + i


def _e2e_build_mesh():
    from lagrangian_backtracking.mesh import NativeMesh

    n_nodes = _E2E_NX * _E2E_NY
    node_xy = np.zeros((n_nodes, 2), dtype=np.float64)
    for j in range(_E2E_NY):
        for i in range(_E2E_NX):
            node_xy[_e2e_node_index(i, j)] = (i * _E2E_SPACING_M, j * _E2E_SPACING_M)
    dp = np.full(n_nodes, _E2E_DP_M, dtype=np.float64)
    triangles: list[tuple[int, int, int]] = []
    for j in range(_E2E_NY - 1):
        for i in range(_E2E_NX - 1):
            n00 = _e2e_node_index(i, j)
            n10 = _e2e_node_index(i + 1, j)
            n11 = _e2e_node_index(i + 1, j + 1)
            n01 = _e2e_node_index(i, j + 1)
            triangles.append((n00, n10, n11))
            triangles.append((n00, n11, n01))
    n_faces = len(triangles)
    face_nodes_local = np.full((n_faces, 4), -1, dtype=np.int64)
    for face_index, nodes in enumerate(triangles):
        face_nodes_local[face_index, :3] = nodes
    face_node_count = np.full(n_faces, 3, dtype=np.int64)
    return NativeMesh(
        node_lon=np.linspace(120.0, 120.01, n_nodes),
        node_lat=np.linspace(24.0, 24.01, n_nodes),
        node_xy=node_xy,
        source_depth_m=dp,
        source_node_bottom_index=np.zeros(n_nodes, dtype=np.int64),
        face_nodes_local=face_nodes_local,
        face_node_count=face_node_count,
        source_face_global_index=np.arange(n_faces, dtype=np.int64),
        bin_size_m=1000.0,
    )


def _e2e_particle_table() -> ParticleIdentityTable:
    """兩個到達時刻、一個沉降速度、一個站別／區域，每個到達時刻 3 顆粒子。"""

    domain_x = (_E2E_NX - 1) * _E2E_SPACING_M
    domain_y = (_E2E_NY - 1) * _E2E_SPACING_M
    center_x, center_y = domain_x / 2.0, domain_y / 2.0
    rows: list[dict] = []
    for arrival_offset_hours in (0, 1):
        arrival = (100 + arrival_offset_hours) * _HOUR_NS
        for offset in (-50.0, 0.0, 50.0):
            rows.append(
                {
                    "study_site_id": "e2e_site",
                    "analysis_region_id": "E2E",
                    "arrival_time_utc_ns": arrival,
                    "settling_velocity_mps": -0.0005,
                    "x0_m": center_x + offset,
                    "y0_m": center_y,
                    "z0_m": -5.0,
                }
            )
    return _build_table(rows)


def _e2e_resolve_baytrace_root() -> str | None:
    return os.environ.get("LBT_BAYTRACE_ROOT")


def _e2e_build_worker_resources(worker_id: int) -> dict:
    from lagrangian_backtracking.native_runner import backward_driver as bd
    from lagrangian_backtracking.native_runner import build_native_grid_arrays

    root = _e2e_resolve_baytrace_root()
    if not root:
        raise RuntimeError("需要 LBT_BAYTRACE_ROOT 才能建立端到端 worker 資源")
    handles = bd.load_baytrace_modules(root)
    mesh = _e2e_build_mesh()
    n_nodes = mesh.node_xy.shape[0]
    zcor_reference = np.zeros((n_nodes, _E2E_NVRT), dtype=np.float64)
    zcor_reference[:, 0] = -_E2E_DP_M
    zcor_reference[:, 1] = 0.0
    eta_reference = np.zeros(n_nodes, dtype=np.float64)
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor_reference, eta_reference=eta_reference)
    driver_grid = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)
    domain_x = (_E2E_NX - 1) * _E2E_SPACING_M
    domain_y = (_E2E_NY - 1) * _E2E_SPACING_M
    from shapely.geometry import box

    interest_region_polygon = box(-1.0, -1.0, domain_x + 1.0, domain_y + 1.0)
    return {
        "handles": handles,
        "driver_grid": driver_grid,
        "interest_region_polygon": interest_region_polygon,
    }


class _E2ESoa:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, row_particle_index: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.z0 = z
        self.row_particle_index = row_particle_index


def _e2e_run_unit(resources: dict, spec: orch.NativeWorkUnitSpec) -> dict:
    from lagrangian_backtracking.native_runner import backward_driver as bd
    from lagrangian_backtracking.native_runner.forcing_bridge import HourlyFieldPair

    handles = resources["handles"]
    driver_grid = resources["driver_grid"]
    interest_region_polygon = resources["interest_region_polygon"]

    table = _e2e_particle_table()
    indices = spec.particle_row_indices
    soa = _E2ESoa(
        x=table.x0_m[indices].copy(),
        y=table.y0_m[indices].copy(),
        z=table.z0_m[indices].copy(),
        row_particle_index=indices,
    )
    n_nodes = int(driver_grid.grid.grid_x.shape[0])
    field_shape = (_E2E_NVRT, n_nodes)
    zero = np.zeros(field_shape, dtype=np.float64)
    eta = np.zeros(n_nodes, dtype=np.float64)
    diffusivity = np.full(field_shape, 1.0e-4, dtype=np.float64)
    missing = np.zeros(field_shape, dtype=bool)
    dry_face = np.zeros(field_shape, dtype=bool)

    def _forcing_resolver(target_utc_ns: int) -> HourlyFieldPair:
        return HourlyFieldPair(
            time_before_utc_ns=target_utc_ns,
            time_after_utc_ns=target_utc_ns - _HOUR_NS,
            trat=0.0,
            ibf=-1,
            u_field1=zero.copy(),
            v_field1=zero.copy(),
            w_field1=zero.copy(),
            eta_field1=eta.copy(),
            diffusivity_field1=diffusivity.copy(),
            missing_mask1=missing.copy(),
            dry_face_flag1=dry_face.copy(),
            month_id1="synthetic",
            time_origin1="test",
            u_field2=zero.copy(),
            v_field2=zero.copy(),
            w_field2=zero.copy(),
            eta_field2=eta.copy(),
            diffusivity_field2=diffusivity.copy(),
            missing_mask2=missing.copy(),
            dry_face_flag2=dry_face.copy(),
            month_id2="synthetic",
            time_origin2="test",
        )

    config = bd.BackwardDriverConfig(
        arrival_time_utc_ns=int(spec.arrival_time_utc_ns),
        settling_velocity_mps=float(spec.settling_velocity_mps),
        max_backtrack_seconds=2.0 * 3600.0,
        enable_diffusion=False,
        master_seed=20260921,
    )
    seed_hex_values = [format(int(row), "032x") for row in indices]
    result = bd.run_backward_batch(
        driver_grid,
        handles,
        soa,
        seed_hex_values=seed_hex_values,
        config=config,
        forcing_resolver=_forcing_resolver,
        interest_region_polygon=interest_region_polygon,
    )
    status_counts: dict[str, int] = {}
    for status in result.final_status.tolist():
        key = str(int(status))
        status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "particle_count": int(indices.size),
        "hours_run": int(result.hours_run),
        "final_status_counts": status_counts,
    }


def test_end_to_end_run_backward_batch_via_orchestrator_two_workers(tmp_path: Path) -> None:
    if not _e2e_resolve_baytrace_root():
        pytest.skip("需要 LBT_BAYTRACE_ROOT 指向 BayTrace 原始碼根目錄才能執行端到端測試")

    table = _e2e_particle_table()
    units = orch.plan_native_work_units(table, region_to_flow_domain={"E2E": "e2e_flow_v1"})
    assert len(units) == 2  # 兩個到達時刻各成一個工作單元（同站、同沉降速度）

    summary = orch.run_native_plan(
        units,
        worker_count=2,
        build_worker_resources=_e2e_build_worker_resources,
        run_unit=_e2e_run_unit,
        output_dir=tmp_path,
    )

    assert summary.completed_unit_count == 2
    assert summary.failed_unit_count == 0, summary.failed_unit_ids

    import json

    for unit in units:
        payload = json.loads((tmp_path / f"{unit.unit_id}.result.json").read_text(encoding="utf-8"))
        assert payload["particle_count"] == 3
        assert payload["hours_run"] >= 1
        assert sum(payload["final_status_counts"].values()) == 3
