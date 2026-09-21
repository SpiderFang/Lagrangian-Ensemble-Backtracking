"""WP-4 情境矩陣 → BayTrace 粒子批次輸入（``native_runner.particle_input``）驗收測試。

測試涵蓋：
1. 合成小情境矩陣的粒子數、身分欄位完整性、``particle_id`` 無重複。
2. 同設定重跑，內容雜湊（``content_sha256``）完全相同（可重現性）。
3. SoA 陣列的 shape、dtype、contiguity 與 BayTrace kernel 要求一致。
4. 依到達時刻、依沉降速度、依兩者同時分批：批次內同質、聯集覆蓋全體且無重疊。
5. 沉降速度正確對應到每顆粒子（依 ``material_id`` 回查）。
6. ``.npz`` 保存／讀回後內容雜湊一致，且能偵測竄改。
7. 規模測試：五站 × 10 沉降速度 × 20 受體 × 50 到達時刻，M=1（50,000 顆）與
   M=10（500,000 顆）的建構耗時與記憶體用量。
"""

from __future__ import annotations

import resource
import time
from pathlib import Path

import numpy as np
import pytest

from lagrangian_backtracking.native_runner import (
    ParticleInputError,
    build_baytrace_particle_soa,
    build_particle_identity_table,
    group_indices_by_arrival_time,
    group_indices_by_settling_velocity,
    group_indices_by_settling_velocity_and_arrival_time,
    load_particle_identity_table_npz,
    save_particle_identity_table_npz,
    summarize_batch_layout,
    validate_arrival_time_homogeneous_batches,
    validate_batch_partition,
    validate_settling_velocity_homogeneous_batches,
)
from lagrangian_backtracking.runner import RunUnit, plan_scenario_shards
from lagrangian_backtracking.scenarios import (
    BASELINE_BEHAVIORS,
    ArrivalTime,
    Behavior,
    Receptor,
    build_scenarios,
)

_BASE_UTC_NS = 1_700_000_000_000_000_000  # 任意固定的 UTC 奈秒起點，只求可重現


def _synthetic_receptors(site_id: str, region_id: str, count: int) -> list[Receptor]:
    """建立一站、指定數量的合成受體，座標與內容只求彼此互異、可重現。"""

    return [
        Receptor(
            receptor_id=f"rcp_{site_id}_{index}",
            study_site_id=site_id,
            analysis_region_id=region_id,
            lon=120.0 + index * 0.001,
            lat=23.0 + index * 0.001,
            z_m_positive_up=-1.0 - index,
            vertical_id=f"v{index}",
            metadata={},
        )
        for index in range(count)
    ]


def _synthetic_arrival_times(site_id: str, site_offset: int, count: int) -> list[ArrivalTime]:
    """建立一站、指定數量的合成到達時刻；不同站點以 ``site_offset`` 錯開，
    確保跨站點到達時刻不重複，方便觀察分批切分行為。
    """

    return [
        ArrivalTime(
            arrival_time_id=f"arr_{site_id}_{index}",
            study_site_id=site_id,
            time_utc_ns=_BASE_UTC_NS + site_offset * 10_000_000_000_000 + index * 3_600_000_000_000,
            year=2025,
            season="synthetic",
            tide_class="synthetic",
            phase_or_event="synthetic",
            metadata={},
        )
        for index in range(count)
    ]


def _resolve_initial_position(unit: RunUnit) -> tuple[float, float, float]:
    """確定性的合成初始位置解析器，模擬真正的受體×到達時刻位置查詢。

    只依 ``receptor_id`` 與 ``arrival_time_id`` 決定位置（與 member_id 無關，
    因為同一情境的所有系集成員理論上共用同一個逆向追蹤起點，差異只在後續
    隨機擴散），刻意不依賴任何全域可變狀態，確保重跑可重現。
    """

    scenario = unit.scenario
    receptor_hash = abs(hash((scenario.receptor_id, "x"))) % 100_000
    arrival_hash = abs(hash((scenario.arrival_time_id, "y"))) % 100_000
    x_m = 300_000.0 + receptor_hash * 0.1
    y_m = 2_500_000.0 + arrival_hash * 0.1
    z_m = -5.0 - (receptor_hash % 50) * 0.5
    return x_m, y_m, z_m


def _build_synthetic_matrix(
    *, site_count: int, receptor_count: int, arrival_count: int, members_per_scenario: int
):
    """建立合成情境矩陣並展開成 shard 序列，回傳 (shards, scenarios)。"""

    behaviors: tuple[Behavior, ...] = BASELINE_BEHAVIORS[: 2]
    site_ids = [f"site{i}" for i in range(site_count)]
    receptors = []
    arrivals = []
    for site_index, site_id in enumerate(site_ids):
        region_id = "A" if site_index % 2 == 0 else "B"
        receptors.extend(_synthetic_receptors(site_id, region_id, receptor_count))
        arrivals.extend(_synthetic_arrival_times(site_id, site_index, arrival_count))

    scenarios = build_scenarios(
        behaviors=behaviors,
        receptors=receptors,
        arrival_times=arrivals,
        design_version="test_design_v1",
    )
    shards = plan_scenario_shards(
        scenarios,
        members_per_scenario=members_per_scenario,
        shard_scenario_count=10_000,
        experiment_case_id="test_case_no_stokes",
    )
    return shards, scenarios


# ── 1. 合成小情境矩陣：粒子數、身分完整性、particle_id 無重複 ──────────────


def test_small_matrix_identity_table_shape_and_uniqueness() -> None:
    site_count, receptor_count, arrival_count, materials, members = 2, 3, 2, 2, 2
    shards, scenarios = _build_synthetic_matrix(
        site_count=site_count,
        receptor_count=receptor_count,
        arrival_count=arrival_count,
        members_per_scenario=members,
    )
    expected_scenario_count = site_count * materials * receptor_count * arrival_count
    assert len(scenarios) == expected_scenario_count

    table = build_particle_identity_table(
        shards, master_seed=12345, resolve_initial_position=_resolve_initial_position
    )
    expected_particle_count = expected_scenario_count * members
    assert len(table) == expected_particle_count
    assert table.particle_id.shape[0] == expected_particle_count

    # particle_id 全表唯一（__post_init__ 已驗證過，這裡再從外部確認一次）。
    assert np.unique(table.particle_id).size == expected_particle_count

    # 身分欄位不得有空字串或缺值。
    for name in (
        "particle_id", "scenario_id", "study_site_id", "analysis_region_id",
        "receptor_id", "material_id", "arrival_time_id", "seed_hex",
    ):
        array = getattr(table, name)
        assert np.all(array != ""), f"{name} 含空字串"

    # 每個站點的 receptor_id 集合應恰為該站合成受體的 ID。
    site0_mask = table.study_site_id == "site0"
    assert set(table.receptor_id[site0_mask].tolist()) == {
        f"rcp_site0_{i}" for i in range(receptor_count)
    }

    # 每個 scenario 恰好對應 members 顆粒子。
    unique_scenarios, counts = np.unique(table.scenario_id, return_counts=True)
    assert unique_scenarios.size == expected_scenario_count
    assert np.all(counts == members)

    # member_id 在每個 scenario 內部應恰為 0..members-1。
    for scenario_id in unique_scenarios[:5]:
        member_ids = sorted(table.member_id[table.scenario_id == scenario_id].tolist())
        assert member_ids == list(range(members))


def test_missing_or_empty_shards_rejected() -> None:
    with pytest.raises(ParticleInputError):
        build_particle_identity_table([], master_seed=1, resolve_initial_position=_resolve_initial_position)


def test_mixed_experiment_case_id_rejected() -> None:
    shards, _ = _build_synthetic_matrix(
        site_count=1, receptor_count=2, arrival_count=1, members_per_scenario=1
    )
    # 人為捏造一個不同 experiment_case_id 的 shard，混進同一次建構應被拒絕。
    tampered = shards[0]
    from dataclasses import replace

    other = replace(tampered, experiment_case_id="a_different_case")
    with pytest.raises(ParticleInputError):
        build_particle_identity_table(
            [*shards, other], master_seed=1, resolve_initial_position=_resolve_initial_position
        )


# ── 2. 可重現性：同設定重跑內容雜湊完全相同 ─────────────────────────────


def test_reproducible_content_hash_across_rebuilds() -> None:
    kwargs = dict(site_count=2, receptor_count=3, arrival_count=2, members_per_scenario=3)
    shards_a, _ = _build_synthetic_matrix(**kwargs)
    shards_b, _ = _build_synthetic_matrix(**kwargs)

    table_a = build_particle_identity_table(
        shards_a, master_seed=999, resolve_initial_position=_resolve_initial_position
    )
    table_b = build_particle_identity_table(
        shards_b, master_seed=999, resolve_initial_position=_resolve_initial_position
    )
    assert table_a.content_sha256 == table_b.content_sha256
    assert table_a.content_sha256 != ""

    # 換一個 master_seed，雜湊必須不同（否則雜湊沒有真的涵蓋種子）。
    table_c = build_particle_identity_table(
        shards_a, master_seed=1000, resolve_initial_position=_resolve_initial_position
    )
    assert table_c.content_sha256 != table_a.content_sha256
    # 種子不同也必須反映在實際粒子 seed_hex 上（否則雜湊差異只是巧合）。
    assert not np.array_equal(table_a.seed_hex, table_c.seed_hex)


# ── 3. SoA 陣列 shape／dtype／contiguity ────────────────────────────────


def test_soa_conversion_matches_baytrace_kernel_dtypes() -> None:
    shards, _ = _build_synthetic_matrix(
        site_count=2, receptor_count=2, arrival_count=2, members_per_scenario=2
    )
    table = build_particle_identity_table(
        shards, master_seed=7, resolve_initial_position=_resolve_initial_position
    )
    soa = build_baytrace_particle_soa(table)

    assert len(soa) == len(table)
    for name in ("x", "y", "z", "z0", "settling_velocity_mps"):
        array = getattr(soa, name)
        assert array.dtype == np.float64
        assert array.flags["C_CONTIGUOUS"]
        assert array.shape == (len(table),)
    for name in ("elem", "status", "row_particle_index"):
        array = getattr(soa, name)
        assert array.dtype == np.int64
        assert array.flags["C_CONTIGUOUS"]

    assert np.all(soa.elem == -1)
    assert np.all(soa.status == 0)
    assert np.array_equal(soa.row_particle_index, np.arange(len(table), dtype=np.int64))
    assert np.array_equal(soa.x, table.x0_m)
    assert np.array_equal(soa.y, table.y0_m)
    assert np.array_equal(soa.z, table.z0_m)
    assert np.array_equal(soa.z0, table.z0_m)
    assert np.array_equal(soa.settling_velocity_mps, table.settling_velocity_mps)


def test_soa_rejects_non_sentinel_elem_or_status() -> None:
    shards, _ = _build_synthetic_matrix(
        site_count=1, receptor_count=2, arrival_count=1, members_per_scenario=1
    )
    table = build_particle_identity_table(
        shards, master_seed=1, resolve_initial_position=_resolve_initial_position
    )
    soa = build_baytrace_particle_soa(table)
    from dataclasses import replace

    with pytest.raises(ParticleInputError):
        replace(soa, elem=np.zeros_like(soa.elem))
    with pytest.raises(ParticleInputError):
        replace(soa, status=np.ones_like(soa.status))


# ── 4. 分批：到達時刻、沉降速度、兩者同時 ───────────────────────────────


def test_batching_by_arrival_time_is_homogeneous_and_covers_all() -> None:
    shards, _ = _build_synthetic_matrix(
        site_count=2, receptor_count=3, arrival_count=4, members_per_scenario=2
    )
    table = build_particle_identity_table(
        shards, master_seed=42, resolve_initial_position=_resolve_initial_position
    )
    groups = group_indices_by_arrival_time(table)

    validate_batch_partition(table, groups)
    validate_arrival_time_homogeneous_batches(table, groups)

    # 到達時刻總數：2 站 × 4 個到達時刻，且跨站不重複（合成資料刻意錯開）。
    assert len(groups) == 2 * 4
    total = sum(int(group.indices.size) for group in groups)
    assert total == len(table)


def test_batching_by_settling_velocity_is_homogeneous_and_covers_all() -> None:
    shards, _ = _build_synthetic_matrix(
        site_count=2, receptor_count=3, arrival_count=2, members_per_scenario=2
    )
    table = build_particle_identity_table(
        shards, master_seed=42, resolve_initial_position=_resolve_initial_position
    )
    groups = group_indices_by_settling_velocity(table)

    validate_batch_partition(table, groups)
    validate_settling_velocity_homogeneous_batches(table, groups)

    # 合成矩陣只用了 BASELINE_BEHAVIORS 前兩種材料，因此恰好兩個沉降速度分組。
    assert len(groups) == 2
    seen_velocities = {key_value for group in groups for key, key_value in group.key}
    assert seen_velocities == set(table.settling_velocity_mps.tolist())


def test_batching_by_settling_velocity_and_arrival_time_combined() -> None:
    shards, _ = _build_synthetic_matrix(
        site_count=2, receptor_count=3, arrival_count=2, members_per_scenario=2
    )
    table = build_particle_identity_table(
        shards, master_seed=42, resolve_initial_position=_resolve_initial_position
    )
    groups = group_indices_by_settling_velocity_and_arrival_time(table)

    validate_batch_partition(table, groups)
    validate_arrival_time_homogeneous_batches(table, groups)
    validate_settling_velocity_homogeneous_batches(table, groups)

    # 2 種沉降速度 × (2 站 × 2 到達時刻) = 8 組。
    assert len(groups) == 2 * 2 * 2

    report = summarize_batch_layout("settling_and_arrival", groups)
    assert report.batch_count == len(groups)
    assert report.particle_count_total == len(table)
    # 每組粒子數 = 3 個受體 × 2 個成員 = 6，因此各批粒子數應完全相同。
    assert report.min_particles_per_batch == report.max_particles_per_batch == 6


def test_homogeneity_validator_detects_deliberately_merged_batches() -> None:
    shards, _ = _build_synthetic_matrix(
        site_count=2, receptor_count=2, arrival_count=2, members_per_scenario=1
    )
    table = build_particle_identity_table(
        shards, master_seed=1, resolve_initial_position=_resolve_initial_position
    )
    groups = list(group_indices_by_arrival_time(table))
    assert len(groups) >= 2

    from dataclasses import replace

    merged_indices = np.sort(np.concatenate([groups[0].indices, groups[1].indices]))
    tampered_group = replace(groups[0], key=groups[0].key, indices=merged_indices)

    # 錯誤地把兩個不同到達時刻的批次合併成一批：批次內到達時刻不一致，
    # 同質性檢查必須攔截。
    with pytest.raises(ParticleInputError):
        validate_arrival_time_homogeneous_batches(table, [tampered_group, *groups[2:]])

    # 若合併批次與原本的 group[1] 同時出現在分批清單中（例如中途重試邏輯
    # 不慎重複附加），group[1] 的列會被重複計入兩個批次；聯集檢查必須攔截
    # 這種重疊，而不是只因為總數仍等於全體粒子數就誤判通過。
    with pytest.raises(ParticleInputError):
        validate_batch_partition(table, [tampered_group, groups[1], *groups[2:]])


# ── 5. 沉降速度正確對應到每顆粒子 ───────────────────────────────────────


def test_settling_velocity_matches_material_definition() -> None:
    shards, _ = _build_synthetic_matrix(
        site_count=1, receptor_count=2, arrival_count=1, members_per_scenario=1
    )
    table = build_particle_identity_table(
        shards, master_seed=1, resolve_initial_position=_resolve_initial_position
    )
    behaviors_by_id = {behavior.material_id: behavior for behavior in BASELINE_BEHAVIORS}
    for material_id in np.unique(table.material_id):
        expected = behaviors_by_id[str(material_id)].settling_velocity_mps
        actual = table.settling_velocity_mps[table.material_id == material_id]
        assert np.all(actual == expected)


# ── 6. .npz 保存／讀回：內容雜湊一致，且能偵測竄改 ─────────────────────


def test_npz_round_trip_preserves_content_hash(tmp_path: Path) -> None:
    shards, _ = _build_synthetic_matrix(
        site_count=2, receptor_count=2, arrival_count=2, members_per_scenario=2
    )
    table = build_particle_identity_table(
        shards, master_seed=55, resolve_initial_position=_resolve_initial_position
    )
    path = tmp_path / "identity_table.npz"
    save_particle_identity_table_npz(table, path)
    reloaded = load_particle_identity_table_npz(path)

    assert reloaded.content_sha256 == table.content_sha256
    assert np.array_equal(reloaded.particle_id, table.particle_id)
    assert np.array_equal(reloaded.x0_m, table.x0_m)
    assert reloaded.master_seed == table.master_seed
    assert reloaded.experiment_case_id == table.experiment_case_id


def test_npz_tampering_is_detected(tmp_path: Path) -> None:
    shards, _ = _build_synthetic_matrix(
        site_count=1, receptor_count=2, arrival_count=1, members_per_scenario=1
    )
    table = build_particle_identity_table(
        shards, master_seed=1, resolve_initial_position=_resolve_initial_position
    )
    path = tmp_path / "identity_table.npz"
    save_particle_identity_table_npz(table, path)

    # 直接竄改其中一個陣列欄位（把第一個 x0_m 加 1），模擬檔案損壞或被覆寫。
    with np.load(path, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}
    data["x0_m"] = data["x0_m"].copy()
    data["x0_m"][0] += 1.0
    np.savez(path, **data)

    with pytest.raises(ParticleInputError):
        load_particle_identity_table_npz(path)


# ── 7. 規模測試：M=1（50,000）與 M=10（500,000） ────────────────────────


def _build_full_baseline_shards(*, members_per_scenario: int):
    """依真實 5 站 × 10 沉降速度 × 20 受體 × 50 到達時刻建立 shard 序列。

    這裡的站點、受體、到達時刻本身是合成資料（不連到任何真實 OCM mesh 或
    NetCDF 檔案），但材料行為（``BASELINE_BEHAVIORS``）與矩陣規模完全比照
    正式設計，用來驗證 WP-4 在正式規模下的建構效能與記憶體用量。
    """

    site_ids = ["gongliao", "guishan", "hsinchu", "nanwan", "lienchiang"]
    receptors = []
    arrivals = []
    for site_index, site_id in enumerate(site_ids):
        region_id = "A" if site_id in ("gongliao", "guishan") else site_id[0].upper()
        receptors.extend(_synthetic_receptors(site_id, region_id, 20))
        arrivals.extend(_synthetic_arrival_times(site_id, site_index, 50))

    scenarios = build_scenarios(
        behaviors=BASELINE_BEHAVIORS,
        receptors=receptors,
        arrival_times=arrivals,
        design_version="scale_test_v1",
    )
    assert len(scenarios) == 50_000
    shards = plan_scenario_shards(
        scenarios,
        members_per_scenario=members_per_scenario,
        shard_scenario_count=10_000,
        experiment_case_id="scale_test_case",
    )
    return shards


@pytest.mark.parametrize("members_per_scenario,expected_total", [(1, 50_000), (10, 500_000)])
def test_scale_build_timing_and_memory(members_per_scenario: int, expected_total: int) -> None:
    shards = _build_full_baseline_shards(members_per_scenario=members_per_scenario)

    # 用 ``resource.getrusage`` 的最大常駐記憶體（RSS）量測，而不是
    # ``tracemalloc``：tracemalloc 會替每一次記憶體配置記錄呼叫堆疊，在
    # 50 萬列、數百萬次小物件配置（字串、tuple、dict）的規模下，量測本身
    # 的額外開銷會嚴重扭曲耗時數字（實測可讓總時間膨脹超過一倍），不適合
    # 用來回報「這個建構步驟實際要跑多久」。RSS 的缺點是包含程序其餘部分
    # （例如已載入模組）的記憶體，因此只能當作「這次呼叫期間常駐記憶體增量」
    # 的近似值，不是精確的物件配置量；陣列位元組數（``array_nbytes``）才是
    # 精確值，兩者一起回報。
    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    table = build_particle_identity_table(
        shards, master_seed=20260916, resolve_initial_position=_resolve_initial_position
    )
    elapsed_s = time.perf_counter() - start
    rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert len(table) == expected_total

    array_nbytes = sum(
        getattr(table, name).nbytes
        for name in (
            "particle_id", "scenario_id", "member_id", "study_site_id",
            "analysis_region_id", "receptor_id", "material_id", "arrival_time_id",
            "settling_velocity_mps", "arrival_time_utc_ns", "seed_hex",
            "x0_m", "y0_m", "z0_m",
        )
    )

    soa_start = time.perf_counter()
    soa = build_baytrace_particle_soa(table)
    soa_elapsed_s = time.perf_counter() - soa_start
    assert len(soa) == expected_total

    print(
        f"\n[規模測試] members_per_scenario={members_per_scenario} "
        f"particles={expected_total} "
        f"identity_build={elapsed_s:.3f}s "
        f"soa_build={soa_elapsed_s:.3f}s "
        f"identity_array_bytes={array_nbytes} ({array_nbytes / 1e6:.2f} MB) "
        f"peak_rss_after={rss_after_kb / 1e3:.2f} MB "
        f"rss_delta≈{(rss_after_kb - rss_before_kb) / 1e3:.2f} MB"
    )

    groups = group_indices_by_settling_velocity_and_arrival_time(table)
    validate_batch_partition(table, groups)
    validate_arrival_time_homogeneous_batches(table, groups)
    validate_settling_velocity_homogeneous_batches(table, groups)
    report = summarize_batch_layout("settling_and_arrival_full_scale", groups)
    print(
        f"[分批統計] scheme={report.scheme_id} batch_count={report.batch_count} "
        f"min={report.min_particles_per_batch} max={report.max_particles_per_batch} "
        f"mean={report.mean_particles_per_batch:.2f}"
    )
