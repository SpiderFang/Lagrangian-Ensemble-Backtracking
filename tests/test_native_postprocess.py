"""WP-6 後處理模組（``native_runner.postprocess``）的驗收測試。

測試策略
--------
本檔案完全不呼叫任何 BayTrace 原始碼、不需要 ``LBT_BAYTRACE_ROOT``：WP-6 的
輸入是 ``BackwardDriverResult``（WP-5 的輸出資料類別），因此測試直接手造
合成的 ``BackwardDriverResult`` 與對應的 ``ParticleIdentityTable``，驗證
``postprocess`` 模組自己的邏輯（串流合併、離開紀錄表、邊界通量統計、條件式
來源足跡、軌跡抽樣、manifest、一致性檢查），不重新驗證 WP-5 驅動層本身的
物理正確性（那是 ``tests/test_native_backward_driver.py`` 的範圍）。
"""

from __future__ import annotations

import resource

import numpy as np
import pyarrow.parquet as pq
import pytest

from lagrangian_backtracking.native_runner.backward_driver import (
    NATIVE_STATUS_ACTIVE,
    NATIVE_STATUS_DEPOSITED,
    NATIVE_STATUS_FORCING_MISSING,
    NATIVE_STATUS_LEFT_INTEREST_REGION,
    NATIVE_STATUS_MAX_BACKTRACK,
    NATIVE_STATUS_NUMERICAL_FAILURE,
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION,
    NATIVE_STATUS_STRANDED,
    BackwardDriverResult,
    DepartureRecord,
    HourlyOverheadDiagnostics,
    SubstepDiagnostics,
)
from lagrangian_backtracking.native_runner.particle_input import ParticleIdentityTable
from lagrangian_backtracking.native_runner.postprocess import (
    KDE_STATUS_AVAILABLE,
    KDE_STATUS_COLLINEAR_POINTS,
    KDE_STATUS_INSUFFICIENT_POINTS,
    PostprocessAccumulator,
    PostprocessConfig,
    PostprocessError,
    build_source_footprint_kde,
    compute_batch_result_fingerprint,
)

_HOUR_NS = 3_600_000_000_000
_DAY_NS = 24 * _HOUR_NS

# 四個固定落在不同季節（DJF/MAM/JJA/SON）的到達 UTC（2024 年，奈秒）。
_JAN_2024_UTC_NS = 1_704_110_400_000_000_000  # 2024-01-01T12:00:00Z -> DJF
_APR_2024_UTC_NS = 1_712_059_200_000_000_000  # 2024-04-02T12:00:00Z -> MAM
_JUL_2024_UTC_NS = 1_719_921_600_000_000_000  # 2024-07-02T12:00:00Z -> JJA
_OCT_2024_UTC_NS = 1_727_870_400_000_000_000  # 2024-10-02T12:00:00Z -> SON


def _make_identity_table(
    *,
    particle_ids: list[str],
    study_site_id: list[str],
    settling_velocity_mps: list[float],
    arrival_time_utc_ns: list[int],
    z0_m: list[float] | None = None,
    experiment_case_id: str = "case-001",
    master_seed: int = 42,
) -> ParticleIdentityTable:
    """建立測試用的合成粒子身分表，欄位語意與 particle_input.py 完全一致。"""

    n = len(particle_ids)
    if z0_m is None:
        z0_m = [-1.0] * n
    return ParticleIdentityTable(
        particle_id=np.asarray(particle_ids, dtype="<U64"),
        scenario_id=np.asarray([f"scn-{i}" for i in range(n)], dtype="<U64"),
        member_id=np.zeros(n, dtype=np.int32),
        study_site_id=np.asarray(study_site_id, dtype="<U64"),
        analysis_region_id=np.asarray(["region-a"] * n, dtype="<U64"),
        receptor_id=np.asarray([f"recv-{i % 5}" for i in range(n)], dtype="<U64"),
        material_id=np.asarray(["microplastic"] * n, dtype="<U64"),
        arrival_time_id=np.asarray([f"arrival-{i}" for i in range(n)], dtype="<U64"),
        settling_velocity_mps=np.asarray(settling_velocity_mps, dtype=np.float64),
        arrival_time_utc_ns=np.asarray(arrival_time_utc_ns, dtype=np.int64),
        seed_hex=np.asarray([f"{i:032x}" for i in range(n)], dtype="<U32"),
        x0_m=np.zeros(n, dtype=np.float64),
        y0_m=np.zeros(n, dtype=np.float64),
        z0_m=np.asarray(z0_m, dtype=np.float64),
        experiment_case_id=experiment_case_id,
        master_seed=master_seed,
        random_stream_id=None,
    )


def _make_result(
    *,
    rows: list[int],
    final_status: list[int],
    final_x_m: list[float] | None = None,
    final_y_m: list[float] | None = None,
    final_z_m: list[float] | None = None,
    final_utc_ns: list[int] | None = None,
    departures: list[DepartureRecord] | None = None,
    n_samples: int = 2,
) -> BackwardDriverResult:
    """建立測試用的合成 ``BackwardDriverResult``，departures 依 final_status 自動配對。"""

    n = len(rows)
    if final_x_m is None:
        final_x_m = [float(1000 * r) for r in rows]
    if final_y_m is None:
        final_y_m = [float(2000 * r) for r in rows]
    if final_z_m is None:
        final_z_m = [-5.0] * n
    if final_utc_ns is None:
        final_utc_ns = [_JAN_2024_UTC_NS - _DAY_NS] * n
    if departures is None:
        departures = [
            DepartureRecord(
                # 與驅動層一致：DepartureRecord 存的是批次內的局部列位置 0..n-1。
                row_particle_index=i,
                status=final_status[i],
                x_m=final_x_m[i],
                y_m=final_y_m[i],
                z_m=final_z_m[i],
                utc_ns=final_utc_ns[i],
            )
            for i in range(n)
            if final_status[i] != NATIVE_STATUS_ACTIVE
        ]
    trajectory_x = np.tile(np.asarray(final_x_m, dtype=np.float64), (n_samples, 1))
    trajectory_y = np.tile(np.asarray(final_y_m, dtype=np.float64), (n_samples, 1))
    trajectory_z = np.tile(np.asarray(final_z_m, dtype=np.float64), (n_samples, 1))
    trajectory_status = np.tile(np.asarray(final_status, dtype=np.int32), (n_samples, 1))
    return BackwardDriverResult(
        row_particle_index=np.asarray(rows, dtype=np.int64),
        final_status=np.asarray(final_status, dtype=np.int32),
        final_x_m=np.asarray(final_x_m, dtype=np.float64),
        final_y_m=np.asarray(final_y_m, dtype=np.float64),
        final_z_m=np.asarray(final_z_m, dtype=np.float64),
        final_utc_ns=np.asarray(final_utc_ns, dtype=np.int64),
        trajectory_utc_ns=np.asarray(
            [_JAN_2024_UTC_NS - i * _HOUR_NS for i in range(n_samples)], dtype=np.int64
        ),
        trajectory_x_m=trajectory_x,
        trajectory_y_m=trajectory_y,
        trajectory_z_m=trajectory_z,
        trajectory_status=trajectory_status,
        departures=tuple(departures),
        diagnostics=SubstepDiagnostics(),
        overhead_diagnostics=HourlyOverheadDiagnostics(),
        hours_run=n_samples,
    )


# ── 基本合併、離開紀錄表與邊界通量統計 ──────────────────────────────────────


def test_merge_departure_table_and_flux_denominators(tmp_path):
    """三站、混合終止狀態的小型合成資料：驗證離開表內容與通量分母正確。

    設計八顆粒子：4 顆成功穿越邊界（分布在站 A、B、C），其餘 4 顆分別對應
    「達回溯上限」「擱淺」「沉底」「流場缺值」四種未穿越狀態；驗證通量表的
    分母是「全部 8 顆」而不是只算穿越或只算未穿越的子集，且未穿越者確實依
    狀態分開列出，不會被合併成單一數字或被當成零。
    """

    identity = _make_identity_table(
        particle_ids=[f"p{i}" for i in range(8)],
        study_site_id=["A", "A", "B", "B", "C", "C", "A", "B"],
        settling_velocity_mps=[-0.001] * 8,
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * 8,
    )
    statuses = [
        NATIVE_STATUS_LEFT_INTEREST_REGION,
        NATIVE_STATUS_LEFT_INTEREST_REGION,
        NATIVE_STATUS_LEFT_INTEREST_REGION,
        NATIVE_STATUS_MAX_BACKTRACK,
        NATIVE_STATUS_LEFT_INTEREST_REGION,
        NATIVE_STATUS_STRANDED,
        NATIVE_STATUS_DEPOSITED,
        NATIVE_STATUS_FORCING_MISSING,
    ]
    result = _make_result(rows=list(range(8)), final_status=statuses)

    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path, config=PostprocessConfig())
    accumulator.add_batch("batch-0", result)
    manifest = accumulator.finalize()

    assert manifest.particle_count_total == 8
    assert sum(manifest.status_counts.values()) == 8
    assert manifest.status_counts[NATIVE_STATUS_LEFT_INTEREST_REGION] == 4
    assert manifest.status_counts[NATIVE_STATUS_MAX_BACKTRACK] == 1
    assert manifest.status_counts[NATIVE_STATUS_STRANDED] == 1
    assert manifest.status_counts[NATIVE_STATUS_DEPOSITED] == 1
    assert manifest.status_counts[NATIVE_STATUS_FORCING_MISSING] == 1

    departures = pq.read_table(tmp_path / "departures.parquet").to_pylist()
    assert len(departures) == 8
    assert {row["row_particle_index"] for row in departures} == set(range(8))
    crossed_rows = [row for row in departures if row["crossed_interest_region_boundary"]]
    assert len(crossed_rows) == 4
    for row in departures:
        assert row["departure_lon_deg"] is None  # 未提供 lonlat_projector，必須是明確 null
        assert row["departure_lat_deg"] is None
        assert row["backtrack_duration_seconds"] >= 0.0

    flux_by_site = pq.read_table(tmp_path / "boundary_flux_by_site.parquet").to_pylist()
    totals_by_site = {row["study_site_id"]: row["total_particle_count"] for row in flux_by_site}
    assert totals_by_site == {"A": 3, "B": 3, "C": 2}
    for row in flux_by_site:
        total = row["total_particle_count"]
        # 分母必須恆等於全部狀態計數之和（穿越 + 五種未穿越），不可只計穿越者。
        summed = row["count_crossed"] + sum(
            row[f"count_status_{code}"]
            for code in (
                NATIVE_STATUS_MAX_BACKTRACK,
                NATIVE_STATUS_STRANDED,
                NATIVE_STATUS_DEPOSITED,
                NATIVE_STATUS_FORCING_MISSING,
                NATIVE_STATUS_NUMERICAL_FAILURE,
            )
        )
        assert summed == total
        assert row["fraction_crossed"] == pytest.approx(row["count_crossed"] / total)


def test_season_grouping_uses_shared_season_by_month(tmp_path):
    """四顆粒子分別在四個不同季節到達，驗證季節分組表確實區分四季。"""

    identity = _make_identity_table(
        particle_ids=[f"p{i}" for i in range(4)],
        study_site_id=["A"] * 4,
        settling_velocity_mps=[-0.001] * 4,
        arrival_time_utc_ns=[_JAN_2024_UTC_NS, _APR_2024_UTC_NS, _JUL_2024_UTC_NS, _OCT_2024_UTC_NS],
    )
    result = _make_result(
        rows=[0, 1, 2, 3],
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION] * 4,
        final_utc_ns=[
            _JAN_2024_UTC_NS - _DAY_NS,
            _APR_2024_UTC_NS - _DAY_NS,
            _JUL_2024_UTC_NS - _DAY_NS,
            _OCT_2024_UTC_NS - _DAY_NS,
        ],
    )
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path)
    accumulator.add_batch("batch-0", result)
    accumulator.finalize()

    seasons = {row["season"] for row in pq.read_table(tmp_path / "departures.parquet").to_pylist()}
    assert seasons == {"DJF", "MAM", "JJA", "SON"}
    flux_by_season = pq.read_table(tmp_path / "boundary_flux_by_season.parquet").to_pylist()
    assert {row["season"] for row in flux_by_season} == {"DJF", "MAM", "JJA", "SON"}
    for row in flux_by_season:
        assert row["total_particle_count"] == 1
        assert row["count_crossed"] == 1


def test_lonlat_projector_is_used_when_provided(tmp_path):
    """提供 lonlat_projector 時，離開表的經緯度欄位必須被正確填入且不是 null。"""

    identity = _make_identity_table(
        particle_ids=["p0", "p1"],
        study_site_id=["A", "A"],
        settling_velocity_mps=[-0.001, -0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS, _JAN_2024_UTC_NS],
    )
    result = _make_result(
        rows=[0, 1],
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION, NATIVE_STATUS_STRANDED],
        final_x_m=[100.0, 200.0],
        final_y_m=[300.0, 400.0],
    )

    def projector(site_ids, x_m, y_m):
        # 簡單的偽投影：只用於驗證資料有沒有被正確傳遞與寫入，不代表任何真實地理投影。
        return x_m / 1000.0, y_m / 1000.0

    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path, lonlat_projector=projector)
    accumulator.add_batch("batch-0", result)
    accumulator.finalize()

    departure_table = pq.read_table(tmp_path / "departures.parquet").to_pylist()
    rows = {row["row_particle_index"]: row for row in departure_table}
    assert rows[0]["departure_lon_deg"] == pytest.approx(0.1)
    assert rows[0]["departure_lat_deg"] == pytest.approx(0.3)
    assert rows[1]["departure_lon_deg"] == pytest.approx(0.2)


# ── 一致性檢查：必須抓到故意製造的錯誤 ──────────────────────────────────────


def test_rejects_duplicate_particle_across_batches(tmp_path):
    identity = _make_identity_table(
        particle_ids=["p0", "p1"],
        study_site_id=["A", "A"],
        settling_velocity_mps=[-0.001, -0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS, _JAN_2024_UTC_NS],
    )
    result = _make_result(rows=[0, 1], final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION] * 2)
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path)
    accumulator.add_batch("batch-0", result)
    with pytest.raises(PostprocessError, match="已處理過"):
        accumulator.add_batch("batch-1", result)


def test_rejects_final_status_still_active(tmp_path):
    identity = _make_identity_table(
        particle_ids=["p0"],
        study_site_id=["A"],
        settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    result = _make_result(
        rows=[0],
        final_status=[NATIVE_STATUS_ACTIVE],
        departures=[],
    )
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path)
    with pytest.raises(PostprocessError, match="ACTIVE"):
        accumulator.add_batch("batch-0", result)


def test_rejects_departure_status_mismatch(tmp_path):
    """故意讓 DepartureRecord.status 與 final_status 不一致，必須被抓到。"""

    identity = _make_identity_table(
        particle_ids=["p0"],
        study_site_id=["A"],
        settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    bad_departure = DepartureRecord(
        row_particle_index=0,
        status=NATIVE_STATUS_STRANDED,  # 與下面 final_status 故意不同
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        utc_ns=_JAN_2024_UTC_NS - _DAY_NS,
    )
    result = _make_result(
        rows=[0],
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION],
        departures=[bad_departure],
    )
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path)
    with pytest.raises(PostprocessError, match="不一致"):
        accumulator.add_batch("batch-0", result)


def test_rejects_missing_departure_record(tmp_path):
    """終止狀態非 ACTIVE 卻沒有任何 DepartureRecord，必須被抓到。"""

    identity = _make_identity_table(
        particle_ids=["p0"],
        study_site_id=["A"],
        settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    result = _make_result(rows=[0], final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION], departures=[])
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path)
    with pytest.raises(PostprocessError, match="DepartureRecord"):
        accumulator.add_batch("batch-0", result)


def test_finalize_rejects_missing_particles(tmp_path):
    """身分表有 3 顆粒子，卻只餵進 2 顆的批次結果：finalize 必須拒絕。"""

    identity = _make_identity_table(
        particle_ids=["p0", "p1", "p2"],
        study_site_id=["A", "A", "A"],
        settling_velocity_mps=[-0.001] * 3,
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * 3,
    )
    result = _make_result(rows=[0, 1], final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION] * 2)
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path)
    accumulator.add_batch("batch-0", result)
    with pytest.raises(PostprocessError, match="從未出現"):
        accumulator.finalize()


def test_rejects_out_of_range_row_index(tmp_path):
    identity = _make_identity_table(
        particle_ids=["p0"],
        study_site_id=["A"],
        settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    result = _make_result(rows=[5], final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION])
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path)
    with pytest.raises(PostprocessError, match="超出身分表範圍"):
        accumulator.add_batch("batch-0", result)


# ── 條件式來源足跡（式 11）：解析對照 ────────────────────────────────────


def test_kde_peak_matches_known_cluster_location():
    """一群緊密聚集在已知位置附近的離開點，KDE 峰值格必須落在該位置附近，
    且格網內機率總和必須是 1（既有 conditional_kde_2d 本身已保證正規化，
    這裡額外驗證本模組的薄轉接層沒有把這個保證弄丟）。
    """

    rng = np.random.default_rng(12345)
    center = np.array([50_000.0, -20_000.0])
    points = center + rng.normal(scale=25.0, size=(200, 2))

    product = build_source_footprint_kde(points, grid_cell_count=80, minimum_point_count=30)

    assert product.status == KDE_STATUS_AVAILABLE
    assert product.raw_point_count == 200
    total_probability = float(np.sum(product.cell_probability))
    assert total_probability == pytest.approx(1.0, abs=1.0e-9)

    peak_index = np.unravel_index(np.argmax(product.cell_probability), product.cell_probability.shape)
    peak_x = product.x_centers_m[peak_index[1]]
    peak_y = product.y_centers_m[peak_index[0]]
    # 聚集點的標準差只有 25 公尺，格網解析度遠小於此；峰值格中心應該非常接近
    # 真正的群聚中心（容許一個格寬左右的誤差）。
    cell_width = float(product.x_centers_m[1] - product.x_centers_m[0])
    assert abs(peak_x - center[0]) < 5 * cell_width
    assert abs(peak_y - center[1]) < 5 * cell_width

    for level in product.hdr_levels:
        mask = product.hdr_masks[level]
        covered = float(np.sum(product.cell_probability[mask]))
        # HDR 遮罩依機率遞減累加、首次「達到」門檻即停止選格，因此涵蓋的累積
        # 機率必然大於等於門檻本身（離散格網下通常略高於門檻，不會低於）。
        assert covered >= level - 1.0e-9


def test_kde_insufficient_points_status():
    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    product = build_source_footprint_kde(points, minimum_point_count=30)
    assert product.status == KDE_STATUS_INSUFFICIENT_POINTS
    assert product.raw_point_count == 2
    assert product.density_per_m2 is None
    assert product.hdr_masks is None


def test_kde_collinear_points_status():
    points = np.stack([np.linspace(0.0, 100.0, 40), np.linspace(0.0, 100.0, 40)], axis=1)
    product = build_source_footprint_kde(points, minimum_point_count=10)
    assert product.status == KDE_STATUS_COLLINEAR_POINTS
    assert product.raw_point_count == 40


def test_kde_products_written_per_site_and_site_velocity(tmp_path):
    """站級與站×沉降速度級的 KDE 產品都必須輸出，且樣本太少時明確標示不可估計。"""

    rng = np.random.default_rng(7)
    n_per_site = 40
    rows: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    sites: list[str] = []
    velocities: list[float] = []
    row_counter = 0
    for site, center in (("A", (0.0, 0.0)), ("B", (100_000.0, 100_000.0))):
        cluster = np.array(center) + rng.normal(scale=50.0, size=(n_per_site, 2))
        for x, y in cluster:
            rows.append(row_counter)
            xs.append(float(x))
            ys.append(float(y))
            sites.append(site)
            velocities.append(-0.001)
            row_counter += 1

    identity = _make_identity_table(
        particle_ids=[f"p{i}" for i in range(row_counter)],
        study_site_id=sites,
        settling_velocity_mps=velocities,
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * row_counter,
    )
    result = _make_result(
        rows=rows,
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION] * row_counter,
        final_x_m=xs,
        final_y_m=ys,
    )
    config = PostprocessConfig(kde_minimum_point_count=30, kde_grid_cell_count=32)
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path, config=config)
    accumulator.add_batch("batch-0", result)
    accumulator.finalize()

    kde_site_table = pq.read_table(tmp_path / "kde_by_site.parquet").to_pylist()
    kde_by_site = {row["study_site_id"]: row for row in kde_site_table}
    assert kde_by_site["A"]["status"] == KDE_STATUS_AVAILABLE
    assert kde_by_site["A"]["raw_point_count"] == n_per_site
    assert kde_by_site["B"]["status"] == KDE_STATUS_AVAILABLE
    expected_cell_count = kde_by_site["A"]["grid_rows"] * kde_by_site["A"]["grid_cols"]
    assert len(kde_by_site["A"]["density_per_m2_flat"]) == expected_cell_count

    kde_by_site_velocity = pq.read_table(tmp_path / "kde_by_site_settling_velocity.parquet").to_pylist()
    assert len(kde_by_site_velocity) == 2
    for row in kde_by_site_velocity:
        assert row["status"] == KDE_STATUS_AVAILABLE
        assert row["raw_point_count"] == n_per_site


# ── 軌跡抽樣：固定容量、與批次到達順序無關 ──────────────────────────────────


def _build_many_particle_case(n_per_site: int = 60):
    sites = []
    rows = []
    for site_index, site in enumerate(("A", "B")):
        for local in range(n_per_site):
            rows.append(site_index * n_per_site + local)
            sites.append(site)
    identity = _make_identity_table(
        particle_ids=[f"p{i}" for i in range(len(rows))],
        study_site_id=sites,
        settling_velocity_mps=[-0.001] * len(rows),
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * len(rows),
    )
    statuses = [NATIVE_STATUS_LEFT_INTEREST_REGION] * len(rows)
    return identity, rows, statuses


def test_trajectory_sampling_is_order_independent_and_reproducible(tmp_path):
    """把同一批粒子拆成不同的批次順序餵進兩個獨立的 accumulator，
    驗證抽樣結果（保留哪些 particle_id）完全相同，證明抽樣與批次到達順序無關。
    """

    identity, rows, statuses = _build_many_particle_case(n_per_site=60)
    config = PostprocessConfig(trajectory_sample_size_per_group=10, trajectory_sample_seed=999)

    # 順序一：依 row 遞增，每批 20 顆。
    out_dir_1 = tmp_path / "run1"
    accumulator_1 = PostprocessAccumulator(identity, output_dir=out_dir_1, config=config)
    chunk = 20
    for start in range(0, len(rows), chunk):
        batch_rows = rows[start : start + chunk]
        batch_status = statuses[start : start + chunk]
        accumulator_1.add_batch(f"chunk-{start}", _make_result(rows=batch_rows, final_status=batch_status))
    accumulator_1.finalize()

    # 順序二：把同樣的粒子隨機打亂順序、切成不同大小的批次。
    rng = np.random.default_rng(0)
    permuted = list(rows)
    rng.shuffle(permuted)
    out_dir_2 = tmp_path / "run2"
    accumulator_2 = PostprocessAccumulator(identity, output_dir=out_dir_2, config=config)
    boundaries = [0, 7, 23, 50, 90, len(permuted)]
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        batch_rows = permuted[start:stop]
        if not batch_rows:
            continue
        batch_status = [NATIVE_STATUS_LEFT_INTEREST_REGION] * len(batch_rows)
        accumulator_2.add_batch(f"shuffled-{start}", _make_result(rows=batch_rows, final_status=batch_status))
    accumulator_2.finalize()

    table_1 = pq.read_table(out_dir_1 / "trajectory_samples_by_site.parquet").to_pylist()
    table_2 = pq.read_table(out_dir_2 / "trajectory_samples_by_site.parquet").to_pylist()
    ids_1 = {row["particle_id"] for row in table_1}
    ids_2 = {row["particle_id"] for row in table_2}
    assert ids_1 == ids_2
    assert len(ids_1) == 2 * config.trajectory_sample_size_per_group  # 兩站各保留固定 K 筆


def test_stable_sample_priority_changes_with_seed():
    """抽樣種子改變時，優先序理應變動（否則抽樣種子這個設定形同虛設）。"""

    from lagrangian_backtracking.native_runner.postprocess import _stable_sample_priority

    a = _stable_sample_priority(1, "particle-xyz")
    b = _stable_sample_priority(2, "particle-xyz")
    assert a != b


# ── manifest 完整性 ─────────────────────────────────────────────────────


def test_manifest_contains_provenance_and_matches_output_checksums(tmp_path):
    identity = _make_identity_table(
        particle_ids=["p0", "p1"],
        study_site_id=["A", "A"],
        settling_velocity_mps=[-0.001, -0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS, _JAN_2024_UTC_NS],
    )
    result = _make_result(rows=[0, 1], final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION] * 2)
    config = PostprocessConfig(project_root="/home/claude/lbt")
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path, config=config)
    accumulator.add_batch("batch-0", result)
    manifest = accumulator.finalize()

    assert manifest.identity_table_content_sha256 == identity.content_sha256
    assert manifest.experiment_case_id == identity.experiment_case_id
    assert manifest.master_seed == identity.master_seed
    assert len(manifest.batches) == 1
    assert manifest.batches[0].fingerprint_sha256 == compute_batch_result_fingerprint(result)
    assert "git_available" in manifest.code_provenance
    assert manifest.resource_usage["peak_rss_kib"] > 0.0
    assert manifest.resource_usage["wall_seconds"] >= 0.0
    assert "scott" in manifest.kde_bandwidth_method.lower() or "Scott" in manifest.kde_bandwidth_method

    from lagrangian_backtracking.outputs import sha256_file

    for name, digest in manifest.output_files_sha256.items():
        assert sha256_file(tmp_path / name) == digest

    manifest_on_disk = (tmp_path / "manifest.json")
    assert manifest_on_disk.exists()


def test_batch_fingerprint_changes_when_result_changes():
    result_a = _make_result(rows=[0, 1], final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION] * 2)
    result_b = _make_result(
        rows=[0, 1],
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION, NATIVE_STATUS_STRANDED],
        departures=[
            DepartureRecord(0, NATIVE_STATUS_LEFT_INTEREST_REGION, 0.0, 0.0, 0.0, _JAN_2024_UTC_NS - _DAY_NS),
            DepartureRecord(1, NATIVE_STATUS_STRANDED, 0.0, 0.0, 0.0, _JAN_2024_UTC_NS - _DAY_NS),
        ],
    )
    assert compute_batch_result_fingerprint(result_a) != compute_batch_result_fingerprint(result_b)
    # 相同輸入必須得到相同指紋（可重現性）。
    result_a_again = _make_result(rows=[0, 1], final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION] * 2)
    assert compute_batch_result_fingerprint(result_a) == compute_batch_result_fingerprint(result_a_again)


# ── 規模測試：500,000 顆粒子的串流合併記憶體與耗時 ──────────────────────────


def test_streaming_merge_scale_500000_particles_memory_and_time(tmp_path):
    """模擬 M=10 全案規模（500,000 顆粒子），驗證串流合併不會把全部軌跡載入
    記憶體，並回報實際耗時與尖峰記憶體（RSS）供驗收報告引用。

    只造最終狀態與離開點（決定通量與 KDE 結果），軌跡陣列刻意只給極短的
    ``n_samples``（本測試不驗證軌跡抽樣的數值內容，那已由較小規模的
    ``test_trajectory_sampling_is_order_independent_and_reproducible`` 驗證），
    避免規模測試本身因為建構合成資料而變得比它要驗證的對象還慢、還耗記憶體。
    """

    n_total = 500_000
    n_per_batch = 2000
    n_batches = n_total // n_per_batch
    sites = np.array(["A", "B", "C", "D", "E"])
    velocities = np.array([-0.0001, -0.001])

    site_for_row = sites[np.arange(n_total) % sites.size]
    velocity_for_row = velocities[np.arange(n_total) % velocities.size]
    # 大約 20% 的粒子成功穿越邊界，其餘依固定週期分布到四種未穿越狀態，
    # 讓通量統計與一致性檢查在規模測試下仍有非平凡的內容可以驗證。
    status_cycle = np.array(
        [
            NATIVE_STATUS_LEFT_INTEREST_REGION,
            NATIVE_STATUS_LEFT_INTEREST_REGION,
            NATIVE_STATUS_MAX_BACKTRACK,
            NATIVE_STATUS_LEFT_INTEREST_REGION,
            NATIVE_STATUS_STRANDED,
            NATIVE_STATUS_LEFT_INTEREST_REGION,
            NATIVE_STATUS_DEPOSITED,
            NATIVE_STATUS_LEFT_INTEREST_REGION,
            NATIVE_STATUS_FORCING_MISSING,
            NATIVE_STATUS_LEFT_INTEREST_REGION,
        ]
    )
    status_for_row = status_cycle[np.arange(n_total) % status_cycle.size]

    particle_ids = np.array([f"scale-p{i:07d}" for i in range(n_total)], dtype="<U64")
    seed_hex = np.array([f"{i:032x}" for i in range(n_total)], dtype="<U32")
    identity = ParticleIdentityTable(
        particle_id=particle_ids,
        scenario_id=particle_ids,
        member_id=np.zeros(n_total, dtype=np.int32),
        study_site_id=site_for_row.astype("<U64"),
        analysis_region_id=np.asarray(["region-a"] * n_total, dtype="<U64"),
        receptor_id=np.asarray([f"recv-{i % 20}" for i in range(n_total)], dtype="<U64"),
        material_id=np.asarray(["microplastic"] * n_total, dtype="<U64"),
        arrival_time_id=np.asarray([f"arrival-{i % 50}" for i in range(n_total)], dtype="<U64"),
        settling_velocity_mps=velocity_for_row,
        arrival_time_utc_ns=np.full(n_total, _JAN_2024_UTC_NS, dtype=np.int64),
        seed_hex=seed_hex,
        x0_m=np.zeros(n_total, dtype=np.float64),
        y0_m=np.zeros(n_total, dtype=np.float64),
        z0_m=np.linspace(-1.0, -50.0, n_total),
        experiment_case_id="scale-case",
        master_seed=7,
        random_stream_id=None,
    )

    baseline_rss_kib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    config = PostprocessConfig(
        trajectory_sample_size_per_group=10,
        kde_grid_cell_count=16,
        kde_minimum_point_count=30,
    )
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path, config=config)

    site_span = 200_000.0  # 各站在 x 方向錯開，避免五個站的 KDE 群聚互相重疊。
    site_offset = {name: index * site_span for index, name in enumerate(sites.tolist())}

    for batch_index in range(n_batches):
        start = batch_index * n_per_batch
        stop = start + n_per_batch
        rows = list(range(start, stop))
        batch_sites = site_for_row[start:stop]
        offsets = np.asarray([site_offset[value] for value in batch_sites])
        # x／y 分別用互質的週期取模，確保同一站內的離開點不會全部落在同一條
        # 直線上（若 x、y 對每個粒子索引都用相同週期，會讓 KDE 誤判為共線）。
        local_index = np.arange(n_per_batch)
        final_x = offsets + (local_index % 97).astype(np.float64)
        final_y = (local_index % 89).astype(np.float64)
        result = _make_result(
            rows=rows,
            final_status=status_for_row[start:stop].tolist(),
            final_x_m=final_x.tolist(),
            final_y_m=final_y.tolist(),
            n_samples=2,
        )
        accumulator.add_batch(f"scale-batch-{batch_index}", result)

    manifest = accumulator.finalize()

    assert manifest.particle_count_total == n_total
    assert sum(manifest.status_counts.values()) == n_total
    assert len(manifest.batches) == n_batches

    peak_rss_kib = manifest.resource_usage["peak_rss_kib"]
    wall_seconds = manifest.resource_usage["wall_seconds"]

    # 若真的把全部軌跡（500,000 × 720 小時等級）攤平進記憶體，會是數十 GB；
    # 這裡的合成資料軌跡樣本數雖然刻意壓到只有 2，但真正要驗證的是「即使
    # 軌跡樣本數很大，串流設計本身也不會隨粒子數線性長出等比例的軌跡快取」，
    # 因此用一個遠低於「全部物化」量級、但仍夠寬鬆不會誤判的絕對門檻
    # （2 GiB）；同時報告基準值與尖峰值的差異，方便未來調整門檻時參考實際
    # 數字而不是憑空猜測。
    print(
        f"\n[WP-6 規模測試] 粒子數={n_total} 批次數={n_batches} "
        f"耗時={wall_seconds:.2f}s 基準 RSS={baseline_rss_kib / 1024:.1f}MiB "
        f"尖峰 RSS={peak_rss_kib / 1024:.1f}MiB"
    )
    assert peak_rss_kib < 2_000_000.0

    flux_by_site = pq.read_table(tmp_path / "boundary_flux_by_site.parquet").to_pylist()
    assert sum(row["total_particle_count"] for row in flux_by_site) == n_total

    kde_site_table = pq.read_table(tmp_path / "kde_by_site.parquet").to_pylist()
    kde_by_site = {row["study_site_id"]: row for row in kde_site_table}
    assert set(kde_by_site) == set(sites.tolist())
    for row in kde_by_site.values():
        assert row["status"] == KDE_STATUS_AVAILABLE


# ══════════════════════════════════════════════════════════════════════════
# WP-17 任務 3：依開放邊界分段統計相對來源權重（boundary_geometries）
# ══════════════════════════════════════════════════════════════════════════


def _make_boundary_geometry(site_id: str):
    """建一個最小合法 ``BoundaryGeometry``：東側是開放水域線，其餘視為海岸線。"""

    from shapely.geometry import LineString, box

    from lagrangian_backtracking.boundaries import BoundaryGeometry

    local = box(-100.0, -100.0, 100.0, 100.0)
    return BoundaryGeometry(
        own_local_domain=local,
        flow_domain=box(-500.0, -500.0, 500.0, 500.0),
        foreign_local_domains={},
        own_local_open_boundary=LineString([(100.0, -100.0), (100.0, 100.0)]),
        flow_open_boundary=LineString([(500.0, -500.0), (500.0, 500.0)]),
        own_local_boundary_segment_id=f"{site_id}_local_open_boundary",
        flow_boundary_segment_id="flow_open_boundary",
    )


def test_boundary_segment_classification_picks_nearest_open_line() -> None:
    from lagrangian_backtracking.native_runner.postprocess import classify_boundary_segment

    geometry = _make_boundary_geometry("siteA")
    # 離開點就在 own_local_open_boundary 線上（東側），應該分類到本站 local 分段。
    segment_id, distance_m = classify_boundary_segment(100.0, 0.0, geometry)
    assert segment_id == "siteA_local_open_boundary"
    assert distance_m == pytest.approx(0.0, abs=1.0e-9)

    # 離開點離 flow_open_boundary（500 m 處）比較近。
    segment_id, distance_m = classify_boundary_segment(480.0, 0.0, geometry)
    assert segment_id == "flow_open_boundary"
    assert distance_m == pytest.approx(20.0, abs=1.0e-9)


def test_boundary_geometries_populate_departure_columns_and_entry_tables(tmp_path) -> None:
    n = 4
    identity = _make_identity_table(
        particle_ids=[f"p{i}" for i in range(n)],
        study_site_id=["siteA", "siteA", "siteA", "siteA"],
        settling_velocity_mps=[-0.001, -0.001, -0.002, -0.002],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * n,
    )
    result = _make_result(
        rows=[0, 1, 2, 3],
        final_status=[
            NATIVE_STATUS_LEFT_INTEREST_REGION,
            NATIVE_STATUS_LEFT_INTEREST_REGION,
            NATIVE_STATUS_LEFT_INTEREST_REGION,
            NATIVE_STATUS_MAX_BACKTRACK,
        ],
        final_x_m=[100.0, 480.0, 100.0, 0.0],
        final_y_m=[0.0, 0.0, 10.0, 0.0],
    )
    accumulator = PostprocessAccumulator(
        identity,
        output_dir=tmp_path,
        boundary_geometries={"siteA": _make_boundary_geometry("siteA")},
    )
    accumulator.add_batch("batch-0", result)
    manifest = accumulator.finalize()
    assert manifest.config["boundary_segment_classification_available"] is True

    departures = pq.read_table(tmp_path / "departures.parquet").to_pylist()
    by_row = {row["row_particle_index"]: row for row in departures}
    assert by_row[0]["boundary_segment_id"] == "siteA_local_open_boundary"
    assert by_row[1]["boundary_segment_id"] == "flow_open_boundary"
    assert by_row[2]["boundary_segment_id"] == "siteA_local_open_boundary"
    # 未穿越邊界的粒子（達回溯上限）不分類，欄位維持 null。
    assert by_row[3]["boundary_segment_id"] is None
    assert by_row[3]["boundary_classification_distance_m"] is None

    by_site = {
        row["boundary_segment_id"]: row
        for row in pq.read_table(tmp_path / "boundary_entry_by_site.parquet").to_pylist()
    }
    assert by_site["siteA_local_open_boundary"]["crossed_count"] == 2
    assert by_site["flow_open_boundary"]["crossed_count"] == 1
    assert by_site["siteA_local_open_boundary"]["relative_source_weight_within_group"] == pytest.approx(
        2.0 / 3.0
    )
    assert by_site["flow_open_boundary"]["relative_source_weight_within_group"] == pytest.approx(1.0 / 3.0)

    assert (tmp_path / "boundary_entry_by_site.csv").is_file()
    assert (tmp_path / "boundary_flux_by_site.csv").is_file()


def test_boundary_geometries_missing_site_fails_closed(tmp_path) -> None:
    identity = _make_identity_table(
        particle_ids=["p0"], study_site_id=["siteB"], settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    result = _make_result(rows=[0], final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION])
    accumulator = PostprocessAccumulator(
        identity, output_dir=tmp_path, boundary_geometries={"siteA": _make_boundary_geometry("siteA")},
    )
    with pytest.raises(PostprocessError):
        accumulator.add_batch("batch-0", result)


def test_flux_tables_without_boundary_geometries_have_null_segment_columns(tmp_path) -> None:
    identity = _make_identity_table(
        particle_ids=["p0"], study_site_id=["siteA"], settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    result = _make_result(rows=[0], final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION])
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path)
    accumulator.add_batch("batch-0", result)
    manifest = accumulator.finalize()
    assert manifest.config["boundary_segment_classification_available"] is False
    assert not (tmp_path / "boundary_entry_by_site.parquet").exists()
    departures = pq.read_table(tmp_path / "departures.parquet").to_pylist()
    assert departures[0]["boundary_segment_id"] is None


# ══════════════════════════════════════════════════════════════════════════
# WP-17 任務 1／3：pre_window_deposition 狀態必須能進入既有邊界通量統計，
# 並在分母與各類別比例中正確、單獨計數（不得與資料缺口／數值失敗混為一談，
# 也不得讓整個 finalize 因為未登錄的狀態碼而報「內部錯誤」）。
# ══════════════════════════════════════════════════════════════════════════


def test_pre_window_deposition_status_counted_separately_in_flux_tables(tmp_path) -> None:
    """``NATIVE_STATUS_PRE_WINDOW_DEPOSITION`` 必須能被 finalize／通量表正確消化：

    - 計入 ``status_counts`` 與每個通量分組的 ``total_particle_count`` 分母
      （見 ``docs/operations/14...`` 第 3.1 節「pre-window 成員保留在固定的
      total 情境／成員 outcome 分母並單獨計數」）；
    - 不與 ``NATIVE_STATUS_MAX_BACKTRACK``／``NATIVE_STATUS_NUMERICAL_FAILURE``
      等既有五種未穿越狀態合併成同一欄；
    - 不進入穿越邊界分子（``count_crossed``）。
    """

    identity = _make_identity_table(
        particle_ids=[f"p{i}" for i in range(4)],
        study_site_id=["A", "A", "A", "A"],
        settling_velocity_mps=[-0.001] * 4,
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * 4,
    )
    statuses = [
        NATIVE_STATUS_LEFT_INTEREST_REGION,
        NATIVE_STATUS_MAX_BACKTRACK,
        NATIVE_STATUS_PRE_WINDOW_DEPOSITION,
        NATIVE_STATUS_PRE_WINDOW_DEPOSITION,
    ]
    result = _make_result(rows=list(range(4)), final_status=statuses)

    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path, config=PostprocessConfig())
    accumulator.add_batch("batch-0", result)
    manifest = accumulator.finalize()

    assert manifest.particle_count_total == 4
    assert sum(manifest.status_counts.values()) == 4
    assert manifest.status_counts[NATIVE_STATUS_PRE_WINDOW_DEPOSITION] == 2
    assert manifest.status_counts[NATIVE_STATUS_LEFT_INTEREST_REGION] == 1
    assert manifest.status_counts[NATIVE_STATUS_MAX_BACKTRACK] == 1

    flux_by_site = pq.read_table(tmp_path / "boundary_flux_by_site.parquet").to_pylist()
    assert len(flux_by_site) == 1
    row = flux_by_site[0]
    assert row["total_particle_count"] == 4
    assert row["count_crossed"] == 1
    assert row[f"count_status_{NATIVE_STATUS_MAX_BACKTRACK}"] == 1
    assert row[f"count_status_{NATIVE_STATUS_PRE_WINDOW_DEPOSITION}"] == 2
    assert row[f"fraction_status_{NATIVE_STATUS_PRE_WINDOW_DEPOSITION}"] == pytest.approx(0.5)
