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

import json
import resource
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lagrangian_backtracking import cli
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
    LocalDomainFirstExitRecord,
    SubstepDiagnostics,
)
from lagrangian_backtracking.native_runner.hour_major_provider import (
    save_backward_driver_result_npz,
)
from lagrangian_backtracking.native_runner.particle_input import (
    ParticleIdentityTable,
    save_particle_identity_table_npz,
)
from lagrangian_backtracking.native_runner.postprocess import (
    COMPASS_16_SECTOR_NAMES,
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
    x0_m: list[float] | None = None,
    y0_m: list[float] | None = None,
    receptor_id: list[str] | None = None,
    scenario_id: list[str] | None = None,
    member_id: list[int] | None = None,
    experiment_case_id: str = "case-001",
    master_seed: int = 42,
) -> ParticleIdentityTable:
    """建立測試用的合成粒子身分表，欄位語意與 particle_input.py 完全一致。

    ``x0_m``／``y0_m``（逆向追蹤起點，即受體位置的公尺座標）預設全零，與 WP-17
    之前既有測試的行為完全相同；WP-20 的方位扇區測試需要指定非零受體位置作為
    「站受體位置中心」，因此新增這兩個可選覆寫參數，不影響既有呼叫端。

    ``scenario_id``／``member_id``（WP-37 新增可選覆寫）預設分別是「每顆粒子各
    自一個情境」「member_id 全部是 0」，與既有測試 100% 相同的行為（即 M=1）；
    只有驗證系集成員數（M）計算與一致性檢核的測試才會明確覆寫成「多顆粒子共用
    同一個 scenario_id、member_id 各不相同」的組合。
    """

    n = len(particle_ids)
    if z0_m is None:
        z0_m = [-1.0] * n
    if x0_m is None:
        x0_m = [0.0] * n
    if y0_m is None:
        y0_m = [0.0] * n
    if receptor_id is None:
        receptor_id = [f"recv-{i % 5}" for i in range(n)]
    if scenario_id is None:
        scenario_id = [f"scn-{i}" for i in range(n)]
    if member_id is None:
        member_id = [0] * n
    return ParticleIdentityTable(
        particle_id=np.asarray(particle_ids, dtype="<U64"),
        scenario_id=np.asarray(scenario_id, dtype="<U64"),
        member_id=np.asarray(member_id, dtype=np.int32),
        study_site_id=np.asarray(study_site_id, dtype="<U64"),
        analysis_region_id=np.asarray(["region-a"] * n, dtype="<U64"),
        receptor_id=np.asarray(receptor_id, dtype="<U64"),
        material_id=np.asarray(["microplastic"] * n, dtype="<U64"),
        arrival_time_id=np.asarray([f"arrival-{i}" for i in range(n)], dtype="<U64"),
        settling_velocity_mps=np.asarray(settling_velocity_mps, dtype=np.float64),
        arrival_time_utc_ns=np.asarray(arrival_time_utc_ns, dtype=np.int64),
        seed_hex=np.asarray([f"{i:032x}" for i in range(n)], dtype="<U32"),
        x0_m=np.asarray(x0_m, dtype=np.float64),
        y0_m=np.asarray(y0_m, dtype=np.float64),
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
    local_domain_departures: list[LocalDomainFirstExitRecord] | None = None,
    n_samples: int = 2,
) -> BackwardDriverResult:
    """建立測試用的合成 ``BackwardDriverResult``，departures 依 final_status 自動配對。

    ``local_domain_departures``（WP-23）預設 ``None`` -> 空 tuple，維持既有呼叫端
    （不涉及 local domain 的測試）100% 不變的行為；只有 WP-23 新增的測試才會
    明確傳入。
    """

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
        local_domain_departures=tuple(local_domain_departures) if local_domain_departures else (),
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


def test_runtime_bed_no_deposition_manifest_field_round_trips_and_defaults_to_none(tmp_path):
    """WP-26／WP-28：``PostprocessAccumulator`` 收到 ``runtime_bed_no_deposition``
    時要原樣寫進 ``manifest.config.runtime_bed_no_deposition``；沒有提供（呼叫端
    沒有給 ``--plan``）時必須明確是 ``None``，不能臆測成 ``False``（省略計畫檔
    不代表「未啟用」，只代表無法核對，見 ``PostprocessAccumulator.__init__``
    docstring）。統計邏輯本身（``status_counts`` 等）不受這個欄位影響。"""

    identity = _make_identity_table(
        particle_ids=["p0"],
        study_site_id=["A"],
        settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    result = _make_result(rows=[0], final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION])

    declaration = {"bed_no_deposition": True}
    accumulator = PostprocessAccumulator(
        identity, output_dir=tmp_path / "with_plan", runtime_bed_no_deposition=declaration
    )
    accumulator.add_batch("batch-0", result)
    manifest = accumulator.finalize()
    assert manifest.config["runtime_bed_no_deposition"] == declaration
    assert manifest.particle_count_total == 1

    accumulator_no_plan = PostprocessAccumulator(identity, output_dir=tmp_path / "without_plan")
    accumulator_no_plan.add_batch("batch-0", result)
    manifest_no_plan = accumulator_no_plan.finalize()
    assert manifest_no_plan.config["runtime_bed_no_deposition"] is None


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


def _make_boundary_geometry(site_id: str, *, local_equals_flow: bool = False):
    """建一個最小合法 ``BoundaryGeometry``：東側是開放水域線，其餘視為海岸線。

    ``local_equals_flow``（WP-23）預設 ``False``（沿用既有測試的隱含假設：
    local 40 m ⊂ flow 500 m，兩條開放邊界線是不同的線）；WP-23 新增的測試會
    明確傳入 ``True`` 代表「新竹／南灣／連江」這類 local domain 與 flow domain
    相同的測站。
    """

    from shapely.geometry import LineString, box

    from lagrangian_backtracking.boundaries import BoundaryGeometry

    local = box(-100.0, -100.0, 100.0, 100.0)
    return BoundaryGeometry(
        own_local_domain=local,
        flow_domain=box(-500.0, -500.0, 500.0, 500.0),
        foreign_local_domains={},
        own_local_open_boundary=LineString([(100.0, -100.0), (100.0, 100.0)]),
        flow_open_boundary=LineString([(500.0, -500.0), (500.0, 500.0)]),
        local_equals_flow=local_equals_flow,
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


# ══════════════════════════════════════════════════════════════════════════
# WP-29：修正 local_equals_flow=True 站點終止事件的邊界歸屬標籤不一致
# ══════════════════════════════════════════════════════════════════════════


def test_boundary_segment_classification_local_equals_flow_true_excludes_own_local_candidate() -> None:
    """WP-29 核心修法：``local_equals_flow=True`` 時，``own_local_open_boundary``
    完全不參與最短距離比較，即使離開點精確落在 ``own_local`` 線上（距離 0）、
    離 ``flow`` 線還有 400 m，仍然必須分類到 ``flow_boundary_segment_id``。

    這裡刻意讓兩條線的距離差很大（不是幾何重合的退化情況），用來證明修法是
    「先依 ``local_equals_flow`` 收斂候選集合，``own_local`` 直接出局」，不是
    「剛好沒有觸發那個浮點誤差或排序穩定性造成的 tie」——修補前（WP-17）的
    程式碼在這個輸入下會回傳 ``siteC_local_open_boundary``（見下一個測試的
    對照組），是本任務要修的臭蟲最直接的反例。
    """

    from lagrangian_backtracking.native_runner.postprocess import classify_boundary_segment

    geometry = _make_boundary_geometry("siteC", local_equals_flow=True)
    segment_id, distance_m = classify_boundary_segment(100.0, 0.0, geometry)
    assert segment_id == "flow_open_boundary"
    assert distance_m == pytest.approx(400.0, abs=1.0e-9)


def test_boundary_segment_classification_local_equals_flow_false_unaffected() -> None:
    """對照組：``local_equals_flow=False``（貢寮／龜山島）行為完全不變——同一個
    離開點、同一份幾何，只把 ``local_equals_flow`` 從 ``True`` 換成 ``False``，
    分類結果換回 WP-17 原本的「兩條都比距離、取較近者」，證明 WP-29 修法只影響
    ``local_equals_flow=True`` 的站，不影響貢寮／龜山島既有行為。
    """

    from lagrangian_backtracking.native_runner.postprocess import classify_boundary_segment

    geometry = _make_boundary_geometry("siteC", local_equals_flow=False)
    segment_id, distance_m = classify_boundary_segment(100.0, 0.0, geometry)
    assert segment_id == "siteC_local_open_boundary"
    assert distance_m == pytest.approx(0.0, abs=1.0e-9)


def _make_boundary_geometry_with_coincident_open_boundary(site_id: str, *, local_equals_flow: bool):
    """建一個 ``own_local_open_boundary`` 與 ``flow_open_boundary`` 是**同一個**
    shapely 物件（座標逐位元相同）的 ``BoundaryGeometry``，重現診斷報告描述的
    「新竹／南灣／連江」情境：local domain 與 flow domain 的開放邊界其實是同一條
    線，只是登記成兩個不同名稱的分段。

    共用同一個物件確保 ``point.distance(...)`` 對兩個名稱算出來的距離逐位元
    相同（不只是數值上接近），因此可以用「同一份幾何，只把 ``local_equals_flow``
    從 ``True`` 切到 ``False``」這一個位元的差異，精確重現修補前（WP-17，任何
    ``local_equals_flow`` 都不特殊處理）的候選比較邏輯：``local_equals_flow=
    False`` 時 ``own_local`` 與 ``flow`` 都進候選，兩者距離逐位元相同，
    ``list.sort`` 的穩定排序保留先插入者（``own_local``），與診斷報告對新竹／
    南灣／連江實際觀察到的行為（恆定分類到 local 名稱）完全一致；
    ``local_equals_flow=True`` 則是 WP-29 修補後的新行為。兩次呼叫除了
    ``local_equals_flow``（與由它決定的候選集合）以外，其餘設定完全相同，因此
    比較兩次的統計輸出就能證明修法只換了標籤字串，沒有換任何數值。
    """

    from shapely.geometry import LineString, box

    from lagrangian_backtracking.boundaries import BoundaryGeometry

    shared_open_boundary = LineString([(500.0, -500.0), (500.0, 500.0)])
    domain = box(-500.0, -500.0, 500.0, 500.0)
    return BoundaryGeometry(
        own_local_domain=domain,
        flow_domain=domain,
        foreign_local_domains={},
        own_local_open_boundary=shared_open_boundary,
        flow_open_boundary=shared_open_boundary,
        local_equals_flow=local_equals_flow,
        own_local_boundary_segment_id=f"{site_id}_local_open_boundary",
        flow_boundary_segment_id="flow_open_boundary",
    )


def test_local_equals_flow_relabeling_preserves_crossed_counts_and_relative_weights(tmp_path) -> None:
    """WP-29 硬要求的逐值證明：修法只能換標籤字串，不能換任何穿越次數或相對
    來源權重的數值。

    做法：對同一批合成離開點（座標完全相同、只沿共用開放邊界線分散到 4 個不同
    弧段），分別用 ``local_equals_flow=False``（重現修補前恆定選到 local 名稱
    的行為）與 ``local_equals_flow=True``（修補後）各跑一次完整的
    ``PostprocessAccumulator``，其餘輸入（粒子數、座標、狀態碼）逐位元相同。
    比對兩次輸出的 ``boundary_entry_by_site_arc_segment.parquet``：兩次的
    ``base_boundary_segment_id`` 必須分別恆定是 local／flow 名稱（不會同一次
    執行內部就分裂成兩種名稱，這正是診斷報告描述的「五站不一致」問題的最小
    重現），但依 ``arc_segment_index`` 對齊後，``crossed_count``、
    ``relative_source_weight_within_group`` 與弧段中點座標必須逐值相同。
    """

    n = 6
    identity = _make_identity_table(
        particle_ids=[f"p{i}" for i in range(n)],
        study_site_id=["siteD"] * n,
        settling_velocity_mps=[-0.001] * n,
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * n,
    )
    # 開放邊界線是 x=500、y 從 -500 到 500（長 1000 m）；預設 20 段、每段 50 m，
    # 弧長位置＝y_m-(-500)。下面四個 y_m 值分別落在弧段 0／9／15／19，其中弧段
    # 0 與 15 各有兩顆粒子，驗證計數不是只對「有沒有非零」敏感，連「數幾顆」都
    # 逐值相同。
    final_y_values = [-490.0, -490.0, -50.0, 250.0, 250.0, 480.0]
    result = _make_result(
        rows=list(range(n)),
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION] * n,
        final_x_m=[500.0] * n,
        final_y_m=final_y_values,
    )

    def _run(local_equals_flow: bool) -> list[dict]:
        out_dir = tmp_path / f"local_equals_flow_{local_equals_flow}"
        geometry = _make_boundary_geometry_with_coincident_open_boundary(
            "siteD", local_equals_flow=local_equals_flow
        )
        accumulator = PostprocessAccumulator(
            identity, output_dir=out_dir, boundary_geometries={"siteD": geometry},
        )
        accumulator.add_batch("batch-0", result)
        accumulator.finalize()
        return pq.read_table(out_dir / "boundary_entry_by_site_arc_segment.parquet").to_pylist()

    old_style_rows = _run(local_equals_flow=False)  # 重現修補前恆定的 local 標籤
    new_style_rows = _run(local_equals_flow=True)  # WP-29 修補後

    # 修補前全部落在 local 名稱、修補後全部落在 flow 名稱：這就是「同一種物理
    # 事件在五站間被貼上不同名稱標籤」問題本身，也是本任務要修的東西。
    assert {row["base_boundary_segment_id"] for row in old_style_rows} == {"siteD_local_open_boundary"}
    assert {row["base_boundary_segment_id"] for row in new_style_rows} == {"flow_open_boundary"}

    old_by_arc = {row["arc_segment_index"]: row for row in old_style_rows}
    new_by_arc = {row["arc_segment_index"]: row for row in new_style_rows}
    assert set(old_by_arc) == set(new_by_arc) == {0, 9, 15, 19}
    for arc_index in old_by_arc:
        old_row = old_by_arc[arc_index]
        new_row = new_by_arc[arc_index]
        assert new_row["crossed_count"] == old_row["crossed_count"]
        assert new_row["relative_source_weight_within_group"] == pytest.approx(
            old_row["relative_source_weight_within_group"]
        )
        assert new_row["midpoint_x_m"] == pytest.approx(old_row["midpoint_x_m"])
        assert new_row["midpoint_y_m"] == pytest.approx(old_row["midpoint_y_m"])

    assert old_by_arc[0]["crossed_count"] == new_by_arc[0]["crossed_count"] == 2
    assert old_by_arc[15]["crossed_count"] == new_by_arc[15]["crossed_count"] == 2
    assert old_by_arc[9]["crossed_count"] == new_by_arc[9]["crossed_count"] == 1
    assert old_by_arc[19]["crossed_count"] == new_by_arc[19]["crossed_count"] == 1
    assert sum(row["crossed_count"] for row in new_style_rows) == n
    assert sum(row["crossed_count"] for row in old_style_rows) == n


def test_local_domain_departures_unaffected_by_wp29_for_local_equals_flow_false_site(tmp_path) -> None:
    """WP-29 任務要求 2：``local_domain_departures.parquet``（貢寮／龜山島的
    local 首次離開，``local_equals_flow=False``）不受這次修法影響——它們的
    ``own_local_open_boundary`` 本來就是唯一有意義的候選（首次離開 local domain
    的位置理論上必然離自己 local domain 邊界最近，見
    ``postprocess._write_local_domain_departure_batch`` docstring），WP-29 的
    候選收斂規則只在 ``geometry.local_equals_flow=True`` 時才會生效，這裡驗證
    ``local_boundary_segment_id`` 仍然一如既往地分類到 ``own_local`` 名稱。
    """

    identity = _make_identity_table(
        particle_ids=["p0"], study_site_id=["siteA"], settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    local_records = [
        LocalDomainFirstExitRecord(
            row_particle_index=0, x_m=100.0, y_m=0.0, z_m=-5.0,
            utc_ns=_JAN_2024_UTC_NS - _HOUR_NS,
        ),
    ]
    result = _make_result(
        rows=[0],
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION],
        final_x_m=[480.0],
        final_y_m=[0.0],
        local_domain_departures=local_records,
    )
    accumulator = PostprocessAccumulator(
        identity, output_dir=tmp_path,
        boundary_geometries={"siteA": _make_boundary_geometry("siteA", local_equals_flow=False)},
    )
    accumulator.add_batch("batch-0", result)
    accumulator.finalize()

    local_rows = pq.read_table(tmp_path / "local_domain_departures.parquet").to_pylist()
    assert len(local_rows) == 1
    # 離開點就落在 own_local_open_boundary 上（距離 0），與終止事件（分類到
    # flow_open_boundary，因為 (480, 0) 離 flow 線比較近）是兩個不同的分類結果，
    # 互不影響，正是 WP-23 裁決＋WP-29 修法要維持的行為。
    assert local_rows[0]["local_boundary_segment_id"] == "siteA_local_open_boundary"
    assert local_rows[0]["local_boundary_classification_distance_m"] == pytest.approx(0.0, abs=1.0e-9)

    departures = pq.read_table(tmp_path / "departures.parquet").to_pylist()
    assert departures[0]["boundary_segment_id"] == "flow_open_boundary"


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


# ══════════════════════════════════════════════════════════════════════════
# WP-20：開放邊界再細分（方位扇區／沿邊界弧長位置）
# ══════════════════════════════════════════════════════════════════════════


def test_classify_azimuth_sector_known_directions() -> None:
    from lagrangian_backtracking.native_runner.postprocess import classify_azimuth_sector

    # 正北：方位角 0 度，落在扇區 0（N）。
    index, azimuth, name = classify_azimuth_sector(0.0, 100.0, 0.0, 0.0, sector_count=16)
    assert (index, name) == (0, "N")
    assert azimuth == pytest.approx(0.0)

    # 正東：方位角 90 度，落在扇區 4（E）。
    index, azimuth, name = classify_azimuth_sector(100.0, 0.0, 0.0, 0.0, sector_count=16)
    assert (index, name) == (4, "E")
    assert azimuth == pytest.approx(90.0)

    # 正南：方位角 180 度，落在扇區 8（S）。
    index, azimuth, name = classify_azimuth_sector(0.0, -100.0, 0.0, 0.0, sector_count=16)
    assert (index, name) == (8, "S")
    assert azimuth == pytest.approx(180.0)

    # 正西：方位角 270 度，落在扇區 12（W）。
    index, azimuth, name = classify_azimuth_sector(-100.0, 0.0, 0.0, 0.0, sector_count=16)
    assert (index, name) == (12, "W")
    assert azimuth == pytest.approx(270.0)

    # 待分類點與中心點重合（零向量），定義為方位角 0 度、扇區 0，不拋例外。
    index, azimuth, name = classify_azimuth_sector(5.0, 5.0, 5.0, 5.0, sector_count=16)
    assert (index, azimuth) == (0, 0.0)

    # 非 16 的扇區數：退回明確的數字命名，不套用羅盤方位名稱。
    index, azimuth, name = classify_azimuth_sector(100.0, 0.0, 0.0, 0.0, sector_count=8)
    assert name == "sector_002"

    with pytest.raises(PostprocessError):
        classify_azimuth_sector(0.0, 0.0, 0.0, 0.0, sector_count=0)


def test_classify_boundary_arc_position_known_points() -> None:
    from lagrangian_backtracking.native_runner.postprocess import classify_boundary_arc_position

    geometry = _make_boundary_geometry("siteA")
    # own_local_open_boundary：(100,-100) -> (100,100)，長度 200，預設 20 段（每段 10 m）。

    # 線中點（s=100）應落在第 10 段。
    result = classify_boundary_arc_position(
        100.0, 0.0, "siteA_local_open_boundary", geometry, arc_segment_count=20
    )
    assert result is not None
    index, s_m, length_m = result
    assert length_m == pytest.approx(200.0)
    assert s_m == pytest.approx(100.0)
    assert index == 10

    # 起點（s=0）落在第 0 段。
    index, s_m, _length_m = classify_boundary_arc_position(
        100.0, -100.0, "siteA_local_open_boundary", geometry, arc_segment_count=20
    )
    assert (index, s_m) == (0, pytest.approx(0.0))

    # 終點（s=length）必須落在最後一段（index = K-1），不能因浮點誤差溢出成 K。
    index, s_m, _length_m = classify_boundary_arc_position(
        100.0, 100.0, "siteA_local_open_boundary", geometry, arc_segment_count=20
    )
    assert (index, s_m) == (19, pytest.approx(200.0))

    # 不是這個 geometry 已知的任何一個開放邊界 segment id：沒有對應線，回傳 None。
    assert (
        classify_boundary_arc_position(100.0, 0.0, "unknown_segment", geometry, arc_segment_count=20)
        is None
    )

    with pytest.raises(PostprocessError):
        classify_boundary_arc_position(
            100.0, 0.0, "siteA_local_open_boundary", geometry, arc_segment_count=0
        )


def test_boundary_subdivision_tables_and_departure_columns(tmp_path) -> None:
    """完整走一次 accumulator：驗證方位扇區／弧段分類、departures 新欄位、
    統計表組內相對來源權重加總為 1、幾何表内容正確。
    """

    n = 3
    identity = _make_identity_table(
        particle_ids=[f"p{i}" for i in range(n)],
        study_site_id=["siteA"] * n,
        settling_velocity_mps=[-0.001] * n,
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * n,
        # 受體位置全部在原點，與 _make_boundary_geometry 的兩條開放邊界線中心一致。
        x0_m=[0.0] * n,
        y0_m=[0.0] * n,
        receptor_id=["recvX"] * n,  # 同一個受體，中心就是 (0, 0)。
    )
    result = _make_result(
        rows=[0, 1, 2],
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION] * n,
        # 兩顆落在 own_local_open_boundary（正東方向），一顆落在 flow_open_boundary。
        final_x_m=[100.0, 100.0, 480.0],
        final_y_m=[0.0, 10.0, 0.0],
    )
    accumulator = PostprocessAccumulator(
        identity,
        output_dir=tmp_path,
        boundary_geometries={"siteA": _make_boundary_geometry("siteA")},
    )
    accumulator.add_batch("batch-0", result)
    manifest = accumulator.finalize()
    assert manifest.config["boundary_subdivision_available"] is True
    assert manifest.config["azimuth_sector_count"] == 16
    assert manifest.config["boundary_arc_segment_count"] == 20

    departures = pq.read_table(tmp_path / "departures.parquet").to_pylist()
    by_row = {row["row_particle_index"]: row for row in departures}
    assert by_row[0]["departure_azimuth_deg"] == pytest.approx(90.0)
    assert by_row[0]["departure_azimuth_sector_name"] == "E"
    assert by_row[0]["boundary_arc_segment_index"] == 10
    assert by_row[0]["boundary_arc_length_position_m"] == pytest.approx(100.0)
    assert by_row[2]["departure_azimuth_deg"] == pytest.approx(90.0)
    assert by_row[2]["boundary_arc_segment_index"] == 10  # flow line 的 s=500 -> 第 10 段（每段 50 m）

    sector_table = pq.read_table(tmp_path / "boundary_entry_by_site_azimuth_sector.parquet").to_pylist()
    total_sector_count = sum(row["crossed_count"] for row in sector_table)
    assert total_sector_count == n
    weight_sum = sum(row["relative_source_weight_within_group"] for row in sector_table)
    assert weight_sum == pytest.approx(1.0)
    # p0 (100,0) 與 p2 (480,0) 方位角恰好 90 度（正東，E）；p1 (100,10) 方位角約
    # 84.29 度，落在下一個扇區（ENE），因此 E 扇區只累積 2 顆。
    e_sector_rows = [row for row in sector_table if row["sector_name"] == "E"]
    assert len(e_sector_rows) == 1
    assert e_sector_rows[0]["crossed_count"] == 2

    arc_table = pq.read_table(tmp_path / "boundary_entry_by_site_arc_segment.parquet").to_pylist()
    assert sum(row["crossed_count"] for row in arc_table) == n
    arc_weight_sum = sum(row["relative_source_weight_within_group"] for row in arc_table)
    assert arc_weight_sum == pytest.approx(1.0)
    own_local_rows = [
        row for row in arc_table if row["base_boundary_segment_id"] == "siteA_local_open_boundary"
    ]
    assert sum(row["crossed_count"] for row in own_local_rows) == 2
    flow_rows = [row for row in arc_table if row["base_boundary_segment_id"] == "flow_open_boundary"]
    assert sum(row["crossed_count"] for row in flow_rows) == 1

    sector_geometry = pq.read_table(tmp_path / "boundary_sectors_geometry_by_site.parquet").to_pylist()
    assert len(sector_geometry) == 16  # 每站固定輸出全部扇區的幾何，不管有沒有粒子穿越
    assert {row["sector_name"] for row in sector_geometry} == set(COMPASS_16_SECTOR_NAMES)

    arc_geometry = pq.read_table(tmp_path / "boundary_arc_segments_geometry_by_site.parquet").to_pylist()
    # 兩條開放邊界線各 20 段。
    assert len(arc_geometry) == 40
    own_local_geom_rows = [
        row for row in arc_geometry if row["base_boundary_segment_id"] == "siteA_local_open_boundary"
    ]
    assert len(own_local_geom_rows) == 20
    first_segment = next(row for row in own_local_geom_rows if row["arc_segment_index"] == 0)
    assert first_segment["start_s_m"] == pytest.approx(0.0)
    assert first_segment["end_s_m"] == pytest.approx(10.0)
    assert first_segment["start_x_m"] == pytest.approx(100.0)
    assert first_segment["start_y_m"] == pytest.approx(-100.0)

    for csv_name in (
        "boundary_entry_by_site_azimuth_sector.csv",
        "boundary_entry_by_site_settling_velocity_azimuth_sector.csv",
        "boundary_entry_by_site_arc_segment.csv",
        "boundary_entry_by_site_settling_velocity_arc_segment.csv",
        "boundary_sectors_geometry_by_site.csv",
        "boundary_arc_segments_geometry_by_site.csv",
        "boundary_summary_by_site.csv",
    ):
        assert (tmp_path / csv_name).is_file()


def test_boundary_subdivision_empty_group_outputs_empty_table_not_error(tmp_path) -> None:
    """全部粒子都未穿越邊界時，方位扇區／弧段統計表必須是空表而不是拋例外，
    但幾何表（不依賴有無穿越粒子）仍應完整輸出。
    """

    identity = _make_identity_table(
        particle_ids=["p0", "p1"],
        study_site_id=["siteA", "siteA"],
        settling_velocity_mps=[-0.001] * 2,
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * 2,
    )
    result = _make_result(
        rows=[0, 1],
        final_status=[NATIVE_STATUS_MAX_BACKTRACK, NATIVE_STATUS_STRANDED],
    )
    accumulator = PostprocessAccumulator(
        identity,
        output_dir=tmp_path,
        boundary_geometries={"siteA": _make_boundary_geometry("siteA")},
    )
    accumulator.add_batch("batch-0", result)
    accumulator.finalize()

    sector_table = pq.read_table(tmp_path / "boundary_entry_by_site_azimuth_sector.parquet")
    assert sector_table.num_rows == 0
    assert set(sector_table.schema.names) >= {
        "study_site_id", "sector_index", "sector_name", "crossed_count",
        "relative_source_weight_within_group",
    }

    arc_table = pq.read_table(tmp_path / "boundary_entry_by_site_arc_segment.parquet")
    assert arc_table.num_rows == 0

    velocity_sector_table = pq.read_table(
        tmp_path / "boundary_entry_by_site_settling_velocity_azimuth_sector.parquet"
    )
    assert velocity_sector_table.num_rows == 0
    assert velocity_sector_table.schema.field("settling_velocity_mps").type == pa.float64()

    # 幾何表不受「有沒有穿越粒子」影響，只要站有邊界幾何就完整輸出。
    sector_geometry = pq.read_table(tmp_path / "boundary_sectors_geometry_by_site.parquet")
    assert sector_geometry.num_rows == 16
    arc_geometry = pq.read_table(tmp_path / "boundary_arc_segments_geometry_by_site.parquet")
    assert arc_geometry.num_rows == 40


def test_boundary_summary_by_site_matches_flux_definition(tmp_path) -> None:
    """``boundary_summary_by_site`` 的比例欄位必須與既有 ``boundary_flux_by_site``
    的計數換算結果完全一致（同一分母），且不需要 boundary_geometries 就會輸出。
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
        NATIVE_STATUS_STRANDED,
        NATIVE_STATUS_DEPOSITED,
    ]
    result = _make_result(rows=list(range(4)), final_status=statuses)

    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path, config=PostprocessConfig())
    accumulator.add_batch("batch-0", result)
    accumulator.finalize()

    assert (tmp_path / "boundary_summary_by_site.parquet").is_file()
    assert (tmp_path / "boundary_summary_by_site.csv").is_file()

    flux_row = pq.read_table(tmp_path / "boundary_flux_by_site.parquet").to_pylist()[0]
    summary_row = pq.read_table(tmp_path / "boundary_summary_by_site.parquet").to_pylist()[0]

    assert summary_row["total_particle_count"] == flux_row["total_particle_count"] == 4
    assert summary_row["fraction_crossed_interest_region_boundary"] == pytest.approx(
        flux_row["fraction_crossed"]
    )
    assert summary_row["fraction_max_backtrack"] == pytest.approx(
        flux_row[f"fraction_status_{NATIVE_STATUS_MAX_BACKTRACK}"]
    )
    assert summary_row["fraction_stranded"] == pytest.approx(
        flux_row[f"fraction_status_{NATIVE_STATUS_STRANDED}"]
    )
    assert summary_row["fraction_deposited"] == pytest.approx(
        flux_row[f"fraction_status_{NATIVE_STATUS_DEPOSITED}"]
    )


# ══════════════════════════════════════════════════════════════════════════
# WP-37：系集成員數（M）直接由 member_id 相異值算出，並 fail-closed 檢核一致性
# ══════════════════════════════════════════════════════════════════════════


def test_ensemble_member_count_computed_from_member_id_and_written_to_summary_and_manifest(
    tmp_path,
) -> None:
    """單站、2 個情境、各自 3 個系集成員（``member_id`` 0/1/2）：
    ``boundary_summary_by_site.ensemble_member_count`` 與 manifest 的
    ``ensemble_member_count_by_site`` 都必須回報 3，且直接由資料算出（本測試
    完全沒有提供任何「M=3」設定值，只提供 member_id 本身）。"""

    particle_ids = [f"p{i}" for i in range(6)]
    scenario_ids = ["scnA", "scnA", "scnA", "scnB", "scnB", "scnB"]
    member_ids = [0, 1, 2, 0, 1, 2]
    identity = _make_identity_table(
        particle_ids=particle_ids,
        study_site_id=["A"] * 6,
        settling_velocity_mps=[-0.001] * 6,
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * 6,
        scenario_id=scenario_ids,
        member_id=member_ids,
    )
    result = _make_result(rows=list(range(6)), final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION] * 6)

    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path, config=PostprocessConfig())
    accumulator.add_batch("batch-0", result)
    manifest = accumulator.finalize()

    summary_row = pq.read_table(tmp_path / "boundary_summary_by_site.parquet").to_pylist()[0]
    assert summary_row["ensemble_member_count"] == 3
    assert manifest.ensemble_member_count_by_site == {"A": 3}

    manifest_json = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_json["ensemble_member_count_by_site"] == {"A": 3}


def test_ensemble_member_count_fails_closed_when_a_scenario_has_fewer_members(tmp_path) -> None:
    """同一站內某個情境的相異 ``member_id`` 數量與其他情境不同（此處 ``scnA``
    有 3 個成員、``scnB`` 只有 2 個），必須在建構 ``PostprocessAccumulator`` 時
    就 fail-closed（不用等到 ``finalize``），錯誤訊息附上實際數字與情境／站點
    名稱，不嘗試補齊或忽略。"""

    particle_ids = [f"p{i}" for i in range(5)]
    scenario_ids = ["scnA", "scnA", "scnA", "scnB", "scnB"]
    member_ids = [0, 1, 2, 0, 1]
    identity = _make_identity_table(
        particle_ids=particle_ids,
        study_site_id=["A"] * 5,
        settling_velocity_mps=[-0.001] * 5,
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * 5,
        scenario_id=scenario_ids,
        member_id=member_ids,
    )
    with pytest.raises(PostprocessError) as exc_info:
        PostprocessAccumulator(identity, output_dir=tmp_path, config=PostprocessConfig())
    message = str(exc_info.value)
    assert "A" in message
    assert "scnB" in message
    assert "2" in message
    assert "3" in message


def test_ensemble_member_count_fails_closed_when_sites_disagree(tmp_path) -> None:
    """站與站之間系集成員數不同（此處站 A 的情境有 3 個成員、站 B 的情境只有
    2 個），即使每一站「站內各情境互相一致」，仍必須 fail-closed——圖說與副標
    只能寫一個 M，不能對應到不同站不同的實際系集大小。"""

    particle_ids = [f"p{i}" for i in range(5)]
    study_site_id = ["A", "A", "A", "B", "B"]
    scenario_ids = ["scnA", "scnA", "scnA", "scnB", "scnB"]
    member_ids = [0, 1, 2, 0, 1]
    identity = _make_identity_table(
        particle_ids=particle_ids,
        study_site_id=study_site_id,
        settling_velocity_mps=[-0.001] * 5,
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * 5,
        scenario_id=scenario_ids,
        member_id=member_ids,
    )
    with pytest.raises(PostprocessError, match="各站系集成員數不一致"):
        PostprocessAccumulator(identity, output_dir=tmp_path, config=PostprocessConfig())


# ══════════════════════════════════════════════════════════════════════════
# WP-23：終止邊界改為 flow domain；local domain 首次離開（非終止事件）另外出表
# ══════════════════════════════════════════════════════════════════════════


def test_local_domain_departures_written_for_local_not_equals_flow_site(tmp_path) -> None:
    """local_equals_flow=False 的測站（例如貢寮／龜山島）：一顆粒子先產生非終止的
    ``LocalDomainFirstExitRecord``（記錄在 local_domain_departures.parquet），
    之後才在 flow domain 邊界終止（記錄在 departures.parquet，
    also_local_domain_first_exit 必須是 False）。
    """

    identity = _make_identity_table(
        particle_ids=["p0"],
        study_site_id=["siteA"],
        settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    result = _make_result(
        rows=[0],
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION],
        final_x_m=[480.0],
        final_y_m=[0.0],
        final_utc_ns=[_JAN_2024_UTC_NS - 2 * _HOUR_NS],
        local_domain_departures=[
            LocalDomainFirstExitRecord(
                row_particle_index=0, x_m=100.0, y_m=0.0, z_m=-5.0,
                utc_ns=_JAN_2024_UTC_NS - _HOUR_NS,
            )
        ],
    )
    accumulator = PostprocessAccumulator(
        identity, output_dir=tmp_path,
        boundary_geometries={"siteA": _make_boundary_geometry("siteA", local_equals_flow=False)},
    )
    accumulator.add_batch("batch-0", result)
    manifest = accumulator.finalize()

    assert manifest.config["local_equals_flow_by_site"] == {"siteA": False}

    local_rows = pq.read_table(tmp_path / "local_domain_departures.parquet").to_pylist()
    assert len(local_rows) == 1
    row = local_rows[0]
    assert row["row_particle_index"] == 0
    assert row["study_site_id"] == "siteA"
    assert row["local_first_exit_x_m"] == pytest.approx(100.0)
    assert row["local_first_exit_utc_ns"] == _JAN_2024_UTC_NS - _HOUR_NS
    assert row["local_first_exit_backtrack_duration_seconds"] == pytest.approx(_HOUR_NS / 1.0e9)
    # 離開點就在 own_local_open_boundary 線上（東側），應該分類到本站 local 分段。
    assert row["local_boundary_segment_id"] == "siteA_local_open_boundary"
    assert row["local_boundary_classification_distance_m"] == pytest.approx(0.0, abs=1.0e-9)
    assert row["local_first_exit_lon_deg"] is None  # 未提供 lonlat_projector

    departures = pq.read_table(tmp_path / "departures.parquet").to_pylist()
    assert len(departures) == 1
    assert departures[0]["crossed_interest_region_boundary"] is True
    assert departures[0]["also_local_domain_first_exit"] is False

    summary_rows = {
        r["study_site_id"]: r
        for r in pq.read_table(
            tmp_path / "local_domain_first_exit_summary_by_site.parquet"
        ).to_pylist()
    }
    assert summary_rows["siteA"]["local_equals_flow"] is False
    assert summary_rows["siteA"]["local_first_exit_particle_count"] == 1
    assert summary_rows["siteA"]["fraction_local_first_exit"] == pytest.approx(1.0)
    assert summary_rows["siteA"]["local_first_exit_source"] == "local_domain_departures"


def test_local_equals_flow_site_reuses_flow_departure_without_duplicating(tmp_path) -> None:
    """local_equals_flow=True 的測站（例如新竹／南灣／連江）：local 首次離開與 flow
    終止是同一次穿越，只能有一筆 ``DepartureRecord``（``also_local_domain_first_
    exit=True``），不得在 ``local_domain_departures.parquet`` 另外出現一列。
    """

    identity = _make_identity_table(
        particle_ids=["p0", "p1"],
        study_site_id=["siteB", "siteB"],
        settling_velocity_mps=[-0.001, -0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * 2,
    )
    departures = [
        DepartureRecord(
            row_particle_index=0, status=NATIVE_STATUS_LEFT_INTEREST_REGION,
            x_m=100.0, y_m=0.0, z_m=-5.0, utc_ns=_JAN_2024_UTC_NS - _HOUR_NS,
            also_local_domain_first_exit=True,
        ),
        DepartureRecord(
            row_particle_index=1, status=NATIVE_STATUS_MAX_BACKTRACK,
            x_m=0.0, y_m=0.0, z_m=-5.0, utc_ns=_JAN_2024_UTC_NS - 2 * _HOUR_NS,
            also_local_domain_first_exit=False,
        ),
    ]
    result = _make_result(
        rows=[0, 1],
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION, NATIVE_STATUS_MAX_BACKTRACK],
        departures=departures,
    )
    accumulator = PostprocessAccumulator(
        identity, output_dir=tmp_path,
        boundary_geometries={"siteB": _make_boundary_geometry("siteB", local_equals_flow=True)},
    )
    accumulator.add_batch("batch-0", result)
    manifest = accumulator.finalize()

    assert manifest.config["local_equals_flow_by_site"] == {"siteB": True}

    local_rows = pq.read_table(tmp_path / "local_domain_departures.parquet").to_pylist()
    assert local_rows == []  # local_equals_flow=True：不重複寫入

    summary_rows = {
        r["study_site_id"]: r
        for r in pq.read_table(
            tmp_path / "local_domain_first_exit_summary_by_site.parquet"
        ).to_pylist()
    }
    # 只有 p0（also_local_domain_first_exit=True）算一次；p1（MAX_BACKTRACK，
    # 從未離開 local domain）不計入分子，但仍計入分母。
    assert summary_rows["siteB"]["local_equals_flow"] is True
    assert summary_rows["siteB"]["total_particle_count"] == 2
    assert summary_rows["siteB"]["local_first_exit_particle_count"] == 1
    assert summary_rows["siteB"]["fraction_local_first_exit"] == pytest.approx(0.5)
    assert summary_rows["siteB"]["local_first_exit_source"] == "same_as_flow_domain_departure"


def test_local_domain_departures_without_boundary_geometries_have_null_classification(
    tmp_path,
) -> None:
    """沒有提供 ``boundary_geometries`` 時，local_domain_departures.parquet 仍然
    記錄位置與時刻，但分類欄位一律明確 ``null``（不得以任何分段代碼頂替）。
    ``local_equals_flow`` 雖然沒有 ``boundary_geometries`` 可查，但這個站的粒子
    確實出現在 ``local_domain_departures`` 裡，資料本身已經能明確推斷這個站是
    ``local_equals_flow=False``（見 ``_check_local_equals_flow_consistency``
    docstring：沒有提供 boundary_geometries 時改以資料本身推斷並登記）。
    """

    identity = _make_identity_table(
        particle_ids=["p0"], study_site_id=["siteA"], settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    result = _make_result(
        rows=[0],
        final_status=[NATIVE_STATUS_MAX_BACKTRACK],
        local_domain_departures=[
            LocalDomainFirstExitRecord(
                row_particle_index=0, x_m=100.0, y_m=0.0, z_m=-5.0,
                utc_ns=_JAN_2024_UTC_NS - _HOUR_NS,
            )
        ],
    )
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path)
    accumulator.add_batch("batch-0", result)
    manifest = accumulator.finalize()

    assert manifest.config["local_equals_flow_by_site"] == {"siteA": False}

    local_rows = pq.read_table(tmp_path / "local_domain_departures.parquet").to_pylist()
    assert len(local_rows) == 1
    assert local_rows[0]["local_boundary_segment_id"] is None
    assert local_rows[0]["local_boundary_classification_distance_m"] is None
    assert local_rows[0]["local_departure_azimuth_deg"] is None

    summary_rows = {
        r["study_site_id"]: r
        for r in pq.read_table(
            tmp_path / "local_domain_first_exit_summary_by_site.parquet"
        ).to_pylist()
    }
    assert summary_rows["siteA"]["local_equals_flow"] is False
    assert summary_rows["siteA"]["local_first_exit_source"] == "local_domain_departures"
    assert summary_rows["siteA"]["local_first_exit_particle_count"] == 1


def test_local_domain_first_exit_summary_unknown_when_no_evidence_and_no_geometry(tmp_path) -> None:
    """沒有提供 ``boundary_geometries``、且這個站的粒子完全沒有任何 local 首次
    離開的證據（既沒有 local_domain_departures，也沒有 also_local_domain_first_
    exit=True）時，``local_equals_flow`` 真的無法判斷，統計表必須誠實列出
    ``"unknown_local_equals_flow"``，不得臆測，分子維持 0（不是缺值，是「目前
    這批粒子裡沒有任何一顆離開過 local domain」）。
    """

    identity = _make_identity_table(
        particle_ids=["p0"], study_site_id=["siteA"], settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    result = _make_result(rows=[0], final_status=[NATIVE_STATUS_MAX_BACKTRACK])
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path)
    accumulator.add_batch("batch-0", result)
    manifest = accumulator.finalize()

    assert manifest.config["local_equals_flow_by_site"] == {}
    assert pq.read_table(tmp_path / "local_domain_departures.parquet").to_pylist() == []

    summary_rows = {
        r["study_site_id"]: r
        for r in pq.read_table(
            tmp_path / "local_domain_first_exit_summary_by_site.parquet"
        ).to_pylist()
    }
    assert summary_rows["siteA"]["local_equals_flow"] is None
    assert summary_rows["siteA"]["local_first_exit_source"] == "unknown_local_equals_flow"
    assert summary_rows["siteA"]["local_first_exit_particle_count"] == 0


def test_local_domain_departures_and_also_local_flag_cannot_both_be_true(tmp_path) -> None:
    """同一顆粒子不能同時出現在 local_domain_departures 又帶
    also_local_domain_first_exit=True——這兩者分別代表 local_equals_flow=False
    與 True 兩種互斥情況，一致性檢查必須 fail-closed。
    """

    identity = _make_identity_table(
        particle_ids=["p0"], study_site_id=["siteA"], settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    departures = [
        DepartureRecord(
            row_particle_index=0, status=NATIVE_STATUS_LEFT_INTEREST_REGION,
            x_m=480.0, y_m=0.0, z_m=-5.0, utc_ns=_JAN_2024_UTC_NS - _HOUR_NS,
            also_local_domain_first_exit=True,
        ),
    ]
    result = _make_result(
        rows=[0],
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION],
        departures=departures,
        local_domain_departures=[
            LocalDomainFirstExitRecord(
                row_particle_index=0, x_m=100.0, y_m=0.0, z_m=-5.0,
                utc_ns=_JAN_2024_UTC_NS - 2 * _HOUR_NS,
            )
        ],
    )
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path)
    with pytest.raises(PostprocessError):
        accumulator.add_batch("batch-0", result)


def test_local_domain_departures_declared_equals_flow_conflict_fails_closed(tmp_path) -> None:
    """boundary_geometries 宣告某站 local_equals_flow=True，但資料卻對這一站的
    粒子回報了 LocalDomainFirstExitRecord（只有 local_equals_flow=False 的測站
    才應該有這個非終止事件）：設定與資料不同步，必須 fail-closed。
    """

    identity = _make_identity_table(
        particle_ids=["p0"], study_site_id=["siteB"], settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    result = _make_result(
        rows=[0],
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION],
        local_domain_departures=[
            LocalDomainFirstExitRecord(
                row_particle_index=0, x_m=100.0, y_m=0.0, z_m=-5.0,
                utc_ns=_JAN_2024_UTC_NS - _HOUR_NS,
            )
        ],
    )
    accumulator = PostprocessAccumulator(
        identity, output_dir=tmp_path,
        boundary_geometries={"siteB": _make_boundary_geometry("siteB", local_equals_flow=True)},
    )
    with pytest.raises(PostprocessError):
        accumulator.add_batch("batch-0", result)


def test_local_domain_departures_missing_boundary_geometry_for_site_fails_closed(tmp_path) -> None:
    """提供了 boundary_geometries（非 None）但缺少某一站的登錄，該站卻有
    local_domain_departures：與既有 departures.parquet 的 fail-closed 慣例一致
    （見 test_boundary_geometries_missing_site_fails_closed）。
    """

    identity = _make_identity_table(
        particle_ids=["p0"], study_site_id=["siteC"], settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    result = _make_result(
        rows=[0],
        final_status=[NATIVE_STATUS_MAX_BACKTRACK],
        local_domain_departures=[
            LocalDomainFirstExitRecord(
                row_particle_index=0, x_m=100.0, y_m=0.0, z_m=-5.0,
                utc_ns=_JAN_2024_UTC_NS - _HOUR_NS,
            )
        ],
    )
    accumulator = PostprocessAccumulator(
        identity, output_dir=tmp_path,
        boundary_geometries={"siteA": _make_boundary_geometry("siteA")},
    )
    with pytest.raises(PostprocessError):
        accumulator.add_batch("batch-0", result)


# ── WP-23 修補：npz 持久化往返端對端測試 ─────────────────────────────────────
#
# 背景：WP-23 上線後，正式伺服器上發現 ``BackwardDriverResult.local_domain_
# departures`` 與 ``DepartureRecord.also_local_domain_first_exit`` 在記憶體中
# 都正確（以上單元測試全部只用記憶體物件直接餵給 ``PostprocessAccumulator.
# add_batch``，完全沒有經過任何檔案），但 ``hour_major_provider.
# save_backward_driver_result_npz`` 序列化成 ``.npz`` 時把這兩個新欄位整個
# 漏接，導致正式後處理讀回的每一站 ``local_first_exit_particle_count`` 全部
# 是 0（見 reports/WP-23-fix.md）。以下測試刻意插入
# ``save_backward_driver_result_npz`` → ``load_backward_driver_result_npz``
# 這一次完整的檔案往返，才呼叫 ``add_batch``，直接對應「跑完 → 寫檔 → 讀回 →
# 後處理」這條正式管線，才能真正抓到這類「單元測試只驗記憶體物件」的漏洞。


def test_npz_roundtrip_preserves_local_domain_departures_for_local_not_equals_flow_site(
    tmp_path,
) -> None:
    """端對端：``local_equals_flow=False`` 站（例如貢寮／龜山島）的
    ``LocalDomainFirstExitRecord`` 經過 ``save_backward_driver_result_npz`` →
    ``load_backward_driver_result_npz`` 往返後，後處理算出的
    ``local_first_exit_particle_count`` 必須是 1（不是 0）。

    修補前的失敗方式：``save_backward_driver_result_npz`` 沒有寫出任何
    ``local_departure_*``／``n_local_departures`` 欄位，讀回的
    ``BackwardDriverResult.local_domain_departures`` 因此永遠是空 tuple，
    這裡的斷言會失敗在 ``local_first_exit_particle_count == 0``
    （而不是預期的 1）。
    """

    from lagrangian_backtracking.native_runner import hour_major_provider as hmp

    identity = _make_identity_table(
        particle_ids=["p0"],
        study_site_id=["siteA"],
        settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    result = _make_result(
        rows=[0],
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION],
        final_x_m=[480.0],
        final_y_m=[0.0],
        final_utc_ns=[_JAN_2024_UTC_NS - 2 * _HOUR_NS],
        local_domain_departures=[
            LocalDomainFirstExitRecord(
                row_particle_index=0, x_m=100.0, y_m=0.0, z_m=-5.0,
                utc_ns=_JAN_2024_UTC_NS - _HOUR_NS,
            )
        ],
    )

    npz_path = tmp_path / "unit-0.driver_result.npz"
    hmp.save_backward_driver_result_npz(npz_path, result)
    roundtripped = hmp.load_backward_driver_result_npz(npz_path)

    # 往返後的物件必須與寫入前完全一致，這是能正確後處理的前提。
    assert len(roundtripped.local_domain_departures) == 1
    local_record = roundtripped.local_domain_departures[0]
    assert local_record.row_particle_index == 0
    assert local_record.x_m == pytest.approx(100.0)
    assert local_record.y_m == pytest.approx(0.0)
    assert local_record.z_m == pytest.approx(-5.0)
    assert local_record.utc_ns == _JAN_2024_UTC_NS - _HOUR_NS
    assert len(roundtripped.departures) == 1
    assert roundtripped.departures[0].also_local_domain_first_exit is False

    accumulator = PostprocessAccumulator(
        identity, output_dir=tmp_path / "postprocess_out",
        boundary_geometries={"siteA": _make_boundary_geometry("siteA", local_equals_flow=False)},
    )
    accumulator.add_batch("batch-0", roundtripped)
    accumulator.finalize()

    local_rows = pq.read_table(
        tmp_path / "postprocess_out" / "local_domain_departures.parquet"
    ).to_pylist()
    assert len(local_rows) == 1
    assert local_rows[0]["row_particle_index"] == 0

    summary_rows = {
        r["study_site_id"]: r
        for r in pq.read_table(
            tmp_path / "postprocess_out" / "local_domain_first_exit_summary_by_site.parquet"
        ).to_pylist()
    }
    assert summary_rows["siteA"]["local_equals_flow"] is False
    assert summary_rows["siteA"]["local_first_exit_particle_count"] == 1
    assert summary_rows["siteA"]["local_first_exit_source"] == "local_domain_departures"


def test_npz_roundtrip_preserves_also_local_domain_first_exit_for_local_equals_flow_site(
    tmp_path,
) -> None:
    """端對端：``local_equals_flow=True`` 站（例如新竹／南灣／連江）的
    ``DepartureRecord.also_local_domain_first_exit=True`` 經過
    ``save_backward_driver_result_npz`` → ``load_backward_driver_result_npz``
    往返後仍然是 ``True``，後處理算出的 ``local_first_exit_particle_count``
    必須是 1（不是 0）。

    修補前的失敗方式：``departure_also_local_first_exit`` 完全沒有寫進
    ``.npz``，讀回的每一筆 ``DepartureRecord.also_local_domain_first_exit``
    都退回 dataclass 預設值 ``False``，這裡的斷言會失敗。
    """

    from lagrangian_backtracking.native_runner import hour_major_provider as hmp

    identity = _make_identity_table(
        particle_ids=["p0", "p1"],
        study_site_id=["siteB", "siteB"],
        settling_velocity_mps=[-0.001, -0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * 2,
    )
    departures = [
        DepartureRecord(
            row_particle_index=0, status=NATIVE_STATUS_LEFT_INTEREST_REGION,
            x_m=100.0, y_m=0.0, z_m=-5.0, utc_ns=_JAN_2024_UTC_NS - _HOUR_NS,
            also_local_domain_first_exit=True,
        ),
        DepartureRecord(
            row_particle_index=1, status=NATIVE_STATUS_MAX_BACKTRACK,
            x_m=0.0, y_m=0.0, z_m=-5.0, utc_ns=_JAN_2024_UTC_NS - 2 * _HOUR_NS,
            also_local_domain_first_exit=False,
        ),
    ]
    result = _make_result(
        rows=[0, 1],
        final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION, NATIVE_STATUS_MAX_BACKTRACK],
        departures=departures,
    )

    npz_path = tmp_path / "unit-0.driver_result.npz"
    hmp.save_backward_driver_result_npz(npz_path, result)
    roundtripped = hmp.load_backward_driver_result_npz(npz_path)

    by_row = {d.row_particle_index: d for d in roundtripped.departures}
    assert by_row[0].also_local_domain_first_exit is True
    assert by_row[1].also_local_domain_first_exit is False
    assert roundtripped.local_domain_departures == ()

    accumulator = PostprocessAccumulator(
        identity, output_dir=tmp_path / "postprocess_out",
        boundary_geometries={"siteB": _make_boundary_geometry("siteB", local_equals_flow=True)},
    )
    accumulator.add_batch("batch-0", roundtripped)
    accumulator.finalize()

    local_rows = pq.read_table(
        tmp_path / "postprocess_out" / "local_domain_departures.parquet"
    ).to_pylist()
    assert local_rows == []  # local_equals_flow=True：不重複寫入

    summary_rows = {
        r["study_site_id"]: r
        for r in pq.read_table(
            tmp_path / "postprocess_out" / "local_domain_first_exit_summary_by_site.parquet"
        ).to_pylist()
    }
    assert summary_rows["siteB"]["local_equals_flow"] is True
    assert summary_rows["siteB"]["local_first_exit_particle_count"] == 1
    assert summary_rows["siteB"]["fraction_local_first_exit"] == pytest.approx(0.5)
    assert summary_rows["siteB"]["local_first_exit_source"] == "same_as_flow_domain_departure"


def test_npz_roundtrip_backward_compatible_with_pre_wp23_result_file(tmp_path) -> None:
    """向後相容：WP-23 之前（run01 等）寫出的舊 ``.npz`` 沒有
    ``local_departure_*``／``departure_also_local_first_exit`` 欄位，讀回不得
    崩潰，且語意是「這個單元沒有任何 local 首次離開事件」（那個功能寫出當下
    還不存在），不是「未知」——``postprocess.py`` 對「這一站 local_equals_flow
    本身未知」另有專門語意（``unknown_local_equals_flow``），這裡驗證的是
    npz 讀回層級的相容性，不涉及那個分類。

    做法：直接用 ``save_backward_driver_result_npz`` 目前的邏輯手造一份「去掉
    WP-23 新欄位」的 npz（模擬 run01 的舊格式），確認讀回不拋例外、且兩個新
    欄位分別退回空 tuple／``False``，餵給後處理後 ``local_first_exit_
    particle_count`` 是 0（沒有證據，不是因為讀取失敗被默默吃掉）。
    """

    from lagrangian_backtracking.native_runner import hour_major_provider as hmp

    identity = _make_identity_table(
        particle_ids=["p0"], study_site_id=["siteA"], settling_velocity_mps=[-0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS],
    )
    result = _make_result(
        rows=[0],
        final_status=[NATIVE_STATUS_MAX_BACKTRACK],
    )

    npz_path = tmp_path / "unit-0.driver_result.npz"
    hmp.save_backward_driver_result_npz(npz_path, result)

    # 模擬 run01：把目前版本寫出的 npz 讀回，去掉 WP-23 新增的欄位後重新存檔，
    # 得到一份與修補前的正式舊結果檔欄位完全相同的「舊格式」檔案。
    with np.load(str(npz_path), allow_pickle=False) as npz:
        legacy_payload = {
            key: npz[key]
            for key in npz.files
            if key
            not in (
                "departure_also_local_first_exit",
                "local_departure_row",
                "local_departure_x_m",
                "local_departure_y_m",
                "local_departure_z_m",
                "local_departure_utc_ns",
                "n_local_departures",
            )
        }
    legacy_path = tmp_path / "legacy-unit-0.driver_result.npz"
    np.savez(str(legacy_path), **legacy_payload)

    roundtripped = hmp.load_backward_driver_result_npz(legacy_path)
    assert roundtripped.local_domain_departures == ()
    assert all(not d.also_local_domain_first_exit for d in roundtripped.departures)

    accumulator = PostprocessAccumulator(
        identity, output_dir=tmp_path / "postprocess_out",
        boundary_geometries={"siteA": _make_boundary_geometry("siteA", local_equals_flow=False)},
    )
    accumulator.add_batch("batch-0", roundtripped)
    accumulator.finalize()

    summary_rows = {
        r["study_site_id"]: r
        for r in pq.read_table(
            tmp_path / "postprocess_out" / "local_domain_first_exit_summary_by_site.parquet"
        ).to_pylist()
    }
    assert summary_rows["siteA"]["local_first_exit_particle_count"] == 0
    assert summary_rows["siteA"]["local_first_exit_source"] == "local_domain_departures"


# ── WP-30：CLI ``native-postprocess`` 的 --project-root 端對端佈線 ──────────


def _write_native_postprocess_fixture(tmp_path):
    """寫出一份最小可用的 identity table npz 與單一批次 driver_result.npz。

    刻意不經過任何 BayTrace 或 native-run 流程，直接用 ``save_*_npz`` 序列化
    合成資料，只為了驗證 ``lbt native-postprocess`` CLI 這一層有沒有把
    ``--project-root``／``--declared-git-commit``／``--formal`` 正確接到
    ``PostprocessConfig``，不重覆驗證後處理數值邏輯本身（那是本檔其餘測試
    與 ``test_native_hour_major_provider.py`` 的範圍）。
    """

    identity = _make_identity_table(
        particle_ids=["p0", "p1"],
        study_site_id=["A", "A"],
        settling_velocity_mps=[-0.001, -0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS, _JAN_2024_UTC_NS],
    )
    identity_path = tmp_path / "identity.npz"
    save_particle_identity_table_npz(identity, identity_path)

    result = _make_result(rows=[0, 1], final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION] * 2)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    save_backward_driver_result_npz(results_dir / "unit-0.driver_result.npz", result)

    return identity_path, results_dir


def test_cli_native_postprocess_without_project_root_keeps_legacy_skip_note(tmp_path):
    """不給 ``--project-root`` 時，``code_provenance`` 必須與加入本參數前完全相同。

    也就是仍然只有那一句「未提供 project_root，略過程式部署指紋收集」說明，
    不能因為新增了這個選項就意外改變既有輸出（WP-30 任務要求的回溯相容性）。
    """

    identity_path, results_dir = _write_native_postprocess_fixture(tmp_path)
    output_dir = tmp_path / "out_no_root"

    code = cli.run_native_postprocess_command(
        [
            "--identity-table", str(identity_path),
            "--results-dir", str(results_dir),
            "--output", str(output_dir),
        ]
    )
    assert code in (0, None)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["code_provenance"] == {
        "note": "未提供 project_root，略過程式部署指紋收集（僅限測試或探索用途）"
    }


def test_cli_native_postprocess_with_project_root_collects_code_provenance(tmp_path):
    """給了 ``--project-root`` 時，manifest 的 ``code_provenance`` 必須有實質內容。

    「實質內容」指真正呼叫了 ``provenance.collect_code_provenance``：能看到
    ``git_available``、部署樹雜湊、套件與依賴版本等欄位，而不是那句略過說明。
    本機工作目錄未必是 Git repo（例如 SERVER 發布只有原始碼、沒有 ``.git``），
    所以本測試不斷言 ``git_commit`` 一定非 null，只斷言確實收集到了指紋本體。
    """

    identity_path, results_dir = _write_native_postprocess_fixture(tmp_path)
    output_dir = tmp_path / "out_with_root"
    project_root = Path(__file__).resolve().parents[1]

    code = cli.run_native_postprocess_command(
        [
            "--identity-table", str(identity_path),
            "--results-dir", str(results_dir),
            "--output", str(output_dir),
            "--project-root", str(project_root),
        ]
    )
    assert code in (0, None)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    code_provenance = manifest["code_provenance"]
    assert "note" not in code_provenance
    assert "git_available" in code_provenance
    assert code_provenance["deployment_tree_sha256"]
    assert code_provenance["uv_lock_sha256"]
    assert code_provenance["package_version"]
    assert code_provenance["python_version"]
