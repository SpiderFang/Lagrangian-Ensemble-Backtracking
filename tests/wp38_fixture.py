"""WP-38 測試共用的合成資料與輔助函式。

本檔案**只依賴 WP-38 之前就存在的模組**（``particle_input``、``backward_driver``、
``postprocess`` 的舊介面），因此可以在 WP-38 之前的程式碼上執行，用來產生
「既有產品」的回歸基準值（見 ``tests/test_footprint.py`` 的回歸測試）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from lagrangian_backtracking.native_runner.backward_driver import (
    NATIVE_STATUS_LEFT_FLOW_DOMAIN,
    NATIVE_STATUS_MAX_BACKTRACK,
    BackwardDriverResult,
    DepartureRecord,
    HourlyOverheadDiagnostics,
    SubstepDiagnostics,
)
from lagrangian_backtracking.native_runner.particle_input import ParticleIdentityTable

HOUR_NS = 3_600_000_000_000
ARRIVAL_BASE_NS = 1_704_110_400_000_000_000  # 2024-01-01T12:00:00Z

# 線性投影：以 (121.0E, 25.0N) 為原點，x 向東、y 向北（公尺）。測試只需要一個可逆、
# 可手算的對應，不需要 pyproj。
LON0, LAT0 = 121.0, 25.0
_M_PER_DEG_LAT = 110574.0
_M_PER_DEG_LON = 111320.0 * float(np.cos(np.radians(LAT0)))


def linear_projector(site_ids: np.ndarray, x_m: np.ndarray, y_m: np.ndarray):
    """``LonLatProjector``：公尺座標到經緯度的線性對應（所有站共用同一原點）。"""

    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    return LON0 + x / _M_PER_DEG_LON, LAT0 + y / _M_PER_DEG_LAT


def lonlat_to_xy(lon: np.ndarray, lat: np.ndarray):
    """``linear_projector`` 的反函式。"""

    return (np.asarray(lon) - LON0) * _M_PER_DEG_LON, (np.asarray(lat) - LAT0) * _M_PER_DEG_LAT


def build_world(
    *,
    site: str = "siteA",
    velocities: tuple[float, ...] = (-0.001, -0.002),
    members: int = 4,
    receptors: int = 2,
    arrivals: int = 3,
    receptor_xy: tuple[tuple[float, float], ...] | None = None,
):
    """建立粒子身分表與「工作單元」清單。

    一個工作單元＝同一到達時刻、同一沉降速度的全部受體與系集成員粒子（與正式運算的
    單元定義一致）。回傳 ``(identity, units)``，``units`` 是
    ``[(unit_key, 全域列位置陣列)]``。
    """

    if receptor_xy is None:
        receptor_xy = tuple((-30.0 + 60.0 * r, 20.0 * r) for r in range(receptors))
    cols: dict[str, list[Any]] = {
        k: []
        for k in (
            "particle_id",
            "scenario_id",
            "member_id",
            "study_site_id",
            "analysis_region_id",
            "receptor_id",
            "material_id",
            "arrival_time_id",
            "settling_velocity_mps",
            "arrival_time_utc_ns",
            "seed_hex",
            "x0_m",
            "y0_m",
            "z0_m",
        )
    }
    units: list[tuple[str, list[int]]] = []
    row = 0
    for a in range(arrivals):
        for v, velocity in enumerate(velocities):
            unit_rows: list[int] = []
            for r in range(len(receptor_xy)):
                for m in range(members):
                    cols["particle_id"].append(f"{site}-a{a}-v{v}-r{r}-m{m}")
                    cols["scenario_id"].append(f"scn-{site}-a{a}-v{v}-r{r}")
                    cols["member_id"].append(m)
                    cols["study_site_id"].append(site)
                    cols["analysis_region_id"].append("region-a")
                    cols["receptor_id"].append(f"rec-{r}")
                    cols["material_id"].append("microplastic")
                    cols["arrival_time_id"].append(f"arr-{a}")
                    cols["settling_velocity_mps"].append(velocity)
                    cols["arrival_time_utc_ns"].append(ARRIVAL_BASE_NS + a * 24 * HOUR_NS)
                    cols["seed_hex"].append(f"{row:032x}")
                    cols["x0_m"].append(receptor_xy[r][0])
                    cols["y0_m"].append(receptor_xy[r][1])
                    cols["z0_m"].append(-1.0)
                    unit_rows.append(row)
                    row += 1
            units.append((f"{site}-a{a}-v{v}", unit_rows))
    identity = ParticleIdentityTable(
        particle_id=np.asarray(cols["particle_id"], dtype="<U64"),
        scenario_id=np.asarray(cols["scenario_id"], dtype="<U64"),
        member_id=np.asarray(cols["member_id"], dtype=np.int32),
        study_site_id=np.asarray(cols["study_site_id"], dtype="<U64"),
        analysis_region_id=np.asarray(cols["analysis_region_id"], dtype="<U64"),
        receptor_id=np.asarray(cols["receptor_id"], dtype="<U64"),
        material_id=np.asarray(cols["material_id"], dtype="<U64"),
        arrival_time_id=np.asarray(cols["arrival_time_id"], dtype="<U64"),
        settling_velocity_mps=np.asarray(cols["settling_velocity_mps"], dtype=np.float64),
        arrival_time_utc_ns=np.asarray(cols["arrival_time_utc_ns"], dtype=np.int64),
        seed_hex=np.asarray(cols["seed_hex"], dtype="<U32"),
        x0_m=np.asarray(cols["x0_m"], dtype=np.float64),
        y0_m=np.asarray(cols["y0_m"], dtype=np.float64),
        z0_m=np.asarray(cols["z0_m"], dtype=np.float64),
        experiment_case_id="case-wp38",
        master_seed=42,
        random_stream_id=None,
    )
    return identity, [(key, np.asarray(rows_, dtype=np.int64)) for key, rows_ in units]


def make_result(
    identity: ParticleIdentityTable,
    rows: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    status: np.ndarray,
) -> BackwardDriverResult:
    """由 ``(T, n)`` 的軌跡與狀態陣列組出 ``BackwardDriverResult``。

    ``final_status`` 取最後一筆的狀態（0 時視為達回溯上限）；最後位置與時刻取最後一筆。
    """

    rows = np.asarray(rows, dtype=np.int64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    status = np.asarray(status, dtype=np.int32)
    n_time, n = x.shape
    assert status.shape == x.shape == y.shape and n == rows.size
    final_status = np.where(status[-1] == 0, NATIVE_STATUS_MAX_BACKTRACK, status[-1]).astype(np.int32)
    arrival = int(identity.arrival_time_utc_ns[rows[0]])
    utc = np.asarray([arrival - i * HOUR_NS for i in range(n_time)], dtype=np.int64)
    departures = tuple(
        DepartureRecord(
            row_particle_index=i,
            status=int(final_status[i]),
            x_m=float(x[-1, i]),
            y_m=float(y[-1, i]),
            z_m=-5.0,
            utc_ns=int(utc[-1]),
        )
        for i in range(n)
    )
    return BackwardDriverResult(
        row_particle_index=rows,
        final_status=final_status,
        final_x_m=x[-1].copy(),
        final_y_m=y[-1].copy(),
        final_z_m=np.full(n, -5.0),
        final_utc_ns=np.full(n, utc[-1], dtype=np.int64),
        trajectory_utc_ns=utc,
        trajectory_x_m=x,
        trajectory_y_m=y,
        trajectory_z_m=np.full((n_time, n), -5.0),
        trajectory_status=status,
        departures=departures,
        diagnostics=SubstepDiagnostics(),
        overhead_diagnostics=HourlyOverheadDiagnostics(),
        hours_run=n_time - 1,
    )


def random_walk_unit(
    rng: np.random.Generator,
    identity: ParticleIdentityTable,
    rows: np.ndarray,
    *,
    n_time: int,
    step_m: float = 40.0,
    terminate: bool = True,
    east_edge_m: float | None = None,
) -> BackwardDriverResult:
    """隨機漫步軌跡：粒子在隨機時刻以隨機終止狀態結束，之後位置凍結、status 非 0。"""

    n = rows.size
    x = np.empty((n_time, n))
    y = np.empty((n_time, n))
    x[0] = identity.x0_m[rows]
    y[0] = identity.y0_m[rows]
    steps = rng.normal(0.0, step_m, size=(n_time - 1, n, 2))
    status = np.zeros((n_time, n), dtype=np.int32)
    end_time = rng.integers(1, n_time, size=n) if terminate else np.full(n, n_time + 5)
    end_status = rng.choice([1, 1, 2, 3, 4], size=n)
    for t in range(1, n_time):
        x[t] = x[t - 1] + steps[t - 1, :, 0]
        y[t] = y[t - 1] + steps[t - 1, :, 1]
        ended = end_time <= t
        # 已終止的粒子：位置凍結、status 為終止碼。
        first_end = end_time == t
        if east_edge_m is not None:
            hit = first_end & (end_status == NATIVE_STATUS_LEFT_FLOW_DOMAIN)
            x[t] = np.where(hit, east_edge_m, x[t])
        x[t] = np.where(end_time < t, x[t - 1], x[t])
        y[t] = np.where(end_time < t, y[t - 1], y[t])
        status[t] = np.where(ended, end_status, 0)
    return make_result(identity, rows, x, y, status)


# ── 既有產品的回歸基準 ────────────────────────────────────────────────────


def canonical_table_digest(path: Path, *, float_digits: int | None = None) -> str:
    """把 Parquet 表轉成標準化 JSON 後取 SHA-256（與 pyarrow 版本／檔案位元組無關）。

    ``float_digits`` 為 ``None`` 時浮點以 ``repr`` 完整保留（逐位元比較）；給定位數時
    先取有效位數，供核密度這類跨函式庫版本可能有末位差異的表使用。
    """

    def norm(value: Any) -> Any:
        if isinstance(value, float):
            if value != value:
                return "nan"
            return repr(value) if float_digits is None else float(f"{value:.{float_digits}g}")
        if isinstance(value, list):
            return [norm(v) for v in value]
        if isinstance(value, dict):
            return {k: norm(v) for k, v in value.items()}
        return value

    rows = pq.read_table(path).to_pylist()
    payload = json.dumps(norm(rows), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def regression_units():
    """回歸用的固定合成資料集（決定性）：``(identity, [(unit_key, result)])``。"""

    identity, units = build_world(velocities=(-0.001, -0.002, -0.004), members=4, receptors=2, arrivals=3)
    rng = np.random.default_rng(20260929)
    results = [
        (key, random_walk_unit(rng, identity, rows, n_time=9, east_edge_m=500.0)) for key, rows in units
    ]
    return identity, results


FOOTPRINT_STEMS = (
    "residence_density_by_site",
    "residence_density_by_site_settling_velocity",
    "residence_density_by_site_member_half",
    "residence_hdr_by_site",
    "footprint_grid_by_site",
    "lookback_snapshot_by_site",
    "lookback_snapshot_density_by_site",
    "footprint_sea_mask_by_site",
    "boundary_departure_along_arc_by_site",
    "boundary_edge_intervals_by_site",
    "footprint_geometry_by_site",
    "footprint_summary_by_site",
)
"""WP-38 新增產品的檔名主幹；回歸比對時排除。"""


def make_boundary_geometry(site_id: str):
    """最小合法 ``BoundaryGeometry``：flow domain 為 ±500 m 方框，東側 x=500 為開放邊界。"""

    from shapely.geometry import LineString, box

    from lagrangian_backtracking.boundaries import BoundaryGeometry

    return BoundaryGeometry(
        own_local_domain=box(-100.0, -100.0, 100.0, 100.0),
        flow_domain=box(-500.0, -500.0, 500.0, 500.0),
        foreign_local_domains={},
        own_local_open_boundary=LineString([(100.0, -100.0), (100.0, 100.0)]),
        flow_open_boundary=LineString([(500.0, -500.0), (500.0, 500.0)]),
        local_equals_flow=False,
        own_local_boundary_segment_id=f"{site_id}_local_open_boundary",
        flow_boundary_segment_id="flow_open_boundary",
    )


def run_legacy_pipeline(out_dir: Path, **accumulator_kwargs: Any) -> Path:
    """以既有介面跑一次完整後處理（含邊界分段與經緯度），回傳輸出目錄。"""

    from lagrangian_backtracking.native_runner.postprocess import PostprocessAccumulator

    identity, results = regression_units()
    accumulator = PostprocessAccumulator(
        identity,
        output_dir=out_dir,
        lonlat_projector=linear_projector,
        boundary_geometries={"siteA": make_boundary_geometry("siteA")},
        **accumulator_kwargs,
    )
    for key, result in results:
        accumulator.add_batch(key, result)
    accumulator.finalize()
    return Path(out_dir)


def legacy_digests(out_dir: Path) -> dict[str, str]:
    """既有產品（排除 WP-38 新增檔）的標準化雜湊：核密度表取 9 位有效數字，其餘逐位元。"""

    digests: dict[str, str] = {}
    for path in sorted(Path(out_dir).glob("*.parquet")):
        if path.stem in FOOTPRINT_STEMS:
            continue
        digits = 9 if path.stem.startswith("kde_by_site") else None
        digests[path.name] = canonical_table_digest(path, float_digits=digits)
    return digests
