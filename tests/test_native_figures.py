"""WP-21 native 正式成果圖表管線（``native_runner.figures``）的驗收測試。

測試策略
--------
本檔案完全不讀取任何真實 native 後處理輸出（不碰 ``/mnt/user-data/uploads``），
而是用 PyArrow 手造一份最小但欄位齊全的合成後處理目錄（兩站、各數十顆合成
粒子），驗證 ``figures.py`` 自己的邏輯：八張圖是否真的產出、輸出目錄已存在時
是否 fail-closed、缺必要欄位時錯誤訊息是否清楚、``--sites`` 篩選是否正確、
找不到繁體中文字型時是否改用英文標籤並記錄於 manifest、以及 manifest 內的
輸入／輸出 SHA-256 是否與實際檔案一致。不驗證任何海洋物理正確性（那不是本
模組的職責），也不對真實五站資料做逐像素比對（真實資料的視覺 QA 已在
``reports/WP-21.md`` 另行記錄）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.native_runner import figures as native_figures
from lagrangian_backtracking.native_runner.backward_driver import (
    NATIVE_STATUS_DEPOSITED,
    NATIVE_STATUS_DESCRIPTIONS,
    NATIVE_STATUS_FORCING_MISSING,
    NATIVE_STATUS_LEFT_INTEREST_REGION,
    NATIVE_STATUS_MAX_BACKTRACK,
    NATIVE_STATUS_NUMERICAL_FAILURE,
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION,
    NATIVE_STATUS_STRANDED,
)
from lagrangian_backtracking.native_runner.postprocess import COMPASS_16_SECTOR_NAMES

_NON_CROSSING_STATUS_CYCLE = (
    NATIVE_STATUS_MAX_BACKTRACK,
    NATIVE_STATUS_STRANDED,
    NATIVE_STATUS_DEPOSITED,
    NATIVE_STATUS_FORCING_MISSING,
    NATIVE_STATUS_NUMERICAL_FAILURE,
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION,
)

_SITE_CENTERS: dict[str, tuple[float, float]] = {
    "site_a": (121.90, 25.00),
    "site_b": (120.70, 21.90),
}

_PARTICLES_PER_SITE = 24
_CROSSED_PER_SITE = 16  # 前 16 顆穿越邊界，剛好對應 16 個方位扇區各一顆
_SETTLING_VELOCITIES_MPS = (-0.001, -0.01, -0.1)

# WP-23：兩站分別模擬 local_equals_flow 的兩種真實情況——``site_a`` 比照
# 貢寮／龜山島（local ⊂ flow，local 首次離開另外出表）、``site_b`` 比照
# 新竹／南灣／連江（local 與 flow 相同，同一次穿越只用 also_local_domain_
# first_exit 旗標表達）。兩站混在同一份合成後處理目錄裡，兩種語意同時測到。
_SITE_LOCAL_EQUALS_FLOW: dict[str, bool] = {"site_a": False, "site_b": True}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _build_site_departures(site_id: str, row_index_start: int) -> tuple[dict[str, list], DomainProjection]:
    """手造一站的離開紀錄欄位（dict of list），回傳資料與對應的 AEQD 投影。

    前 ``_CROSSED_PER_SITE`` 顆粒子成功穿越開放邊界，方位扇區索引 0..15 各一顆
    （交替分配到 local／flow 兩條開放邊界線）；其餘粒子依序循環六種未穿越狀態。
    經緯度用小範圍隨機偏移（種子固定，結果可重現），公尺座標由同一個
    ``DomainProjection`` 正投影得到，因此模組內部的投影中心反解一定能收斂。
    """

    center_lon, center_lat = _SITE_CENTERS[site_id]
    local_equals_flow = _SITE_LOCAL_EQUALS_FLOW[site_id]
    projection = DomainProjection(center_lon, center_lat)
    rng = np.random.default_rng(abs(hash(site_id)) % (2**32))

    n = _PARTICLES_PER_SITE
    lon_offsets = rng.uniform(-0.03, 0.03, size=n)
    lat_offsets = rng.uniform(-0.03, 0.03, size=n)
    lon_deg = center_lon + lon_offsets
    lat_deg = center_lat + lat_offsets
    x_m, y_m = projection.project(lon_deg, lat_deg)

    particle_id: list[str] = []
    study_site_id: list[str] = []
    receptor_id: list[str] = []
    settling_velocity_mps: list[float] = []
    final_status_code: list[int] = []
    final_status_description_zh: list[str] = []
    crossed: list[bool] = []
    row_particle_index: list[int] = []
    backtrack_duration_seconds: list[float] = []
    azimuth_sector_index: list[int | None] = []
    azimuth_sector_name: list[str | None] = []
    boundary_segment_id: list[str | None] = []
    boundary_arc_segment_index: list[int | None] = []
    boundary_arc_length_position_m: list[float | None] = []

    also_local_domain_first_exit: list[bool] = []

    for i in range(n):
        particle_id.append(f"{site_id}_p{i:03d}")
        study_site_id.append(site_id)
        receptor_id.append("receptor_1")
        settling_velocity_mps.append(_SETTLING_VELOCITIES_MPS[i % len(_SETTLING_VELOCITIES_MPS)])
        row_particle_index.append(row_index_start + i)
        backtrack_duration_seconds.append(3600.0 * (i + 1))

        if i < _CROSSED_PER_SITE:
            sector_index = i % 16
            final_status_code.append(NATIVE_STATUS_LEFT_INTEREST_REGION)
            final_status_description_zh.append(NATIVE_STATUS_DESCRIPTIONS[NATIVE_STATUS_LEFT_INTEREST_REGION])
            crossed.append(True)
            # WP-23：只有 local_equals_flow=True 的站（this 是「local 首次離開」
            # 與「flow 終止」同一次穿越的旗標），此時這一列不會另外出現在
            # local_domain_departures.parquet（見 _build_local_departure_rows）。
            also_local_domain_first_exit.append(local_equals_flow)
            azimuth_sector_index.append(sector_index)
            azimuth_sector_name.append(COMPASS_16_SECTOR_NAMES[sector_index])
            base_id = f"{site_id}_local_open_boundary" if i % 2 == 0 else f"{site_id}_flow_open_boundary"
            boundary_segment_id.append(base_id)
            boundary_arc_segment_index.append(i % 4)
            boundary_arc_length_position_m.append(float(100.0 * (i % 4)))
        else:
            cycle_index = (i - _CROSSED_PER_SITE) % len(_NON_CROSSING_STATUS_CYCLE)
            status_code = _NON_CROSSING_STATUS_CYCLE[cycle_index]
            final_status_code.append(status_code)
            final_status_description_zh.append(NATIVE_STATUS_DESCRIPTIONS[status_code])
            crossed.append(False)
            also_local_domain_first_exit.append(False)
            azimuth_sector_index.append(None)
            azimuth_sector_name.append(None)
            boundary_segment_id.append(None)
            boundary_arc_segment_index.append(None)
            boundary_arc_length_position_m.append(None)

    columns = {
        "particle_id": particle_id,
        "study_site_id": study_site_id,
        "receptor_id": receptor_id,
        "settling_velocity_mps": settling_velocity_mps,
        "final_status_code": final_status_code,
        "final_status_description_zh": final_status_description_zh,
        "crossed_interest_region_boundary": crossed,
        "also_local_domain_first_exit": also_local_domain_first_exit,
        "departure_lon_deg": lon_deg.tolist(),
        "departure_lat_deg": lat_deg.tolist(),
        "departure_x_m": x_m.tolist(),
        "departure_y_m": y_m.tolist(),
        "departure_z_m": [0.0] * n,
        "row_particle_index": row_particle_index,
        "backtrack_duration_seconds": backtrack_duration_seconds,
        "departure_azimuth_sector_index": azimuth_sector_index,
        "departure_azimuth_sector_name": azimuth_sector_name,
        "boundary_segment_id": boundary_segment_id,
        "boundary_arc_segment_index": boundary_arc_segment_index,
        "boundary_arc_length_position_m": boundary_arc_length_position_m,
    }
    return columns, projection


def _build_local_departure_rows(site_id: str, departures: dict[str, list]) -> list[dict[str, object]]:
    """造 ``local_domain_departures.parquet`` 的列：只有 ``local_equals_flow=False``
    的站（本檔案是 ``site_a``）才有內容，每個成功穿越 flow domain 的粒子一列，
    模擬「先離開 local domain（記錄），再離開 flow domain（終止）」——離開 local
    domain 的位置刻意取離開 flow domain位置的一半（回溯起點方向），確保兩個事件
    在圖上是不同的點，能驗證 figures.py 畫出兩個不同標記。
    """

    if _SITE_LOCAL_EQUALS_FLOW[site_id]:
        return []
    rows: list[dict[str, object]] = []
    for i, crossed in enumerate(departures["crossed_interest_region_boundary"]):
        if not crossed:
            continue
        x_m = departures["departure_x_m"][i] * 0.5
        y_m = departures["departure_y_m"][i] * 0.5
        rows.append(
            {
                "particle_id": departures["particle_id"][i],
                "scenario_id": "synthetic_scenario",
                "member_id": 0,
                "study_site_id": site_id,
                "analysis_region_id": site_id,
                "receptor_id": departures["receptor_id"][i],
                "material_id": "synthetic_material",
                "arrival_time_id": "t0",
                "arrival_time_utc_ns": 1_700_000_000_000_000_000,
                "settling_velocity_mps": departures["settling_velocity_mps"][i],
                "season": "synthetic_season",
                "depth_quartile": 0,
                "row_particle_index": departures["row_particle_index"][i],
                "local_first_exit_x_m": x_m,
                "local_first_exit_y_m": y_m,
                "local_first_exit_z_m": 0.0,
                "local_first_exit_lon_deg": None,
                "local_first_exit_lat_deg": None,
                "local_first_exit_utc_ns": 1_699_999_996_400_000_000,
                "local_first_exit_backtrack_duration_seconds": 3600.0,
                "local_boundary_segment_id": f"{site_id}_local_open_boundary",
                "local_boundary_classification_distance_m": 0.0,
                "local_departure_azimuth_deg": float(departures["departure_azimuth_sector_index"][i] or 0)
                * 22.5,
                "local_departure_azimuth_sector_index": departures["departure_azimuth_sector_index"][i],
                "local_departure_azimuth_sector_name": departures["departure_azimuth_sector_name"][i],
                "local_boundary_arc_segment_index": departures["boundary_arc_segment_index"][i],
                "local_boundary_arc_length_position_m": departures["boundary_arc_length_position_m"][i],
            }
        )
    return rows


def _fill_local_first_exit_lonlat(
    rows: list[dict[str, object]], projection: DomainProjection
) -> list[dict[str, object]]:
    """把 :func:`_build_local_departure_rows` 造出的公尺座標投影回經緯度，
    分兩步是因為投影中心要等 ``_build_site_departures`` 算完才知道。"""

    for row in rows:
        lon, lat = projection.unproject(row["local_first_exit_x_m"], row["local_first_exit_y_m"])
        row["local_first_exit_lon_deg"] = float(lon)
        row["local_first_exit_lat_deg"] = float(lat)
    return rows


def _build_local_summary_row(site_id: str, departures: dict[str, list]) -> dict[str, object]:
    """造 ``local_domain_first_exit_summary_by_site.parquet`` 這一站的一列。"""

    local_equals_flow = _SITE_LOCAL_EQUALS_FLOW[site_id]
    total = len(departures["particle_id"])
    if local_equals_flow:
        count = sum(1 for v in departures["also_local_domain_first_exit"] if v)
        source = "same_as_flow_domain_departure"
    else:
        count = sum(1 for v in departures["crossed_interest_region_boundary"] if v)
        source = "local_domain_departures"
    return {
        "study_site_id": site_id,
        "local_equals_flow": local_equals_flow,
        "total_particle_count": total,
        "local_first_exit_particle_count": count,
        "fraction_local_first_exit": count / total if total else float("nan"),
        "local_first_exit_source": source,
    }


def _build_trajectory_rows(
    site_id: str, departures: dict[str, list], projection: DomainProjection, *, sample_count: int = 8
) -> dict[str, list]:
    """為一站挑選前 ``sample_count`` 顆粒子造出最小代表軌跡樣本（3 個時間點）。"""

    study_site_id: list[str] = []
    row_particle_index: list[int] = []
    particle_id: list[str] = []
    trajectory_utc_ns: list[list[int]] = []
    trajectory_x_m: list[list[float]] = []
    trajectory_y_m: list[list[float]] = []
    trajectory_z_m: list[list[float]] = []

    base_utc_ns = 1_700_000_000_000_000_000
    for i in range(sample_count):
        x0 = departures["departure_x_m"][i]
        y0 = departures["departure_y_m"][i]
        study_site_id.append(site_id)
        row_particle_index.append(departures["row_particle_index"][i])
        particle_id.append(departures["particle_id"][i])
        trajectory_utc_ns.append(
            [base_utc_ns, base_utc_ns - 3_600_000_000_000, base_utc_ns - 7_200_000_000_000]
        )
        trajectory_x_m.append([x0, x0 - 50.0 * (i + 1), x0 - 120.0 * (i + 1)])
        trajectory_y_m.append([y0, y0 + 40.0 * (i + 1), y0 + 90.0 * (i + 1)])
        # 起始深度依粒子序位錯開，讓深度序位排序（見 figures.TrajectorySample）有意義。
        trajectory_z_m.append([-2.0 * (i + 1), -8.0 * (i + 1), -20.0 * (i + 1)])

    return {
        "study_site_id": study_site_id,
        "row_particle_index": row_particle_index,
        "particle_id": particle_id,
        "trajectory_utc_ns": trajectory_utc_ns,
        "trajectory_x_m": trajectory_x_m,
        "trajectory_y_m": trajectory_y_m,
        "trajectory_z_m": trajectory_z_m,
    }


def _build_kde_row(site_id: str) -> dict[str, object]:
    rows, cols = 6, 6
    x_centers = np.linspace(-1500.0, 1500.0, cols)
    y_centers = np.linspace(-1500.0, 1500.0, rows)
    density = np.abs(np.random.default_rng(1).normal(size=rows * cols)) + 0.1
    cell_probability = density / density.sum()
    hdr_levels = [0.5, 0.75, 0.9]
    hdr_masks_flat: list[bool] = []
    order = np.argsort(-cell_probability)
    cumulative = np.cumsum(cell_probability[order])
    for level in hdr_levels:
        keep = order[: int(np.searchsorted(cumulative, level) + 1)]
        mask = np.zeros(rows * cols, dtype=bool)
        mask[keep] = True
        hdr_masks_flat.extend(mask.tolist())
    return {
        "study_site_id": site_id,
        "status": "available",
        "raw_point_count": _CROSSED_PER_SITE,
        "bandwidth_factor": 1.0,
        "hdr_levels": hdr_levels,
        "grid_rows": rows,
        "grid_cols": cols,
        "x_centers_m": x_centers.tolist(),
        "y_centers_m": y_centers.tolist(),
        "density_per_m2_flat": density.tolist(),
        "cell_probability_flat": cell_probability.tolist(),
        "hdr_masks_flat": hdr_masks_flat,
    }


def _build_summary_row(site_id: str, departures: dict[str, list]) -> dict[str, object]:
    total = len(departures["particle_id"])
    codes = departures["final_status_code"]

    def _fraction(code: int) -> float:
        return sum(1 for c in codes if c == code) / total

    return {
        "study_site_id": site_id,
        "total_particle_count": total,
        "fraction_crossed_interest_region_boundary": _fraction(NATIVE_STATUS_LEFT_INTEREST_REGION),
        "fraction_max_backtrack": _fraction(NATIVE_STATUS_MAX_BACKTRACK),
        "fraction_stranded": _fraction(NATIVE_STATUS_STRANDED),
        "fraction_deposited": _fraction(NATIVE_STATUS_DEPOSITED),
        "fraction_forcing_missing": _fraction(NATIVE_STATUS_FORCING_MISSING),
        "fraction_numerical_failure": _fraction(NATIVE_STATUS_NUMERICAL_FAILURE),
        "fraction_pre_window_deposition": _fraction(NATIVE_STATUS_PRE_WINDOW_DEPOSITION),
    }


def _build_arc_geometry_rows(site_id: str, projection: DomainProjection) -> list[dict[str, object]]:
    """造兩條開放邊界折線（local／flow）各 4 段弧，座標與 departures 用同一投影。"""

    center_lon, center_lat = _SITE_CENTERS[site_id]
    rows: list[dict[str, object]] = []
    for base_suffix, radius_deg in (("local_open_boundary", 0.05), ("flow_open_boundary", 0.4)):
        base_id = f"{site_id}_{base_suffix}"
        corners_lon = [
            center_lon - radius_deg,
            center_lon + radius_deg,
            center_lon + radius_deg,
            center_lon - radius_deg,
            center_lon - radius_deg,
        ]
        corners_lat = [
            center_lat - radius_deg,
            center_lat - radius_deg,
            center_lat + radius_deg,
            center_lat + radius_deg,
            center_lat - radius_deg,
        ]
        for index in range(4):
            start_lon, end_lon = corners_lon[index], corners_lon[index + 1]
            start_lat, end_lat = corners_lat[index], corners_lat[index + 1]
            start_x, start_y = projection.project(start_lon, start_lat)
            end_x, end_y = projection.project(end_lon, end_lat)
            mid_lon = (start_lon + end_lon) / 2
            mid_lat = (start_lat + end_lat) / 2
            mid_x, mid_y = projection.project(mid_lon, mid_lat)
            rows.append(
                {
                    "study_site_id": site_id,
                    "base_boundary_segment_id": base_id,
                    "arc_segment_index": index,
                    "arc_segment_id": f"{base_id}_arc{index:03d}",
                    "start_x_m": float(start_x),
                    "start_y_m": float(start_y),
                    "end_x_m": float(end_x),
                    "end_y_m": float(end_y),
                    "midpoint_x_m": float(mid_x),
                    "midpoint_y_m": float(mid_y),
                    "start_lon_deg": start_lon,
                    "start_lat_deg": start_lat,
                    "end_lon_deg": end_lon,
                    "end_lat_deg": end_lat,
                    "midpoint_lon_deg": mid_lon,
                    "midpoint_lat_deg": mid_lat,
                }
            )
    return rows


def _write_synthetic_postprocess_dir(
    base_dir: Path, *, sites: tuple[str, ...] = ("site_a", "site_b")
) -> Path:
    """在 ``base_dir`` 下寫出一份最小但欄位齊全的合成 native 後處理目錄。"""

    postprocess_dir = base_dir / "postprocess"
    postprocess_dir.mkdir()

    all_departures: dict[str, list] = {}
    all_trajectories: dict[str, list] = {}
    kde_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    arc_rows: list[dict[str, object]] = []
    local_departure_rows: list[dict[str, object]] = []
    local_summary_rows: list[dict[str, object]] = []

    row_index_start = 0
    for site_id in sites:
        departures, projection = _build_site_departures(site_id, row_index_start)
        row_index_start += _PARTICLES_PER_SITE

        trajectories = _build_trajectory_rows(site_id, departures, projection)
        for key, values in departures.items():
            all_departures.setdefault(key, []).extend(values)
        for key, values in trajectories.items():
            all_trajectories.setdefault(key, []).extend(values)

        kde_rows.append(_build_kde_row(site_id))
        summary_rows.append(_build_summary_row(site_id, departures))
        arc_rows.extend(_build_arc_geometry_rows(site_id, projection))
        local_departure_rows.extend(
            _fill_local_first_exit_lonlat(_build_local_departure_rows(site_id, departures), projection)
        )
        local_summary_rows.append(_build_local_summary_row(site_id, departures))

    pq.write_table(pa.table(all_departures), postprocess_dir / "departures.parquet")
    pq.write_table(pa.table(all_trajectories), postprocess_dir / "trajectory_samples_by_site.parquet")
    pq.write_table(pa.Table.from_pylist(kde_rows), postprocess_dir / "kde_by_site.parquet")
    pq.write_table(pa.Table.from_pylist(summary_rows), postprocess_dir / "boundary_summary_by_site.parquet")
    pq.write_table(
        pa.Table.from_pylist(arc_rows), postprocess_dir / "boundary_arc_segments_geometry_by_site.parquet"
    )
    if local_departure_rows:
        local_departure_table = pa.Table.from_pylist(local_departure_rows)
    else:
        # 全部站點都 local_equals_flow=True 時這張表沒有任何列，但欄位仍須齊全，
        # 否則 figures.py 的 _require_columns 會 fail-closed（見該檔 docstring）。
        local_departure_table = pa.table(
            {name: [] for name in [
                "particle_id", "scenario_id", "member_id", "study_site_id", "analysis_region_id",
                "receptor_id", "material_id", "arrival_time_id", "arrival_time_utc_ns",
                "settling_velocity_mps", "season", "depth_quartile", "row_particle_index",
                "local_first_exit_x_m", "local_first_exit_y_m", "local_first_exit_z_m",
                "local_first_exit_lon_deg", "local_first_exit_lat_deg", "local_first_exit_utc_ns",
                "local_first_exit_backtrack_duration_seconds", "local_boundary_segment_id",
                "local_boundary_classification_distance_m", "local_departure_azimuth_deg",
                "local_departure_azimuth_sector_index", "local_departure_azimuth_sector_name",
                "local_boundary_arc_segment_index", "local_boundary_arc_length_position_m",
            ]}
        )
    pq.write_table(local_departure_table, postprocess_dir / "local_domain_departures.parquet")
    pq.write_table(
        pa.Table.from_pylist(local_summary_rows),
        postprocess_dir / "local_domain_first_exit_summary_by_site.parquet",
    )

    manifest = {
        "experiment_case_id": "synthetic_test_case",
        "master_seed": 12345,
        "config": {"azimuth_sector_count": 16, "boundary_arc_segment_count": 4},
    }
    (postprocess_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    return postprocess_dir


def _write_synthetic_coastline(base_dir: Path) -> Path:
    """造一份最小合法 CRS84 海岸 GeoJSON（單一陸地方塊，僅供地理參照）。"""

    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [121.0, 24.5],
                            [121.2, 24.5],
                            [121.2, 24.7],
                            [121.0, 24.7],
                            [121.0, 24.5],
                        ]
                    ],
                },
            }
        ],
    }
    path = base_dir / "coastline.geojson"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    postprocess_dir = _write_synthetic_postprocess_dir(tmp_path)
    coastline_path = _write_synthetic_coastline(tmp_path)
    return postprocess_dir, coastline_path


def test_all_eight_figures_produced_for_each_site(
    tmp_path: Path, synthetic_inputs: tuple[Path, Path]
) -> None:
    postprocess_dir, coastline_path = synthetic_inputs
    output_dir = tmp_path / "figures-out"
    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir, coastline_path=coastline_path, output_dir=output_dir,
    )

    result_dir = native_figures.build_native_figures(config, command="lbt native-figures --test")
    assert result_dir == output_dir

    for site_id in ("site_a", "site_b"):
        for stem in native_figures.FIGURE_STEMS:
            png_path = output_dir / site_id / f"{stem}.png"
            assert png_path.is_file(), f"missing {png_path}"
            assert png_path.stat().st_size > 0

    assert (output_dir / "README.md").is_file()
    assert (output_dir / "manifest.json").is_file()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["sites"]) == ["site_a", "site_b"]
    assert manifest["figure_stems"] == list(native_figures.FIGURE_STEMS)


def test_output_dir_must_not_already_exist(tmp_path: Path, synthetic_inputs: tuple[Path, Path]) -> None:
    postprocess_dir, coastline_path = synthetic_inputs
    output_dir = tmp_path / "figures-out"
    output_dir.mkdir()  # 事先建立，模擬呼叫端誤把既有目錄當輸出目標

    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir, coastline_path=coastline_path, output_dir=output_dir,
    )
    with pytest.raises(native_figures.FiguresError, match="尚不存在"):
        native_figures.build_native_figures(config, command="lbt native-figures --test")

    # fail-closed 必須發生在寫任何檔案之前，既有目錄內容不應被動過。
    assert list(output_dir.iterdir()) == []


def test_missing_required_column_raises_clear_error(tmp_path: Path) -> None:
    postprocess_dir = _write_synthetic_postprocess_dir(tmp_path)
    coastline_path = _write_synthetic_coastline(tmp_path)

    # 讀回 departures.parquet，刪掉一個必要欄位後覆寫，模擬版本不相容的後處理輸出。
    departures_path = postprocess_dir / "departures.parquet"
    table = pq.read_table(departures_path)
    table = table.drop(["departure_azimuth_sector_index"])
    pq.write_table(table, departures_path)

    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir,
        coastline_path=coastline_path,
        output_dir=tmp_path / "figures-out",
    )
    with pytest.raises(native_figures.FiguresError, match="departure_azimuth_sector_index"):
        native_figures.build_native_figures(config, command="lbt native-figures --test")


def test_sites_filter_selects_only_requested_site(
    tmp_path: Path, synthetic_inputs: tuple[Path, Path]
) -> None:
    postprocess_dir, coastline_path = synthetic_inputs
    output_dir = tmp_path / "figures-out"
    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir,
        coastline_path=coastline_path,
        output_dir=output_dir,
        sites=("site_a",),
    )
    native_figures.build_native_figures(config, command="lbt native-figures --test")

    assert (output_dir / "site_a").is_dir()
    assert not (output_dir / "site_b").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sites"] == ["site_a"]


def test_sites_filter_unknown_site_fails_closed(tmp_path: Path, synthetic_inputs: tuple[Path, Path]) -> None:
    postprocess_dir, coastline_path = synthetic_inputs
    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir,
        coastline_path=coastline_path,
        output_dir=tmp_path / "figures-out",
        sites=("no_such_site",),
    )
    with pytest.raises(native_figures.FiguresError, match="no_such_site"):
        native_figures.build_native_figures(config, command="lbt native-figures --test")


def test_english_fallback_when_no_cjk_font_available(
    tmp_path: Path, synthetic_inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """模擬環境找不到合格繁體中文字型：管線不得失敗，須改英文並記錄於 manifest。"""

    def _raise_no_font() -> None:
        raise ValueError("找不到涵蓋報告必要字元的合格繁體中文字型")

    monkeypatch.setattr(native_figures, "resolve_cjk_font", _raise_no_font)

    postprocess_dir, coastline_path = synthetic_inputs
    output_dir = tmp_path / "figures-out"
    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir, coastline_path=coastline_path, output_dir=output_dir,
    )
    native_figures.build_native_figures(config, command="lbt native-figures --test")

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cjk_labels_used"] is False
    assert manifest["font"]["mode"] == "english_fallback"
    assert "reason" in manifest["font"]

    for site_id in ("site_a", "site_b"):
        for stem in native_figures.FIGURE_STEMS:
            assert (output_dir / site_id / f"{stem}.png").stat().st_size > 0


def test_manifest_sha256_matches_actual_files(tmp_path: Path, synthetic_inputs: tuple[Path, Path]) -> None:
    postprocess_dir, coastline_path = synthetic_inputs
    output_dir = tmp_path / "figures-out"
    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir, coastline_path=coastline_path, output_dir=output_dir,
    )
    native_figures.build_native_figures(config, command="lbt native-figures --test")

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    # 輸入 SHA-256：繪圖前後都必須與磁碟上實際檔案一致（本模組不得改動輸入）。
    for label, record in manifest["inputs_sha256_after"].items():
        # coastline 的 label 是固定字串 "coastline"，其餘 label 取自檔案 basename。
        source_path = coastline_path if label == "coastline" else postprocess_dir / record["basename"]
        assert record["sha256"] == _sha256_file(source_path), f"input sha256 mismatch: {label}"
    assert manifest["inputs_sha256_before"] == manifest["inputs_sha256_after"]

    # 輸出 PNG SHA-256：manifest 記錄必須與實際落地的 PNG bytes 一致。
    for relative_name, record in manifest["output_png_sha256"].items():
        png_path = output_dir / relative_name
        assert record["sha256"] == _sha256_file(png_path), f"output sha256 mismatch: {relative_name}"


# ── WP-23：local domain 首次離開（非終止事件）與 flow domain 終止的雙重語意 ──


def test_boundary_kind_classifies_local_flow_and_unknown_suffixes() -> None:
    """``_boundary_kind`` 是繪圖層唯一判斷「這條線是 local 還是 flow」的地方，
    依據跟 ``manifests.py`` 建構 ``BoundaryGeometry`` 時完全相同的既有命名慣例。
    """

    assert native_figures._boundary_kind("gongliao_local_open_boundary") == "local"
    assert native_figures._boundary_kind("area_a_flow_open_boundary") == "flow"
    assert native_figures._boundary_kind("coastline") == "other"


def test_local_equals_flow_false_site_populates_local_first_exit_fields(
    tmp_path: Path, synthetic_inputs: tuple[Path, Path]
) -> None:
    """``site_a``（本檔案模擬 local ⊂ flow，比照貢寮／龜山島）：
    ``local_equals_flow=False``、``local_departures_by_row`` 非空、且至少一顆
    代表粒子的 ``TrajectorySample.local_first_exit_lon/lat`` 有值（非終止標記
    有東西可畫）。"""

    postprocess_dir, coastline_path = synthetic_inputs
    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir, coastline_path=coastline_path, output_dir=tmp_path / "out",
    )
    inputs = native_figures.load_native_figures_inputs(config)
    site_a = next(s for s in inputs.sites if s.site_id == "site_a")

    assert site_a.local_equals_flow is False
    assert site_a.local_first_exit_particle_count == _CROSSED_PER_SITE
    assert site_a.local_first_exit_fraction == pytest.approx(_CROSSED_PER_SITE / _PARTICLES_PER_SITE)
    assert len(site_a.local_departures_by_row) == _CROSSED_PER_SITE

    samples_with_local_marker = [
        s for s in site_a.trajectories
        if s.local_first_exit_lon_deg is not None and s.local_first_exit_lat_deg is not None
    ]
    assert samples_with_local_marker, "site_a 應該至少有一顆代表粒子帶 local 首次離開標記"
    for sample in samples_with_local_marker:
        # 造合成資料時 local 首次離開位置刻意取終止點的一半（回溯起點方向），
        # 兩者不應該是同一個點，否則兩種標記在圖上會重疊而失去意義。
        assert (sample.local_first_exit_lon_deg, sample.local_first_exit_lat_deg) != (
            sample.lon_deg[-1], sample.lat_deg[-1],
        )


def test_local_equals_flow_true_site_has_no_separate_local_departures(
    tmp_path: Path, synthetic_inputs: tuple[Path, Path]
) -> None:
    """``site_b``（本檔案模擬 local＝flow，比照新竹／南灣／連江）：
    ``local_equals_flow=True``、``local_departures_by_row`` 必須是空的（該事件
    已經用 ``also_local_domain_first_exit`` 表達，不會另外出現一列），且沒有任何
    代表粒子帶 local 首次離開標記（避免同一次穿越在圖上被畫成兩個點）。"""

    postprocess_dir, coastline_path = synthetic_inputs
    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir, coastline_path=coastline_path, output_dir=tmp_path / "out",
    )
    inputs = native_figures.load_native_figures_inputs(config)
    site_b = next(s for s in inputs.sites if s.site_id == "site_b")

    assert site_b.local_equals_flow is True
    assert site_b.local_departures_by_row == {}
    assert site_b.local_first_exit_particle_count == _CROSSED_PER_SITE

    also_local_flags = site_b.departures["also_local_domain_first_exit"]
    crossed_flags = site_b.departures["crossed_interest_region_boundary"]
    assert int(also_local_flags.sum()) == _CROSSED_PER_SITE
    assert bool((also_local_flags <= crossed_flags).all())  # 旗標只可能在穿越那一列為真

    for sample in site_b.trajectories:
        assert sample.local_first_exit_lon_deg is None
        assert sample.local_first_exit_lat_deg is None


def test_missing_local_domain_departures_table_fails_closed(tmp_path: Path) -> None:
    postprocess_dir = _write_synthetic_postprocess_dir(tmp_path)
    coastline_path = _write_synthetic_coastline(tmp_path)
    (postprocess_dir / "local_domain_departures.parquet").unlink()

    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir, coastline_path=coastline_path, output_dir=tmp_path / "out",
    )
    with pytest.raises(native_figures.FiguresError, match="local_domain_departures"):
        native_figures.build_native_figures(config, command="lbt native-figures --test")


def test_missing_also_local_domain_first_exit_column_fails_closed(tmp_path: Path) -> None:
    postprocess_dir = _write_synthetic_postprocess_dir(tmp_path)
    coastline_path = _write_synthetic_coastline(tmp_path)

    departures_path = postprocess_dir / "departures.parquet"
    table = pq.read_table(departures_path)
    table = table.drop(["also_local_domain_first_exit"])
    pq.write_table(table, departures_path)

    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir, coastline_path=coastline_path, output_dir=tmp_path / "out",
    )
    with pytest.raises(native_figures.FiguresError, match="also_local_domain_first_exit"):
        native_figures.build_native_figures(config, command="lbt native-figures --test")


def test_all_eight_figures_still_produced_with_mixed_local_equals_flow_sites(
    tmp_path: Path, synthetic_inputs: tuple[Path, Path]
) -> None:
    """本檔案的合成目錄本身就混了 local_equals_flow=False（site_a）／True
    （site_b）兩種站，完整跑一次八張圖確保新增的邊界分色、local 標記與副標
    邏輯不會讓任何一張圖整個畫圖失敗（成功產出即代表 matplotlib 呼叫序列
    沒有拋例外；圖面內容正確性另由上面兩個 SiteData 欄位測試把關）。"""

    postprocess_dir, coastline_path = synthetic_inputs
    output_dir = tmp_path / "figures-out"
    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir, coastline_path=coastline_path, output_dir=output_dir,
    )
    native_figures.build_native_figures(config, command="lbt native-figures --test")

    for site_id in ("site_a", "site_b"):
        for stem in ("horizontal_overview", "horizontal_local"):
            png_path = output_dir / site_id / f"{stem}.png"
            assert png_path.is_file()
            assert png_path.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════
# WP-24b：圖面不畫警語、圖說另存為 captions.md、--figure-text {full,none} 開關
# ═══════════════════════════════════════════════════════════════════════════


def test_module_no_longer_defines_any_disclaimer_helper() -> None:
    """WP-24b：底部紅字警語整段移除，不是縮短或改寫——這裡直接鎖定舊版繪製警語
    用的兩個私有函式已經從模組消失，防止之後又悄悄加回某種形式的警語文字。"""

    assert not hasattr(native_figures, "_disclaimer_text")
    assert not hasattr(native_figures, "_add_disclaimer")

    # 模組 docstring 仍可能提到「警語」二字來說明這項裁決本身（見 WP-24b 一節），
    # 但舊版真正畫在圖上的那句固定警語全文不應該再出現在任何地方。
    source_text = Path(native_figures.__file__).read_text(encoding="utf-8")
    assert "不得解讀為絕對來源機率或因果歸因" not in source_text
    assert "尚未提升為核准發布幾何" not in source_text


def test_full_mode_figures_carry_no_disclaimer_text_object(
    tmp_path: Path, synthetic_inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """即使在 ``figure_text="full"``（畫標題／副標）模式下，也不應該有任何一個
    ``fig.text`` 物件是舊版警語（紅字、``color="#b3261e"``）。攔截
    ``_save_exclusive`` 在關閉圖面前檢查每個 ``Figure`` 物件的全部文字元素。"""

    postprocess_dir, coastline_path = synthetic_inputs
    output_dir = tmp_path / "figures-out-full-no-disclaimer"

    checked_figures = 0

    original_save = native_figures._save_exclusive

    def _checking_save(fig, path, *, dpi):
        nonlocal checked_figures
        checked_figures += 1
        for text_obj in fig.texts:
            assert text_obj.get_color() != "#b3261e", f"{path.name} 仍有紅字警語顏色的文字物件"
            assert "警語" not in text_obj.get_text()
            assert "不得解讀為" not in text_obj.get_text()
        return original_save(fig, path, dpi=dpi)

    monkeypatch.setattr(native_figures, "_save_exclusive", _checking_save)

    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir, coastline_path=coastline_path, output_dir=output_dir,
    )
    native_figures.build_native_figures(config, command="lbt native-figures --test")

    assert checked_figures == 2 * len(native_figures.FIGURE_STEMS)


def test_captions_markdown_has_one_section_per_figure_and_matches_manifest(
    tmp_path: Path, synthetic_inputs: tuple[Path, Path]
) -> None:
    """``captions.md`` 必須每張圖都有對應段落，且與 manifest 記錄的 ``caption``
    欄位逐字相同（同一份文字，不是兩處各自維護）。"""

    postprocess_dir, coastline_path = synthetic_inputs
    output_dir = tmp_path / "figures-out-captions"
    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir, coastline_path=coastline_path, output_dir=output_dir,
    )
    native_figures.build_native_figures(config, command="lbt native-figures --test")

    captions_path = output_dir / "captions.md"
    assert captions_path.is_file()
    captions_text = captions_path.read_text(encoding="utf-8")

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    png_records = manifest["output_png_sha256"]

    for site_id in ("site_a", "site_b"):
        for stem in native_figures.FIGURE_STEMS:
            key = f"{site_id}/{stem}.png"
            assert key in png_records
            caption = png_records[key]["caption"]
            assert isinstance(caption, str) and caption.strip()
            # captions.md 用 `### `stem.png`` 小節標題，內文緊接著同一段 manifest 文字。
            assert f"`{stem}.png`" in captions_text
            assert caption in captions_text
            # 圖說內容須交代關鍵設定：站點、粒子數、M、終止邊界定義。
            assert "M=1" in caption
            assert "flow domain 開放邊界" in caption
            assert "顆粒子" in caption

    # sha256 與圖說綁在同一筆 record，供事後核對「這段文字對應的就是這個版本的圖」。
    for record in png_records.values():
        assert set(record) == {"sha256", "size_bytes", "caption"}


def test_figure_text_none_mode_omits_titles_and_subtitles(
    tmp_path: Path, synthetic_inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--figure-text none`` 只留座標軸與圖例：攔截 ``_save_exclusive``，在每張
    圖存檔前確認沒有 ``suptitle``、也沒有任何 ``fig.text`` 加上去的副標／說明文字
    （圖例用 ``fig.legend``，不算在 ``fig.texts`` 內，因此圖例存在不受影響）。"""

    postprocess_dir, coastline_path = synthetic_inputs
    output_dir = tmp_path / "figures-out-none"

    checked_figures = 0
    original_save = native_figures._save_exclusive

    def _checking_save(fig, path, *, dpi):
        nonlocal checked_figures
        checked_figures += 1
        assert fig._suptitle is None, f"{path.name} 在 figure_text=none 模式下不應有標題"
        assert fig.texts == [], f"{path.name} 在 figure_text=none 模式下不應有額外文字物件"
        return original_save(fig, path, dpi=dpi)

    monkeypatch.setattr(native_figures, "_save_exclusive", _checking_save)

    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir, coastline_path=coastline_path, output_dir=output_dir,
        figure_text="none",
    )
    native_figures.build_native_figures(config, command="lbt native-figures --test --figure-text none")

    assert checked_figures == 2 * len(native_figures.FIGURE_STEMS)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["figure_text"] == "none"
    assert (output_dir / "captions.md").is_file()

    for site_id in ("site_a", "site_b"):
        for stem in native_figures.FIGURE_STEMS:
            png_path = output_dir / site_id / f"{stem}.png"
            assert png_path.is_file()
            assert png_path.stat().st_size > 0


def test_figure_text_full_mode_draws_a_title_for_at_least_one_figure(
    tmp_path: Path, synthetic_inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """反向對照 :func:`test_figure_text_none_mode_omits_titles_and_subtitles`：
    確認預設 ``full`` 模式底下標題機制本身仍然有畫，不是兩種模式都畫不出來而讓
    上一個測試變得沒有意義。"""

    postprocess_dir, coastline_path = synthetic_inputs
    output_dir = tmp_path / "figures-out-full-title"

    saw_suptitle = False
    original_save = native_figures._save_exclusive

    def _checking_save(fig, path, *, dpi):
        nonlocal saw_suptitle
        if fig._suptitle is not None:
            saw_suptitle = True
        return original_save(fig, path, dpi=dpi)

    monkeypatch.setattr(native_figures, "_save_exclusive", _checking_save)

    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir, coastline_path=coastline_path, output_dir=output_dir,
    )
    native_figures.build_native_figures(config, command="lbt native-figures --test")

    assert saw_suptitle, "full 模式下應該至少有一張圖畫出標題"


def test_figure_text_rejects_unknown_value(tmp_path: Path, synthetic_inputs: tuple[Path, Path]) -> None:
    postprocess_dir, coastline_path = synthetic_inputs
    with pytest.raises(native_figures.FiguresError, match="figure_text"):
        native_figures.NativeFiguresConfig(
            postprocess_dir=postprocess_dir, coastline_path=coastline_path,
            output_dir=tmp_path / "figures-out-bad-mode", figure_text="partial",
        )
