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

    pq.write_table(pa.table(all_departures), postprocess_dir / "departures.parquet")
    pq.write_table(pa.table(all_trajectories), postprocess_dir / "trajectory_samples_by_site.parquet")
    pq.write_table(pa.Table.from_pylist(kde_rows), postprocess_dir / "kde_by_site.parquet")
    pq.write_table(pa.Table.from_pylist(summary_rows), postprocess_dir / "boundary_summary_by_site.parquet")
    pq.write_table(
        pa.Table.from_pylist(arc_rows), postprocess_dir / "boundary_arc_segments_geometry_by_site.parquet"
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
