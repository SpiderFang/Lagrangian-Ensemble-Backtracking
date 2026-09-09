"""`build_pilot_coastline_preview.py` 的小型契約測試。

本檔案透過 `tmp_path` 建立可重現的合成展示資料，不讀取被忽略的 `work/`、SERVER
或下載海岸資料。fixture 只填入目標 script 實際讀取的 CSV、summary、domain、open
及 GeoJSON 欄位；它代表測試用的圖面資料，不代表任何真實模擬結果或科學結論。
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shlex
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, mapping

from lagrangian_backtracking.geometry import DomainProjection

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_pilot_coastline_preview.py"
_APPROVED_COASTLINE_SHA = "9e2e0ac9bc527aca87d89332cd428fdcb776eefbf94a85dd70f887f729b95fdd"
_VERTICAL_LEVELS = (
    "near_bed",
    "mid_lower_water_column",
    "mid_upper_water_column",
    "upper_water_column",
)
_TERMINAL_STATUSES = (
    "flow_domain_open_exit",
    "coast_contact",
    "surface_regime_exit",
    "deposited",
    "forcing_start",
    "data_gap",
    "max_age",
    "numerical_failure",
)


def _load_target_module():
    """以檔案路徑載入尚未成為 `src` 模組的目標 script，固定測試穩定公開 API。"""

    spec = importlib.util.spec_from_file_location("pilot_coastline_preview_under_test", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"無法建立 script module spec：{_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TARGET = _load_target_module()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """以 UTF-8 排序 key 寫入 fixture JSON，讓 raw bytes 與語意 hash 都可重建。"""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_hash(payload: dict[str, Any]) -> str:
    """複製專案 manifest 的 canonical JSON 規則，供合成 summary 綁定幾何語意。"""

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _file_hash(path: Path) -> str:
    """讀取小型測試檔的完整 bytes SHA-256，檢查 renderer 沒有改動輸入。"""

    return sha256(path.read_bytes()).hexdigest()


def _file_contract(path: Path) -> dict[str, Any]:
    """建立 preview manifest 需要的檔案大小與 raw SHA-256 契約。"""

    return {"size_bytes": path.stat().st_size, "sha256": _file_hash(path)}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """以固定欄位順序寫入合成 CSV；不額外加入 script 不會讀取的科學欄位。"""

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _refresh_summary_contract(preview_dir: Path, summary: dict[str, Any]) -> None:
    """改寫 summary 後同步 preview manifest 的 summary bytes 契約。"""

    summary_path = preview_dir / "summary.json"
    _write_json(summary_path, summary)
    manifest_path = preview_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["summary.json"] = _file_contract(summary_path)
    _write_json(manifest_path, manifest)


@pytest.fixture
def synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    """建立 B 區 20/203 合成展示 fixture，並以 summary canonical hash 綁定幾何。

    五個水平面板各有四個不同初始水層粒子，每個基礎條件只有一顆粒子（M=1）。前三
    顆粒子各保存 11 筆觀測，其餘 17 顆各保存 10 筆，合計 203 筆；這個分配只為覆蓋
    計數與完整識別碼，不具有模擬意義。九顆粒子到達回溯上限、八顆以開放邊界離域、
    三顆以數值失敗停止，終止 age 分別為 3600 與 1800 秒，僅用來讓展示 renderer
    保留三種狀態標記。海岸使用 1905 個小型有效 Polygon，並明示 OGC CRS84，以
    覆蓋 loader 的數量與 CRS gate，而不引入真實海岸線來源。
    """

    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    domain_path = tmp_path / "domain.json"
    open_path = tmp_path / "open.json"
    coastline_path = tmp_path / "coastline.json"

    domain_polygon = Polygon(
        [
            (120.0, 24.7),
            (120.2, 24.7),
            (120.2, 24.9),
            (120.0, 24.9),
            (120.0, 24.7),
        ]
    )
    open_line = LineString(domain_polygon.exterior.coords)
    domain_payload: dict[str, Any] = {
        "manifest_kind": "domain_geometry_manifest",
        "schema_version": "1.0.0",
        "status": "approved",
        "coordinate_reference": "EPSG:4326",
        "provenance": {"method_id": "synthetic_fixture"},
        "records": [
            {
                "analysis_region_id": "B",
                "flow_domain_id": "B-flow",
                "geometry": mapping(domain_polygon),
            }
        ],
    }
    open_payload: dict[str, Any] = {
        "manifest_kind": "open_boundary_manifest",
        "schema_version": "1.0.0",
        "status": "approved",
        "coordinate_reference": "EPSG:4326",
        "records": [
            {
                "owner_kind": "flow_domain",
                "owner_id": "B-flow",
                "analysis_region_id": "B",
                "segment_id": "B-exterior",
                "geometry": mapping(open_line),
            }
        ],
    }
    _write_json(domain_path, domain_payload)
    _write_json(open_path, open_payload)

    # 45 欄 × 43 列取前 1905 個小 polygon；全數落在合成 B domain 內且各自有效。
    coastline_features = []
    for index in range(1905):
        column = index % 45
        row = index // 45
        lon = 120.005 + column * 0.004
        lat = 24.705 + row * 0.004
        land = Polygon(
            [
                (lon, lat),
                (lon + 0.0008, lat),
                (lon + 0.0008, lat + 0.0008),
                (lon, lat + 0.0008),
                (lon, lat),
            ]
        )
        coastline_features.append({"type": "Feature", "properties": {}, "geometry": mapping(land)})
    coastline_payload: dict[str, Any] = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": coastline_features,
    }
    _write_json(coastline_path, coastline_payload)

    center_lon, center_lat = 120.1, 24.8
    projection = DomainProjection(center_lon, center_lat)
    statuses = ("max_age",) * 9 + ("flow_domain_open_exit",) * 8 + ("numerical_failure",) * 3
    panels = []
    particles: list[dict[str, str]] = []
    observations: list[dict[str, str]] = []
    for panel_number in range(1, 6):
        receptor_lon = 120.035 + 0.032 * (panel_number - 1)
        receptor_lat = 24.735 + 0.028 * (panel_number - 1)
        panels.append(
            {
                "panel": panel_number,
                "receptor_lon": receptor_lon,
                "receptor_lat": receptor_lat,
            }
        )
        for level_index, vertical_id in enumerate(_VERTICAL_LEVELS):
            particle_id = f"B-H{panel_number}-{vertical_id}"
            particle_status = statuses[len(particles)]
            terminal_age = 3600.0 if particle_status == "max_age" else 1800.0
            longitude = receptor_lon + 0.001 * (level_index - 1.5)
            latitude = receptor_lat + 0.0005 * (level_index - 1.5)
            x_m, y_m = projection.project(longitude, latitude)
            particle_x = float(x_m)
            particle_y = float(y_m)
            receptor_x_m, receptor_y_m = projection.project(receptor_lon, receptor_lat)
            receptor_x = float(receptor_x_m)
            receptor_y = float(receptor_y_m)
            particles.append(
                {
                    "particle_id": particle_id,
                    "horizontal_panel": str(panel_number),
                    "vertical_id": vertical_id,
                    "status": particle_status,
                    "longitude": f"{longitude:.12f}",
                    "latitude": f"{latitude:.12f}",
                    "x_m": f"{particle_x:.12f}",
                    "y_m": f"{particle_y:.12f}",
                    "age_seconds": f"{terminal_age:.12f}",
                    "failure_reason": (
                        "invalid_velocity_sample" if particle_status == "numerical_failure" else "unknown"
                    ),
                    "failure_stage": "step_start" if particle_status == "numerical_failure" else "unknown",
                    "qc_flags": "16" if particle_status == "numerical_failure" else "",
                }
            )

            observation_count = 11 if len(particles) <= 3 else 10
            for observation_index in range(observation_count):
                # age=0 共用同 panel receptor 原點，之後依保存終點做線性展示連線。
                fraction = observation_index / (observation_count - 1)
                observations.append(
                    {
                        "particle_id": particle_id,
                        "age_seconds": f"{terminal_age * fraction:.12f}",
                        "x_m": f"{receptor_x + fraction * (particle_x - receptor_x):.12f}",
                        "y_m": f"{receptor_y + fraction * (particle_y - receptor_y):.12f}",
                        "vertical_id": vertical_id,
                        "z_m": f"{-10.0 - 4.0 * level_index - fraction:.12f}",
                        "eta_m": "" if observation_index == observation_count - 1 else "1.0",
                        "bed_z_m": "" if observation_index == observation_count - 1 else "-40.0",
                        "environment_sample_status": "not_sampled"
                        if observation_index == observation_count - 1
                        else "valid",
                        "environment_qc_flags": "" if observation_index == observation_count - 1 else "0",
                        "forcing_month_id": "" if observation_index == observation_count - 1 else "202501",
                    }
                )

    _write_csv(
        preview_dir / "particles.csv",
        [
            "particle_id",
            "horizontal_panel",
            "vertical_id",
            "status",
            "longitude",
            "latitude",
            "x_m",
            "y_m",
            "age_seconds",
            "failure_reason",
            "failure_stage",
            "qc_flags",
        ],
        particles,
    )
    _write_csv(
        preview_dir / "observations.csv",
        [
            "particle_id",
            "age_seconds",
            "x_m",
            "y_m",
            "vertical_id",
            "z_m",
            "eta_m",
            "bed_z_m",
            "environment_sample_status",
            "environment_qc_flags",
            "forcing_month_id",
        ],
        observations,
    )

    summary: dict[str, Any] = {
        "artifact_kind": "pilot-preview-v1",
        # 與 production build 的 r2 run gate 對齊；內容仍全為合成 fixture，不代表真資料。
        "run_id": "b-fishinggear-m1-1h-r2",
        "arrival_utc": "2025-01-01T12:00:00+00:00",
        "study_site_id": "hsinchu",
        "settings": {"horizon_seconds": 3600.0},
        "particle_count": 20,
        "observation_count": 203,
        "material_id": "synthetic-settling-material",
        "projection": {
            "analysis_region_id": "B",
            "center_lonlat": [center_lon, center_lat],
            "kind": "source_geometry_AEQD",
            "units": "m",
            "geometry_canonical_hashes": {
                "domain": _canonical_hash(domain_payload),
                "open_boundary": _canonical_hash(open_payload),
            },
        },
        "vertical_order": list(_VERTICAL_LEVELS),
        "horizontal_panels": panels,
        "settling_velocity_mps": -0.002,
        "members_per_scenario": 1,
    }
    terminal_counts = {
        status: sum(row["status"] == status for row in particles) for status in _TERMINAL_STATUSES
    }
    summary["terminal_counts"] = terminal_counts
    summary["terminal_counts_by_vertical"] = {
        vertical_id: {
            status: sum(row["vertical_id"] == vertical_id and row["status"] == status for row in particles)
            for status in _TERMINAL_STATUSES
        }
        for vertical_id in _VERTICAL_LEVELS
    }
    summary["failure_details"] = [
        {
            "particle_id": row["particle_id"],
            "status": "numerical_failure",
            "failure_reason": "invalid_velocity_sample",
            "qc_flags": 16,
        }
        for row in particles
        if row["status"] == "numerical_failure"
    ]
    summary["diagnostics"] = [
        {
            "status": "numerical_failure",
            "count": terminal_counts["numerical_failure"],
            "failure_reason": "invalid_velocity_sample",
            "qc_flags": 16,
        }
    ]
    _write_json(preview_dir / "summary.json", summary)
    manifest: dict[str, Any] = {
        "artifact_kind": "pilot-preview-v1",
        "run_id": summary["run_id"],
        "files": {
            name: _file_contract(preview_dir / name)
            for name in ("summary.json", "particles.csv", "observations.csv")
        },
    }
    _write_json(preview_dir / "manifest.json", manifest)
    return {
        "preview_dir": preview_dir,
        "domain_path": domain_path,
        "open_path": open_path,
        "coastline_path": coastline_path,
    }


@pytest.fixture
def allow_synthetic_coastline_sha(synthetic_inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """只隔離固定海岸檔身分 gate，不把合成 payload 冒充成真實海岸資料。

    frozen script 以已核定的 raw SHA-256 白名單辨識真實海岸來源；本測試不能依賴
    ignored work/ 或下載圖資，因此只在 fixture coastline 的明確路徑上替換該一個
    raw SHA。`_read_json` 仍負責解析 payload，canonical SHA 與實際路徑原樣保留，
    其餘 preview、domain、open 路徑完全走原函式。monkeypatch 會在每個測試結束時
    還原，不影響其他測試或生產 script。
    """

    original_read_json = _TARGET._read_json
    coastline_path = synthetic_inputs["coastline_path"].resolve()

    def read_json(path: str | Path):
        """僅將此測試 coastline 的 raw identity 對齊 frozen allow-list。"""

        payload, raw_sha, canonical_sha, actual_path = original_read_json(path)
        if Path(path).resolve() != coastline_path:
            return payload, raw_sha, canonical_sha, actual_path
        return payload, _APPROVED_COASTLINE_SHA, canonical_sha, actual_path

    monkeypatch.setattr(_TARGET, "_read_json", read_json)


@pytest.fixture
def writable_mpl_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """把 Matplotlib 的快取與字型設定導向可寫入的暫存目錄。"""

    config_dir = tmp_path / "matplotlib-config"
    config_dir.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(config_dir))
    monkeypatch.setenv("MPLBACKEND", "Agg")


def _load_inputs(fixture: dict[str, Path]) -> dict[str, Any]:
    """用穩定四參數 API 載入合成 fixture，集中避免測試重複組路徑。"""

    return _TARGET.load_plot_inputs(
        fixture["preview_dir"],
        fixture["domain_path"],
        fixture["open_path"],
        fixture["coastline_path"],
    )


def _rewrite_particle_statuses(
    preview_dir: Path,
    statuses: list[str],
    ages: list[float],
) -> None:
    """只在測試 fixture 內重寫粒子終止欄位，並同步所有既有計數契約。

    這個 helper 用來建立 forcing_start 的完成／提前反例；它模擬的是已存在的
    ``particles.csv`` 與 ``summary.json``，不會呼叫引擎或改變 renderer 的資料層規則。
    每顆粒子的 ``status`` 與 ``age_seconds`` 仍分開保存，測試才能確認圖面分類沒有把
    提前 forcing_start 改寫成 max_age 或其他狀態。
    """

    particles_path = preview_dir / "particles.csv"
    particle_rows = list(csv.DictReader(particles_path.open(newline="", encoding="utf-8")))
    assert len(particle_rows) == len(statuses) == len(ages)
    for row, status, age in zip(particle_rows, statuses, ages, strict=True):
        row["status"] = status
        row["age_seconds"] = f"{age:.12f}"
        if status == "numerical_failure":
            row["failure_reason"] = "invalid_velocity_sample"
            row["failure_stage"] = "step_start"
            row["qc_flags"] = "16"
        else:
            row["failure_reason"] = "unknown"
            row["failure_stage"] = "unknown"
            row["qc_flags"] = ""
    with particles_path.open(newline="", encoding="utf-8") as stream:
        fields = list(csv.DictReader(stream).fieldnames or [])
    _write_csv(particles_path, fields, particle_rows)

    manifest_path = preview_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["particles.csv"] = _file_contract(particles_path)
    _write_json(manifest_path, manifest)

    summary_path = preview_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["terminal_counts"] = {
        status: sum(row["status"] == status for row in particle_rows) for status in _TERMINAL_STATUSES
    }
    summary["terminal_counts_by_vertical"] = {
        vertical_id: {
            status: sum(
                row["vertical_id"] == vertical_id and row["status"] == status for row in particle_rows
            )
            for status in _TERMINAL_STATUSES
        }
        for vertical_id in _VERTICAL_LEVELS
    }
    summary["failure_details"] = [
        {
            "particle_id": row["particle_id"],
            "status": "numerical_failure",
            "failure_reason": "invalid_velocity_sample",
            "qc_flags": 16,
        }
        for row in particle_rows
        if row["status"] == "numerical_failure"
    ]
    summary["diagnostics"] = [
        {
            "status": "numerical_failure",
            "count": summary["terminal_counts"]["numerical_failure"],
            "failure_reason": "invalid_velocity_sample",
            "qc_flags": 16,
        }
    ]
    _refresh_summary_contract(preview_dir, summary)


def test_load_plot_inputs_happy_path_preserves_counts_ids_and_denominators(
    synthetic_inputs: dict[str, Path],
    allow_synthetic_coastline_sha: None,
) -> None:
    """確認 happy path 保留 20 顆粒子、203 筆觀測及五面板四層分母語意。"""

    data = _load_inputs(synthetic_inputs)
    expected_ids = {
        f"B-H{panel_number}-{vertical_id}" for panel_number in range(1, 6) for vertical_id in _VERTICAL_LEVELS
    }
    particle_ids = {row["particle_id"] for row in data["particles"]}
    observation_ids = {row["particle_id"] for row in data["observations"]}

    assert len(data["particles"]) == 20
    assert len(data["observations"]) == 203
    assert particle_ids == expected_ids
    assert observation_ids == expected_ids
    assert len(data["summary"]["horizontal_panels"]) == 5
    assert data["summary"]["members_per_scenario"] == 1
    assert {row["horizontal_panel"] for row in data["particles"]} == {"1", "2", "3", "4", "5"}
    assert Counter(row["status"] for row in data["particles"]) == Counter(
        {"max_age": 9, "flow_domain_open_exit": 8, "numerical_failure": 3}
    )
    assert len(data["land_geometries"]) == 1905


def test_loader_rejects_csv_checksum_mismatch(synthetic_inputs: dict[str, Path]) -> None:
    """確認 CSV bytes 改變而未更新 manifest 時會在讀取前拒絕。"""

    particles_path = synthetic_inputs["preview_dir"] / "particles.csv"
    original = particles_path.read_text(encoding="utf-8")
    changed = original.replace("max_age", "numerical_failure", 1)
    assert changed != original
    particles_path.write_text(changed, encoding="utf-8")

    with pytest.raises(ValueError, match="preview particles.csv bytes/checksum 不符"):
        _load_inputs(synthetic_inputs)


def test_loader_rejects_domain_canonical_mismatch(synthetic_inputs: dict[str, Path]) -> None:
    """確認 domain 幾何語意變更且未同步 summary binding 時會被拒絕。"""

    domain_path = synthetic_inputs["domain_path"]
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    domain["records"][0]["geometry"]["coordinates"][0][0][0] += 0.001
    _write_json(domain_path, domain)

    with pytest.raises(ValueError, match="domain/open canonical hash"):
        _load_inputs(synthetic_inputs)


def test_loader_rejects_unpatched_synthetic_coastline_sha(synthetic_inputs: dict[str, Path]) -> None:
    """確認未隔離 frozen 海岸 raw SHA 時，合成 coastline 會被明確拒絕。"""

    with pytest.raises(ValueError, match="海岸檔 SHA-256 與本次已確認來源不符"):
        _load_inputs(synthetic_inputs)


def test_loader_rejects_wrong_open_owner_or_exterior(synthetic_inputs: dict[str, Path]) -> None:
    """確認同步 open canonical binding 後，錯 owner 與錯外框仍不能進入繪圖。"""

    open_path = synthetic_inputs["open_path"]
    summary_path = synthetic_inputs["preview_dir"] / "summary.json"
    original_open = json.loads(open_path.read_text(encoding="utf-8"))
    original_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for mutation, message in (("owner", "flow owner"), ("exterior", "domain exterior")):
        _write_json(open_path, original_open)
        open_boundary = json.loads(open_path.read_text(encoding="utf-8"))
        record = open_boundary["records"][0]
        if mutation == "owner":
            record["owner_id"] = "not-B-flow"
        else:
            record["geometry"]["coordinates"][1][0] += 0.001
        _write_json(open_path, open_boundary)

        summary = json.loads(json.dumps(original_summary))
        summary["projection"]["geometry_canonical_hashes"]["open_boundary"] = _canonical_hash(open_boundary)
        _refresh_summary_contract(synthetic_inputs["preview_dir"], summary)

        with pytest.raises(ValueError, match=message):
            _load_inputs(synthetic_inputs)


def test_projection_roundtrip_matches_fixture_xy(
    synthetic_inputs: dict[str, Path], allow_synthetic_coastline_sha: None
) -> None:
    """確認 AEQD 公尺座標與經緯度交換可逆，且 CSV particle XY 由同一投影產生。"""

    data = _load_inputs(synthetic_inputs)
    projection = data["projection"]
    longitude = np.asarray([120.01, 120.1, 120.19])
    latitude = np.asarray([24.71, 24.8, 24.89])
    x_m, y_m = projection.project(longitude, latitude)
    recovered_lon, recovered_lat = projection.unproject(x_m, y_m)
    np.testing.assert_allclose(recovered_lon, longitude, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(recovered_lat, latitude, rtol=0.0, atol=1e-10)

    for particle in data["particles"]:
        expected_x, expected_y = projection.project(float(particle["longitude"]), float(particle["latitude"]))
        assert float(particle["x_m"]) == pytest.approx(float(expected_x), abs=1e-6)
        assert float(particle["y_m"]) == pytest.approx(float(expected_y), abs=1e-6)


def test_render_pngs_decode_and_leave_input_hashes_unchanged(
    synthetic_inputs: dict[str, Path],
    tmp_path: Path,
    writable_mpl_config: None,
    allow_synthetic_coastline_sha: None,
) -> None:
    """確認兩種 renderer 產生可解碼 PNG，且不改寫任何 preview 或幾何輸入 bytes。"""

    data = _load_inputs(synthetic_inputs)
    source_paths = [
        synthetic_inputs["preview_dir"] / name
        for name in ("manifest.json", "summary.json", "particles.csv", "observations.csv")
    ]
    source_paths.extend(synthetic_inputs[name] for name in ("domain_path", "open_path", "coastline_path"))
    before = {path: _file_hash(path) for path in source_paths}

    overview_path = tmp_path / "rendered" / "overview.png"
    local_path = tmp_path / "rendered" / "local.png"
    assert _TARGET.render_overview(data, overview_path) == overview_path
    assert _TARGET.render_local(data, local_path) == local_path

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.image as mpimg

    for output_path in (overview_path, local_path):
        decoded = mpimg.imread(output_path)
        assert decoded.ndim >= 2
        assert decoded.size > 0
        assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert {path: _file_hash(path) for path in source_paths} == before


def test_render_refuses_existing_destination_without_overwrite(
    synthetic_inputs: dict[str, Path],
    tmp_path: Path,
    writable_mpl_config: None,
    allow_synthetic_coastline_sha: None,
) -> None:
    """確認總覽圖與局部圖都以排他建立，既有目的 PNG 內容維持原樣。"""

    data = _load_inputs(synthetic_inputs)
    for renderer, filename in (
        (_TARGET.render_overview, "overview.png"),
        (_TARGET.render_local, "local.png"),
    ):
        output_path = tmp_path / filename
        sentinel = b"existing-output-sentinel"
        output_path.write_bytes(sentinel)
        with pytest.raises(FileExistsError):
            renderer(data, output_path)
        assert output_path.read_bytes() == sentinel


def test_baytrace_build_emits_four_pngs_readme_and_rebuild_contract(
    synthetic_inputs: dict[str, Path],
    tmp_path: Path,
    writable_mpl_config: None,
    allow_synthetic_coastline_sha: None,
) -> None:
    """確認新版輸出、繁中標籤、來源 SHA 與重建命令的 style/output-dir 位置都正確。"""

    output = tmp_path / "baytrace-style"
    manifest = _TARGET.build_coastline_preview(
        synthetic_inputs["preview_dir"],
        synthetic_inputs["domain_path"],
        synthetic_inputs["open_path"],
        synthetic_inputs["coastline_path"],
        output,
        style="baytrace",
    )

    expected_files = {
        "horizontal_overview.png",
        "horizontal_local.png",
        "depth_age.png",
        "terminal_counts.png",
        "README.md",
        "manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected_files
    assert manifest["style"] == "baytrace"
    assert manifest["style_version"] == _TARGET._BAYTRACE_STYLE_VERSION
    assert manifest["particle_count"] == 20
    assert manifest["observation_count"] == 203
    assert manifest["terminal_counts"]["max_age"] == 9
    assert manifest["terminal_counts"]["flow_domain_open_exit"] == 8
    assert manifest["terminal_counts"]["numerical_failure"] == 3
    assert set(manifest["files"]) == {
        "horizontal_overview.png",
        "horizontal_local.png",
        "depth_age.png",
        "terminal_counts.png",
        "README.md",
    }
    assert set(manifest["sources_before"]) == {
        "manifest.json",
        "summary.json",
        "particles.csv",
        "observations.csv",
        "domain",
        "open_boundary",
        "coastline",
    }
    assert manifest["sources_before"] == manifest["sources_after"]

    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "BayTrace v4.7.2" in readme
    assert "scripts/analyze_cases.py" in readme
    assert "plot_case" in readme
    assert "初始位置（回溯起點）" in readme
    assert "最終位置｜到達設定回溯時間上限（1小時）" in readme
    assert "到達：2025-01-01 12:00 UTC；回溯至：2025-01-01 11:00 UTC" in readme
    assert "位置1–5是五個受體／指定位置，不是時間順序" in readme
    assert "各位置以本身回溯起點為 (0,0)" in readme
    assert "各粒子所在位置的海床高度（每圖 5 條）" in readme
    assert "每條彩色實線代表 1 顆粒子的高度軌跡" in readme
    assert "B／新竹外海" in readme or "B區" in readme
    assert readme.count("| 位置") == 5
    for position in range(1, 6):
        assert f"| 位置{position} | 原 H{position} |" in readme

    # 重建命令中的 style 必須仍是 baytrace，output-dir 則要指向另一個全新目錄；
    # 這個斷言直接鎖定曾發生過的 args[-1] 誤改 style 值回歸問題。
    rebuild_args = shlex.split(manifest["rebuild_command"])
    assert rebuild_args[rebuild_args.index("--style") + 1] == "baytrace"
    rebuild_output = Path(rebuild_args[rebuild_args.index("--output-dir") + 1])
    assert rebuild_output == output.with_name(output.name + "-rebuild")
    assert rebuild_output != output
    for filename in expected_files - {"manifest.json"}:
        if filename.endswith(".png"):
            assert (output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_baytrace_build_uses_summary_horizon_run_counts_and_depth_axis(
    synthetic_inputs: dict[str, Path],
    tmp_path: Path,
    writable_mpl_config: None,
    allow_synthetic_coastline_sha: None,
) -> None:
    """確認新版圖面由 summary 驅動 12 小時、run 身分與非舊版停止計數。

    這個回歸案例刻意把合成 fixture 改成 12 小時、202 筆保存紀錄及 6／10／4
    三種停止計數；若 renderer 仍依賴舊 r2 的 20／203、9／8／3 或 0--60 分鐘
    常數，應在產圖前失敗。資料仍保留五個位置、四個水層與每層五顆粒子的既有
    圖面拓樸，讓測試只隔離本次必要的資料驅動泛化行為。
    """

    preview_dir = synthetic_inputs["preview_dir"]
    particles_path = preview_dir / "particles.csv"
    particle_rows = list(csv.DictReader(particles_path.open(newline="", encoding="utf-8")))
    status_sequence = (
        ("max_age",) * 6
        + ("flow_domain_open_exit",) * 10
        + ("numerical_failure",) * 4
    )
    assert len(particle_rows) == len(status_sequence) == 20
    status_by_particle: dict[str, str] = {}
    for row, status in zip(particle_rows, status_sequence, strict=True):
        row["status"] = status
        status_by_particle[row["particle_id"]] = status
        if status == "numerical_failure":
            row["failure_reason"] = "invalid_velocity_sample"
            row["failure_stage"] = "step_start"
            row["qc_flags"] = "16"
        else:
            row["failure_reason"] = "unknown"
            row["failure_stage"] = "unknown"
            row["qc_flags"] = ""
        row["age_seconds"] = "43200.000000000000" if status == "max_age" else "1800.000000000000"
    with particles_path.open(newline="", encoding="utf-8") as stream:
        particle_fields = list(csv.DictReader(stream).fieldnames or [])
    _write_csv(particles_path, particle_fields, particle_rows)

    observations_path = preview_dir / "observations.csv"
    observations = list(csv.DictReader(observations_path.open(newline="", encoding="utf-8")))
    assert observations
    for row in observations:
        particle_status = status_by_particle[row["particle_id"]]
        original_horizon = 3600.0 if particle_status == "max_age" else 1800.0
        fraction = float(row["age_seconds"]) / original_horizon
        new_horizon = 43200.0 if particle_status == "max_age" else 1800.0
        row["age_seconds"] = f"{fraction * new_horizon:.12f}"
    observations.pop()
    with observations_path.open(newline="", encoding="utf-8") as stream:
        observation_fields = list(csv.DictReader(stream).fieldnames or [])
    _write_csv(observations_path, observation_fields, observations)

    preview_manifest_path = preview_dir / "manifest.json"
    preview_manifest = json.loads(preview_manifest_path.read_text(encoding="utf-8"))
    preview_manifest["files"]["particles.csv"] = _file_contract(particles_path)
    preview_manifest["files"]["observations.csv"] = _file_contract(observations_path)
    _write_json(preview_manifest_path, preview_manifest)

    summary_path = preview_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_id"] = "b-fishinggear-m1-12h-r3"
    preview_manifest["run_id"] = summary["run_id"]
    _write_json(preview_manifest_path, preview_manifest)
    summary["settings"]["horizon_seconds"] = 43200.0
    summary["observation_count"] = len(observations)
    summary["terminal_counts"] = {
        status: sum(row["status"] == status for row in particle_rows) for status in _TERMINAL_STATUSES
    }
    summary["terminal_counts_by_vertical"] = {
        vertical_id: {
            status: sum(
                row["vertical_id"] == vertical_id and row["status"] == status for row in particle_rows
            )
            for status in _TERMINAL_STATUSES
        }
        for vertical_id in _VERTICAL_LEVELS
    }
    summary["failure_details"] = [
        {
            "particle_id": row["particle_id"],
            "status": "numerical_failure",
            "failure_reason": "invalid_velocity_sample",
            "qc_flags": 16,
        }
        for row in particle_rows
        if row["status"] == "numerical_failure"
    ]
    summary["diagnostics"] = [
        {
            "status": "numerical_failure",
            "count": summary["terminal_counts"]["numerical_failure"],
            "failure_reason": "invalid_velocity_sample",
            "qc_flags": 16,
        }
    ]
    _refresh_summary_contract(preview_dir, summary)

    data = _load_inputs(synthetic_inputs)
    assert _TARGET._baytrace_status_labels(data)["max_age"] == "到達設定回溯時間上限（12小時）"
    assert _TARGET._depth_axis_limits_and_ticks(summary) == (720.0, [0.0, 180.0, 360.0, 540.0, 720.0])

    output = tmp_path / "baytrace-style-12h"
    manifest = _TARGET.build_coastline_preview(
        preview_dir,
        synthetic_inputs["domain_path"],
        synthetic_inputs["open_path"],
        synthetic_inputs["coastline_path"],
        output,
        style="baytrace",
    )
    assert manifest["run_id"] == "b-fishinggear-m1-12h-r3"
    assert manifest["particle_count"] == 20
    assert manifest["observation_count"] == 202
    assert manifest["terminal_counts"] == {
        "coast_contact": 0,
        "data_gap": 0,
        "deposited": 0,
        "flow_domain_open_exit": 10,
        "forcing_start": 0,
        "max_age": 6,
        "numerical_failure": 4,
        "surface_regime_exit": 0,
    }
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "回溯上限12小時" in readme
    assert "到達設定回溯時間上限（12小時） 6 顆" in readme
    assert "202 筆模型保存紀錄" in readme
    assert "1h" not in readme
    assert "1小時" not in readme
    assert "203 筆" not in readme
    assert "9/8/3" not in readme
    assert "0–60 分鐘" not in readme


def test_legacy_build_keeps_two_png_contract_and_no_baytrace_style(
    synthetic_inputs: dict[str, Path],
    tmp_path: Path,
    writable_mpl_config: None,
    allow_synthetic_coastline_sha: None,
) -> None:
    """確認 default legacy 仍只有原兩張水平圖與舊 manifest，不混入新版診斷圖。"""

    output = tmp_path / "legacy-style"
    manifest = _TARGET.build_coastline_preview(
        synthetic_inputs["preview_dir"],
        synthetic_inputs["domain_path"],
        synthetic_inputs["open_path"],
        synthetic_inputs["coastline_path"],
        output,
    )

    assert {path.name for path in output.iterdir()} == {
        "horizontal_overview.png",
        "horizontal_local.png",
        "README.md",
        "manifest.json",
    }
    assert "style" not in manifest
    assert set(manifest["files"]) == {"horizontal_overview.png", "horizontal_local.png", "README.md"}


def test_baytrace_artist_markers_and_labels_keep_initial_final_semantics(
    synthetic_inputs: dict[str, Path],
    tmp_path: Path,
    writable_mpl_config: None,
    allow_synthetic_coastline_sha: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """確認新版 artist 實際保留 20 個綠色起點及三種既有終點的獨立形狀。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.axes import Axes

    data = _load_inputs(synthetic_inputs)
    labels = _TARGET._baytrace_status_labels(data)
    handles = _TARGET._baytrace_endpoint_handles(
        labels,
        include_boundary=False,
        statuses={"max_age", "flow_domain_open_exit", "numerical_failure"},
    )
    assert [handle.get_label() for handle in handles] == [
        "初始位置（回溯起點）",
        "最終位置｜離開計算範圍",
        "最終位置｜到達設定回溯時間上限（1小時）",
        "最終位置｜取樣失敗停止",
    ]
    assert all(handle.get_color() == _TARGET._BAYTRACE_FINAL_COLOR for handle in handles[1:])

    original_plot = Axes.plot
    marker_calls: list[tuple[str, str | None, str | None]] = []

    def traced_plot(self, *args: Any, **kwargs: Any):
        """收集實際座標軸端點 artist，仍交回 Matplotlib 完成正常繪圖。"""

        marker = kwargs.get("marker")
        if marker in {"o", "^", "s", "X"}:
            marker_calls.append((marker, kwargs.get("markerfacecolor"), kwargs.get("markeredgecolor")))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", traced_plot)
    _TARGET.render_baytrace_overview(data, tmp_path / "artist-overview.png")

    assert sum(marker == "o" for marker, _, _ in marker_calls) == 20
    assert Counter(marker for marker, _, _ in marker_calls if marker in {"^", "s", "X"}) == Counter(
        {"^": 9, "s": 8, "X": 3}
    )
    assert all(face == _TARGET._BAYTRACE_INITIAL_COLOR for marker, face, _ in marker_calls if marker == "o")
    assert all(
        face == _TARGET._BAYTRACE_FINAL_COLOR
        for marker, face, _ in marker_calls
        if marker in {"^", "s", "X"}
    )


def test_forcing_start_near_horizon_uses_data_window_label_and_full_time_summary(
    synthetic_inputs: dict[str, Path],
    tmp_path: Path,
    writable_mpl_config: None,
    allow_synthetic_coastline_sha: None,
) -> None:
    """確認容差內 forcing_start 顯示資料窗端點，且不改寫原始狀態或零計數分類。"""

    horizon = 3600.0
    # 這個差值比固定 1e-4 秒容差小，模擬真資料約 1.6e-5 秒的浮點誤差。
    _rewrite_particle_statuses(
        synthetic_inputs["preview_dir"],
        ["forcing_start"] * 20,
        [horizon - 1.6e-5] * 20,
    )
    data = _load_inputs(synthetic_inputs)

    assert {row["status"] for row in data["particles"]} == {"forcing_start"}
    assert set(_TARGET._forcing_start_age_states(data)) == {"completed"}
    assert _TARGET._forcing_start_label(data) == "到達本次資料時間窗起點（完成1小時回溯）"
    assert _TARGET._time_window_label(data["summary"]) == (
        "到達：2025-01-01 12:00 UTC；回溯至：2025-01-01 11:00 UTC"
    )
    assert _TARGET._baytrace_terminal_summary(data) == (
        "20 顆到達本次資料時間窗起點（完成1小時回溯）、0 顆離域、0 顆數值失敗。"
    )

    labels = _TARGET._baytrace_status_labels(data)
    assert labels["forcing_start"] != labels["max_age"]
    assert labels["max_age"] == "到達設定回溯時間上限（1小時）"
    handles = _TARGET._baytrace_endpoint_handles(
        labels,
        include_boundary=False,
        forcing_start_states={"completed"},
        forcing_start_summary=data["summary"],
        statuses={"forcing_start", "max_age", "numerical_failure"},
    )
    forcing_handle = next(handle for handle in handles if "資料時間窗起點" in handle.get_label())
    max_age_handle = next(handle for handle in handles if "設定回溯時間上限" in handle.get_label())
    numerical_handle = next(handle for handle in handles if "數值停止" in handle.get_label())
    assert forcing_handle.get_marker() == "D"
    assert forcing_handle.get_color() == _TARGET._BAYTRACE_FORCING_START_COLOR
    assert max_age_handle.get_marker() == "^"
    assert max_age_handle.get_label() == "最終位置｜到達設定回溯時間上限（1小時）"
    assert numerical_handle.get_marker() == "X"
    assert numerical_handle.get_color() == _TARGET._BAYTRACE_FINAL_COLOR

    output = tmp_path / "forcing-start-completed"
    manifest = _TARGET.build_coastline_preview(
        synthetic_inputs["preview_dir"],
        synthetic_inputs["domain_path"],
        synthetic_inputs["open_path"],
        synthetic_inputs["coastline_path"],
        output,
        style="baytrace",
    )
    assert manifest["terminal_counts"]["forcing_start"] == 20
    assert manifest["terminal_counts"]["max_age"] == 0
    assert manifest["forcing_start_age_tolerance_seconds"] == pytest.approx(1e-4)
    assert manifest["forcing_start_age_state_counts"] == {"completed": 20}
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "到達：2025-01-01 12:00 UTC；回溯至：2025-01-01 11:00 UTC" in readme
    assert "20 顆到達本次資料時間窗起點（完成1小時回溯）、0 顆離域、0 顆數值失敗。" in readme
    assert "最終位置｜到達設定回溯時間上限（1小時）" in readme
    assert "0 顆完成回溯" not in readme


def test_forcing_start_early_than_horizon_is_not_completion(
    synthetic_inputs: dict[str, Path],
    allow_synthetic_coastline_sha: None,
) -> None:
    """確認明顯早於 horizon 的 forcing_start 顯示提前，資料層仍不變。

    反例刻意提前一秒；相較 1e-4 秒固定容差仍有四個數量級差距，不能被 output
    interval 300 秒這種過寬界線吞掉。
    """

    horizon = 3600.0
    _rewrite_particle_statuses(
        synthetic_inputs["preview_dir"],
        ["forcing_start"] * 20,
        [horizon - 1.0] * 20,
    )
    data = _load_inputs(synthetic_inputs)

    assert _TARGET._forcing_start_age_state(horizon - 0.5e-4, horizon) == "completed"
    assert _TARGET._forcing_start_age_state(horizon - 2.0e-4, horizon) == "early"
    assert set(_TARGET._forcing_start_age_states(data)) == {"early"}
    assert _TARGET._forcing_start_label(data) == "提前到達驅動資料起點"
    assert _TARGET._baytrace_terminal_summary(data) == "20 顆提前到達驅動資料起點、0 顆離域、0 顆數值失敗。"
    assert {row["status"] for row in data["particles"]} == {"forcing_start"}
    marker, color = _TARGET._baytrace_endpoint_style("forcing_start", "early")
    assert marker == "D"
    assert color == _TARGET._BAYTRACE_FORCING_START_EARLY_COLOR
    assert color != _TARGET._BAYTRACE_FINAL_COLOR


def test_baytrace_terminal_markers_keep_all_registered_states_distinct(
    synthetic_inputs: dict[str, Path],
    allow_synthetic_coastline_sha: None,
) -> None:
    """確認離域、海岸接觸、資料缺口等狀態各有 marker，不會退回同一個預設菱形。"""

    data = _load_inputs(synthetic_inputs)
    labels = _TARGET._baytrace_status_labels(data)
    assert set(_TARGET._STATUS_MARKERS) == set(_TERMINAL_STATUSES)
    assert len(set(_TARGET._STATUS_MARKERS.values())) == len(_TERMINAL_STATUSES)
    assert labels["coast_contact"] == "接觸海岸"
    assert labels["surface_regime_exit"] == "離開表面適用範圍"
    assert labels["deposited"] == "沉積"
    assert labels["data_gap"] == "資料缺口"
    assert _TARGET._baytrace_endpoint_style("coast_contact")[0] == "P"
    assert _TARGET._baytrace_endpoint_style("numerical_failure") == (
        "X",
        _TARGET._BAYTRACE_FINAL_COLOR,
    )
    assert _TARGET._baytrace_endpoint_style("forcing_start", "completed") == (
        "D",
        _TARGET._BAYTRACE_FORCING_START_COLOR,
    )


def test_baytrace_depth_keeps_missing_eta_and_bed_blank(
    synthetic_inputs: dict[str, Path],
    tmp_path: Path,
    writable_mpl_config: None,
    allow_synthetic_coastline_sha: None,
) -> None:
    """確認 eta_m 與 bed_z_m 各 20 筆空白在數列中仍是 NaN，不被補成零。"""

    data = _load_inputs(synthetic_inputs)
    for field in ("eta_m", "bed_z_m"):
        values = _TARGET._plot_series(data["observations"], field)
        expected_values = [
            float(row[field]) if row[field].strip() else np.nan for row in data["observations"]
        ]
        assert len(values) == len(expected_values)
        for actual, expected in zip(values, expected_values, strict=True):
            if np.isnan(expected):
                assert np.isnan(actual)
            else:
                assert actual == pytest.approx(expected, rel=0.0, abs=0.0)
        assert int(np.isnan(values).sum()) == 20

    output = tmp_path / "depth.png"
    assert _TARGET.render_baytrace_depth(data, output) == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_baytrace_depth_labels_describe_per_particle_lines_and_dynamic_bed_count() -> None:
    """確認垂向圖說明逐粒子高度線，海床圖例只在同數時標示精確每圖數量。"""

    assert _TARGET._baytrace_depth_particle_note(5) == "每條彩色實線代表 1 顆粒子的高度軌跡（共 5 顆）"
    assert _TARGET._baytrace_depth_bed_label([5, 5, 5, 5]) == "各粒子所在位置的海床高度（每圖 5 條）"
    assert _TARGET._baytrace_depth_bed_label([5, 4, 5, 4]) == "各粒子所在位置的海床高度（各圖數量依子圖標示）"


def test_baytrace_terminal_table_rejects_swapped_summary_cells(
    synthetic_inputs: dict[str, Path],
    allow_synthetic_coastline_sha: None,
) -> None:
    """確認水層內停止計數逐格對 CSV，單純維持每層五顆仍不能掩蓋對調。"""

    data = _load_inputs(synthetic_inputs)
    row = data["summary"]["terminal_counts_by_vertical"]["near_bed"]
    row["max_age"], row["flow_domain_open_exit"] = row["flow_domain_open_exit"], row["max_age"]
    with pytest.raises(ValueError, match="逐格計數"):
        _TARGET._baytrace_terminal_table(data)


def test_baytrace_build_refuses_existing_directory_and_broken_symlink(
    synthetic_inputs: dict[str, Path],
    tmp_path: Path,
    allow_synthetic_coastline_sha: None,
) -> None:
    """確認新版 build 對既有目錄與失效符號連結均在讀取前拒絕，且不刪除目標。"""

    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _TARGET.build_coastline_preview(
            synthetic_inputs["preview_dir"],
            synthetic_inputs["domain_path"],
            synthetic_inputs["open_path"],
            synthetic_inputs["coastline_path"],
            existing,
            style="baytrace",
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    broken = tmp_path / "broken-link"
    broken.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)
    with pytest.raises(FileExistsError):
        _TARGET.build_coastline_preview(
            synthetic_inputs["preview_dir"],
            synthetic_inputs["domain_path"],
            synthetic_inputs["open_path"],
            synthetic_inputs["coastline_path"],
            broken,
            style="baytrace",
        )
    assert broken.is_symlink()
