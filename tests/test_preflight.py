"""metadata-only preflight 的 schema、status、時間與敏感路徑測試。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from lagrangian_backtracking.config import ProjectConfig
from lagrangian_backtracking.preflight import run_preflight

ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> ProjectConfig:
    """沿用正式 schema，但把年份保留 2024；測試呼叫只盤點一個月份。"""

    payload = yaml.safe_load(
        (ROOT / "configs" / "lagrangian_backtracking.example.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return ProjectConfig.model_validate(deepcopy(payload))


def _write_month(
    root: Path,
    domain: str,
    *,
    product: str,
    status: str,
    month: str = "202401",
    times: np.ndarray | None = None,
) -> None:
    """建立不含大型物理值的小型月份契約；所有必要陣列只寫 shape 正確的占位值。"""

    month_dir = root / domain / "months" / month
    month_dir.mkdir(parents=True)
    if times is None:
        times = np.array([1_704_067_200_000_000_000, 1_704_070_800_000_000_000], dtype=np.int64)
    np.save(month_dir / "time_utc_ns.npy", times, allow_pickle=False)
    if product == "ocm_native":
        names = [
            "hvel.npy",
            "vertical_velocity.npy",
            "zcor.npy",
            "elev.npy",
            "wetdry_elem.npy",
            "diffusivity.npy",
        ]
        schema_key = "cache_schema_version"
        schema_value = "3.0.0"
        cache_kind = "standard_month"
    else:
        names = [
            "significant_wave_height.npy",
            "peak_frequency.npy",
            "peak_direction_raw_deg.npy",
            "valid_mask_wave.npy",
            "qc_flags.npy",
        ]
        schema_key = "schema_version"
        schema_value = "1.0.0"
        cache_kind = "ocm_analysis_grid_resample_from_nww3_native"
    arrays = {"time_utc_ns.npy": {"shape": [2], "dtype": "int64"}}
    for name in names:
        dtype = (
            np.bool_ if name == "valid_mask_wave.npy" else np.uint16 if name == "qc_flags.npy" else np.float32
        )
        np.save(month_dir / name, np.zeros((2, 1), dtype=dtype), allow_pickle=False)
        arrays[name] = {"shape": [2, 1], "dtype": str(np.dtype(dtype))}
    metadata = {schema_key: schema_value, "status": status, "cache_kind": cache_kind, "arrays": arrays}
    (month_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _write_grid(root: Path, domain: str, *, product: str) -> None:
    """建立 preflight 所需的最小靜態 grid metadata 與占位陣列。"""

    grid_dir = root / domain / "grid"
    grid_dir.mkdir(parents=True)
    if product == "ocm_native":
        names = [
            "source_lon.npy",
            "source_lat.npy",
            "source_face_nodes_local.npy",
            "source_face_node_count.npy",
            "source_depth_m.npy",
            "source_node_bottom_index.npy",
            "source_face_global_index.npy",
        ]
        metadata = {"cache_schema_version": "3.0.0", "domain": {"domain_id": domain}}
    else:
        names = ["lon.npy", "lat.npy", "mask_static.npy"]
        metadata = {"schema_version": "1.0.0", "flow_domain_id": domain}
    for name in names:
        np.save(grid_dir / name, np.zeros(1, dtype=np.float32), allow_pickle=False)
    (grid_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_preflight_accepts_ready_fixture_and_tokenizes_paths(tmp_path: Path) -> None:
    """完整 ready fixture 應無 finding，報告不得洩漏 tmp 絕對路徑。"""

    config = _small_config()
    ocm_root = tmp_path / "ocm"
    nww_root = tmp_path / "nww"
    for domain in config.domains:
        _write_grid(ocm_root, domain.flow_domain_id, product="ocm_native")
        _write_grid(nww_root, domain.flow_domain_id, product="nww3_analysis")
        _write_month(ocm_root, domain.flow_domain_id, product="ocm_native", status="ready")
        _write_month(nww_root, domain.flow_domain_id, product="nww3_analysis", status="ready")
    report = run_preflight(config, ocm_native_root=ocm_root, nww_analysis_root=nww_root, months=["202401"])
    assert report.formal_ready
    assert len(report.inventories) == 8
    serialized = json.dumps(report.to_dict(), ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert "$OCM_NATIVE_ROOT/" in serialized


def test_preflight_blocks_trial_ready_nww(tmp_path: Path) -> None:
    """SERVER 現況的 NWW ``trial_ready`` 只能供 pilot，正式 status gate 必須報錯。"""

    config = _small_config()
    ocm_root = tmp_path / "ocm"
    nww_root = tmp_path / "nww"
    for domain in config.domains:
        _write_grid(ocm_root, domain.flow_domain_id, product="ocm_native")
        _write_grid(nww_root, domain.flow_domain_id, product="nww3_analysis")
        _write_month(ocm_root, domain.flow_domain_id, product="ocm_native", status="ready")
        _write_month(nww_root, domain.flow_domain_id, product="nww3_analysis", status="trial_ready")
    report = run_preflight(config, ocm_native_root=ocm_root, nww_analysis_root=nww_root, months=["202401"])
    assert not report.formal_ready
    assert {finding.code for finding in report.findings} == {"STATUS_NOT_READY"}
    assert len(report.findings) == 4


def test_preflight_checks_actual_npy_header_against_metadata(tmp_path: Path) -> None:
    """大型陣列即使檔案存在，實際 shape/dtype 與 metadata 不同仍須阻擋。"""

    config = _small_config()
    ocm_root = tmp_path / "ocm"
    nww_root = tmp_path / "nww"
    for domain in config.domains:
        _write_grid(ocm_root, domain.flow_domain_id, product="ocm_native")
        _write_grid(nww_root, domain.flow_domain_id, product="nww3_analysis")
        _write_month(ocm_root, domain.flow_domain_id, product="ocm_native", status="ready")
        _write_month(nww_root, domain.flow_domain_id, product="nww3_analysis", status="ready")
    first_domain = config.domains[0].flow_domain_id
    np.save(
        nww_root / first_domain / "months" / "202401" / "significant_wave_height.npy",
        np.zeros((3, 1), dtype=np.float32),
        allow_pickle=False,
    )
    report = run_preflight(config, ocm_native_root=ocm_root, nww_analysis_root=nww_root, months=["202401"])
    mismatches = [finding for finding in report.findings if finding.code == "ARRAY_CONTRACT_MISMATCH"]
    assert len(mismatches) == 1
    assert "significant_wave_height.npy" in mismatches[0].location


def test_preflight_checks_cross_month_time_gap(tmp_path: Path) -> None:
    """各月內部皆為 1 小時仍不足；相鄰月份邊界的 10 小時缺口也必須被找出。"""

    config = _small_config()
    ocm_root = tmp_path / "ocm"
    nww_root = tmp_path / "nww"
    january = np.array(["2024-01-31T22:00", "2024-01-31T23:00"], dtype="datetime64[ns]").astype(np.int64)
    february = np.array(["2024-02-01T09:00", "2024-02-01T10:00"], dtype="datetime64[ns]").astype(np.int64)
    for domain in config.domains:
        _write_grid(ocm_root, domain.flow_domain_id, product="ocm_native")
        _write_grid(nww_root, domain.flow_domain_id, product="nww3_analysis")
        for month, times in (("202401", january), ("202402", february)):
            _write_month(
                ocm_root,
                domain.flow_domain_id,
                product="ocm_native",
                status="ready",
                month=month,
                times=times,
            )
            _write_month(
                nww_root,
                domain.flow_domain_id,
                product="nww3_analysis",
                status="ready",
                month=month,
                times=times,
            )
    report = run_preflight(
        config,
        ocm_native_root=ocm_root,
        nww_analysis_root=nww_root,
        months=["202402", "202401"],
    )
    gaps = [finding for finding in report.findings if finding.code == "CROSS_MONTH_TIME_GAP_EXCEEDED"]
    assert len(gaps) == 4
    assert all("gap=10 h" in finding.message for finding in gaps)
