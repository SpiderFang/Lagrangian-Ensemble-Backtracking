"""WP-10（``native_runner.gap_fill``）相鄰太陰日位移混合的驗收測試。

所有 fixture 都是小型合成 ``ocm_native``／``nww3_analysis`` 產品（參考
``tests/test_ocm_reconstruction.py``、``tests/test_native_forcing_bridge.py``、
``tests/test_native_production_wiring.py`` 既有慣例），不觸碰任何真實伺服器
資料。測試重點：輸出格式與既有讀取端（``forcing.OCMReconstructionPatchMonth``、
``forcing_window._ReconstructionPatchStore``、
``native_runner.forcing_bridge.inventory_forcing_gaps``、
``forcing.NWWAnalysisMonth``）完全相容、25 小時位移確實優於 24 小時位移、缺口
兩端銜接處沒有跳變、乾濕與垂向順序約束成立，以及既有 1 小時／23--49 小時契約
仍優先交給核心重建方法。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import lagrangian_backtracking.forcing_window as forcing_window_module
from lagrangian_backtracking.forcing import NWWAnalysisMonth, OCMReconstructionPatchMonth
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.native_runner.forcing_bridge import inventory_forcing_gaps
from lagrangian_backtracking.native_runner.gap_fill import (
    GAP_FILL_METHOD_ID,
    QUALITY_LUNAR_DAY_BLEND,
    QUALITY_SINGLE_SIDED_BLEND,
    GapFillConfig,
    build_gap_fill_patches,
    build_wave_gap_fill_patches,
    lunar_day_shift_phase_error_report,
    validate_gap_fill_blocked,
    write_gap_fill_root_index,
)
from lagrangian_backtracking.ocm_reconstruction import (
    HOURLY_NS,
    ORIGIN_RECONSTRUCTED_SHORT,
    ORIGIN_RECONSTRUCTED_STATE_SPACE,
    ReconstructionConfig,
)

M2_PERIOD_HOURS = 12.4206
K1_PERIOD_HOURS = 23.9345
_START = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)


def _times(hours: int) -> np.ndarray:
    """建立從 2024-01-01T00:00:00Z 起、逐時遞增的 UTC 軸（奈秒）。"""

    return _START + np.arange(hours, dtype=np.int64) * HOURLY_NS


def _tidal_signal(hours: int) -> np.ndarray:
    """M2＋K1 合成潮流訊號，供缺口重建與 baseline 比較共用同一組真值。"""

    index = np.arange(hours, dtype=np.float64)
    return np.sin(2.0 * np.pi * index / M2_PERIOD_HOURS) + 0.5 * np.sin(2.0 * np.pi * index / K1_PERIOD_HOURS)


def _write_ocm_grid(domain_root: Path) -> None:
    """建立 ``ocm_reconstruction`` fingerprint 與 ``NativeMesh`` 共用的最小原生網格。"""

    grid = domain_root / "grid"
    grid.mkdir(parents=True)
    np.save(grid / "source_lon.npy", np.asarray([120.0, 120.1, 120.0]))
    np.save(grid / "source_lat.npy", np.asarray([24.0, 24.0, 24.1]))
    np.save(grid / "source_depth_m.npy", np.asarray([5.0, 5.0, 5.0]))
    np.save(grid / "source_node_bottom_index.npy", np.zeros(3, dtype=np.int64))
    np.save(grid / "source_face_nodes_local.npy", np.asarray([[0, 1, 2, -1]], dtype=np.int64))
    np.save(grid / "source_face_node_count.npy", np.asarray([3], dtype=np.int64))
    np.save(grid / "source_face_global_index.npy", np.asarray([10], dtype=np.int64))


def _month_id(time_ns: int) -> str:
    return datetime.fromtimestamp(time_ns / 1_000_000_000, tz=UTC).strftime("%Y%m")


def _write_ocm_months(domain_root: Path, times: np.ndarray, signal: np.ndarray) -> tuple[str, ...]:
    """把觀測子集依月份寫成 ``ocm_native`` 月檔，回傳實際寫出的月份清單。"""

    by_month: dict[str, list[int]] = {}
    for index, time_ns in enumerate(times):
        by_month.setdefault(_month_id(int(time_ns)), []).append(index)
    for month_id, indices in by_month.items():
        idx = np.asarray(indices, dtype=np.int64)
        month_dir = domain_root / "months" / month_id
        month_dir.mkdir(parents=True)
        count = int(idx.size)
        np.save(month_dir / "time_utc_ns.npy", times[idx])
        hvel = np.zeros((count, 3, 2, 2), dtype=np.float32)
        hvel[..., 0] = signal[idx][:, None, None]
        np.save(month_dir / "hvel.npy", hvel)
        np.save(month_dir / "vertical_velocity.npy", np.zeros((count, 3, 2), dtype=np.float32))
        zcor = np.broadcast_to(np.asarray([-5.0, 0.0], dtype=np.float32), (count, 3, 2)).copy()
        np.save(month_dir / "zcor.npy", zcor)
        elev = (0.05 * signal[idx])[:, None].astype(np.float32) * np.ones((count, 3), dtype=np.float32)
        np.save(month_dir / "elev.npy", elev)
        np.save(month_dir / "wetdry_elem.npy", np.zeros((count, 1), dtype=np.int8))
        np.save(month_dir / "diffusivity.npy", np.full((count, 3, 2), 0.01, dtype=np.float32))
    return tuple(sorted(by_month))


def _load_patch_month(output_root: Path, flow_id: str, month_id: str) -> OCMReconstructionPatchMonth:
    """依既有讀取端契約（見 ``forcing_window._ReconstructionPatchStore.load_month``）
    手動載入一份輸出 patch，驗證 ``OCMReconstructionPatchMonth`` 建構不需要任何修改。
    """

    month_dir = output_root / flow_id / "months" / month_id
    arrays = {
        name: np.load(month_dir / f"{name}.npy")
        for name in (
            "time_utc_ns", "hvel", "vertical_velocity", "zcor", "elev",
            "wetdry_elem", "diffusivity", "origin_code", "quality_flags",
        )
    }
    metadata = json.loads((month_dir / "metadata.json").read_text(encoding="utf-8"))
    return OCMReconstructionPatchMonth(month_id=month_id, metadata=metadata, **arrays)


def test_lunar_day_shift_phase_error_report_prefers_25_hours() -> None:
    """25 小時位移的四分潮 RMS 相位誤差必須明顯小於 24 小時位移，且 M2／K1 個別更準。"""

    report = lunar_day_shift_phase_error_report()
    by_shift = {row["shift_hours"]: row for row in report["shift_candidates"]}
    rms_24h = by_shift[24.0]["rms_phase_error_hours"]
    rms_25h = by_shift[25.0]["rms_phase_error_hours"]
    assert rms_25h < rms_24h

    # M2（主導分潮）在 25 小時位移下個別相位誤差也必須更小；K1／S2／O1 不保證
    # 逐一勝出（見模組 docstring），只有四分潮 RMS（上面已驗證）才是選擇 25 小時
    # 的依據。
    m2_24h = by_shift[24.0]["constituents"]["M2"]["phase_error_hours"]
    m2_25h = by_shift[25.0]["constituents"]["M2"]["phase_error_hours"]
    assert m2_25h < m2_24h

    # 手算驗證（避免函式本身的算法錯誤被自我印證）：25 對 12.4206 取模＝0.1588。
    assert m2_25h == pytest.approx(25.0 % M2_PERIOD_HOURS, abs=1.0e-9)


def test_multi_day_gap_blend_matches_signal_and_loads_via_existing_reader(tmp_path: Path) -> None:
    """5 天缺口：混合結果貼近真值、無邊界跳變、約束成立，且既有讀取端不需修改。"""

    domain_root = tmp_path / "ocm" / "domainA"
    _write_ocm_grid(domain_root)
    hours = 24 * 20
    times = _times(hours)
    signal = _tidal_signal(hours)
    gap_start, gap_len = 200, 120
    keep = np.ones(hours, dtype=bool)
    keep[gap_start : gap_start + gap_len] = False
    observed_months = _write_ocm_months(domain_root, times[keep], signal[keep])

    output_root = tmp_path / "gap_fill_out"
    report = build_gap_fill_patches(
        domain_root, output_root,
        start_utc_ns=int(times[0]), end_utc_ns=int(times[-1]), config=GapFillConfig(),
    )
    assert report.row_count == gap_len
    assert report.existing_method_gap_ids == ()
    assert len(report.blend_method_gap_ids) == 1
    assert report.unfilled_time_utc_ns == ()

    patch = _load_patch_month(output_root, "domainA", report.month_ids[0])
    assert patch.valid
    assert set(np.unique(np.asarray(patch.origin_code)).tolist()) == {int(ORIGIN_RECONSTRUCTED_STATE_SPACE)}
    quality = np.asarray(patch.quality_flags)
    assert np.all((quality & QUALITY_LUNAR_DAY_BLEND) != 0)

    # 逐列 provenance：記錄來源時刻與權重，且權重加總為 1。
    metadata = patch.metadata
    provenance = metadata["gap_fill"]["row_provenance"]
    assert len(provenance) == gap_len
    for row in provenance:
        assert row["method"] == GAP_FILL_METHOD_ID
        assert row["left_weight"] + row["right_weight"] == pytest.approx(1.0, abs=1.0e-9)

    # 準確度：對合成潮流訊號，混合值應遠比「單純複製前一個曆法日」準確。
    predicted = np.asarray(patch.hvel)[:, 0, 0, 0]
    truth = signal[gap_start : gap_start + gap_len]
    rmse_blend = float(np.sqrt(np.mean((predicted - truth) ** 2)))
    naive_24h = signal[gap_start - 24 : gap_start - 24 + gap_len]
    rmse_naive = float(np.sqrt(np.mean((naive_24h - truth) ** 2)))
    assert rmse_blend < rmse_naive

    # 缺口邊界無跳變：第一與最後一個補值列，應與相鄰觀測列的差距遠小於訊號整體振幅。
    first_boundary_jump = abs(predicted[0] - signal[gap_start - 1])
    last_boundary_jump = abs(predicted[-1] - signal[gap_start + gap_len])
    assert first_boundary_jump < 0.5
    assert last_boundary_jump < 0.5

    # zcor 垂向嚴格遞增、wetdry 沒有被內插成非法中間值。
    zcor = np.asarray(patch.zcor)
    assert np.all(np.diff(zcor, axis=2) > 0.0)
    wetdry = np.asarray(patch.wetdry_elem)
    finite_wetdry = wetdry[np.isfinite(wetdry)]
    assert np.all(np.isin(finite_wetdry, [0.0, 1.0]))

    # 既有讀取端：inventory_forcing_gaps 補值後應回報「無未登錄缺口」。
    month_dirs = {month_id: str(domain_root / "months" / month_id) for month_id in observed_months}
    patches = {
        month_id: _load_patch_month(output_root, "domainA", month_id) for month_id in report.month_ids
    }
    inventory = inventory_forcing_gaps(
        month_dirs,
        expected_start_utc_ns=int(times[0]),
        expected_end_utc_ns=int(times[-1]),
        patches=patches,
        flow_id="domainA",
    )
    assert inventory.is_run_ready
    assert inventory.unregistered_hour_count == 0

    # 完整正式讀取路徑：write_gap_fill_root_index + _ReconstructionPatchStore。
    write_gap_fill_root_index(output_root, ["domainA"])
    mesh = NativeMesh.from_directory(domain_root / "grid", projection=DomainProjection(121.0, 25.0))
    store = forcing_window_module._ReconstructionPatchStore(
        root=output_root, flow_domain_id="domainA", mesh=mesh
    )
    from_store = store.load_month(report.month_ids[0])
    assert from_store is not None
    assert from_store.valid
    np.testing.assert_array_equal(np.asarray(from_store.time_utc_ns), np.asarray(patch.time_utc_ns))


def test_cross_month_gap_splits_into_two_month_patches(tmp_path: Path) -> None:
    """跨月份缺口：輸出應同時在兩個月份目錄各自發布合法 patch。"""

    domain_root = tmp_path / "ocm" / "domainA"
    _write_ocm_grid(domain_root)
    start = int(datetime(2024, 1, 25, tzinfo=UTC).timestamp() * 1_000_000_000)
    hours = 24 * 15
    times = start + np.arange(hours, dtype=np.int64) * HOURLY_NS
    signal = _tidal_signal(hours)
    gap_start, gap_len = 100, 80
    keep = np.ones(hours, dtype=bool)
    keep[gap_start : gap_start + gap_len] = False
    _write_ocm_months(domain_root, times[keep], signal[keep])

    output_root = tmp_path / "gap_fill_out"
    report = build_gap_fill_patches(
        domain_root, output_root,
        start_utc_ns=int(times[0]), end_utc_ns=int(times[-1]), config=GapFillConfig(),
    )
    assert report.month_ids == ("202401", "202402")
    assert report.row_count == gap_len
    total_rows = 0
    for month_id in report.month_ids:
        patch = _load_patch_month(output_root, "domainA", month_id)
        assert patch.valid
        total_rows += patch.time_utc_ns.size
    assert total_rows == gap_len


def test_single_sided_gap_at_domain_start_flags_single_sided(tmp_path: Path) -> None:
    """domain 起點的 edge gap 沒有左側支援，應只用右側並標記 QUALITY_SINGLE_SIDED_BLEND。"""

    domain_root = tmp_path / "ocm" / "domainA"
    _write_ocm_grid(domain_root)
    hours = 24 * 10
    times = _times(hours)
    signal = _tidal_signal(hours)
    keep = np.ones(hours, dtype=bool)
    keep[0:40] = False
    _write_ocm_months(domain_root, times[keep], signal[keep])

    output_root = tmp_path / "gap_fill_out"
    report = build_gap_fill_patches(
        domain_root, output_root,
        start_utc_ns=int(times[0]), end_utc_ns=int(times[-1]), config=GapFillConfig(),
    )
    assert report.row_count == 40
    assert report.unfilled_time_utc_ns == ()
    assert len(report.single_sided_gap_ids) == 1

    patch = _load_patch_month(output_root, "domainA", report.month_ids[0])
    quality = np.asarray(patch.quality_flags)
    assert np.all((quality & QUALITY_SINGLE_SIDED_BLEND) != 0)
    provenance = patch.metadata["gap_fill"]["row_provenance"]
    for row in provenance:
        assert row["left_source_time_utc_ns"] is None
        assert row["right_source_time_utc_ns"] is not None
        assert row["left_weight"] == 0.0
        assert row["right_weight"] == 1.0


def test_short_and_state_space_length_gaps_use_existing_method(tmp_path: Path) -> None:
    """1 小時與 30 小時缺口應優先交給既有核心方法，而不是相鄰日混合。"""

    domain_root = tmp_path / "ocm" / "domainA"
    _write_ocm_grid(domain_root)
    hours = 24 * 10
    times = _times(hours)
    signal = _tidal_signal(hours)
    keep = np.ones(hours, dtype=bool)
    keep[50:51] = False
    keep[100:130] = False
    _write_ocm_months(domain_root, times[keep], signal[keep])

    output_root = tmp_path / "gap_fill_out"
    report = build_gap_fill_patches(
        domain_root, output_root,
        start_utc_ns=int(times[0]), end_utc_ns=int(times[-1]), config=GapFillConfig(),
    )
    assert report.blend_method_gap_ids == ()
    assert report.fallback_to_blend_gap_ids == ()
    assert len(report.existing_method_gap_ids) == 2
    assert report.row_count == 31

    patch = _load_patch_month(output_root, "domainA", report.month_ids[0])
    origin = np.asarray(patch.origin_code)
    quality = np.asarray(patch.quality_flags)
    assert int(ORIGIN_RECONSTRUCTED_SHORT) in origin.tolist()
    assert int(ORIGIN_RECONSTRUCTED_STATE_SPACE) in origin.tolist()
    # 既有方法的列不應帶有本模組專屬的相鄰日混合旗標。
    assert not np.any((quality & QUALITY_LUNAR_DAY_BLEND) != 0)


def test_force_lunar_day_blend_only_bypasses_existing_method(tmp_path: Path) -> None:
    """``force_lunar_day_blend_only`` 開啟時，連 1 小時缺口都改用相鄰日混合。"""

    domain_root = tmp_path / "ocm" / "domainA"
    _write_ocm_grid(domain_root)
    hours = 24 * 10
    times = _times(hours)
    signal = _tidal_signal(hours)
    keep = np.ones(hours, dtype=bool)
    keep[50:51] = False
    _write_ocm_months(domain_root, times[keep], signal[keep])

    output_root = tmp_path / "gap_fill_out"
    config = GapFillConfig(force_lunar_day_blend_only=True)
    report = build_gap_fill_patches(
        domain_root, output_root, start_utc_ns=int(times[0]), end_utc_ns=int(times[-1]), config=config
    )
    assert report.existing_method_gap_ids == ()
    assert len(report.blend_method_gap_ids) == 1


def test_fallback_to_blend_when_existing_method_lacks_two_sided_support(tmp_path: Path) -> None:
    """核心方法因雙側支援不足失敗時，應自動退回相鄰日混合而不是讓整個 domain 停止。"""

    domain_root = tmp_path / "ocm" / "domainA"
    _write_ocm_grid(domain_root)
    hours = 24 * 10
    times = _times(hours)
    signal = _tidal_signal(hours)
    keep = np.ones(hours, dtype=bool)
    # 30 小時缺口長度落在既有 23-49 契約內，但把 gap 前緣的觀測歷史削到只剩 2 筆，
    # 讓 ``min_state_history``（預設 8）無法滿足，核心方法必須 fail 才能觸發 fallback。
    keep[0:38] = False
    keep[40:70] = False
    _write_ocm_months(domain_root, times[keep], signal[keep])

    output_root = tmp_path / "gap_fill_out"
    small_history_config = GapFillConfig(
        reconstruction_config=ReconstructionConfig(min_state_history=8, context_hours=96)
    )
    report = build_gap_fill_patches(
        domain_root,
        output_root,
        start_utc_ns=int(times[0]),
        end_utc_ns=int(times[-1]),
        config=small_history_config,
    )
    assert "domainA_" + str(int(times[40])) + "_" + str(int(times[69])) in report.fallback_to_blend_gap_ids
    assert report.existing_method_gap_ids == ()


def test_all_gap_lengths_produce_no_unregistered_gap_after_patching(tmp_path: Path) -> None:
    """混合長度（1／30／120 小時）全部補完後，正式盤點應完全沒有未登錄缺口。"""

    domain_root = tmp_path / "ocm" / "domainA"
    _write_ocm_grid(domain_root)
    hours = 24 * 30
    times = _times(hours)
    signal = _tidal_signal(hours)
    keep = np.ones(hours, dtype=bool)
    keep[50:51] = False
    keep[100:130] = False
    keep[300:420] = False
    observed_months = _write_ocm_months(domain_root, times[keep], signal[keep])

    output_root = tmp_path / "gap_fill_out"
    report = build_gap_fill_patches(
        domain_root, output_root,
        start_utc_ns=int(times[0]), end_utc_ns=int(times[-1]), config=GapFillConfig(),
    )
    assert report.unfilled_time_utc_ns == ()

    month_dirs = {month_id: str(domain_root / "months" / month_id) for month_id in observed_months}
    patches = {
        month_id: _load_patch_month(output_root, "domainA", month_id) for month_id in report.month_ids
    }
    inventory = inventory_forcing_gaps(
        month_dirs,
        expected_start_utc_ns=int(times[0]),
        expected_end_utc_ns=int(times[-1]),
        patches=patches,
        flow_id="domainA",
    )
    assert inventory.is_run_ready


def test_unfillable_interior_rows_are_left_out_not_fabricated(tmp_path: Path) -> None:
    """把位移搜尋上限縮到極小時，缺口中段找不到來源的列必須被剔除，不得補零或最近值。"""

    domain_root = tmp_path / "ocm" / "domainA"
    _write_ocm_grid(domain_root)
    hours = 24 * 20
    times = _times(hours)
    signal = _tidal_signal(hours)
    gap_start, gap_len = 200, 120
    keep = np.ones(hours, dtype=bool)
    keep[gap_start : gap_start + gap_len] = False
    _write_ocm_months(domain_root, times[keep], signal[keep])

    output_root = tmp_path / "gap_fill_out"
    # max_shift_multiples=1：只有落在缺口前後 25 小時內的列才找得到來源，中段
    # （距任一端超過 25 小時）注定無單側支援，必須被剔除而不是勉強發布。
    config = GapFillConfig(max_shift_multiples=1)
    report = build_gap_fill_patches(
        domain_root, output_root,
        start_utc_ns=int(times[0]), end_utc_ns=int(times[-1]), config=config,
    )
    assert 0 < report.row_count < gap_len
    assert report.unfilled_time_utc_ns
    filled_times = set()
    for month_id in report.month_ids:
        patch = _load_patch_month(output_root, "domainA", month_id)
        filled_times.update(int(t) for t in np.asarray(patch.time_utc_ns))
    unfilled = set(report.unfilled_time_utc_ns)
    gap_times = {int(t) for t in times[gap_start : gap_start + gap_len]}
    assert filled_times | unfilled == gap_times
    assert filled_times.isdisjoint(unfilled)
    # 中段（缺口正中央）必須是未填補，而不是被冒充成一列合法資料。
    center_time = int(times[gap_start + gap_len // 2])
    assert center_time in unfilled
    assert center_time not in filled_times


# ── NWW3：完整替代月份 ────────────────────────────────────────────────────


def _write_nww3_grid(domain_root: Path, lon: np.ndarray, lat: np.ndarray) -> None:
    grid = domain_root / "grid"
    grid.mkdir(parents=True)
    np.save(grid / "lon.npy", lon)
    np.save(grid / "lat.npy", lat)


def _write_nww3_months(
    domain_root: Path,
    times: np.ndarray,
    hs: np.ndarray,
    fp: np.ndarray,
    direction: np.ndarray,
    lat_size: int,
    lon_size: int,
) -> tuple[str, ...]:
    by_month: dict[str, list[int]] = {}
    for index, time_ns in enumerate(times):
        by_month.setdefault(_month_id(int(time_ns)), []).append(index)
    for month_id, indices in by_month.items():
        idx = np.asarray(indices, dtype=np.int64)
        month_dir = domain_root / "months" / month_id
        month_dir.mkdir(parents=True)
        count = int(idx.size)
        np.save(month_dir / "time_utc_ns.npy", times[idx])
        np.save(
            month_dir / "significant_wave_height.npy",
            np.broadcast_to(hs[idx][:, None, None], (count, lat_size, lon_size)).copy(),
        )
        np.save(
            month_dir / "peak_frequency.npy",
            np.broadcast_to(fp[idx][:, None, None], (count, lat_size, lon_size)).copy(),
        )
        np.save(
            month_dir / "peak_direction_raw_deg.npy",
            np.broadcast_to(direction[idx][:, None, None], (count, lat_size, lon_size)).copy(),
        )
        np.save(month_dir / "valid_mask_wave.npy", np.ones((count, lat_size, lon_size), dtype=bool))
        np.save(month_dir / "qc_flags.npy", np.zeros((count, lat_size, lon_size), dtype=np.uint16))
    return tuple(sorted(by_month))


def test_wave_gap_fill_wraps_direction_and_keeps_untouched_months_symlinked(tmp_path: Path) -> None:
    """NWW3 缺時刻補值：波向跨 0/360° 正確混合、qc_flags 保持乾淨、未缺月份用符號連結。"""

    domain_root = tmp_path / "nww3" / "domainA"
    lon = np.asarray([120.0, 120.5, 121.0])
    lat = np.asarray([24.0, 24.5])
    _write_nww3_grid(domain_root, lon, lat)

    hours = 24 * 40  # 跨兩個月份，確保有一個月份完全不缺
    times = _times(hours)
    index = np.arange(hours, dtype=np.float64)
    direction = (350.0 + 20.0 * np.sin(2.0 * np.pi * index / 24.0)) % 360.0
    hs = 1.0 + 0.2 * np.sin(2.0 * np.pi * index / M2_PERIOD_HOURS)
    fp = 0.1 + 0.01 * np.cos(2.0 * np.pi * index / 12.0)
    gap_start, gap_len = 100, 30
    keep = np.ones(hours, dtype=bool)
    keep[gap_start : gap_start + gap_len] = False
    _write_nww3_months(domain_root, times[keep], hs[keep], fp[keep], direction[keep], lat.size, lon.size)

    output_root = tmp_path / "gap_fill_out"
    report = build_wave_gap_fill_patches(
        domain_root, output_root,
        start_utc_ns=int(times[0]), end_utc_ns=int(times[-1]), config=GapFillConfig(),
    )
    assert report.blended_row_count == gap_len
    assert report.unfilled_time_utc_ns == ()
    assert len(report.filled_month_ids) == 1
    assert len(report.symlinked_month_ids) >= 1
    for month_id in report.symlinked_month_ids:
        link = output_root / "domainA" / "months" / month_id
        assert link.is_symlink()

    filled_month = report.filled_month_ids[0]
    filled_month_dir = output_root / "domainA" / "months" / filled_month
    month = NWWAnalysisMonth.from_directories(output_root / "domainA" / "grid", filled_month_dir)
    assert int(np.count_nonzero(month.qc_flags)) == 0  # 見模組文件：不得污染 qc_flags

    loaded_times = np.asarray(month.time_utc_ns)
    loaded_dir = np.asarray(month.peak_direction_raw_deg)
    gap_mask = np.isin(loaded_times, times[gap_start : gap_start + gap_len])
    assert int(gap_mask.sum()) == gap_len
    filled_dir = loaded_dir[gap_mask, 0, 0]
    truth_dir = direction[gap_start : gap_start + gap_len]
    circular_error = np.abs((filled_dir - truth_dir + 180.0) % 360.0 - 180.0)
    assert np.max(circular_error) < 10.0

    # 有效遮罩交集：兩側都有效時仍應有效（合成資料本身沒有域外／無效點）。
    assert bool(np.asarray(month.valid_mask_wave)[gap_mask].all())


def test_validate_gap_fill_blocked_prefers_lunar_day_over_naive_and_persistence(tmp_path: Path) -> None:
    """遮蔽交叉驗證：長缺口的相鄰日混合應優於 persistence 與 24 小時 naive 位移。"""

    domain_root = tmp_path / "ocm" / "domainA"
    _write_ocm_grid(domain_root)
    hours = 24 * 60
    times = _times(hours)
    signal = _tidal_signal(hours)
    _write_ocm_months(domain_root, times, signal)

    config = GapFillConfig(
        reconstruction_config=ReconstructionConfig(context_hours=48, min_state_history=8, eof_rank=4)
    )
    report = validate_gap_fill_blocked(
        domain_root, block_lengths_hours=(24, 72, 168), n_blocks=2, seed=7, config=config
    )
    assert report.acceptance["formal_scientific_claim"] is False
    by_length = {item["block_length_hours"]: item for item in report.results}
    for length in (72, 168):
        result = by_length[length]
        assert result["trial_count"] > 0
        assert result["accepted"] is True
        assert result["mean_rmse"]["gap_fill"] < result["mean_rmse"]["shift_24h_naive"]
        assert result["mean_rmse"]["gap_fill"] < result["mean_rmse"]["persistence"]
