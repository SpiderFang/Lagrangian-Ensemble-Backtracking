"""方法 v2（潮汐調和＋有界殘差橋接）與「取代觀測列」宣告的合成資料測試。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from lagrangian_backtracking.forcing import OCMNativeMonth, OCMReconstructionPatchMonth
from lagrangian_backtracking.native_runner.forcing_bridge import (
    ForcingBridgeError,
    NativeForcingBridgeReader,
    _load_raw_month,
    _merge_month_with_patch,
    inventory_forcing_gaps,
)
from lagrangian_backtracking.native_runner.production_wiring import ReconstructionPatchTimeAxis
from lagrangian_backtracking.ocm_reconstruction import (
    SUPERSEDES_OBSERVED_KEY,
    superseded_observed_times_from_metadata,
)
from lagrangian_backtracking.ocm_reconstruction_v2 import (
    CleanMonthsSource,
    GapCluster,
    HarmonicBridgeConfig,
    TIDAL_SPEED_DEG_PER_HOUR,
    apply_envelope,
    bridge_predict,
    find_gap_clusters,
    reconstruct_cluster_field,
    reconstruct_feature_block,
    repair_zcor_order,
)

H = 3_600_000_000_000


def _t(hour: int) -> int:
    return int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000) + hour * H


def _tide(hours: np.ndarray, amp: float, phase: float, name: str = "M2") -> np.ndarray:
    return amp * np.cos(np.deg2rad(TIDAL_SPEED_DEG_PER_HOUR[name]) * hours + phase)


# ── 方法本體 ────────────────────────────────────────────────────────────────


def test_bridge_recovers_tide_plus_slow_residual_and_is_bounded() -> None:
    hours = np.arange(-168, 169, dtype=np.float64)
    gap_mask = (hours >= 0) & (hours < 48)
    truth = (
        _tide(hours, 1.0, 0.3)[:, None]
        + _tide(hours, 0.4, 1.1, "S2")[:, None]
        + _tide(hours, 0.5, 2.0, "K1")[:, None]
        + 0.2 * np.sin(hours / 90.0)[:, None] * np.ones((1, 3))
    )
    cfg = HarmonicBridgeConfig(source_fingerprint="t")
    train = ~gap_mask
    prediction = bridge_predict(hours[train], truth[train], hours[gap_mask], cfg)
    assert prediction.shape == (48, 3)
    assert np.sqrt(np.mean((prediction - truth[gap_mask]) ** 2)) < 0.05
    bounded = apply_envelope(prediction * 10.0, truth[train], cfg)
    assert bounded.max() <= truth[train].max() * 1.0 + 0.05 * np.ptp(truth[train]) + 1e-12


def test_bridge_does_not_diverge_for_long_gaps() -> None:
    """舊方法（二階自迴歸外推）在長缺口會指數發散；v2 為內插，72 小時缺口仍有界。"""

    hours = np.arange(-168, 241, dtype=np.float64)
    gap = (hours >= 0) & (hours < 72)
    rng = np.random.default_rng(1)
    truth = _tide(hours, 1.5, 0.0)[:, None] * np.ones((1, 5)) + 0.05 * rng.standard_normal((hours.size, 5))
    cfg = HarmonicBridgeConfig(source_fingerprint="t")
    block, bad = reconstruct_feature_block(hours[~gap], truth[~gap], hours[gap], cfg, kind="scalar")
    assert bad == 0
    assert np.all(np.isfinite(block))
    assert np.abs(block).max() <= np.abs(truth[~gap]).max() * 1.06


def test_feature_block_keeps_nan_mask_and_counts_inconsistent_features() -> None:
    hours = np.arange(-60, 84, dtype=np.float64)
    gap = (hours >= 0) & (hours < 24)
    values = np.tile(_tide(hours, 1.0, 0.0)[:, None], (1, 4))
    values[:, 1] = np.nan  # 全部為 NaN：海床以下
    values[5, 2] = np.nan  # 只有一列為 NaN：局部有效，輸出 NaN 並計數
    block, bad = reconstruct_feature_block(hours[~gap], values[~gap], hours[gap], HarmonicBridgeConfig(source_fingerprint="t"), kind="scalar")
    assert np.isnan(block[:, 1]).all() and np.isnan(block[:, 2]).all()
    assert np.isfinite(block[:, [0, 3]]).all()
    assert bad == 1


def test_hvel_speed_never_exceeds_training_maximum() -> None:
    hours = np.arange(-100, 124, dtype=np.float64)
    gap = (hours >= 0) & (hours < 24)
    u = _tide(hours, 2.0, 0.0)
    v = _tide(hours, 1.0, 1.0)
    values = np.stack((u, v), axis=1)  # (time, 2) = 1 個節點層的 u,v
    cfg = HarmonicBridgeConfig(source_fingerprint="t", speed_bound_factor=1.0, envelope_margin_fraction=0.5)
    block, _ = reconstruct_feature_block(hours[~gap], values[~gap], hours[gap], cfg, kind="hvel")
    limit = np.hypot(values[~gap, 0], values[~gap, 1]).max()
    assert np.hypot(block[:, 0], block[:, 1]).max() <= limit + 1e-9


def test_wetdry_is_binary() -> None:
    hours = np.arange(-80, 104, dtype=np.float64)
    gap = (hours >= 0) & (hours < 24)
    wet = (np.cos(np.deg2rad(28.9841042) * hours) > 0.3).astype(np.float64)[:, None]
    block, _ = reconstruct_feature_block(hours[~gap], wet[~gap], hours[gap], HarmonicBridgeConfig(source_fingerprint="t"), kind="wetdry")
    assert set(np.unique(block)).issubset({0.0, 1.0})


def test_repair_zcor_order_only_touches_inversions() -> None:
    z = np.array([[[-10.0, -6.0, -2.0, 0.0], [-10.0, -3.0, -4.0, 0.0]]])
    changed = repair_zcor_order(z)
    assert changed >= 1
    assert np.all(np.diff(z, axis=2) > 0)
    assert z[0, 0].tolist() == [-10.0, -6.0, -2.0, 0.0]


def test_find_gap_clusters_merges_replay_and_true_hole() -> None:
    times = np.asarray([_t(h) for h in list(range(0, 10)) + list(range(30, 40))], dtype=np.int64)
    replay = np.asarray([_t(h) for h in range(5, 10)], dtype=np.int64)
    # 時間軸已排除 replay：5..9 已不在 times（0..9 其實含 5..9），所以先移除
    clean = np.asarray([t for t in times if int(t) not in set(int(v) for v in replay)], dtype=np.int64)
    clusters = find_gap_clusters(clean, replay, flow_id="f")
    assert len(clusters) == 1
    c = clusters[0]
    assert c.missing_times_ns[0] == _t(5) and c.missing_times_ns[-1] == _t(29)
    assert c.superseded_ns.size == 5


def _write_month(root: Path, month_id: str, hours: list[int], nodes: int = 3, layers: int = 2, faces: int = 1, fill: float = 0.0) -> Path:
    d = root / month_id
    d.mkdir(parents=True, exist_ok=True)
    n = len(hours)
    np.save(d / "time_utc_ns.npy", np.asarray([_t(h) for h in hours], dtype=np.int64))
    base = np.asarray(hours, dtype=np.float64)[:, None, None] + np.zeros((n, nodes, layers))
    np.save(d / "hvel.npy", np.stack((base + fill, -base - fill), axis=-1).astype(np.float32))
    np.save(d / "vertical_velocity.npy", (base * 0.01).astype(np.float32))
    np.save(d / "zcor.npy", np.broadcast_to(-5.0 * np.arange(layers), (n, nodes, layers)).astype(np.float32).copy())
    np.save(d / "elev.npy", np.zeros((n, nodes), dtype=np.float32))
    np.save(d / "wetdry_elem.npy", np.zeros((n, faces), dtype=np.float32))
    return d


def _patch(month_id: str, hours: list[int], superseded_hours: list[int], nodes: int = 3, layers: int = 2, faces: int = 1) -> OCMReconstructionPatchMonth:
    n = len(hours)
    meta = {"method": "ocm_harmonic_residual_bridge_bounded_v2"}
    if superseded_hours:
        meta[SUPERSEDES_OBSERVED_KEY] = {"policy": "test", "time_utc_ns": [_t(h) for h in superseded_hours]}
    return OCMReconstructionPatchMonth(
        month_id=month_id,
        time_utc_ns=np.asarray([_t(h) for h in hours], dtype=np.int64),
        hvel=np.full((n, nodes, layers, 2), 7.0, dtype=np.float32),
        vertical_velocity=np.zeros((n, nodes, layers), dtype=np.float32),
        zcor=np.broadcast_to(-5.0 * np.arange(layers), (n, nodes, layers)).astype(np.float32).copy(),
        elev=np.zeros((n, nodes), dtype=np.float32),
        wetdry_elem=np.zeros((n, faces), dtype=np.float32),
        diffusivity=np.full((n, nodes, layers), np.nan, dtype=np.float32),
        origin_code=np.full(n, 2, dtype=np.uint8),
        quality_flags=np.full(n, 2, dtype=np.uint16),
        metadata=meta,
    )


# ── 取代觀測列（重播列）的讀取契約 ───────────────────────────────────────────


def test_superseded_times_metadata_roundtrip() -> None:
    assert superseded_observed_times_from_metadata(None).size == 0
    assert superseded_observed_times_from_metadata({}).size == 0
    got = superseded_observed_times_from_metadata({SUPERSEDES_OBSERVED_KEY: {"time_utc_ns": [5, 3, 5]}})
    assert got.tolist() == [3, 5]


def test_overlap_without_declaration_is_rejected(tmp_path: Path) -> None:
    d = _write_month(tmp_path, "202401", [0, 1, 2, 3, 4])
    raw = _load_raw_month(d, load_diffusivity=False)
    with pytest.raises(ForcingBridgeError, match="重疊"):
        _merge_month_with_patch("202401", raw, _patch("202401", [2], []), load_diffusivity=False)


def test_declared_overlap_replaces_observed_row_and_uses_patch_values(tmp_path: Path) -> None:
    d = _write_month(tmp_path, "202401", [0, 1, 2, 3, 4], fill=1.0)
    raw = _load_raw_month(d, load_diffusivity=False)
    merged = _merge_month_with_patch("202401", raw, _patch("202401", [2, 3], [2, 3]), load_diffusivity=False)
    assert merged.time_utc_ns.tolist() == [_t(h) for h in range(5)]
    kinds = merged.source_kind.tolist()
    assert kinds == [0, 0, 1, 1, 0]
    # 未被取代的觀測列仍指向原陣列的正確列號（第 4 小時是原陣列第 4 列）
    assert merged.source_index.tolist() == [0, 1, 0, 1, 4]


def test_declaration_must_be_subset_of_patch_times(tmp_path: Path) -> None:
    d = _write_month(tmp_path, "202401", [0, 1, 2, 3, 4])
    raw = _load_raw_month(d, load_diffusivity=False)
    with pytest.raises(ForcingBridgeError, match="宣告"):
        _merge_month_with_patch("202401", raw, _patch("202401", [2], [2, 3]), load_diffusivity=False)


def test_inventory_counts_superseded_rows_as_reconstructed(tmp_path: Path) -> None:
    d = _write_month(tmp_path, "202401", [0, 1, 2, 3, 4])
    axis = ReconstructionPatchTimeAxis(
        month_id="202401",
        time_utc_ns=np.asarray([_t(2), _t(3)], dtype=np.int64),
        superseded_observed_utc_ns=np.asarray([_t(2), _t(3)], dtype=np.int64),
    )
    report = inventory_forcing_gaps(
        {"202401": d}, expected_start_utc_ns=_t(0), expected_end_utc_ns=_t(4), patches={"202401": axis}
    )
    assert report.is_run_ready
    assert report.observed_hour_count == 3
    assert report.reconstructed_hour_count == 2
    bad_axis = ReconstructionPatchTimeAxis(month_id="202401", time_utc_ns=np.asarray([_t(2)], dtype=np.int64))
    with pytest.raises(ForcingBridgeError, match="重疊"):
        inventory_forcing_gaps({"202401": d}, expected_start_utc_ns=_t(0), expected_end_utc_ns=_t(4), patches={"202401": bad_axis})


def test_reader_returns_patch_values_for_superseded_hour(tmp_path: Path) -> None:
    d = _write_month(tmp_path, "202401", [0, 1, 2, 3, 4], fill=1.0)
    reader = NativeForcingBridgeReader(
        {"202401": d}, ibf=1, patches={"202401": _patch("202401", [2], [2])}, load_diffusivity=False,
        max_reconstructed_speed_mps=None,
    )
    pair = reader.resolve(_t(2) + H // 2)
    assert {pair.time_origin1, pair.time_origin2} == {"reconstructed", "observed"}


# ── 來源物件 ────────────────────────────────────────────────────────────────


def test_clean_months_source_excludes_replay_hours_and_reconstructs(tmp_path: Path) -> None:
    hours = list(range(0, 120))
    d = tmp_path / "flow" / "months"
    month = _write_month(d, "202401", hours, nodes=2, layers=1)
    # 以潮汐信號覆寫 hvel，讓重建有意義
    t = np.asarray(hours, dtype=np.float64)
    u = _tide(t, 1.0, 0.0)[:, None, None] * np.ones((1, 2, 1))
    np.save(month / "hvel.npy", np.stack((u, 0.5 * u), axis=-1).astype(np.float32))
    bad = np.asarray([_t(h) for h in range(60, 72)], dtype=np.int64)
    src = CleanMonthsSource(tmp_path / "flow", exclude_utc_ns=bad)
    assert src.times_utc_ns.size == 108
    clusters = find_gap_clusters(src.times_utc_ns, bad, flow_id="flow")
    assert len(clusters) == 1 and clusters[0].missing_times_ns.size == 12
    out = np.empty((12, 4))

    def sink(a: int, b: int, block: np.ndarray) -> None:
        out[:, a:b] = block

    cfg = HarmonicBridgeConfig(source_fingerprint="t", context_hours=48, min_context_hours=12)
    reconstruct_cluster_field(src, clusters[0], "hvel", cfg, sink=sink)
    truth = np.stack((u[60:72], 0.5 * u[60:72]), axis=-1).reshape(12, -1)
    assert np.sqrt(np.mean((out - truth) ** 2)) < 0.05
