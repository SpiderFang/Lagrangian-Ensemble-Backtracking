"""OCM sparse reconstruction 的 synthetic 契約、約束與 immutable 發布測試。

所有 fixture 都是小型完整逐時序列，不代表真實海洋結果；測試重點是確認缺口重建
遵守雙側支援、無最近值／零值／跳時、deterministic、chunked access 與 provenance
hash。長缺口 validation 只衡量 synthetic blocked-mask，不能替代實際 OCM 的
blocked cross-validation 或 Lagrangian 敏感度分析。
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import lagrangian_backtracking.ocm_reconstruction as reconstruction_module
from lagrangian_backtracking.ocm_reconstruction import (
    HOURLY_NS,
    ORIGIN_RECONSTRUCTED_SHORT,
    ORIGIN_RECONSTRUCTED_STATE_SPACE,
    ArraySequenceSource,
    NpyDomainSource,
    ReconstructionConfig,
    ReconstructionError,
    build_reconstruction_patch,
    validate_blocked_masks,
    validate_reconstruction_patch,
)


def _write_npy_domain_month(
    domain_root: Path,
    month_id: str,
    times: np.ndarray,
    elev_values: np.ndarray,
) -> None:
    """建立最小 schema 3 月份，供跨月 halo 的 prefer-last 索引測試。"""

    month = domain_root / "months" / month_id
    month.mkdir(parents=True)
    count = int(times.size)
    np.save(month / "time_utc_ns.npy", np.asarray(times, dtype=np.int64))
    np.save(month / "hvel.npy", np.zeros((count, 3, 2, 2), dtype=np.float32))
    np.save(month / "vertical_velocity.npy", np.zeros((count, 3, 2), dtype=np.float32))
    np.save(
        month / "zcor.npy",
        np.broadcast_to(np.asarray([-5.0, 0.0], dtype=np.float32), (count, 3, 2)).copy(),
    )
    np.save(month / "elev.npy", np.broadcast_to(elev_values[:, None], (count, 3)).copy())
    np.save(month / "wetdry_elem.npy", np.zeros((count, 1), dtype=np.int8))
    np.save(month / "diffusivity.npy", np.ones((count, 3, 2), dtype=np.float32))


def _write_npy_domain_grid(domain_root: Path) -> None:
    """建立 fingerprint 所需的最小原生網格，不建立站點投影座標。"""

    grid = domain_root / "grid"
    grid.mkdir(parents=True)
    np.save(grid / "source_lon.npy", np.asarray([120.0, 120.1, 120.0]))
    np.save(grid / "source_lat.npy", np.asarray([24.0, 24.0, 24.1]))
    np.save(grid / "source_depth_m.npy", np.asarray([5.0, 5.0, 5.0]))
    np.save(grid / "source_node_bottom_index.npy", np.zeros(3, dtype=np.int64))
    np.save(grid / "source_face_nodes_local.npy", np.asarray([[0, 1, 2, -1]], dtype=np.int64))
    np.save(grid / "source_face_node_count.npy", np.asarray([3], dtype=np.int64))
    np.save(grid / "source_face_global_index.npy", np.asarray([10], dtype=np.int64))


def _source(hours: int = 320) -> tuple[ArraySequenceSource, dict[str, np.ndarray]]:
    """建立可重現的潮汐 synthetic source，連續場 shape 接近 OCM node/layer 介面。"""

    start = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
    times = start + np.arange(hours, dtype=np.int64) * HOURLY_NS
    index = np.arange(hours, dtype=np.float64)
    signal = np.sin(2.0 * np.pi * index / 12.4206) + 0.4 * np.cos(2.0 * np.pi * index / 23.9345)
    hvel = np.empty((hours, 2, 3, 2), dtype=np.float32)
    hvel[..., 0] = signal[:, None, None]
    hvel[..., 1] = (0.2 * np.cos(2.0 * np.pi * index / 12.0))[:, None, None]
    vertical = (0.01 * signal)[:, None, None] * np.ones((hours, 2, 3), dtype=np.float32)
    zcor = np.broadcast_to(np.asarray([-10.0, -5.0, 0.0], dtype=np.float32), (hours, 2, 3)).copy()
    elev = (0.1 * signal)[:, None] * np.ones((hours, 2), dtype=np.float32)
    wetdry = np.zeros((hours, 4), dtype=np.int8)
    diffusivity = (0.01 + 0.002 * signal)[:, None, None] * np.ones((hours, 2, 3), dtype=np.float32)
    fields = {
        "hvel": hvel,
        "vertical_velocity": vertical.astype(np.float32),
        "zcor": zcor,
        "elev": elev.astype(np.float32),
        "wetdry_elem": wetdry,
        "diffusivity": diffusivity.astype(np.float32),
    }
    return ArraySequenceSource(times, fields), fields


def _config() -> ReconstructionConfig:
    """縮小 feature block 以讓測試同時驗證 chunked access。"""

    return ReconstructionConfig(context_hours=48, min_state_history=8, feature_block_size=4, eof_rank=4)


def test_one_hour_gap_uses_exact_component_linear_and_origin_code(tmp_path: Path) -> None:
    """單小時缺口必須等於兩端 component 線性值，而非 persistence。"""

    source, fields = _source()
    missing = source.times_utc_ns[100:101]
    masked = source.masked(missing)
    build_reconstruction_patch(masked, tmp_path, flow_id="A", config=_config())
    month = tmp_path / "A" / "months" / "202401"
    hvel = np.load(month / "hvel.npy")
    expected = (fields["hvel"][99:100] + fields["hvel"][101:102]) / 2.0
    np.testing.assert_allclose(hvel, expected, rtol=0.0, atol=0.0)
    assert np.load(month / "origin_code.npy").tolist() == [int(ORIGIN_RECONSTRUCTED_SHORT)]
    assert np.load(month / "quality_flags.npy").tolist() == [1]


def test_long_gap_state_space_beats_persistence_and_endpoint_linear(tmp_path: Path) -> None:
    """23 小時潮汐缺口的 EOF/harmonic/state prediction 應優於兩個簡單 baseline。"""

    source, fields = _source()
    missing = source.times_utc_ns[120:143]
    build_reconstruction_patch(source.masked(missing), tmp_path, flow_id="B", config=_config())
    predicted = np.load(tmp_path / "B" / "months" / "202401" / "hvel.npy")
    truth = fields["hvel"][120:143]
    persistence = np.repeat(fields["hvel"][119:120], 23, axis=0)
    alpha = np.arange(1, 24, dtype=np.float64)[:, None, None, None] / 24.0
    endpoint = fields["hvel"][119:120] + alpha * (fields["hvel"][143:144] - fields["hvel"][119:120])
    prediction_rmse = float(np.sqrt(np.mean(np.square(predicted - truth))))
    persistence_rmse = float(np.sqrt(np.mean(np.square(persistence - truth))))
    endpoint_rmse = float(np.sqrt(np.mean(np.square(endpoint - truth))))
    origin = np.load(tmp_path / "B" / "months" / "202401" / "origin_code.npy")
    assert int(origin[0]) == int(ORIGIN_RECONSTRUCTED_STATE_SPACE)
    assert prediction_rmse < persistence_rmse
    assert prediction_rmse < endpoint_rmse


def test_long_gap_is_deterministic_and_patch_manifest_is_hash_checked(tmp_path: Path) -> None:
    """同一來源／設定應產生相同 bytes；array tamper 必須被 validator 拒絕。"""

    source, _ = _source()
    missing = source.times_utc_ns[140:164]
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_reconstruction_patch(source.masked(missing), first, flow_id="C", config=_config())
    build_reconstruction_patch(source.masked(missing), second, flow_id="C", config=_config())
    first_file = first / "C" / "months" / "202401" / "hvel.npy"
    second_file = second / "C" / "months" / "202401" / "hvel.npy"
    first_hash = hashlib.sha256(first_file.read_bytes()).hexdigest()
    second_hash = hashlib.sha256(second_file.read_bytes()).hexdigest()
    assert first_hash == second_hash
    assert validate_reconstruction_patch(first / "C")["flow_id"] == "C"
    with first_file.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ReconstructionError, match="checksum"):
        validate_reconstruction_patch(first / "C")


def test_patch_rejects_existing_destination_and_no_two_sided_support(tmp_path: Path) -> None:
    """immutable 目的地與 edge gap 均 fail closed，不改寫 source 或產生單側預測。"""

    source, _ = _source()
    missing = source.times_utc_ns[80:81]
    output = tmp_path / "output"
    build_reconstruction_patch(source.masked(missing), output, flow_id="D", config=_config())
    with pytest.raises(ReconstructionError, match="已存在"):
        build_reconstruction_patch(source.masked(missing), output, flow_id="D", config=_config())
    edge_missing = source.times_utc_ns[0:1]
    with pytest.raises(ReconstructionError, match="雙側 exact support"):
        build_reconstruction_patch(
            source.masked(edge_missing),
            tmp_path / "edge",
            flow_id="D",
            config=_config(),
            expected_start_ns=int(edge_missing[0]),
        )


def test_blocked_validation_reports_all_requested_lengths_and_callback_boundary() -> None:
    """blocked-mask API 回報 Eulerian 指標，Lagrangian callback 只保留 caller payload。"""

    source, _ = _source()
    callback_seen: dict[str, object] = {}

    def callback(payload: dict[str, object]) -> dict[str, object]:
        callback_seen.update(payload)
        return {"schema": "caller", "accepted": False}

    payload = validate_blocked_masks(
        source,
        block_lengths_hours=(1, 23, 24, 25, 49),
        config=_config(),
        lagrangian_callback=callback,
    )
    assert payload["block_lengths_hours"] == [1, 23, 24, 25, 49]
    assert len(payload["results"]) == 5
    assert payload["acceptance"]["formal_scientific_claim"] is False
    assert payload["lagrangian_validation"]["status"] == "caller_supplied"
    assert callback_seen["schema_version"] == "ocm_reconstruction_lagrangian_callback_input_v1"


def test_chunked_source_never_reads_more_than_configured_feature_block(tmp_path: Path) -> None:
    """核心以 flat feature block 讀取，不能因 constraint gate 退化成整場一次讀取。"""

    source, _ = _source()
    recorded: list[int] = []
    masked = source.masked(source.times_utc_ns[150:174])
    original = masked.read_times

    def recording_read(*args: object, **kwargs: object) -> np.ndarray:
        start = kwargs.get("flat_start")
        stop = kwargs.get("flat_stop")
        if start is not None and stop is not None:
            recorded.append(int(stop) - int(start))
        return original(*args, **kwargs)

    masked.read_times = recording_read  # type: ignore[method-assign]
    build_reconstruction_patch(masked, tmp_path, flow_id="E", config=_config())
    assert recorded
    assert max(recorded) <= _config().feature_block_size


def test_one_hour_gap_reads_each_field_once_without_small_nfs_blocks(tmp_path: Path) -> None:
    """短缺口應一次讀兩端完整欄位，避免正式 NFS 月檔出現數千次小讀取。"""

    source, _ = _source()
    masked = source.masked(source.times_utc_ns[100:101])
    calls: list[tuple[str, object, object]] = []
    original = masked.read_times

    def recording_read(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(
            (
                str(args[1]),
                kwargs.get("flat_start"),
                kwargs.get("flat_stop"),
            )
        )
        return original(*args, **kwargs)

    masked.read_times = recording_read  # type: ignore[method-assign]
    build_reconstruction_patch(masked, tmp_path, flow_id="short", config=_config())
    assert len(calls) == 6
    assert {name for name, _, _ in calls} == {
        "hvel",
        "vertical_velocity",
        "zcor",
        "elev",
        "diffusivity",
        "wetdry_elem",
    }
    assert all(
        (start is None and stop is None)
        or (name == "wetdry_elem" and start == 0 and stop == 4)
        for name, start, stop in calls
    )


def test_wetdry_mixed_support_is_nan_and_flagged(tmp_path: Path) -> None:
    """兩端一乾一濕不得創造海域，patch 以 NaN 與 quality flag 保存限制。"""

    source, fields = _source()
    fields["wetdry_elem"][99, 0] = 1
    fields["wetdry_elem"][101, 0] = 0
    source = ArraySequenceSource(source.times_utc_ns, fields)
    missing = source.times_utc_ns[100:101]
    build_reconstruction_patch(source.masked(missing), tmp_path, flow_id="F", config=_config())
    month = tmp_path / "F" / "months" / "202401"
    wetdry = np.load(month / "wetdry_elem.npy")
    assert np.isnan(wetdry[0, 0])
    assert int(np.load(month / "quality_flags.npy")[0]) & 4


def test_npy_domain_source_accepts_month_halo_and_prefers_later_month(tmp_path: Path) -> None:
    """跨月重疊 UTC 是 schema 3 halo；canonical index 必須 stable prefer-last。"""

    domain = tmp_path / "flow"
    _write_npy_domain_grid(domain)
    start = np.int64(1_700_000_000_000_000_000)
    _write_npy_domain_month(
        domain,
        "202401",
        start + np.asarray([0, 1], dtype=np.int64) * HOURLY_NS,
        np.asarray([10.0, 11.0], dtype=np.float32),
    )
    _write_npy_domain_month(
        domain,
        "202402",
        start + np.asarray([1, 2], dtype=np.int64) * HOURLY_NS,
        np.asarray([21.0, 22.0], dtype=np.float32),
    )
    source = NpyDomainSource(domain)
    assert source.times_utc_ns.tolist() == [
        int(start),
        int(start + HOURLY_NS),
        int(start + 2 * HOURLY_NS),
    ]
    overlap = source.read_times(np.asarray([start + HOURLY_NS]), "elev")
    np.testing.assert_allclose(overlap, np.asarray([[21.0, 21.0, 21.0]]))


def test_ar2_coefficients_are_estimated_independently_per_eof_mode() -> None:
    """每個 EOF score 應得到兩個自身係數，不可誤取 2*rank 聯合矩陣前兩欄。"""

    scores = np.asarray(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [5.0, 3.0], [8.0, 7.0], [13.0, 10.0]],
        dtype=np.float64,
    )
    ridge = 1.0e-6
    actual = reconstruction_module._ar_coefficients(scores, ridge)
    expected = np.empty((2, 2), dtype=np.float64)
    for mode in range(2):
        design = np.column_stack((scores[1:-1, mode], scores[:-2, mode]))
        gram = design.T @ design + ridge * np.eye(2)
        expected[mode] = np.linalg.solve(gram, design.T @ scores[2:, mode])
    assert actual.shape == (2, 2)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)
