"""WP-2 流場橋接（``native_runner.forcing_bridge``）的合成資料驗收測試。

所有測試只使用 ``tmp_path`` 造出的合成 ``ocm_native`` 月份 ``.npy`` 產品，不觸碰
``/data/OCM-Preprocessed-Data``（容器內本來就讀不到）。呼叫 BayTrace 原版
``batch_interpolate_velocity`` 的測試需要 ``LBT_BAYTRACE_ROOT``，比照
``test_baytrace_native_cpu.py`` 既有慣例，未設定時直接跳過。
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from lagrangian_backtracking.forcing import OCMReconstructionPatchMonth
from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.native_runner.forcing_bridge import (
    ForcingBridgeError,
    NativeForcingBridgeReader,
    UnregisteredForcingGapError,
    UnresolvedForcingGapError,
    _build_snapshot,
    _load_raw_month,
    _merge_month_with_patch,
    _resolve_field_assignment,
    inventory_forcing_gaps,
    project_face_dryness_to_nodes,
)

HOUR_NS = 3_600_000_000_000


def _t(hour_offset: int) -> int:
    """把「自 2024-01-01T00:00:00Z 起第幾個小時」換成 UTC 奈秒，方便測試閱讀。"""

    base = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
    return base + hour_offset * HOUR_NS


def _write_month(
    root: Path,
    month_id: str,
    *,
    time_hours: list[int],
    node_count: int,
    layer_count: int,
    face_count: int,
    hvel: np.ndarray | None = None,
    vertical_velocity: np.ndarray | None = None,
    zcor: np.ndarray | None = None,
    elev: np.ndarray | None = None,
    wetdry_elem: np.ndarray | None = None,
    diffusivity: np.ndarray | None = None,
) -> Path:
    """在 ``root/month_id`` 造出一個最小可用的合成 ``ocm_native`` 月份。

    未指定的欄位使用可重現、彼此不同的預設值（依 time/node/layer 索引線性
    展開），方便測試斷言轉置後的數值仍可追溯回原始索引。
    """

    month_dir = root / month_id
    month_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray([_t(h) for h in time_hours], dtype=np.int64)
    time_count = times.size

    def _default_tnl() -> np.ndarray:
        t_idx, n_idx, l_idx = np.meshgrid(
            np.arange(time_count), np.arange(node_count), np.arange(layer_count), indexing="ij"
        )
        return (t_idx * 100 + n_idx * 10 + l_idx).astype(np.float64)

    if hvel is None:
        base = _default_tnl()
        hvel = np.stack((base, -base), axis=-1)
    if vertical_velocity is None:
        vertical_velocity = _default_tnl() * 0.01
    if zcor is None:
        # 每個節點的層由淺到深固定間距 -5m，第 0 層最靠近海面。
        zcor = -5.0 * np.arange(layer_count)[None, None, :].astype(np.float64)
        zcor = np.broadcast_to(zcor, (time_count, node_count, layer_count)).copy()
    if elev is None:
        elev = np.zeros((time_count, node_count), dtype=np.float64)
    if wetdry_elem is None:
        wetdry_elem = np.zeros((time_count, face_count), dtype=np.float64)
    if diffusivity is None:
        diffusivity = np.full((time_count, node_count, layer_count), 1.0e-3, dtype=np.float64)

    np.save(month_dir / "time_utc_ns.npy", times)
    np.save(month_dir / "hvel.npy", hvel)
    np.save(month_dir / "vertical_velocity.npy", vertical_velocity)
    np.save(month_dir / "zcor.npy", zcor)
    np.save(month_dir / "elev.npy", elev)
    np.save(month_dir / "wetdry_elem.npy", wetdry_elem)
    np.save(month_dir / "diffusivity.npy", diffusivity)
    return month_dir


# ── 1. 載入與軸轉置正確性 ───────────────────────────────────────────────


def test_load_and_transpose_matches_source(tmp_path: Path) -> None:
    month_dir = _write_month(
        tmp_path, "202401", time_hours=[0, 1, 2], node_count=4, layer_count=3, face_count=1
    )
    raw = _load_raw_month(month_dir)
    merged = _merge_month_with_patch("202401", raw, None)
    snapshot = _build_snapshot(merged, 1)  # 第二個小時（hour offset=1）

    expected_u = np.asarray(raw.hvel[1, :, :, 0], dtype=np.float64).T
    expected_v = np.asarray(raw.hvel[1, :, :, 1], dtype=np.float64).T
    expected_w = np.asarray(raw.vertical_velocity[1], dtype=np.float64).T

    assert snapshot.u.shape == (3, 4)  # (nvrt, nnode)
    np.testing.assert_array_equal(snapshot.u, expected_u)
    np.testing.assert_array_equal(snapshot.v, expected_v)
    np.testing.assert_array_equal(snapshot.w, expected_w)
    assert snapshot.u.flags["C_CONTIGUOUS"]
    assert snapshot.v.flags["C_CONTIGUOUS"]
    assert snapshot.w.flags["C_CONTIGUOUS"]
    assert snapshot.diffusivity.flags["C_CONTIGUOUS"]
    assert snapshot.eta.shape == (4,)
    assert snapshot.time_origin == "observed"
    assert snapshot.month_id == "202401"
    assert not np.any(snapshot.missing_mask)  # 合成資料全部有限，不應有缺值


# ── 2. 滑動前進：field2 變 field1，且不重複讀取 ────────────────────────


def test_sliding_window_reuses_field2_as_field1_without_reread(tmp_path: Path) -> None:
    _write_month(tmp_path, "202401", time_hours=[0, 1, 2, 3], node_count=3, layer_count=2, face_count=1)
    reader = NativeForcingBridgeReader({"202401": tmp_path / "202401"}, ibf=1)

    pair_a = reader.resolve(_t(0))
    assert reader.read_counts[_t(0)] == 1
    assert reader.read_counts[_t(1)] == 1

    pair_b = reader.resolve(_t(1))
    # 前進一小時：原本 pair_a 的 field2 應該直接變成 pair_b 的 field1，
    # 且不重新讀取／轉置（同一個 ndarray 物件，不只是數值相等）。
    assert pair_b.u_field1 is pair_a.u_field2
    assert pair_b.v_field1 is pair_a.v_field2
    assert pair_b.w_field1 is pair_a.w_field2
    assert reader.read_counts[_t(1)] == 1  # 沒有因為 pair_b 而再讀一次
    assert reader.read_counts[_t(2)] == 1

    # 重複查詢同一小時、或回頭查詢仍在快取窗內的時刻，也不應該增加讀取次數。
    reader.resolve(_t(1) + HOUR_NS // 2)
    assert reader.read_counts[_t(1)] == 1
    assert reader.read_counts[_t(2)] == 1

    pair_c = reader.resolve(_t(2))
    assert pair_c.u_field1 is pair_b.u_field2
    assert reader.read_counts[_t(2)] == 1
    assert reader.read_counts[_t(3)] == 1
    assert sum(reader.read_counts.values()) == 4  # 4 個小時、各只讀一次


def test_sliding_window_reverse_direction(tmp_path: Path) -> None:
    """逆向（``ibf=-1``）滑動時，field2 一樣直接變成下一次的 field1。"""

    _write_month(tmp_path, "202401", time_hours=[0, 1, 2, 3], node_count=3, layer_count=2, face_count=1)
    reader = NativeForcingBridgeReader({"202401": tmp_path / "202401"}, ibf=-1)

    pair_a = reader.resolve(_t(3))
    pair_b = reader.resolve(_t(2))
    assert pair_b.u_field1 is pair_a.u_field2
    assert reader.read_counts[_t(3)] == 1
    assert reader.read_counts[_t(2)] == 1
    assert reader.read_counts[_t(1)] == 1


# ── 3. trat 正向／逆向正確 ─────────────────────────────────────────────


def test_resolve_field_assignment_forward() -> None:
    lo, hi = _t(0), _t(1)
    target = lo + (hi - lo) // 4  # 25% 處
    before, after, trat = _resolve_field_assignment(lo, hi, target, ibf=1)
    assert before == lo
    assert after == hi
    assert trat == pytest.approx(0.25)

    before0, after0, trat0 = _resolve_field_assignment(lo, hi, lo, ibf=1)
    assert (before0, after0, trat0) == (lo, hi, 0.0)


def test_resolve_field_assignment_reverse() -> None:
    lo, hi = _t(0), _t(1)
    target = lo + (hi - lo) // 4  # 距 hi 75%、距 lo 25%
    before, after, trat = _resolve_field_assignment(lo, hi, target, ibf=-1)
    # 逆向：起點（field1）是較晚的 hi，走到 target 已經「往回」走了 75%。
    assert before == hi
    assert after == lo
    assert trat == pytest.approx(0.75)

    # 逆向剛好落在較晚端點：trat 應為 0（純 field1）。
    before_hi, after_hi, trat_hi = _resolve_field_assignment(lo, hi, hi, ibf=-1)
    assert (before_hi, after_hi) == (hi, lo)
    assert trat_hi == pytest.approx(0.0)


def test_reader_trat_matches_ibf_direction(tmp_path: Path) -> None:
    _write_month(tmp_path, "202401", time_hours=[0, 1], node_count=3, layer_count=2, face_count=1)
    forward = NativeForcingBridgeReader({"202401": tmp_path / "202401"}, ibf=1)
    reverse = NativeForcingBridgeReader({"202401": tmp_path / "202401"}, ibf=-1)

    target = _t(0) + HOUR_NS // 4
    forward_pair = forward.resolve(target)
    reverse_pair = reverse.resolve(target)

    assert forward_pair.time_before_utc_ns == _t(0)
    assert forward_pair.time_after_utc_ns == _t(1)
    assert forward_pair.trat == pytest.approx(0.25)

    assert reverse_pair.time_before_utc_ns == _t(1)
    assert reverse_pair.time_after_utc_ns == _t(0)
    assert reverse_pair.trat == pytest.approx(0.75)


# ── 4. 缺口偵測、拒絕與盤點 ─────────────────────────────────────────────


def test_reader_rejects_unregistered_gap_with_missing_utc_in_message(tmp_path: Path) -> None:
    _write_month(
        tmp_path, "202401", time_hours=[0, 1, 2, 4], node_count=3, layer_count=2, face_count=1
    )  # 缺 hour_offset=3
    reader = NativeForcingBridgeReader({"202401": tmp_path / "202401"}, ibf=1)

    target = _t(2) + HOUR_NS // 2  # 落在 t2~t4 的未登錄缺口內
    with pytest.raises(UnregisteredForcingGapError) as excinfo:
        reader.resolve(target)
    message = str(excinfo.value)
    missing_iso = datetime.fromtimestamp(_t(3) / 1_000_000_000, tz=UTC).isoformat()
    assert missing_iso in message
    assert "缺" in message


def test_inventory_forcing_gaps_reports_missing_hour(tmp_path: Path) -> None:
    _write_month(
        tmp_path, "202401", time_hours=[0, 1, 2, 4], node_count=3, layer_count=2, face_count=1
    )
    report = inventory_forcing_gaps(
        {"202401": tmp_path / "202401"},
        expected_start_utc_ns=_t(0),
        expected_end_utc_ns=_t(4),
    )
    assert not report.is_run_ready
    assert report.expected_hour_count == 5
    assert report.observed_hour_count == 4
    assert report.reconstructed_hour_count == 0
    assert len(report.unregistered_gaps) == 1
    gap = report.unregistered_gaps[0]
    assert list(gap.missing_times_ns) == [_t(3)]
    assert report.unregistered_hour_count == 1
    assert datetime.fromtimestamp(_t(3) / 1_000_000_000, tz=UTC).isoformat() in report.describe()


def test_inventory_forcing_gaps_clean_period_is_run_ready(tmp_path: Path) -> None:
    _write_month(
        tmp_path, "202401", time_hours=[0, 1, 2, 3, 4], node_count=3, layer_count=2, face_count=1
    )
    report = inventory_forcing_gaps(
        {"202401": tmp_path / "202401"},
        expected_start_utc_ns=_t(0),
        expected_end_utc_ns=_t(4),
    )
    assert report.is_run_ready
    assert report.unregistered_gaps == ()
    assert report.observed_hour_count == 5


def _minimal_patch_for_gap(month_id: str, missing_hour: int, node_count: int, layer_count: int, face_count: int) -> OCMReconstructionPatchMonth:
    """建立一個只覆蓋單一缺失小時的最小合法重建 patch，供「已登錄缺口」測試使用。"""

    time_utc_ns = np.asarray([_t(missing_hour)], dtype=np.int64)
    hvel = np.zeros((1, node_count, layer_count, 2), dtype=np.float64)
    vertical_velocity = np.zeros((1, node_count, layer_count), dtype=np.float64)
    zcor = np.broadcast_to(
        -5.0 * np.arange(layer_count)[None, None, :].astype(np.float64), (1, node_count, layer_count)
    ).copy()
    elev = np.zeros((1, node_count), dtype=np.float64)
    wetdry_elem = np.zeros((1, face_count), dtype=np.float64)
    diffusivity = np.full((1, node_count, layer_count), 1.0e-3, dtype=np.float64)
    return OCMReconstructionPatchMonth(
        month_id=month_id,
        time_utc_ns=time_utc_ns,
        hvel=hvel,
        vertical_velocity=vertical_velocity,
        zcor=zcor,
        elev=elev,
        wetdry_elem=wetdry_elem,
        diffusivity=diffusivity,
        origin_code=np.array([1], dtype=np.int64),
        quality_flags=np.array([1], dtype=np.int64),
        metadata={"method": "short_linear_v1", "gap_id": "test-gap"},
    )


def test_registered_patch_resolves_gap_without_error(tmp_path: Path) -> None:
    _write_month(
        tmp_path, "202401", time_hours=[0, 1, 2, 4], node_count=3, layer_count=2, face_count=1
    )
    patch = _minimal_patch_for_gap("202401", missing_hour=3, node_count=3, layer_count=2, face_count=1)

    report = inventory_forcing_gaps(
        {"202401": tmp_path / "202401"},
        expected_start_utc_ns=_t(0),
        expected_end_utc_ns=_t(4),
        patches={"202401": patch},
    )
    assert report.is_run_ready
    assert report.reconstructed_hour_count == 1

    reader = NativeForcingBridgeReader(
        {"202401": tmp_path / "202401"}, ibf=1, patches={"202401": patch}
    )
    pair = reader.resolve(_t(3))
    assert pair.time_origin1 == "reconstructed" or pair.time_origin2 == "reconstructed"


# ── 5. 乾點與缺值分別保存、互不混淆 ─────────────────────────────────────


def _two_triangle_mesh() -> NativeMesh:
    """建立兩個不共用節點的三角形：0/1/2 之後會標成全乾，3/4/5 標成濕。"""

    node_lon = np.array([0.0, 1.0, 0.0, 10.0, 11.0, 10.0])
    node_lat = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    node_xy = np.column_stack(
        (np.array([0.0, 10.0, 0.0, 100.0, 110.0, 100.0]), np.array([0.0, 0.0, 10.0, 0.0, 0.0, 10.0]))
    )
    return NativeMesh(
        node_lon=node_lon,
        node_lat=node_lat,
        node_xy=node_xy,
        source_depth_m=np.full(6, 10.0),
        source_node_bottom_index=np.zeros(6, dtype=np.int64),
        face_nodes_local=np.array([[0, 1, 2, -1], [3, 4, 5, -1]], dtype=np.int64),
        face_node_count=np.array([3, 3], dtype=np.int64),
        source_face_global_index=np.array([0, 1], dtype=np.int64),
    )


def test_dry_and_missing_are_kept_separate(tmp_path: Path) -> None:
    mesh = _two_triangle_mesh()
    node_count, layer_count, face_count = 6, 2, 2

    hvel = np.zeros((1, node_count, layer_count, 2), dtype=np.float64)
    hvel[:] = 1.0
    hvel[0, [0, 1, 2], :, :] = np.nan  # 全乾三角形（face 0）：缺值可解釋為乾涸
    hvel[0, 4, :, :] = np.nan  # 濕三角形（face 1）中的節點 4：無法解釋的真實缺值

    vertical_velocity = np.zeros((1, node_count, layer_count), dtype=np.float64)
    zcor = np.broadcast_to(
        -5.0 * np.arange(layer_count)[None, None, :].astype(np.float64), (1, node_count, layer_count)
    ).copy()
    elev = np.zeros((1, node_count), dtype=np.float64)
    wetdry_elem = np.array([[1.0, 0.0]])  # face0=乾, face1=濕
    diffusivity = np.full((1, node_count, layer_count), 1.0e-3, dtype=np.float64)

    _write_month(
        tmp_path,
        "202401",
        time_hours=[0],
        node_count=node_count,
        layer_count=layer_count,
        face_count=face_count,
        hvel=hvel,
        vertical_velocity=vertical_velocity,
        zcor=zcor,
        elev=elev,
        wetdry_elem=wetdry_elem,
        diffusivity=diffusivity,
    )
    raw = _load_raw_month(tmp_path / "202401")
    merged = _merge_month_with_patch("202401", raw, None)
    snapshot = _build_snapshot(merged, 0)

    # 缺值遮罩應同時標到「乾涸節點」與「真實缺值節點」，因為兩者在原生資料裡
    # 都是非有限值；missing_mask 本身不負責分辨兩者。
    assert np.all(snapshot.missing_mask[:, [0, 1, 2, 4]])
    assert not np.any(snapshot.missing_mask[:, [3, 5]])

    node_is_dry, node_is_known = project_face_dryness_to_nodes(mesh, snapshot.dry_face_flag)
    assert node_is_known.all()
    np.testing.assert_array_equal(node_is_dry, [True, True, True, False, False, False])

    # 建一組「field2 完全乾淨」的假快照湊成一對，只測試 field1 那一側的邏輯。
    from lagrangian_backtracking.native_runner.forcing_bridge import HourlyFieldPair

    clean = np.ones_like(snapshot.u)
    clean_eta = np.zeros_like(snapshot.eta)
    clean_mask = np.zeros_like(snapshot.missing_mask)
    pair = HourlyFieldPair(
        time_before_utc_ns=_t(0),
        time_after_utc_ns=_t(1),
        trat=0.0,
        ibf=1,
        u_field1=snapshot.u,
        v_field1=snapshot.v,
        w_field1=snapshot.w,
        eta_field1=snapshot.eta,
        diffusivity_field1=snapshot.diffusivity,
        missing_mask1=snapshot.missing_mask,
        dry_face_flag1=snapshot.dry_face_flag,
        month_id1="202401",
        time_origin1="observed",
        u_field2=clean,
        v_field2=clean,
        w_field2=clean,
        eta_field2=clean_eta,
        diffusivity_field2=clean,
        missing_mask2=clean_mask,
        dry_face_flag2=np.zeros(2),
        month_id2="202401",
        time_origin2="observed",
    )

    # 已知乾濕狀態排除掉「乾涸解釋得通」的節點 0/1/2 之後，節點 4 的真實缺值
    # 仍應該讓呼叫端 fail closed，不能被靜默吃掉。
    with pytest.raises(UnresolvedForcingGapError) as excinfo:
        pair.raise_if_unresolved_gap(node_is_dry1=node_is_dry, node_is_dry2=None)
    assert "4" in str(excinfo.value) or "field1" in str(excinfo.value)

    # 完全不提供乾濕資訊時，乾涸節點也會被視為未解釋，一樣要 fail closed——
    # 這證明「乾點」與「缺值」是兩個獨立狀態，只有在明確提供乾濕證據時才會
    # 排除乾涸節點，不會被本模組自作主張地悄悄放行。
    with pytest.raises(UnresolvedForcingGapError):
        pair.raise_if_unresolved_gap()


# ── 6. 欄位可直接餵進 BayTrace 原版 batch_interpolate_velocity ──────────


def test_fields_feed_real_batch_interpolate_velocity(tmp_path: Path) -> None:
    raw_root = os.environ.get("LBT_BAYTRACE_ROOT")
    if not raw_root:
        pytest.skip("未設定 LBT_BAYTRACE_ROOT；略過需要 PI BayTrace 原始 artifact 的測試")
    source_root = Path(raw_root)
    if not source_root.exists():
        pytest.fail(f"LBT_BAYTRACE_ROOT 不存在：{source_root}")

    from lagrangian_backtracking.baytrace_native_cpu import BayTraceGridArrays, load_original_cpu

    kernels = load_original_cpu(source_root, tmp_path / "numba-cache")

    node_count, layer_count = 3, 3
    _write_month(
        tmp_path, "202401", time_hours=[0, 1], node_count=node_count, layer_count=layer_count, face_count=1
    )
    reader = NativeForcingBridgeReader({"202401": tmp_path / "202401"}, ibf=1)
    pair = reader.resolve(_t(0) + HOUR_NS // 3)
    assert not np.any(pair.missing_mask1)
    assert not np.any(pair.missing_mask2)

    grid = BayTraceGridArrays(
        i34=np.array([3], dtype=np.int64),
        elnode=np.array([[0, 1, 2, -1]], dtype=np.int64),
        grid_x=np.array([0.0, 10.0, 0.0]),
        grid_y=np.array([0.0, 0.0, 10.0]),
        ic3=np.full((4, 1), -1, dtype=np.int64),
        nvrt=layer_count,
        kbp=np.zeros(node_count, dtype=np.int64),
        kbe=np.zeros(1, dtype=np.int64),
        sigma_lcl=np.array([[-1.0, -1.0, -1.0], [-0.5, -0.5, -0.5], [0.0, 0.0, 0.0]]),
        dp=np.full(node_count, 10.0),
    )

    px = np.array([2.0])
    py = np.array([2.0])
    pz = np.array([-1.0])
    p_elem = np.array([0], dtype=np.int64)
    p_status = np.array([1], dtype=np.int32)
    u_out = np.zeros(1)
    v_out = np.zeros(1)
    w_out = np.zeros(1)

    kernels.batch_interpolate_velocity(
        px, py, pz, p_elem, p_status,
        grid.i34, grid.elnode, grid.grid_x, grid.grid_y,
        grid.nvrt, grid.kbp, grid.kbe, grid.sigma_lcl, grid.dp,
        pair.u_field1, pair.v_field1, pair.w_field1, pair.eta_field1,
        pair.u_field2, pair.v_field2, pair.w_field2,
        pair.trat, 0.0,
        u_out, v_out, w_out,
    )

    assert np.all(np.isfinite(u_out))
    assert np.all(np.isfinite(v_out))
    assert np.all(np.isfinite(w_out))


# ── 建構參數與例外檢查 ───────────────────────────────────────────────────


def test_reader_rejects_invalid_ibf(tmp_path: Path) -> None:
    _write_month(tmp_path, "202401", time_hours=[0, 1], node_count=3, layer_count=2, face_count=1)
    with pytest.raises(ForcingBridgeError):
        NativeForcingBridgeReader({"202401": tmp_path / "202401"}, ibf=0)


def test_reader_rejects_out_of_range_query(tmp_path: Path) -> None:
    _write_month(tmp_path, "202401", time_hours=[0, 1], node_count=3, layer_count=2, face_count=1)
    reader = NativeForcingBridgeReader({"202401": tmp_path / "202401"}, ibf=1)
    with pytest.raises(ForcingBridgeError):
        reader.resolve(_t(0) - HOUR_NS)
