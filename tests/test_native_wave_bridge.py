"""WP-5 補充：NWW3→追蹤網格節點波浪橋接（``native_runner.wave_bridge``）驗收測試。

測試涵蓋：

1. **雙線性內插與波向分量平均正確性**：以隨機合成 NWW3 網格，逐一對照
   ``forcing.NWWAnalysisMonth.sample_at_time_index``（既有、已驗證過的逐點
   內插邏輯）當作地面真相，驗證本模組完全向量化的
   ``gather_nww3_hour_at_nodes`` 對每個節點算出的 ``Hs``／``fp``／方向與
   有效旗標都與逐點參考實作一致。
2. **跨 0°/360° 邊界的方向平均**：專門構造一組四角方向分別接近 350°／10°
   的合成資料，驗證加權平均後的方向落在 0°附近，而不是「直接線性平均角度」
   會錯誤得到的 180°附近。
3. **四角遮罩政策**：任一角點無效時，節點整體判定為無效（不用其餘三角點
   退化內插頂替）。
4. **NWW3 涵蓋範圍外的節點**：明確標記為無效，不論當下資料本身是否有效。
5. **時間精確對齊**：``NativeWaveBridgeReader.resolve`` 對不存在的精確 UTC
   小時直接拒絕，不做時間內插或最近值頂替。
6. **快取行為**：同一 UTC 小時重複查詢不重複計算；快取大小限制下，被淘汰
   的 UTC 再次查詢會重新計算一次。
7. **端對端**：把 ``NativeWaveBridgeReader`` 接上
   ``backward_driver.run_backward_batch`` 的 ``wave_field_resolver``，驗證
   啟用 Stokes 後的位移與停用時不同（確認整合鏈路真正把 Stokes 貢獻送進了
   RK4，而不只是橋接模組自己算得出正確的節點波浪場）。
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from shapely.geometry import Polygon

from lagrangian_backtracking.forcing import NWWAnalysisMonth
from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.native_runner import backward_driver as bd
from lagrangian_backtracking.native_runner.forcing_bridge import HourlyFieldPair
from lagrangian_backtracking.native_runner.grid_bridge import build_native_grid_arrays
from lagrangian_backtracking.native_runner.wave_bridge import (
    NativeWaveBridgeReader,
    WaveBridgeError,
    build_nww3_node_interpolation_weights,
    gather_nww3_hour_at_nodes,
)

_HOUR_NS = 3_600_000_000_000


# ── 共用小工具：直接以記憶體陣列建立 NWWAnalysisMonth，不落地檔案 ───────────


def _make_month(
    *,
    month_id: str,
    lon: np.ndarray,
    lat: np.ndarray,
    time_utc_ns: np.ndarray,
    hs: np.ndarray,
    fp: np.ndarray,
    direction_deg: np.ndarray,
    valid: np.ndarray | None = None,
    qc: np.ndarray | None = None,
) -> NWWAnalysisMonth:
    shape = hs.shape
    return NWWAnalysisMonth(
        month_id=month_id,
        lon=np.asarray(lon, dtype=np.float64),
        lat=np.asarray(lat, dtype=np.float64),
        time_utc_ns=np.asarray(time_utc_ns, dtype=np.int64),
        significant_wave_height=np.asarray(hs, dtype=np.float64),
        peak_frequency=np.asarray(fp, dtype=np.float64),
        peak_direction_raw_deg=np.asarray(direction_deg, dtype=np.float64),
        valid_mask_wave=(np.ones(shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)),
        qc_flags=(np.zeros(shape, dtype=np.uint16) if qc is None else np.asarray(qc, dtype=np.uint16)),
    )


# ── 驗收 1：雙線性內插與波向分量平均，逐點對照 forcing.py 既有邏輯 ─────────


def test_gather_matches_forcing_reference_sample_at_time_index() -> None:
    """隨機合成網格：向量化收集結果須與既有逐點參考實作 (forcing.py) 一致。"""

    rng = np.random.default_rng(20240921)
    lon = np.array([120.0, 120.4, 121.1, 121.6])
    lat = np.array([23.5, 23.9, 24.4])
    shape = (1, lat.size, lon.size)
    hs = rng.uniform(0.2, 3.0, size=shape)
    fp = rng.uniform(0.05, 0.3, size=shape)
    direction_deg = rng.uniform(0.0, 360.0, size=shape)

    month = _make_month(
        month_id="synthetic",
        lon=lon,
        lat=lat,
        time_utc_ns=np.array([0], dtype=np.int64),
        hs=hs,
        fp=fp,
        direction_deg=direction_deg,
    )

    # 節點座標刻意避開網格邊界與格線上（那些退化情形在驗收 2、4 另外測試），
    # 只在內部隨機取點，確保四個角點都在網格範圍內。
    n_nodes = 25
    node_lon = rng.uniform(lon[0] + 0.01, lon[-1] - 0.01, size=n_nodes)
    node_lat = rng.uniform(lat[0] + 0.01, lat[-1] - 0.01, size=n_nodes)

    weights = build_nww3_node_interpolation_weights(node_lon, node_lat, month.lon, month.lat)
    assert np.all(weights.node_covered_by_nww3)

    result = gather_nww3_hour_at_nodes(month, 0, weights)

    for i in range(n_nodes):
        reference = month.sample_at_time_index(float(node_lon[i]), float(node_lat[i]), 0)
        assert bool(result.valid_mask_wave[i]) == reference.valid
        assert reference.valid, "本測試合成資料全部有效，任何節點判定無效都代表內插邏輯有分歧"
        assert result.significant_wave_height_m[i] == pytest.approx(reference.significant_wave_height_m, rel=1e-12)
        assert result.peak_frequency_hz[i] == pytest.approx(reference.peak_frequency_hz, rel=1e-12)
        # 方向是圓週期量，用最小夾角比較，避免 359.9999 vs 0.0001 這種數值上
        # 幾乎相等、但直接相減會誤判為不相等的邊界情況。
        diff = abs(result.direction_raw_deg[i] - reference.peak_direction_raw_deg) % 360.0
        diff = min(diff, 360.0 - diff)
        assert diff == pytest.approx(0.0, abs=1e-9)


# ── 驗收 2：跨 0°/360° 邊界的方向平均 ────────────────────────────────────────


def test_direction_average_wraps_correctly_across_zero_degrees() -> None:
    """四角方向在 350°/10° 之間，加權平均應落在 0°附近，而非錯誤的 180°附近。"""

    lon = np.array([0.0, 1.0])
    lat = np.array([0.0, 1.0])
    direction_deg = np.array([[[350.0, 10.0], [350.0, 10.0]]])  # (time=1, lat=2, lon=2)
    hs = np.ones((1, 2, 2))
    fp = np.full((1, 2, 2), 0.1)

    month = _make_month(
        month_id="wrap", lon=lon, lat=lat, time_utc_ns=np.array([0], dtype=np.int64),
        hs=hs, fp=fp, direction_deg=direction_deg,
    )

    # 查詢點在正中央：四個角點權重相等（各 0.25），若誤用線性角度平均會得到
    # (350+10+350+10)/4 = 180，與正確的圓平均（應接近 0/360）南轅北轍。
    weights = build_nww3_node_interpolation_weights(
        np.array([0.5]), np.array([0.5]), month.lon, month.lat
    )
    result = gather_nww3_hour_at_nodes(month, 0, weights)
    assert bool(result.valid_mask_wave[0])
    direction = float(result.direction_raw_deg[0])
    distance_from_zero = min(direction, 360.0 - direction)
    assert distance_from_zero < 1e-6, f"預期方向接近 0/360 度，實際得到 {direction} 度"

    # 與既有 forcing.py 逐點參考實作交叉核對，確保不是本模組自己重新定義了
    # 一套只在這個測試裡自洽、但與既有慣例不一致的方向平均方式。
    reference = month.sample_at_time_index(0.5, 0.5, 0)
    assert reference.valid
    assert reference.peak_direction_raw_deg == pytest.approx(direction, abs=1e-9)


# ── 驗收 3：四角遮罩政策 ───────────────────────────────────────────────────


def test_any_invalid_corner_makes_node_invalid_not_degraded_interpolation() -> None:
    """四個角點只要有一個無效，節點整體無效；不接受用其餘三角點頂替內插。"""

    lon = np.array([0.0, 1.0])
    lat = np.array([0.0, 1.0])
    hs = np.ones((1, 2, 2))
    fp = np.full((1, 2, 2), 0.1)
    direction_deg = np.full((1, 2, 2), 45.0)
    valid = np.ones((1, 2, 2), dtype=bool)
    valid[0, 0, 0] = False  # (y0, x0) 角點無效

    month = _make_month(
        month_id="masked", lon=lon, lat=lat, time_utc_ns=np.array([0], dtype=np.int64),
        hs=hs, fp=fp, direction_deg=direction_deg, valid=valid,
    )

    weights = build_nww3_node_interpolation_weights(
        np.array([0.5]), np.array([0.5]), month.lon, month.lat
    )
    result = gather_nww3_hour_at_nodes(month, 0, weights)
    assert not bool(result.valid_mask_wave[0])
    assert np.isnan(result.significant_wave_height_m[0])
    assert np.isnan(result.peak_frequency_hz[0])
    assert np.isnan(result.direction_raw_deg[0])

    reference = month.sample_at_time_index(0.5, 0.5, 0)
    assert not reference.valid, "既有 forcing.py 邏輯應該也判定此點無效，兩者政策必須一致"


def test_qc_flags_are_unioned_across_four_corners_even_when_invalid() -> None:
    """qc_flags 一律回傳四角聯集，供 QC 稽核，即使節點本身因遮罩被判定無效。"""

    lon = np.array([0.0, 1.0])
    lat = np.array([0.0, 1.0])
    hs = np.ones((1, 2, 2))
    fp = np.full((1, 2, 2), 0.1)
    direction_deg = np.full((1, 2, 2), 45.0)
    qc = np.zeros((1, 2, 2), dtype=np.uint16)
    qc[0, 0, 0] = 0b0001
    qc[0, 1, 1] = 0b0010

    month = _make_month(
        month_id="qc", lon=lon, lat=lat, time_utc_ns=np.array([0], dtype=np.int64),
        hs=hs, fp=fp, direction_deg=direction_deg, qc=qc,
    )
    weights = build_nww3_node_interpolation_weights(
        np.array([0.5]), np.array([0.5]), month.lon, month.lat
    )
    result = gather_nww3_hour_at_nodes(month, 0, weights)
    assert int(result.qc_flags[0]) == 0b0011
    # 本測試 valid_mask 全真，唯一使節點無效的原因會是 qc 非零，交由呼叫端
    # （stokes_batch）的 QC 閘門處理；wave_bridge 本身的 valid_mask_wave 只
    # 反映空間遮罩與物理防呆，不因 qc 非零而自行判定無效——這與
    # WaveFieldAtNodes docstring「qc_flags 非零視為不可信」的下游職責分工
    # 一致：wave_bridge 負責誠實回報 qc，是否因此拒收由 stokes_batch 決定。
    assert bool(result.valid_mask_wave[0])


# ── 驗收 4：NWW3 涵蓋範圍外的節點 ────────────────────────────────────────────


def test_node_outside_nww3_coverage_is_always_invalid() -> None:
    """經緯度落在 NWW3 網格範圍外的節點，任何小時都應判定無效，不得補值。"""

    lon = np.array([120.0, 121.0])
    lat = np.array([24.0, 25.0])
    hs = np.ones((1, 2, 2))
    fp = np.full((1, 2, 2), 0.1)
    direction_deg = np.full((1, 2, 2), 90.0)

    month = _make_month(
        month_id="coverage", lon=lon, lat=lat, time_utc_ns=np.array([0], dtype=np.int64),
        hs=hs, fp=fp, direction_deg=direction_deg,
    )

    node_lon = np.array([120.5, 200.0, np.nan])
    node_lat = np.array([24.5, 25.0, 24.5])
    weights = build_nww3_node_interpolation_weights(node_lon, node_lat, month.lon, month.lat)
    assert weights.node_covered_by_nww3.tolist() == [True, False, False]

    result = gather_nww3_hour_at_nodes(month, 0, weights)
    assert bool(result.valid_mask_wave[0])
    assert not bool(result.valid_mask_wave[1])
    assert not bool(result.valid_mask_wave[2])
    assert np.isnan(result.significant_wave_height_m[1])
    assert np.isnan(result.significant_wave_height_m[2])


# ── 驗收 5、6：NativeWaveBridgeReader 的時間精確對齊與快取 ─────────────────


def _write_nww3_month(root, *, grid_lon, grid_lat, month_id, time_utc_ns, hs, fp, direction_deg):
    """把一個月份的 NWW3 陣列落地成 ``nww3_analysis`` schema 1 的檔案配置，供
    ``NWWAnalysisMonth.from_directories``／``NativeWaveBridgeReader`` 讀取。
    """

    grid_dir = root / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    np.save(grid_dir / "lon.npy", np.asarray(grid_lon, dtype=np.float64))
    np.save(grid_dir / "lat.npy", np.asarray(grid_lat, dtype=np.float64))

    month_dir = root / month_id
    month_dir.mkdir(parents=True, exist_ok=True)
    shape = hs.shape
    np.save(month_dir / "time_utc_ns.npy", np.asarray(time_utc_ns, dtype=np.int64))
    np.save(month_dir / "significant_wave_height.npy", np.asarray(hs, dtype=np.float64))
    np.save(month_dir / "peak_frequency.npy", np.asarray(fp, dtype=np.float64))
    np.save(month_dir / "peak_direction_raw_deg.npy", np.asarray(direction_deg, dtype=np.float64))
    np.save(month_dir / "valid_mask_wave.npy", np.ones(shape, dtype=bool))
    np.save(month_dir / "qc_flags.npy", np.zeros(shape, dtype=np.uint16))
    return grid_dir, month_dir


def test_reader_requires_exact_hour_match_and_rejects_missing_hour(tmp_path) -> None:
    grid_lon = np.array([120.0, 121.0])
    grid_lat = np.array([24.0, 25.0])
    times = np.array([0, _HOUR_NS, 2 * _HOUR_NS], dtype=np.int64)
    shape = (times.size, 2, 2)
    hs = np.full(shape, 1.2)
    fp = np.full(shape, 0.1)
    direction_deg = np.full(shape, 30.0)

    grid_dir, month_dir = _write_nww3_month(
        tmp_path, grid_lon=grid_lon, grid_lat=grid_lat, month_id="m1",
        time_utc_ns=times, hs=hs, fp=fp, direction_deg=direction_deg,
    )

    reader = NativeWaveBridgeReader(
        grid_dir=grid_dir,
        month_dirs={"m1": month_dir},
        node_lon=np.array([120.5]),
        node_lat=np.array([24.5]),
    )

    result = reader.resolve(int(times[1]))
    assert bool(result.valid_mask_wave[0])
    assert result.significant_wave_height_m[0] == pytest.approx(1.2)

    with pytest.raises(WaveBridgeError):
        reader.resolve(int(times[1]) + 123)


def test_reader_cache_avoids_recompute_and_evicts_oldest(tmp_path) -> None:
    grid_lon = np.array([120.0, 121.0])
    grid_lat = np.array([24.0, 25.0])
    times = np.array([0, _HOUR_NS, 2 * _HOUR_NS], dtype=np.int64)
    shape = (times.size, 2, 2)
    hs = np.full(shape, 1.0)
    fp = np.full(shape, 0.1)
    direction_deg = np.full(shape, 0.0)

    grid_dir, month_dir = _write_nww3_month(
        tmp_path, grid_lon=grid_lon, grid_lat=grid_lat, month_id="m1",
        time_utc_ns=times, hs=hs, fp=fp, direction_deg=direction_deg,
    )
    reader = NativeWaveBridgeReader(
        grid_dir=grid_dir, month_dirs={"m1": month_dir},
        node_lon=np.array([120.5]), node_lat=np.array([24.5]), cache_size=2,
    )

    reader.resolve(int(times[0]))
    reader.resolve(int(times[0]))  # 快取命中，不應重複計算
    assert reader.read_counts[int(times[0])] == 1

    reader.resolve(int(times[1]))
    reader.resolve(int(times[2]))  # 快取大小 2：times[0] 應被淘汰
    reader.resolve(int(times[0]))  # 重新計算一次
    assert reader.read_counts[int(times[0])] == 2


# ── 驗收 7：端對端接上 run_backward_batch ───────────────────────────────────

_NX = 6
_NY = 6
_SPACING_M = 200.0
_NVRT = 2
_DP_M = 20.0


def _node_index(i: int, j: int) -> int:
    return j * _NX + i


def _build_small_mesh() -> NativeMesh:
    n_nodes = _NX * _NY
    node_xy = np.zeros((n_nodes, 2), dtype=np.float64)
    for j in range(_NY):
        for i in range(_NX):
            node_xy[_node_index(i, j)] = (i * _SPACING_M, j * _SPACING_M)
    dp = np.full(n_nodes, _DP_M, dtype=np.float64)

    triangles: list[tuple[int, int, int]] = []
    for j in range(_NY - 1):
        for i in range(_NX - 1):
            n00 = _node_index(i, j)
            n10 = _node_index(i + 1, j)
            n11 = _node_index(i + 1, j + 1)
            n01 = _node_index(i, j + 1)
            triangles.append((n00, n10, n11))
            triangles.append((n00, n11, n01))
    n_faces = len(triangles)
    face_nodes_local = np.full((n_faces, 4), -1, dtype=np.int64)
    for face_index, nodes in enumerate(triangles):
        face_nodes_local[face_index, :3] = nodes
    face_node_count = np.full(n_faces, 3, dtype=np.int64)

    # 節點經緯度刻意落在下面合成 NWW3 網格 [119.9, 120.2] x [23.9, 24.2] 的
    # 內部，確保端對端測試裡全部節點都在 NWW3 涵蓋範圍內。
    return NativeMesh(
        node_lon=np.linspace(120.0, 120.05, n_nodes),
        node_lat=np.linspace(24.0, 24.05, n_nodes),
        node_xy=node_xy,
        source_depth_m=dp,
        source_node_bottom_index=np.zeros(n_nodes, dtype=np.int64),
        face_nodes_local=face_nodes_local,
        face_node_count=face_node_count,
        source_face_global_index=np.arange(n_faces, dtype=np.int64),
        bin_size_m=400.0,
    )


def _resolve_baytrace_root() -> str:
    root = os.environ.get("LBT_BAYTRACE_ROOT")
    if not root:
        pytest.skip("需要 LBT_BAYTRACE_ROOT 指向 BayTrace 原始碼根目錄才能執行端對端測試")
    return root


class _SyntheticSoa:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, row_particle_index: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.z0 = z
        self.row_particle_index = row_particle_index


def test_end_to_end_wave_field_resolver_changes_trajectory(tmp_path) -> None:
    """把 wave_bridge 接上 run_backward_batch，確認啟用 Stokes 後軌跡真的改變。"""

    handles = bd.load_baytrace_modules(_resolve_baytrace_root())
    mesh = _build_small_mesh()
    n_nodes = mesh.node_xy.shape[0]
    zcor_reference = np.zeros((n_nodes, _NVRT), dtype=np.float64)
    zcor_reference[:, 0] = -_DP_M
    zcor_reference[:, 1] = 0.0
    eta_reference = np.zeros(n_nodes, dtype=np.float64)
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor_reference, eta_reference=eta_reference)
    driver_grid = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)
    assert driver_grid.ic3_cross_check_ok

    grid_lon = np.array([119.9, 120.2])
    grid_lat = np.array([23.9, 24.2])
    arrival_utc = 8_000 * _HOUR_NS
    times = np.array([arrival_utc - 2 * _HOUR_NS, arrival_utc - _HOUR_NS, arrival_utc], dtype=np.int64)
    shape = (times.size, 2, 2)
    hs = np.full(shape, 2.0)
    fp = np.full(shape, 0.1)
    direction_deg = np.full(shape, 90.0)  # 來自正東，Stokes 主要沿 -x 方向推。

    grid_dir, month_dir = _write_nww3_month(
        tmp_path, grid_lon=grid_lon, grid_lat=grid_lat, month_id="m1",
        time_utc_ns=times, hs=hs, fp=fp, direction_deg=direction_deg,
    )
    reader = NativeWaveBridgeReader(
        grid_dir=grid_dir, month_dirs={"m1": month_dir},
        node_lon=mesh.node_lon, node_lat=mesh.node_lat,
    )
    assert np.all(reader.weights.node_covered_by_nww3), "端對端測試假設全部節點都在 NWW3 涵蓋範圍內"

    u_curr, v_curr = 0.02, 0.0

    def resolver(target_utc_ns: int) -> HourlyFieldPair:
        hi = target_utc_ns
        lo = target_utc_ns - _HOUR_NS
        n_nodes_local = n_nodes
        shape_field = (_NVRT, n_nodes_local)
        eta = np.zeros(n_nodes_local, dtype=np.float64)
        diffusivity = np.full(shape_field, 1.0e-4, dtype=np.float64)
        missing = np.zeros(shape_field, dtype=bool)
        dry_face = np.zeros(shape_field, dtype=bool)
        return HourlyFieldPair(
            time_before_utc_ns=hi, time_after_utc_ns=lo, trat=0.0, ibf=-1,
            u_field1=np.full(shape_field, u_curr), v_field1=np.full(shape_field, v_curr),
            w_field1=np.zeros(shape_field), eta_field1=eta.copy(),
            diffusivity_field1=diffusivity.copy(), missing_mask1=missing.copy(), dry_face_flag1=dry_face.copy(),
            month_id1="synthetic", time_origin1="test",
            u_field2=np.full(shape_field, u_curr), v_field2=np.full(shape_field, v_curr),
            w_field2=np.zeros(shape_field), eta_field2=eta.copy(),
            diffusivity_field2=diffusivity.copy(), missing_mask2=missing.copy(), dry_face_flag2=dry_face.copy(),
            month_id2="synthetic", time_origin2="test",
        )

    interest_polygon = Polygon([(-1.0e6, -1.0e6), (1.0e6, -1.0e6), (1.0e6, 1.0e6), (-1.0e6, 1.0e6)])
    cx, cy = (_NX - 1) * _SPACING_M / 2.0, (_NY - 1) * _SPACING_M / 2.0
    soa = _SyntheticSoa(
        x=np.array([cx]), y=np.array([cy]), z=np.array([-5.0]),
        row_particle_index=np.array([0], dtype=np.int64),
    )

    config = bd.BackwardDriverConfig(
        arrival_time_utc_ns=arrival_utc,
        settling_velocity_mps=0.0,
        max_backtrack_seconds=2.0 * 3600.0,
        enable_diffusion=False,
        master_seed=0,
    )

    result_with_waves = bd.run_backward_batch(
        driver_grid, handles, soa,
        seed_hex_values=["0" * 32], config=config,
        forcing_resolver=resolver, interest_region_polygon=interest_polygon,
        wave_field_resolver=reader.as_wave_field_resolver(),
    )
    result_without_waves = bd.run_backward_batch(
        driver_grid, handles, soa,
        seed_hex_values=["0" * 32], config=config,
        forcing_resolver=resolver, interest_region_polygon=interest_polygon,
        wave_field_resolver=None,
    )

    dx_with = result_with_waves.final_x_m[0] - cx
    dx_without = result_without_waves.final_x_m[0] - cx
    assert dx_with != pytest.approx(dx_without, abs=1e-9), (
        "啟用 wave_field_resolver 後的位移應該與停用時不同，"
        "否則代表 Stokes 貢獻沒有真正透過 wave_bridge 進入 RK4"
    )
