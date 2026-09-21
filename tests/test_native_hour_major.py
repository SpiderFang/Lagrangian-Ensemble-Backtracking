"""WP-9：小時主軸排程（``native_runner.hour_major``）的驗收測試。

測試涵蓋（對應工作包驗收要求，逐項編號；詳見 ``hour_major.py`` 模組文件）：

1. **鏡像等價（擴散關閉）**：小時主軸合併呼叫的任一虛擬單元子批次，最終
   狀態、軌跡、departures、``hours_run`` 必須與單獨呼叫 ``run_backward_batch``
   位元相同。
2. **鏡像等價（擴散啟用）**：同一顆粒子單獨跑 vs. 併入更大批次跑，計數器型
   擴散序列必須位元相同；同一批粒子換不同的「其他同批粒子」組合，結果不變
   （批次組成無關性）。
3. **多測站關注區域**：同一小時合併呼叫混合不同測站（不同多邊形）的粒子，
   逐列判定結果必須與各自獨立呼叫（單一多邊形）完全一致。
4. **回溯上限**：精確在設定的秒數觸發 ``NATIVE_STATUS_MAX_BACKTRACK``。
5. **到達時刻精確控制**：不同到達時刻的粒子在正確的小時才開始移動。
6. **Checkpoint／續跑**：中斷後續跑的最終結果，與未中斷執行位元相同。
7. **效能**：與逐工作單元路徑比較，量測固定開銷的攤提效果。

計數器型擴散亂數本身的統計品質（均值、變異數、批次組成無關性、
SplitMix64＋Box-Muller 的獨立性）已經在 ``tests/test_native_backward_driver.py``
（``derive_particle_diffusion_seed_lo_hi_u64``／``counter_based_gaussian_pairs``
的定義處）完整驗證，本檔案只驗證「小時主軸排程正確地接上同一套機制」，不
重複驗證亂數演算法本身的統計性質。
"""

from __future__ import annotations

import os
import time as _time

import numpy as np
import pytest
from shapely.geometry import Polygon

from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.native_runner import backward_driver as bd
from lagrangian_backtracking.native_runner import build_native_grid_arrays
from lagrangian_backtracking.native_runner import hour_major as hm
from lagrangian_backtracking.native_runner.forcing_bridge import HourlyFieldPair
from lagrangian_backtracking.native_runner.orchestrator import NativeWorkUnit

_HOUR_NS = 3_600_000_000_000

_NX = 10
_NY = 10
_SPACING_M = 200.0
_NVRT = 2
_DP_M = 20.0
_DOMAIN_X = (_NX - 1) * _SPACING_M
_DOMAIN_Y = (_NY - 1) * _SPACING_M


def _node_index(i: int, j: int) -> int:
    return j * _NX + i


def _build_mesh() -> NativeMesh:
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

    return NativeMesh(
        node_lon=np.linspace(120.0, 120.01, n_nodes),
        node_lat=np.linspace(24.0, 24.01, n_nodes),
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
        pytest.skip("需要 LBT_BAYTRACE_ROOT 指向 BayTrace 原始碼根目錄才能執行 WP-9 測試")
    return root


@pytest.fixture(scope="module")
def handles() -> bd.BayTraceModuleHandles:
    return bd.load_baytrace_modules(_resolve_baytrace_root())


@pytest.fixture(scope="module")
def driver_grid(handles: bd.BayTraceModuleHandles) -> bd.NativeBackwardDriverGrid:
    mesh = _build_mesh()
    n_nodes = mesh.node_xy.shape[0]
    zcor_reference = np.zeros((n_nodes, _NVRT), dtype=np.float64)
    zcor_reference[:, 0] = -_DP_M
    zcor_reference[:, 1] = 0.0
    eta_reference = np.zeros(n_nodes, dtype=np.float64)
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor_reference, eta_reference=eta_reference)
    built = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)
    assert built.ic3_cross_check_ok
    return built


def _n_nodes(driver_grid: bd.NativeBackwardDriverGrid) -> int:
    return int(driver_grid.grid.grid_x.shape[0])


def _uniform_field_pair(
    driver_grid: bd.NativeBackwardDriverGrid,
    *,
    u1: float,
    v1: float,
    w1: float = 0.0,
    kh: float = 1.0e-4,
    time_before_utc_ns: int,
    time_after_utc_ns: int,
) -> HourlyFieldPair:
    # 注意：垂直層數必須讀 ``driver_grid`` 實際的 ``nvrt``，不能沿用模組層級
    # 給小型測試網格用的 ``_NVRT=2`` 常數——效能測試用的「真實規模」網格
    # （見 ``_build_realistic_grid``）是 20 層，用錯層數會讓這裡建出的欄位
    # 陣列在垂直維度太小，讓 BayTrace 的內插 kernel 讀到陣列外的記憶體，
    # 輕則算出天文數字的速度值，重則直接 segfault（實際踩過這個臭蟲，見
    # 最終報告）。
    n_nodes = _n_nodes(driver_grid)
    nvrt = int(driver_grid.schism_grid.nvrt)
    shape = (nvrt, n_nodes)
    eta = np.zeros(n_nodes, dtype=np.float64)
    diffusivity = np.full(shape, kh, dtype=np.float64)
    missing = np.zeros(shape, dtype=bool)
    dry_face = np.zeros(shape, dtype=bool)
    return HourlyFieldPair(
        time_before_utc_ns=time_before_utc_ns, time_after_utc_ns=time_after_utc_ns, trat=0.0, ibf=-1,
        u_field1=np.full(shape, u1), v_field1=np.full(shape, v1), w_field1=np.full(shape, w1),
        eta_field1=eta.copy(), diffusivity_field1=diffusivity.copy(),
        missing_mask1=missing.copy(), dry_face_flag1=dry_face.copy(),
        month_id1="synthetic", time_origin1="test",
        u_field2=np.full(shape, u1), v_field2=np.full(shape, v1), w_field2=np.full(shape, w1),
        eta_field2=eta.copy(), diffusivity_field2=diffusivity.copy(),
        missing_mask2=missing.copy(), dry_face_flag2=dry_face.copy(),
        month_id2="synthetic", time_origin2="test",
    )


def _make_resolver(
    driver_grid: bd.NativeBackwardDriverGrid, *, u: float, v: float, w: float = 0.0, kh: float = 1.0e-4
):
    def resolver(target_utc_ns: int) -> HourlyFieldPair:
        return _uniform_field_pair(
            driver_grid, u1=u, v1=v, w1=w, kh=kh,
            time_before_utc_ns=target_utc_ns, time_after_utc_ns=target_utc_ns - _HOUR_NS,
        )

    return resolver


class _FakeIdentityTable:
    """只提供 ``build_hour_major_domain_batch`` 實際會讀取的四個欄位。"""

    def __init__(self, x0_m: np.ndarray, y0_m: np.ndarray, z0_m: np.ndarray, seed_hex: np.ndarray) -> None:
        self.x0_m = x0_m
        self.y0_m = y0_m
        self.z0_m = z0_m
        self.seed_hex = seed_hex


class _FakeSoa:
    """欄位相容的最小 ``BayTraceParticleSoA`` 替身，供 ``run_backward_batch`` 使用。"""

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, row_particle_index: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.z0 = z
        self.row_particle_index = row_particle_index


_BIG_POLYGON = Polygon(
    [(-1.0e6, -1.0e6), (1.0e6, -1.0e6), (1.0e6, 1.0e6), (-1.0e6, 1.0e6)]
)


def _make_unit(
    *, unit_id: str, region: str, domain: str, site: str, arrival_utc: int, settling: float, rows: np.ndarray,
) -> NativeWorkUnit:
    return NativeWorkUnit(
        unit_id=unit_id,
        analysis_region_id=region,
        flow_domain_id=domain,
        study_site_id=site,
        arrival_time_utc_ns=int(arrival_utc),
        settling_velocity_mps=float(settling),
        particle_row_indices=rows.astype(np.int64),
        input_hash="0" * 12,
    )


def _seed_hex_for(row_ids: np.ndarray) -> np.ndarray:
    return np.asarray([f"{int(r):032x}" for r in row_ids], dtype="<U32")


def _run_unit_standalone(
    driver_grid, handles, *, table: _FakeIdentityTable, unit: NativeWorkUnit,
    hm_config: hm.HourMajorConfig, resolver, wave_field_resolver=None,
) -> bd.BackwardDriverResult:
    """獨立呼叫 ``run_backward_batch`` 跑同一個工作單元，作為等價驗收的對照組。

    ``run_backward_batch`` 一律不啟用 WP-14 任務 B 的端點重用（該路徑只在
    ``hour_major.py`` 的小時主軸迴圈接線，見該模組
    ``run_flow_domain_hour_major`` docstring），因此這個對照組天然是「完全
    不重用」的基準，適合用來驗證「啟用端點重用」與「完全不重用」兩條路徑
    逐位元相同。
    """

    rows = unit.particle_row_indices
    soa = _FakeSoa(
        x=table.x0_m[rows].copy(), y=table.y0_m[rows].copy(), z=table.z0_m[rows].copy(),
        row_particle_index=rows.copy(),
    )
    config = hm.hour_major_config_to_backward_driver_config(
        hm_config,
        arrival_time_utc_ns=unit.arrival_time_utc_ns,
        settling_velocity_mps=unit.settling_velocity_mps,
    )
    return bd.run_backward_batch(
        driver_grid, handles, soa,
        seed_hex_values=list(table.seed_hex[rows]),
        config=config,
        forcing_resolver=resolver,
        interest_region_polygon=_BIG_POLYGON,
        wave_field_resolver=wave_field_resolver,
    )


def _assert_results_bit_identical(
    a: bd.BackwardDriverResult, b: bd.BackwardDriverResult, *, label: str
) -> None:
    assert np.array_equal(a.row_particle_index, b.row_particle_index), f"{label}: row_particle_index 不同"
    assert np.array_equal(a.final_status, b.final_status), f"{label}: final_status 不同"
    assert np.array_equal(a.final_x_m, b.final_x_m), f"{label}: final_x_m 不同"
    assert np.array_equal(a.final_y_m, b.final_y_m), f"{label}: final_y_m 不同"
    assert np.array_equal(a.final_z_m, b.final_z_m), f"{label}: final_z_m 不同"
    assert np.array_equal(a.final_utc_ns, b.final_utc_ns), f"{label}: final_utc_ns 不同"
    assert np.array_equal(a.trajectory_utc_ns, b.trajectory_utc_ns), f"{label}: trajectory_utc_ns 不同"
    assert np.array_equal(a.trajectory_x_m, b.trajectory_x_m), f"{label}: trajectory_x_m 不同"
    assert np.array_equal(a.trajectory_y_m, b.trajectory_y_m), f"{label}: trajectory_y_m 不同"
    assert np.array_equal(a.trajectory_z_m, b.trajectory_z_m), f"{label}: trajectory_z_m 不同"
    assert np.array_equal(a.trajectory_status, b.trajectory_status), f"{label}: trajectory_status 不同"
    assert a.hours_run == b.hours_run, f"{label}: hours_run 不同 ({a.hours_run} vs {b.hours_run})"
    assert len(a.departures) == len(b.departures), f"{label}: departures 數量不同"
    for da, db in zip(
        sorted(a.departures, key=lambda d: d.row_particle_index),
        sorted(b.departures, key=lambda d: d.row_particle_index),
        strict=True,
    ):
        assert da.row_particle_index == db.row_particle_index
        assert da.status == db.status
        assert da.x_m == db.x_m and da.y_m == db.y_m and da.z_m == db.z_m
        assert da.utc_ns == db.utc_ns


# ══════════════════════════════════════════════════════════════════════════
# 情境建構：兩個到達時刻交錯的沉降速度分組，供多個測試重複使用
# ══════════════════════════════════════════════════════════════════════════


def _build_staggered_domain(*, n_per_unit: int = 3):
    """建立 2 個沉降速度分組、每組 2 個到達時刻交錯釋放的合成輸入。

    ``settle_a=0.0`` 分組有 unit A0（到達時刻 T0）與 unit A1（到達時刻
    T0-1hr）；``settle_b=-0.0005`` 分組只有 unit B0（到達時刻 T0-1hr）。
    T0-1hr 這個小時會同時出現「A1 剛釋放」與「A0 延續中」，正是驗證
    hf1／hf2 拆分邏輯是否正確的關鍵時刻。
    """

    t0 = 20_000 * _HOUR_NS
    n = n_per_unit
    rng = np.random.default_rng(7)
    x0_a0 = _DOMAIN_X / 2.0 + rng.uniform(-50.0, 50.0, size=n)
    y0_a0 = _DOMAIN_Y / 2.0 + rng.uniform(-50.0, 50.0, size=n)
    x0_a1 = _DOMAIN_X / 2.0 + rng.uniform(-50.0, 50.0, size=n)
    y0_a1 = _DOMAIN_Y / 2.0 + rng.uniform(-50.0, 50.0, size=n)
    x0_b0 = _DOMAIN_X / 2.0 + rng.uniform(-50.0, 50.0, size=n)
    y0_b0 = _DOMAIN_Y / 2.0 + rng.uniform(-50.0, 50.0, size=n)

    row_offsets = {"a0": (0, n), "a1": (n, 2 * n), "b0": (2 * n, 3 * n)}
    total = 3 * n
    x0_m = np.zeros(total)
    y0_m = np.zeros(total)
    z0_m = np.full(total, -5.0)
    x0_m[0:n], y0_m[0:n] = x0_a0, y0_a0
    x0_m[n : 2 * n], y0_m[n : 2 * n] = x0_a1, y0_a1
    x0_m[2 * n : 3 * n], y0_m[2 * n : 3 * n] = x0_b0, y0_b0
    row_ids = np.arange(total, dtype=np.int64)
    table = _FakeIdentityTable(x0_m=x0_m, y0_m=y0_m, z0_m=z0_m, seed_hex=_seed_hex_for(row_ids))

    unit_a0 = _make_unit(
        unit_id="A0", region="R", domain="D", site="S1", arrival_utc=t0, settling=0.0,
        rows=np.arange(*row_offsets["a0"]),
    )
    unit_a1 = _make_unit(
        unit_id="A1", region="R", domain="D", site="S1", arrival_utc=t0 - _HOUR_NS, settling=0.0,
        rows=np.arange(*row_offsets["a1"]),
    )
    unit_b0 = _make_unit(
        unit_id="B0", region="R", domain="D", site="S1", arrival_utc=t0 - _HOUR_NS, settling=-0.0005,
        rows=np.arange(*row_offsets["b0"]),
    )
    interest_regions = {"S1": _BIG_POLYGON}
    return table, [unit_a0, unit_a1, unit_b0], interest_regions, t0


# ══════════════════════════════════════════════════════════════════════════
# 驗收 1：擴散關閉時的位元等價
# ══════════════════════════════════════════════════════════════════════════


def test_hour_major_diffusion_off_matches_standalone_per_unit(driver_grid, handles) -> None:
    table, units, interest_regions, _t0 = _build_staggered_domain()
    resolver = _make_resolver(driver_grid, u=0.03, v=-0.01)
    config = hm.HourMajorConfig(
        max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=False, master_seed=123,
    )

    batch = hm.build_hour_major_domain_batch(table, units, interest_regions=interest_regions)
    result = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=config, forcing_resolver=resolver,
    )

    assert set(result.unit_results) == {"A0", "A1", "B0"}
    for unit in units:
        standalone = _run_unit_standalone(
            driver_grid, handles, table=table, unit=unit, hm_config=config, resolver=resolver,
        )
        _assert_results_bit_identical(result.unit_results[unit.unit_id], standalone, label=unit.unit_id)


# ══════════════════════════════════════════════════════════════════════════
# 驗收 2：擴散啟用時的位元等價（含 hf1/hf2 混合小時的正確處理）
# ══════════════════════════════════════════════════════════════════════════


def test_hour_major_diffusion_on_matches_standalone_per_unit(driver_grid, handles) -> None:
    table, units, interest_regions, _t0 = _build_staggered_domain()
    resolver = _make_resolver(driver_grid, u=0.02, v=0.01, kh=2.0e-4)
    config = hm.HourMajorConfig(
        max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=True, master_seed=99,
    )

    batch = hm.build_hour_major_domain_batch(table, units, interest_regions=interest_regions)
    result = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=config, forcing_resolver=resolver,
    )

    for unit in units:
        standalone = _run_unit_standalone(
            driver_grid, handles, table=table, unit=unit, hm_config=config, resolver=resolver,
        )
        _assert_results_bit_identical(result.unit_results[unit.unit_id], standalone, label=unit.unit_id)


def test_hour_major_endpoint_reuse_matches_standalone_and_reports_hits(driver_grid, handles) -> None:
    """WP-14 任務 B：端點重用（field1 沿用上一小時的 field2）啟用時，
    小時主軸路徑的結果仍須與完全不重用的逐單元路徑逐位元相同，且
    ``endpoint_reuse_hits`` 必須大於 0（否則這個測試沒有鑑別力，等於沒有
    真正驗證到重用路徑）。

    用非均勻（隨節點變化）的波浪場，確保 Stokes 全網格計算真的被觸發、
    且每個節點的計算結果不是平凡的常數，才能有效區分「端點重用算對」與
    「端點重用算錯但恰好因為資料太簡單而看不出來」。
    """

    table, units, interest_regions, _t0 = _build_staggered_domain()
    resolver = _make_resolver(driver_grid, u=0.02, v=0.015, kh=1.0e-4)
    n_nodes = _n_nodes(driver_grid)
    rng = np.random.default_rng(20260921)
    wave = bd.WaveFieldAtNodes(
        significant_wave_height_m=rng.uniform(0.5, 2.5, size=n_nodes),
        peak_frequency_hz=rng.uniform(0.08, 0.25, size=n_nodes),
        direction_raw_deg=rng.uniform(0.0, 360.0, size=n_nodes),
        valid_mask_wave=np.ones(n_nodes, dtype=bool),
        qc_flags=np.zeros(n_nodes, dtype=np.int64),
    )

    def wave_resolver(_utc_ns: int) -> bd.WaveFieldAtNodes:
        return wave

    config = hm.HourMajorConfig(
        max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=True, master_seed=2026,
    )

    batch = hm.build_hour_major_domain_batch(table, units, interest_regions=interest_regions)
    result = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=config,
        forcing_resolver=resolver, wave_field_resolver=wave_resolver,
    )

    assert result.domain_overhead_diagnostics.endpoint_reuse_hits > 0, (
        "這個情境（固定波浪場、eta 全程為 0、單一資料來源）每個小時的 field1 "
        "都應該能沿用上一小時的 field2，端點重用命中數應該大於 0，否則本測試"
        "沒有真正驗證到 WP-14 任務 B 的重用路徑"
    )
    assert result.domain_overhead_diagnostics.endpoint_reuse_misses == 0

    for unit in units:
        standalone = _run_unit_standalone(
            driver_grid, handles, table=table, unit=unit, hm_config=config, resolver=resolver,
            wave_field_resolver=wave_resolver,
        )
        _assert_results_bit_identical(result.unit_results[unit.unit_id], standalone, label=unit.unit_id)


def test_hour_major_diffusion_on_batch_composition_invariance(driver_grid, handles) -> None:
    """unit A0 併入不同的「其他單元」組合，自己的結果必須完全不變。"""

    table, units, interest_regions, _t0 = _build_staggered_domain()
    unit_a0, unit_a1, unit_b0 = units
    resolver = _make_resolver(driver_grid, u=0.02, v=0.01, kh=2.0e-4)
    config = hm.HourMajorConfig(
        max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=True, master_seed=99,
    )

    batch_all = hm.build_hour_major_domain_batch(table, units, interest_regions=interest_regions)
    result_all = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch_all, config=config, forcing_resolver=resolver,
    )

    batch_alone = hm.build_hour_major_domain_batch(table, [unit_a0], interest_regions=interest_regions)
    result_alone = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch_alone, config=config, forcing_resolver=resolver,
    )

    batch_pair = hm.build_hour_major_domain_batch(
        table, [unit_a0, unit_b0], interest_regions=interest_regions
    )
    result_pair = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch_pair, config=config, forcing_resolver=resolver,
    )

    _assert_results_bit_identical(
        result_all.unit_results["A0"], result_alone.unit_results["A0"], label="A0 (all vs alone)"
    )
    _assert_results_bit_identical(
        result_all.unit_results["A0"], result_pair.unit_results["A0"], label="A0 (all vs pair)"
    )


# ══════════════════════════════════════════════════════════════════════════
# 驗收 3：多測站關注區域（不同多邊形）合併呼叫的等價性
# ══════════════════════════════════════════════════════════════════════════


def test_hour_major_multi_station_polygons_matches_separate_calls(driver_grid, handles) -> None:
    t0 = 21_000 * _HOUR_NS
    n = 4
    rng = np.random.default_rng(11)
    x0 = _DOMAIN_X / 2.0 + rng.uniform(-30.0, 30.0, size=2 * n)
    y0 = _DOMAIN_Y / 2.0 + rng.uniform(-30.0, 30.0, size=2 * n)
    z0 = np.full(2 * n, -5.0)
    row_ids = np.arange(2 * n, dtype=np.int64)
    table = _FakeIdentityTable(x0_m=x0, y0_m=y0, z0_m=z0, seed_hex=_seed_hex_for(row_ids))

    # 測站 SMALL 的關注區域很小，粒子會在流場推動下很快離開；
    # 測站 BIG 用全域大多邊形，粒子在短暫的回溯上限內都不會離開。
    small_polygon = Polygon(
        [
            (_DOMAIN_X / 2.0 - 40.0, _DOMAIN_Y / 2.0 - 40.0),
            (_DOMAIN_X / 2.0 + 40.0, _DOMAIN_Y / 2.0 - 40.0),
            (_DOMAIN_X / 2.0 + 40.0, _DOMAIN_Y / 2.0 + 40.0),
            (_DOMAIN_X / 2.0 - 40.0, _DOMAIN_Y / 2.0 + 40.0),
        ]
    )
    interest_regions = {"SMALL": small_polygon, "BIG": _BIG_POLYGON}

    unit_small = _make_unit(
        unit_id="SMALL_UNIT", region="R", domain="D", site="SMALL", arrival_utc=t0, settling=0.0,
        rows=np.arange(0, n),
    )
    unit_big = _make_unit(
        unit_id="BIG_UNIT", region="R", domain="D", site="BIG", arrival_utc=t0, settling=0.0,
        rows=np.arange(n, 2 * n),
    )
    resolver = _make_resolver(driver_grid, u=0.05, v=0.0)
    config = hm.HourMajorConfig(max_backtrack_seconds=4.0 * 3600.0, enable_diffusion=False, master_seed=1)

    batch = hm.build_hour_major_domain_batch(
        table, [unit_small, unit_big], interest_regions=interest_regions
    )
    result = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=config, forcing_resolver=resolver,
    )

    for unit, polygon in ((unit_small, small_polygon), (unit_big, _BIG_POLYGON)):
        rows = unit.particle_row_indices
        soa = _FakeSoa(
            x=table.x0_m[rows].copy(), y=table.y0_m[rows].copy(), z=table.z0_m[rows].copy(),
            row_particle_index=rows.copy(),
        )
        driver_config = hm.hour_major_config_to_backward_driver_config(
            config, arrival_time_utc_ns=unit.arrival_time_utc_ns, settling_velocity_mps=0.0
        )
        standalone = bd.run_backward_batch(
            driver_grid, handles, soa,
            seed_hex_values=list(table.seed_hex[rows]), config=driver_config,
            forcing_resolver=resolver, interest_region_polygon=polygon,
        )
        _assert_results_bit_identical(result.unit_results[unit.unit_id], standalone, label=unit.unit_id)

    # 明確確認「小多邊形」確實提早以離開關注區域終止，「大多邊形」則是回溯上限；
    # 這保證上面的位元相同比對不是兩邊剛好都用同一種終止原因而巧合通過。
    assert np.all(result.unit_results["SMALL_UNIT"].final_status == bd.NATIVE_STATUS_LEFT_INTEREST_REGION)
    assert np.all(result.unit_results["BIG_UNIT"].final_status == bd.NATIVE_STATUS_MAX_BACKTRACK)


# ══════════════════════════════════════════════════════════════════════════
# 驗收 4：回溯上限精確觸發
# ══════════════════════════════════════════════════════════════════════════


def test_hour_major_max_backtrack_triggers_at_exact_hour(driver_grid, handles) -> None:
    t0 = 22_000 * _HOUR_NS
    n = 2
    x0 = np.full(n, _DOMAIN_X / 2.0)
    y0 = np.full(n, _DOMAIN_Y / 2.0)
    z0 = np.full(n, -5.0)
    row_ids = np.arange(n, dtype=np.int64)
    table = _FakeIdentityTable(x0_m=x0, y0_m=y0, z0_m=z0, seed_hex=_seed_hex_for(row_ids))
    unit = _make_unit(
        unit_id="CAP", region="R", domain="D", site="S1", arrival_utc=t0, settling=0.0, rows=row_ids
    )
    resolver = _make_resolver(driver_grid, u=0.0, v=0.0)
    config = hm.HourMajorConfig(max_backtrack_seconds=2.0 * 3600.0, enable_diffusion=False, master_seed=0)

    batch = hm.build_hour_major_domain_batch(table, [unit], interest_regions={"S1": _BIG_POLYGON})
    result = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=config, forcing_resolver=resolver,
    )
    unit_result = result.unit_results["CAP"]
    assert np.all(unit_result.final_status == bd.NATIVE_STATUS_MAX_BACKTRACK)
    assert unit_result.hours_run == 2
    assert np.all(unit_result.final_utc_ns == t0 - 2 * _HOUR_NS)


# ══════════════════════════════════════════════════════════════════════════
# 驗收 5：不同到達時刻精確控制釋放
# ══════════════════════════════════════════════════════════════════════════


def test_hour_major_releases_exactly_at_own_arrival_hour(driver_grid, handles) -> None:
    table, units, interest_regions, t0 = _build_staggered_domain(n_per_unit=1)
    unit_a0, unit_a1, _unit_b0 = units
    resolver = _make_resolver(driver_grid, u=0.02, v=0.0)
    config = hm.HourMajorConfig(max_backtrack_seconds=1.0 * 3600.0, enable_diffusion=False, master_seed=0)

    batch = hm.build_hour_major_domain_batch(table, [unit_a0, unit_a1], interest_regions=interest_regions)
    result = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=config, forcing_resolver=resolver,
    )
    # A0 到達時刻是 t0，A1 到達時刻是 t0-1hr；兩者的 max_backtrack 都是 1 小時，
    # 所以兩者的軌跡取樣長度應該相同（各自從自己的到達時刻起算 1 小時後觸頂），
    # 但 A1 的 final_utc_ns 應該比 A0 的 final_utc_ns 早整整 1 小時——這正是
    # 「A1 比 A0 晚一小時才開始」的直接證據。
    r_a0 = result.unit_results["A0"]
    r_a1 = result.unit_results["A1"]
    assert r_a0.hours_run == 1
    assert r_a1.hours_run == 1
    assert int(r_a0.final_utc_ns[0]) - int(r_a1.final_utc_ns[0]) == _HOUR_NS
    # 在 A1 尚未到達之前（t0 這個小時），A1 的粒子不應該移動：用一顆從未經歷
    # 流場推動的粒子來確認——A1 唯一的一次輸出取樣就是牠自己 hours_run=1 後的
    # 終點，位移量應該恰好是「1 小時、u=0.02」，而不是 2 小時份的位移。
    displacement = float(r_a1.final_x_m[0] - table.x0_m[unit_a1.particle_row_indices[0]])
    assert displacement == pytest.approx(-0.02 * 3600.0, abs=1.0e-6)


# ══════════════════════════════════════════════════════════════════════════
# 驗收 6：checkpoint／續跑的位元相同性
# ══════════════════════════════════════════════════════════════════════════


def test_hour_major_checkpoint_resume_bit_identical(driver_grid, handles, tmp_path) -> None:
    table, units, interest_regions, _t0 = _build_staggered_domain()
    resolver = _make_resolver(driver_grid, u=0.02, v=-0.015, kh=1.5e-4)
    config = hm.HourMajorConfig(
        max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=True, master_seed=42,
    )

    batch_full = hm.build_hour_major_domain_batch(table, units, interest_regions=interest_regions)
    result_full = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch_full, config=config, forcing_resolver=resolver,
    )

    checkpoint_dir = tmp_path / "hm_checkpoint"
    batch_partial = hm.build_hour_major_domain_batch(table, units, interest_regions=interest_regions)
    partial = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch_partial, config=config, forcing_resolver=resolver,
        checkpoint_dir=checkpoint_dir, checkpoint_every_hours=1, _max_master_hours=1,
    )
    assert partial.domain_hours_walked == 1
    assert (checkpoint_dir / "meta.json").is_file()

    batch_resume = hm.build_hour_major_domain_batch(table, units, interest_regions=interest_regions)
    result_resumed = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch_resume, config=config, forcing_resolver=resolver,
        checkpoint_dir=checkpoint_dir, checkpoint_every_hours=1,
    )

    assert set(result_resumed.unit_results) == set(result_full.unit_results)
    for unit_id in result_full.unit_results:
        _assert_results_bit_identical(
            result_full.unit_results[unit_id], result_resumed.unit_results[unit_id], label=unit_id
        )


# ══════════════════════════════════════════════════════════════════════════
# 驗收 8（WP-13 任務 2／3）：小時內平行化（史托克斯子行程池）位元等價
# ══════════════════════════════════════════════════════════════════════════
#
# ``run_flow_domain_hour_major`` 每小時只呼叫一次的
# ``compose_current_and_stokes_fields`` 新增了 ``stokes_pool``／
# ``stokes_worker_count`` 參數（見 ``hour_major.py`` 該函式 docstring 與
# ``backward_driver._evaluate_batch_finite_depth_stokes``）：史托克斯漂移
# （Stokes drift）逐點求解彼此獨立、沒有跨節點化簡（reduction），因此切成
# 多塊分派到子行程池平行計算，理論上不該改變任何數值結果。這裡直接驗證
# 「不給行程池（預設、單一行程逐點）」與「給 2 個工作行程的子行程池」在
# 同一個小時主軸批次上跑出來的最終狀態、軌跡、departures 完全位元相同。


def test_hour_major_stokes_pool_matches_sequential_bit_identical(driver_grid, handles) -> None:
    from concurrent.futures import ProcessPoolExecutor

    table, units, interest_regions, _t0 = _build_staggered_domain()
    resolver = _make_resolver(driver_grid, u=0.02, v=0.015, kh=1.0e-4)
    n_nodes = _n_nodes(driver_grid)
    wave = bd.WaveFieldAtNodes(
        significant_wave_height_m=np.full(n_nodes, 1.2),
        peak_frequency_hz=np.full(n_nodes, 0.15),
        direction_raw_deg=np.full(n_nodes, 30.0),
        valid_mask_wave=np.ones(n_nodes, dtype=bool),
        qc_flags=np.zeros(n_nodes, dtype=np.int64),
    )

    def wave_resolver(_utc_ns: int) -> bd.WaveFieldAtNodes:
        return wave

    config = hm.HourMajorConfig(
        max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=True, master_seed=7,
    )

    batch_sequential = hm.build_hour_major_domain_batch(table, units, interest_regions=interest_regions)
    result_sequential = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch_sequential, config=config,
        forcing_resolver=resolver, wave_field_resolver=wave_resolver,
    )

    batch_parallel = hm.build_hour_major_domain_batch(table, units, interest_regions=interest_regions)
    with ProcessPoolExecutor(max_workers=2) as pool:
        result_parallel = hm.run_flow_domain_hour_major(
            driver_grid, handles, batch_parallel, config=config,
            forcing_resolver=resolver, wave_field_resolver=wave_resolver,
            stokes_pool=pool, stokes_worker_count=2,
        )

    assert set(result_sequential.unit_results) == set(result_parallel.unit_results) == {"A0", "A1", "B0"}
    for unit_id in result_sequential.unit_results:
        _assert_results_bit_identical(
            result_sequential.unit_results[unit_id], result_parallel.unit_results[unit_id], label=unit_id
        )


# ══════════════════════════════════════════════════════════════════════════
# 驗收 7：效能——與逐工作單元路徑比較固定開銷攤提效果
# ══════════════════════════════════════════════════════════════════════════


def _build_realistic_grid(handles: bd.BayTraceModuleHandles, node_target: int) -> bd.NativeBackwardDriverGrid:
    side = max(2, int(round(node_target**0.5)))
    n_nodes = side * side
    node_xy = np.zeros((n_nodes, 2), dtype=np.float64)
    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="xy")
    node_xy[:, 0] = (ii * 200.0).ravel()
    node_xy[:, 1] = (jj * 200.0).ravel()
    dp = np.full(n_nodes, 20.0, dtype=np.float64)
    idx = np.arange(n_nodes).reshape(side, side)
    n00 = idx[:-1, :-1].ravel()
    n10 = idx[:-1, 1:].ravel()
    n11 = idx[1:, 1:].ravel()
    n01 = idx[1:, :-1].ravel()
    triangles = np.concatenate(
        [np.stack([n00, n10, n11], axis=1), np.stack([n00, n11, n01], axis=1)], axis=0
    )
    n_faces = triangles.shape[0]
    face_nodes_local = np.full((n_faces, 4), -1, dtype=np.int64)
    face_nodes_local[:, :3] = triangles
    face_node_count = np.full(n_faces, 3, dtype=np.int64)
    mesh = NativeMesh(
        node_lon=np.linspace(120.0, 120.5, n_nodes),
        node_lat=np.linspace(24.0, 24.5, n_nodes),
        node_xy=node_xy,
        source_depth_m=dp,
        source_node_bottom_index=np.zeros(n_nodes, dtype=np.int64),
        face_nodes_local=face_nodes_local,
        face_node_count=face_node_count,
        source_face_global_index=np.arange(n_faces, dtype=np.int64),
        bin_size_m=400.0,
    )
    nvrt = 20
    sigma_1d = np.linspace(-1.0, 0.0, nvrt)
    zcor_reference = sigma_1d[None, :] * mesh.source_depth_m[:, None]
    eta_reference = np.zeros(n_nodes, dtype=np.float64)
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor_reference, eta_reference=eta_reference)
    driver_grid = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)
    assert driver_grid.ic3_cross_check_ok
    return driver_grid


@pytest.mark.parametrize("members_per_scenario", [1, 10])
def test_hour_major_performance_vs_per_unit_path(handles, members_per_scenario) -> None:
    """量測小時主軸路徑 vs. 逐工作單元路徑的壁鐘時間，回報攤提效果。

    受限於測試執行時間，本測試用縮小過的規模驗證「同一機制」（固定開銷
    攤提），而不是逐字重跑工作書指定的完整正式規模（30,000 節點、5 個
    相隔 15 天的到達時刻、10 個沉降速度、20/200 顆粒子）；縮小規模與
    外插到正式規模的完整推導、假設與量化結果見最終報告，這裡只確認
    「小時主軸路徑在小批次情境下明顯更快」這個核心工程結論，且方向與
    既有 ``test_realistic_grid_scale_hourly_overhead_benchmark``（60,000
    節點、單一小時、3 個沉降速度分組）的量測結果一致。
    """

    driver_grid = _build_realistic_grid(handles, 8_000)
    n_arrivals = 3
    n_settling = 4
    n_particles = members_per_scenario * 2
    t_latest = 30_000 * _HOUR_NS
    arrival_gap_hours = 2  # 到達時刻彼此相隔 2 小時，確保回溯窗大量重疊
    max_backtrack_hours = 4

    settling_speeds = [0.0, -2.0e-4, -4.0e-4, 2.0e-4][:n_settling]
    resolver = _make_resolver(driver_grid, u=0.01, v=-0.005)
    config = hm.HourMajorConfig(
        max_backtrack_seconds=float(max_backtrack_hours * 3600),
        enable_diffusion=False,
        master_seed=0,
    )

    rng = np.random.default_rng(3)
    units: list[NativeWorkUnit] = []
    x0_parts, y0_parts, z0_parts = [], [], []
    cursor = 0
    for ai in range(n_arrivals):
        arrival = t_latest - ai * arrival_gap_hours * _HOUR_NS
        for si, settle in enumerate(settling_speeds):
            rows = np.arange(cursor, cursor + n_particles, dtype=np.int64)
            cursor += n_particles
            x0_parts.append(_DOMAIN_X_LARGE(driver_grid) / 2.0 + rng.uniform(-100.0, 100.0, size=n_particles))
            y0_parts.append(_DOMAIN_Y_LARGE(driver_grid) / 2.0 + rng.uniform(-100.0, 100.0, size=n_particles))
            z0_parts.append(np.full(n_particles, -5.0))
            units.append(
                _make_unit(
                    unit_id=f"a{ai}_s{si}", region="R", domain="D", site="S1",
                    arrival_utc=arrival, settling=settle, rows=rows,
                )
            )
    table = _FakeIdentityTable(
        x0_m=np.concatenate(x0_parts), y0_m=np.concatenate(y0_parts), z0_m=np.concatenate(z0_parts),
        seed_hex=_seed_hex_for(np.arange(cursor, dtype=np.int64)),
    )
    interest_regions = {"S1": _BIG_POLYGON}

    # ── 新路徑：一次 run_flow_domain_hour_major 呼叫 ──────────────────────
    batch = hm.build_hour_major_domain_batch(table, units, interest_regions=interest_regions)
    t0 = _time.perf_counter()
    hour_major_result = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=config, forcing_resolver=resolver,
    )
    hour_major_elapsed = _time.perf_counter() - t0

    # ── 舊路徑：逐工作單元各自呼叫 run_backward_batch，完全不共用任何快取 ──
    t0 = _time.perf_counter()
    per_unit_results = {}
    for unit in units:
        per_unit_results[unit.unit_id] = _run_unit_standalone(
            driver_grid, handles, table=table, unit=unit, hm_config=config, resolver=resolver,
        )
    per_unit_elapsed = _time.perf_counter() - t0

    for unit in units:
        _assert_results_bit_identical(
            hour_major_result.unit_results[unit.unit_id], per_unit_results[unit.unit_id], label=unit.unit_id
        )

    speedup = per_unit_elapsed / hour_major_elapsed if hour_major_elapsed > 0 else float("inf")
    total_particle_steps = sum(
        r.hours_run * unit.particle_row_indices.size for r, unit in zip(
            (hour_major_result.unit_results[u.unit_id] for u in units), units, strict=True
        )
    )
    print(
        f"\n[hour_major benchmark] M={members_per_scenario} "
        f"n_units={len(units)} particles/unit={n_particles} "
        f"grid_nodes~{_n_nodes(driver_grid)} "
        f"per_unit={per_unit_elapsed:.3f}s hour_major={hour_major_elapsed:.3f}s "
        f"speedup={speedup:.2f}x "
        f"domain_hours_walked={hour_major_result.domain_hours_walked} "
        f"kernel_calls={hour_major_result.domain_kernel_call_count} "
        f"compose_hits={hour_major_result.domain_overhead_diagnostics.compose_base_cache_hits} "
        f"compose_misses={hour_major_result.domain_overhead_diagnostics.compose_base_cache_misses} "
        f"total_particle_steps={total_particle_steps}"
    )
    assert speedup > 1.0, "小時主軸路徑理應快於逐工作單元路徑（固定開銷已攤提）"
    # 同一小時最多 10 個沉降速度分組；本測試 n_settling <= 10，混合到達的
    # 小時最多讓一個分組多付 1 次呼叫，核心呼叫次數應該遠少於「舊路徑等價
    # 呼叫次數」（每個工作單元、每個自己的小時各一次）。
    assert hour_major_result.domain_kernel_call_count <= n_settling * (
        n_arrivals * max_backtrack_hours + n_arrivals
    )


def _DOMAIN_X_LARGE(driver_grid: bd.NativeBackwardDriverGrid) -> float:
    return float(driver_grid.grid.grid_x.max() - driver_grid.grid.grid_x.min())


def _DOMAIN_Y_LARGE(driver_grid: bd.NativeBackwardDriverGrid) -> float:
    return float(driver_grid.grid.grid_y.max() - driver_grid.grid.grid_y.min())
