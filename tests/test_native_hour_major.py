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

import dataclasses
import os
import time as _time
import tracemalloc

import numpy as np
import pytest
from shapely.geometry import Polygon

from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.native_runner import backward_driver as bd
from lagrangian_backtracking.native_runner import build_native_grid_arrays
from lagrangian_backtracking.native_runner import horizon_derivation as hd
from lagrangian_backtracking.native_runner import hour_major as hm
from lagrangian_backtracking.native_runner.backtrack_horizon import ResolvedUnitHorizon
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


def _flow_only_kwargs(interest_regions: dict[str, Polygon]) -> dict[str, object]:
    """WP-23：把既有測試的 ``interest_regions`` 轉成
    ``build_hour_major_domain_batch`` 現在要求的三個關鍵字參數。

    本檔案絕大多數測試驗證的是小時主軸排程本身（鏡像等價、批次組成無關性、
    checkpoint 續跑等），不是 WP-23 新增的 local domain 語意，所以一律把
    ``local_equals_flow`` 設成 ``True``、``local_domains`` 填成與
    ``interest_regions`` 相同的多邊形物件——這對應
    ``resources.DomainResources`` 對新竹／南灣／連江（local domain＝flow
    domain）的填法，讓這些既有測試在新程式下維持與 WP-23 之前逐位元等價的
    行為（``local_first_exit_pending`` 從一開始就是 ``False``，子步迴圈的
    check_mask 天生跳過 local 判定）。真正測試 local domain 非終止語意的
    案例見 ``test_hour_major_local_domain_*`` 系列，那裡會顯式傳入
    ``local_equals_flow=False`` 與獨立的 local 多邊形。
    """

    return {
        "local_domains": dict(interest_regions),
        "local_equals_flow": {site_id: True for site_id in interest_regions},
    }


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
        assert da.also_local_domain_first_exit == db.also_local_domain_first_exit, (
            f"{label}: row {da.row_particle_index} 的 also_local_domain_first_exit 不同"
        )
    # WP-23：local domain 首次離開（非終止）事件也要逐列位元比對，不只比對終止事件。
    assert len(a.local_domain_departures) == len(b.local_domain_departures), (
        f"{label}: local_domain_departures 數量不同"
    )
    for la, lb in zip(
        sorted(a.local_domain_departures, key=lambda d: d.row_particle_index),
        sorted(b.local_domain_departures, key=lambda d: d.row_particle_index),
        strict=True,
    ):
        assert la.row_particle_index == lb.row_particle_index
        assert la.x_m == lb.x_m and la.y_m == lb.y_m and la.z_m == lb.z_m
        assert la.utc_ns == lb.utc_ns
    # WP-28：bed_no_deposition 的非終止事件（一顆粒子可能出現多次，見
    # BedNoDepositionEventRecord docstring）也要逐筆位元比對，不能只比對
    # departures／local_domain_departures 就宣稱合併路徑與獨立路徑相同。
    assert len(a.bed_no_deposition_events) == len(b.bed_no_deposition_events), (
        f"{label}: bed_no_deposition_events 數量不同"
    )
    for ba, bb in zip(
        sorted(a.bed_no_deposition_events, key=lambda d: (d.row_particle_index, d.utc_ns)),
        sorted(b.bed_no_deposition_events, key=lambda d: (d.row_particle_index, d.utc_ns)),
        strict=True,
    ):
        assert ba.row_particle_index == bb.row_particle_index
        assert ba.x_m == bb.x_m and ba.y_m == bb.y_m and ba.z_m == bb.z_m
        assert ba.utc_ns == bb.utc_ns


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

    batch = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
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

    batch = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
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

    batch = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
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

    batch_all = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
    result_all = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch_all, config=config, forcing_resolver=resolver,
    )

    batch_alone = hm.build_hour_major_domain_batch(
        table, [unit_a0], interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
    result_alone = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch_alone, config=config, forcing_resolver=resolver,
    )

    batch_pair = hm.build_hour_major_domain_batch(
        table, [unit_a0, unit_b0], interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
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
        table, [unit_small, unit_big], interest_regions=interest_regions,
        **_flow_only_kwargs(interest_regions),
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
# WP-23：local domain（非終止記錄）與 flow domain（終止）分離語意——合成幾何測試
#
# 三種站點型態混在同一個小時主軸分組裡一次驗證：
#   SUBSET：local domain 是 flow domain 的真子集，粒子應先產生非終止的
#           LocalDomainFirstExitRecord（local_domain_departures），之後才在
#           更外圍的 flow domain 邊界產生終止的 DepartureRecord
#           （status=NATIVE_STATUS_LEFT_FLOW_DOMAIN，also_local_domain_
#           first_exit=False）。
#   EQ：    local_equals_flow=True，local domain 與 flow domain 是同一個多
#           邊形物件，只能產生一筆終止事件，該事件的
#           also_local_domain_first_exit 必須是 True，且不得另外出現在
#           local_domain_departures。
#   CAP：   local_equals_flow=True 但多邊形夠大，回溯上限先觸發（不離開任何
#           邊界）——確認 local_equals_flow=True 站點在沒有離開事件時完全不
#           受影響，且不會意外產生 local_domain_departures。
# 三個單元的小時主軸結果分別與各自單獨呼叫 run_backward_batch 做位元比對，
# 確保合併批次不會互相污染彼此的 local domain 判定。
# ══════════════════════════════════════════════════════════════════════════


def test_hour_major_local_domain_subset_and_equals_flow_mixed_batch(driver_grid, handles) -> None:
    t0 = 23_000 * _HOUR_NS
    n = 2

    def _square(half_width: float) -> Polygon:
        cx, cy = _DOMAIN_X / 2.0, _DOMAIN_Y / 2.0
        return Polygon(
            [
                (cx - half_width, cy - half_width), (cx + half_width, cy - half_width),
                (cx + half_width, cy + half_width), (cx - half_width, cy + half_width),
            ]
        )

    # SUBSET：local 40 m ⊂ flow 121 m，兩者是不同的多邊形物件。flow 半寬
    # 刻意不取 120（＝80 個 1.5 m 子步整數倍，u=0.05 m/s、dtb=30 s 時粒子軌跡
    # 會恰好落在邊界上），避免撞上 `substep_numba.py` 模組文件「與原版行為的
    # 已知差異」第 3 點誠實記載、且 `tests/test_substep_numba.py`
    # （`test_point_in_polygon_boundary_mismatch_is_only_on_exact_boundary`／
    # `test_diffusion_and_interest_region_matches_python_loop_statistically`）
    # 已經量化過的邊界重合問題：座標恰好落在多邊形邊界上時，射線法
    # （numba）與 Shapely ``contains_xy``（python）對「邊界算不算內部」的
    # 浮點捨入慣例可能不同，屬預期中、測度為零的邊界情況，不是 WP-23 新增
    # 的 local domain 判定本身的錯誤——已用最小重現腳本驗證：半寬 120 時
    # numba／python 兩條路徑各自都能正確判定離開（local 首次離開事件逐位元
    # 相同），差異只出現在 flow domain 那筆終止事件恰好卡在邊界的子步；半寬
    # 121 時兩條路徑逐位元相同。
    subset_local_polygon = _square(40.0)
    subset_flow_polygon = _square(121.0)
    # EQ：local_equals_flow=True，local 與 flow 是同一個多邊形物件（80 m）。
    eq_polygon = _square(80.0)
    # CAP：local_equals_flow=True，多邊形夠大，在下面的 max_backtrack 內不會離開。
    cap_polygon = _BIG_POLYGON

    x0 = np.full(3 * n, _DOMAIN_X / 2.0)
    y0 = np.full(3 * n, _DOMAIN_Y / 2.0)
    z0 = np.full(3 * n, -5.0)
    row_ids = np.arange(3 * n, dtype=np.int64)
    table = _FakeIdentityTable(x0_m=x0, y0_m=y0, z0_m=z0, seed_hex=_seed_hex_for(row_ids))

    unit_subset = _make_unit(
        unit_id="SUBSET_UNIT", region="R", domain="D", site="SUBSET", arrival_utc=t0, settling=0.0,
        rows=row_ids[0:n],
    )
    unit_eq = _make_unit(
        unit_id="EQ_UNIT", region="R", domain="D", site="EQ", arrival_utc=t0, settling=0.0,
        rows=row_ids[n:2 * n],
    )
    unit_cap = _make_unit(
        unit_id="CAP_UNIT", region="R", domain="D", site="CAP", arrival_utc=t0, settling=0.0,
        rows=row_ids[2 * n:3 * n],
    )

    interest_regions = {"SUBSET": subset_flow_polygon, "EQ": eq_polygon, "CAP": cap_polygon}
    local_domains = {"SUBSET": subset_local_polygon, "EQ": eq_polygon, "CAP": cap_polygon}
    local_equals_flow = {"SUBSET": False, "EQ": True, "CAP": True}

    resolver = _make_resolver(driver_grid, u=0.05, v=0.0)
    config = hm.HourMajorConfig(max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=False, master_seed=7)

    batch = hm.build_hour_major_domain_batch(
        table, [unit_subset, unit_eq, unit_cap], interest_regions=interest_regions,
        local_domains=local_domains, local_equals_flow=local_equals_flow,
    )
    result = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=config, forcing_resolver=resolver,
    )

    subset_result = result.unit_results["SUBSET_UNIT"]
    eq_result = result.unit_results["EQ_UNIT"]
    cap_result = result.unit_results["CAP_UNIT"]

    # SUBSET：兩顆粒子都應該先產生非終止的 local 首次離開記錄，之後才在 flow
    # domain 邊界終止；兩個事件都必須存在，且 local 早於 flow。
    assert np.all(subset_result.final_status == bd.NATIVE_STATUS_LEFT_FLOW_DOMAIN)
    assert len(subset_result.local_domain_departures) == n
    local_dep_by_row = {d.row_particle_index: d for d in subset_result.local_domain_departures}
    flow_dep_by_row = {d.row_particle_index: d for d in subset_result.departures}
    assert set(local_dep_by_row) == set(range(n))
    assert set(flow_dep_by_row) == set(range(n))
    for row in range(n):
        # 這是逆向（backward）積分：utc_ns 隨著回溯過程單調遞減，越晚離開（走
        # 越遠）的事件 utc_ns 越小。local domain（40 m）比 flow domain（121 m）
        # 先被離開，所以 local 事件的 utc_ns 必須「比較大」（比較接近到達時刻）。
        assert local_dep_by_row[row].utc_ns > flow_dep_by_row[row].utc_ns
        assert flow_dep_by_row[row].also_local_domain_first_exit is False

    # EQ：local domain 與 flow domain 相同，只能有一筆終止事件，且必須標記
    # also_local_domain_first_exit=True；不得另外出現在 local_domain_departures。
    assert np.all(eq_result.final_status == bd.NATIVE_STATUS_LEFT_FLOW_DOMAIN)
    assert len(eq_result.departures) == n
    assert all(d.also_local_domain_first_exit is True for d in eq_result.departures)
    assert len(eq_result.local_domain_departures) == 0

    # CAP：多邊形夠大、在回溯上限內不會離開邊界——確認 local_equals_flow=True
    # 且從未離開邊界的站點完全不受 WP-23 新增邏輯影響，不會憑空冒出 local
    # 首次離開事件。注意：`departures`（`DepartureRecord`）記錄的是「任何終止
    # 原因」（見其 docstring），達回溯上限（`NATIVE_STATUS_MAX_BACKTRACK`）本
    # 身就是一種終止，因此每顆粒子仍然會有一筆 `DepartureRecord`——這是既有、
    # WP-23 之前就有的行為（`postprocess.py` 也依賴這個不變量：每顆非 ACTIVE
    # 的粒子都必須對應一筆 `DepartureRecord`，否則視為資料不一致而報錯）。這裡
    # 真正該驗的是：狀態碼是 MAX_BACKTRACK 而不是 LEFT_FLOW_DOMAIN、且完全沒有
    # local 事件（不論是 `also_local_domain_first_exit` 或
    # `local_domain_departures`）。
    assert np.all(cap_result.final_status == bd.NATIVE_STATUS_MAX_BACKTRACK)
    assert len(cap_result.departures) == n
    assert all(d.status == bd.NATIVE_STATUS_MAX_BACKTRACK for d in cap_result.departures)
    assert all(d.also_local_domain_first_exit is False for d in cap_result.departures)
    assert len(cap_result.local_domain_departures) == 0

    # 逐站與獨立呼叫 run_backward_batch 位元比對，確保混合批次不會互相污染。
    for unit, polygon, local_equals, local_polygon in (
        (unit_subset, subset_flow_polygon, False, subset_local_polygon),
        (unit_eq, eq_polygon, True, None),
        (unit_cap, cap_polygon, True, None),
    ):
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
            local_equals_flow=local_equals, local_domain_polygon=local_polygon,
        )
        _assert_results_bit_identical(result.unit_results[unit.unit_id], standalone, label=unit.unit_id)


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

    batch = hm.build_hour_major_domain_batch(
        table, [unit], interest_regions={"S1": _BIG_POLYGON}, **_flow_only_kwargs({"S1": _BIG_POLYGON})
    )
    result = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=config, forcing_resolver=resolver,
    )
    unit_result = result.unit_results["CAP"]
    assert np.all(unit_result.final_status == bd.NATIVE_STATUS_MAX_BACKTRACK)
    assert unit_result.hours_run == 2
    assert np.all(unit_result.final_utc_ns == t0 - 2 * _HOUR_NS)


# ══════════════════════════════════════════════════════════════════════════
# WP-17：逐單元的 resolved_max_backtrack_seconds 覆寫（fixed_calendar_window 的
# 運算層支援）——同一個沉降速度分組裡混合兩個不同「有效回溯時間」的到達時刻，
# 驗證各自在自己的時限觸頂，而不是被 config 的共用值或彼此覆蓋。
# ══════════════════════════════════════════════════════════════════════════


def test_hour_major_per_unit_resolved_max_backtrack_seconds_override(driver_grid, handles) -> None:
    t0 = 22_000 * _HOUR_NS
    x0 = np.full(2, _DOMAIN_X / 2.0)
    y0 = np.full(2, _DOMAIN_Y / 2.0)
    z0 = np.full(2, -5.0)
    row_ids = np.array([0, 1], dtype=np.int64)
    table = _FakeIdentityTable(x0_m=x0, y0_m=y0, z0_m=z0, seed_hex=_seed_hex_for(row_ids))

    short_unit = dataclasses.replace(
        _make_unit(
            unit_id="SHORT", region="R", domain="D", site="S1", arrival_utc=t0, settling=0.0,
            rows=row_ids[:1],
        ),
        resolved_max_backtrack_seconds=1.0 * 3600.0,
    )
    long_unit = dataclasses.replace(
        _make_unit(
            unit_id="LONG", region="R", domain="D", site="S1", arrival_utc=t0, settling=0.0,
            rows=row_ids[1:],
        ),
        resolved_max_backtrack_seconds=3.0 * 3600.0,
    )
    resolver = _make_resolver(driver_grid, u=0.0, v=0.0)
    # config 自己的 max_backtrack_seconds 故意設一個兩個單元都不相等的值，確認
    # 兩個單元確實各自使用自己 resolved 的值，不是退回這個共用值。
    config = hm.HourMajorConfig(max_backtrack_seconds=10.0 * 3600.0, enable_diffusion=False, master_seed=0)

    batch = hm.build_hour_major_domain_batch(
        table, [short_unit, long_unit], interest_regions={"S1": _BIG_POLYGON},
        **_flow_only_kwargs({"S1": _BIG_POLYGON}),
    )
    result = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=config, forcing_resolver=resolver,
    )
    short_result = result.unit_results["SHORT"]
    long_result = result.unit_results["LONG"]
    assert np.all(short_result.final_status == bd.NATIVE_STATUS_MAX_BACKTRACK)
    assert short_result.hours_run == 1
    assert np.all(short_result.final_utc_ns == t0 - 1 * _HOUR_NS)
    assert np.all(long_result.final_status == bd.NATIVE_STATUS_MAX_BACKTRACK)
    assert long_result.hours_run == 3
    assert np.all(long_result.final_utc_ns == t0 - 3 * _HOUR_NS)


def test_build_hour_major_domain_batch_rejects_pre_window_units(driver_grid, handles) -> None:
    del driver_grid, handles
    row_ids = np.array([0], dtype=np.int64)
    table = _FakeIdentityTable(
        x0_m=np.zeros(1), y0_m=np.zeros(1), z0_m=np.full(1, -5.0), seed_hex=_seed_hex_for(row_ids)
    )
    unit = dataclasses.replace(
        _make_unit(
            unit_id="PRE", region="R", domain="D", site="S1", arrival_utc=22_000 * _HOUR_NS,
            settling=0.0, rows=row_ids,
        ),
        pre_window_deposition=True,
    )
    with pytest.raises(hm.HourMajorError):
        hm.build_hour_major_domain_batch(
            table, [unit], interest_regions={"S1": _BIG_POLYGON}, **_flow_only_kwargs({"S1": _BIG_POLYGON})
        )


# ══════════════════════════════════════════════════════════════════════════
# 驗收 5：不同到達時刻精確控制釋放
# ══════════════════════════════════════════════════════════════════════════


def test_hour_major_releases_exactly_at_own_arrival_hour(driver_grid, handles) -> None:
    table, units, interest_regions, t0 = _build_staggered_domain(n_per_unit=1)
    unit_a0, unit_a1, _unit_b0 = units
    resolver = _make_resolver(driver_grid, u=0.02, v=0.0)
    config = hm.HourMajorConfig(max_backtrack_seconds=1.0 * 3600.0, enable_diffusion=False, master_seed=0)

    batch = hm.build_hour_major_domain_batch(
        table, [unit_a0, unit_a1], interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
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

    batch_full = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
    result_full = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch_full, config=config, forcing_resolver=resolver,
    )

    checkpoint_dir = tmp_path / "hm_checkpoint"
    batch_partial = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
    partial = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch_partial, config=config, forcing_resolver=resolver,
        checkpoint_dir=checkpoint_dir, checkpoint_every_hours=1, _max_master_hours=1,
    )
    assert partial.domain_hours_walked == 1
    assert (checkpoint_dir / "meta.json").is_file()

    batch_resume = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
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

    batch_sequential = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
    result_sequential = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch_sequential, config=config,
        forcing_resolver=resolver, wave_field_resolver=wave_resolver,
    )

    batch_parallel = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
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
    batch = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
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


def test_hour_major_numba_smagorinsky_impl_runs_and_matches_on_uniform_flow(driver_grid, handles) -> None:
    """``smagorinsky_impl="numba"`` 接線可用；均勻流場應變率為零，兩種實作結果應完全相同。

    非均勻流場下兩者差 1–2 個最後有效位元（統計等價），由
    ``tests/test_native_smagorinsky_batch.py`` 與伺服器真實資料比對驗證。
    """

    table, units, interest_regions, _t0 = _build_staggered_domain()
    resolver = _make_resolver(driver_grid, u=0.02, v=0.01, kh=2.0e-4)
    batch = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
    results = {}
    for impl in ("baytrace", "numba"):
        config = hm.HourMajorConfig(
            max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=True, master_seed=99, smagorinsky_impl=impl,
        )
        results[impl] = hm.run_flow_domain_hour_major(
            driver_grid, handles, batch, config=config, forcing_resolver=resolver,
        )
    for unit in units:
        _assert_results_bit_identical(
            results["baytrace"].unit_results[unit.unit_id],
            results["numba"].unit_results[unit.unit_id],
            label=f"{unit.unit_id} (baytrace vs numba)",
        )


# ══════════════════════════════════════════════════════════════════════════
# WP-16 任務 1：substep_policy（local_cfl／global_cfl）
# ══════════════════════════════════════════════════════════════════════════


def test_hour_major_substep_policy_global_cfl_matches_legacy_default_bit_identical(
    driver_grid, handles
) -> None:
    """``substep_policy="global_cfl"`` 必須與 WP-16 新增這個欄位前的唯一行為
    （即 ``run_backward_batch`` 一律使用的 ``estimate_substep_count``）逐位元
    相同——這是「舊行為必須仍可用」驗收標準的直接測試，與
    ``test_hour_major_diffusion_off_matches_standalone_per_unit`` 用同一組
    合成情境，只是本測試明確傳入 ``substep_policy="global_cfl"``，不依賴
    ``HourMajorConfig`` 的預設值恰好是什麼。
    """

    table, units, interest_regions, _t0 = _build_staggered_domain()
    resolver = _make_resolver(driver_grid, u=0.03, v=-0.01)
    config = hm.HourMajorConfig(
        max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=False, master_seed=123,
        substep_policy="global_cfl",
    )

    batch = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
    result = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=config, forcing_resolver=resolver,
    )

    for unit in units:
        standalone = _run_unit_standalone(
            driver_grid, handles, table=table, unit=unit, hm_config=config, resolver=resolver,
        )
        _assert_results_bit_identical(result.unit_results[unit.unit_id], standalone, label=unit.unit_id)


def test_hour_major_config_rejects_unknown_substep_policy(driver_grid, handles) -> None:
    table, units, interest_regions, _t0 = _build_staggered_domain()
    resolver = _make_resolver(driver_grid, u=0.02, v=0.0)
    config = hm.HourMajorConfig(max_backtrack_seconds=1.0 * 3600.0, substep_policy="quantum_cfl")
    batch = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
    with pytest.raises(hm.HourMajorError, match="substep_policy"):
        hm.run_flow_domain_hour_major(driver_grid, handles, batch, config=config, forcing_resolver=resolver)


def test_estimate_substep_count_local_cfl_matches_global_on_uniform_edge_grid(driver_grid) -> None:
    """本檔案共用的合成網格是規則直角三角形網格，全部元素邊長相同，因此
    「只看鄰域」與「看全網格」的最短邊必然相同；活動粒子子集、位置不同，
    局部版本仍應與全域版本算出完全相同的 ``ndeltp``（不是巧合，是這個
    網格本身邊長均勻的必然結果，見 ``estimate_substep_count_local_cfl``
    docstring 對半徑保守性的論證）。
    """

    n_nodes = _n_nodes(driver_grid)
    nvrt = int(driver_grid.schism_grid.nvrt)
    u_field = np.full((nvrt, n_nodes), 0.05)
    v_field = np.full((nvrt, n_nodes), -0.02)

    spatial_index = bd.build_local_cfl_spatial_index(driver_grid)
    rng = np.random.default_rng(2026)
    active_x = rng.uniform(0.0, _DOMAIN_X, size=7)
    active_y = rng.uniform(0.0, _DOMAIN_Y, size=7)

    ndeltp_global = bd.estimate_substep_count(driver_grid, u_field, v_field, dt_max_seconds=30.0)
    ndeltp_local = bd.estimate_substep_count_local_cfl(
        driver_grid, u_field, v_field,
        active_x=active_x, active_y=active_y, spatial_index=spatial_index,
        dt_max_seconds=30.0,
    )
    assert ndeltp_local == ndeltp_global


def test_estimate_substep_count_local_cfl_falls_back_to_dt_max_without_active_particles(driver_grid) -> None:
    n_nodes = _n_nodes(driver_grid)
    nvrt = int(driver_grid.schism_grid.nvrt)
    u_field = np.full((nvrt, n_nodes), 5.0)  # 全網格極快的速度，若被用到 ndeltp 會爆炸大
    v_field = np.zeros((nvrt, n_nodes))
    spatial_index = bd.build_local_cfl_spatial_index(driver_grid)

    ndeltp = bd.estimate_substep_count_local_cfl(
        driver_grid, u_field, v_field,
        active_x=np.zeros(0), active_y=np.zeros(0), spatial_index=spatial_index,
        dt_seconds=3600.0, dt_max_seconds=30.0,
    )
    assert ndeltp == 120  # ceil(3600/30)，沒有活動粒子時只受 dt_max 限制


def _build_mesh_with_tiny_edge(shift_m: float) -> NativeMesh:
    """在 ``_build_mesh()`` 同一個 10x10 規則網格上，把角落節點 (9,9) 平移到
    幾乎貼齊節點 (8,9)，人工製造一段極短邊（長度約 ``shift_m``），供
    WP-16 任務 1「局部 CFL 忽略遠處極小元素、但不忽略鄰近的極小元素」測試
    使用。其餘 98 個節點與三角化拓樸完全不變。
    """

    mesh = _build_mesh()
    node_xy = mesh.node_xy.copy()
    far_node = _node_index(8, 9)
    near_node = _node_index(9, 9)
    node_xy[near_node, 0] = node_xy[far_node, 0] + shift_m
    node_xy[near_node, 1] = node_xy[far_node, 1]
    return NativeMesh(
        node_lon=mesh.node_lon, node_lat=mesh.node_lat, node_xy=node_xy,
        source_depth_m=mesh.source_depth_m, source_node_bottom_index=mesh.source_node_bottom_index,
        face_nodes_local=mesh.face_nodes_local, face_node_count=mesh.face_node_count,
        source_face_global_index=mesh.source_face_global_index, bin_size_m=mesh.bin_size_m,
    )


@pytest.fixture(scope="module")
def driver_grid_tiny_edge(handles: bd.BayTraceModuleHandles) -> bd.NativeBackwardDriverGrid:
    mesh = _build_mesh_with_tiny_edge(shift_m=0.02)
    n_nodes = mesh.node_xy.shape[0]
    zcor_reference = np.zeros((n_nodes, _NVRT), dtype=np.float64)
    zcor_reference[:, 0] = -_DP_M
    zcor_reference[:, 1] = 0.0
    eta_reference = np.zeros(n_nodes, dtype=np.float64)
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor_reference, eta_reference=eta_reference)
    built = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)
    assert built.ic3_cross_check_ok
    return built


def test_estimate_substep_count_local_cfl_ignores_far_away_tiny_element(driver_grid_tiny_edge) -> None:
    """節點 (9,9)／(8,9) 之間的極短邊（約 0.02 公尺）把全域版本的 ``ndeltp``
    推到很大；活動粒子全部位於對角遠處角落（節點 (0,0) 附近）時，局部版本
    的查詢半徑不該碰到這段極短邊，``ndeltp`` 應該遠小於全域版本、且仍然
    遠大於 1（確認不是退化成完全不做 CFL 判斷）。
    """

    n_nodes = _n_nodes(driver_grid_tiny_edge)
    nvrt = int(driver_grid_tiny_edge.schism_grid.nvrt)
    u_field = np.full((nvrt, n_nodes), 0.02)
    v_field = np.full((nvrt, n_nodes), 0.01)
    spatial_index = bd.build_local_cfl_spatial_index(driver_grid_tiny_edge)

    # 活動粒子集中在左下角，遠離右上角的極短邊（網格對角線約 2545 公尺）。
    active_x = np.full(5, 50.0)
    active_y = np.full(5, 50.0)

    ndeltp_global = bd.estimate_substep_count(
        driver_grid_tiny_edge, u_field, v_field, dt_max_seconds=30.0, cfl_number=0.5,
    )
    ndeltp_local = bd.estimate_substep_count_local_cfl(
        driver_grid_tiny_edge, u_field, v_field,
        active_x=active_x, active_y=active_y, spatial_index=spatial_index,
        dt_max_seconds=30.0, cfl_number=0.5,
    )

    assert ndeltp_local < ndeltp_global / 10.0, (ndeltp_local, ndeltp_global)
    assert ndeltp_local > 1
    dtb_local = 3600.0 / ndeltp_local
    assert dtb_local <= 30.0 * (1.0 + 1.0e-9)


def test_estimate_substep_count_local_cfl_matches_global_near_tiny_element(driver_grid_tiny_edge) -> None:
    """活動粒子就在極短邊旁邊時，局部版本必須偵測到這段極短邊，算出與全域
    版本相同（或更大，因為查詢半徑外擴＋節點鄰域可能多算幾個更短的候選邊，
    但不能更小）的 ``ndeltp``——這是局部 CFL 政策「不能漏掉真正危險的小
    元素」這個安全性要求的直接測試。
    """

    n_nodes = _n_nodes(driver_grid_tiny_edge)
    nvrt = int(driver_grid_tiny_edge.schism_grid.nvrt)
    u_field = np.full((nvrt, n_nodes), 0.02)
    v_field = np.full((nvrt, n_nodes), 0.01)
    spatial_index = bd.build_local_cfl_spatial_index(driver_grid_tiny_edge)

    tiny_edge_x = (_NX - 1) * _SPACING_M
    tiny_edge_y = (_NY - 1) * _SPACING_M
    active_x = np.full(5, tiny_edge_x)
    active_y = np.full(5, tiny_edge_y)

    ndeltp_global = bd.estimate_substep_count(
        driver_grid_tiny_edge, u_field, v_field, dt_max_seconds=30.0, cfl_number=0.5,
    )
    ndeltp_local = bd.estimate_substep_count_local_cfl(
        driver_grid_tiny_edge, u_field, v_field,
        active_x=active_x, active_y=active_y, spatial_index=spatial_index,
        dt_max_seconds=30.0, cfl_number=0.5,
    )

    assert ndeltp_local >= ndeltp_global


# ══════════════════════════════════════════════════════════════════════════
# WP-34：``estimate_substep_count_local_cfl`` 的 ``query_ball_point`` 記憶體
# 剖析與修正（見 reports/WP-34.md 完整數字）。
#
# 背景：``query_ball_point`` 對一整批活動粒子座標一次查詢時，scipy 會把
# 「每顆粒子自己查到的鄰近元素索引」各自組成一個獨立 Python list，``n`` 顆
# 粒子的 list 同時存活，尖峰記憶體與活動粒子數成正比——這是正式規模
# （M=10、500,000 顆粒子）在伺服器上 OOM 的根本原因。修正把查詢切成固定
# 大小（``_LOCAL_CFL_QUERY_CHUNK_SIZE``）的小批次，逐批查詢、逐批把結果併入
# 集合後立刻捨棄暫時 list，讓尖峰記憶體只跟「一個小批次的鄰近元素數」成
# 正比，與活動粒子總數無關。下面兩個測試分別驗證「輸出仍然逐位元相同」與
# 「尖峰記憶體不再隨活動粒子數線性增加」。
# ══════════════════════════════════════════════════════════════════════════


def _build_regular_grid_mesh_for_memory_probe(
    nx: int, ny: int, *, spacing_m: float, dp_m: float = 20.0
) -> NativeMesh:
    """比照 ``tests/test_native_backward_driver.py`` 的
    ``_build_regular_grid_mesh``：全程向量化建構規則三角網格，供 WP-34 的
    記憶體剖析測試使用一個比模組共用的 10x10 網格更大（因此更接近真實網格
    「查詢半徑內元素數很多」這個現象）、但仍然在 CI 可接受時間內建好的網格。
    刻意不 import 那個檔案的同名函式，避免讓兩個測試檔互相依賴彼此的內部
    實作（見該檔案同名函式的完整註解，兩者邏輯逐行相同）。
    """

    n_nodes = nx * ny
    node_xy = np.zeros((n_nodes, 2), dtype=np.float64)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    node_xy[:, 0] = (ii * spacing_m).ravel()
    node_xy[:, 1] = (jj * spacing_m).ravel()
    dp = np.full(n_nodes, dp_m, dtype=np.float64)

    idx = np.arange(n_nodes).reshape(ny, nx)
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

    return NativeMesh(
        node_lon=np.linspace(120.0, 120.2, n_nodes),
        node_lat=np.linspace(24.0, 24.2, n_nodes),
        node_xy=node_xy,
        source_depth_m=dp,
        source_node_bottom_index=np.zeros(n_nodes, dtype=np.int64),
        face_nodes_local=face_nodes_local,
        face_node_count=face_node_count,
        source_face_global_index=np.arange(n_faces, dtype=np.int64),
        bin_size_m=400.0,
    )


@pytest.fixture(scope="module")
def driver_grid_memory_probe(handles: bd.BayTraceModuleHandles) -> bd.NativeBackwardDriverGrid:
    """WP-34 專用、比模組共用 ``driver_grid`` 大上不少的合成網格（63x63＝
    3,969 節點、約 7,688 個元素、20 垂向層）。刻意選這個規模：夠大到「查詢
    半徑內元素數以千計」這個真實現象能重現（模組共用的 10x10 網格只有 162
    個元素，太小以致於修正前後的差異不明顯，見本節開頭說明），又小到建構
    與後續查詢仍能在 CI 的合理時間內跑完（實測建構 <1 秒，見
    ``reports/WP-34.md`` 效能章節）。"""

    mesh = _build_regular_grid_mesh_for_memory_probe(63, 63, spacing_m=150.0)
    n_nodes = mesh.node_xy.shape[0]
    nvrt = 20
    sigma_1d = np.linspace(-1.0, 0.0, nvrt)
    zcor_reference = sigma_1d[None, :] * mesh.source_depth_m[:, None]
    eta_reference = np.zeros(n_nodes, dtype=np.float64)
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor_reference, eta_reference=eta_reference)
    built = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)
    assert built.ic3_cross_check_ok
    return built


def test_estimate_substep_count_local_cfl_chunked_query_matches_unchunked_bit_identical(
    driver_grid_memory_probe,
) -> None:
    """WP-34 修正的位元相同性驗收：把查詢切成
    ``_LOCAL_CFL_QUERY_CHUNK_SIZE``（32）大小的小批次逐批查詢＋聯集，
    與「一次把全部活動粒子座標傳給 ``query_ball_point``」（修正前的寫法，
    這裡直接照抄原始邏輯做對照組，不依賴 git 歷史）比較，活動粒子數刻意
    取遠大於一個 chunk（900 顆），確保測試真的跨越多個 chunk 邊界，不是
    只測到「剛好一個 chunk 打完」這種退化成跟修正前完全一樣呼叫方式的
    情況。
    """

    driver_grid = driver_grid_memory_probe
    n_nodes = _n_nodes(driver_grid)
    nvrt = int(driver_grid.schism_grid.nvrt)
    spatial_index = bd.build_local_cfl_spatial_index(driver_grid)

    rng = np.random.default_rng(340228)
    domain_extent = 62 * 150.0

    def _unchunked_reference(u_field, v_field, active_x, active_y, *, dt_max_seconds, cfl_number):
        """修正前的寫法：一次呼叫 ``query_ball_point``，不分批。直接照抄
        ``backward_driver.estimate_substep_count_local_cfl`` 在 WP-34 之前
        的邏輯，只是把「取得 nearby_elements 集合」以外的部分原封不動，
        確保這個對照組跟現在的函式共用同一套下游公式（``local_min_edge_
        length``／``local_max_speed``／``ndeltp``），只有「怎麼查
        nearby_elements」這一步不同。"""

        dt_seconds = 3600.0
        ndeltp_from_dt_max = int(np.ceil(dt_seconds / dt_max_seconds))
        speed = np.hypot(u_field, v_field)
        finite_speed = speed[np.isfinite(speed)]
        global_max_speed = float(np.max(finite_speed)) if finite_speed.size > 0 else 0.0
        if active_x.size == 0 or global_max_speed <= 0.0:
            return max(1, ndeltp_from_dt_max)
        radius = 3.0 * global_max_speed * dt_seconds
        query_radius = radius + spatial_index.element_max_reach_m
        points = np.column_stack([active_x, active_y])
        neighbor_lists = spatial_index.element_centroid_kdtree.query_ball_point(points, r=query_radius)
        nearby_elements: set[int] = set()
        for neighbors in neighbor_lists:
            nearby_elements.update(neighbors)
        if not nearby_elements:
            return bd.estimate_substep_count(
                driver_grid, u_field, v_field, dt_max_seconds=dt_max_seconds, cfl_number=cfl_number,
            )
        idx = np.fromiter(nearby_elements, dtype=np.int64, count=len(nearby_elements))
        local_min_edge_length = float(np.min(spatial_index.element_min_edge_length[idx]))
        local_nodes = np.unique(
            np.concatenate(
                [
                    spatial_index.element_node0[idx],
                    spatial_index.element_node1[idx],
                    spatial_index.element_node2[idx],
                ]
            )
        )
        local_speed = speed[:, local_nodes]
        local_finite = local_speed[np.isfinite(local_speed)]
        local_max_speed = float(np.max(local_finite)) if local_finite.size > 0 else 0.0
        if local_max_speed <= 0.0:
            ndeltp_cfl = 1
        else:
            dtb_cfl = cfl_number * local_min_edge_length / local_max_speed
            ndeltp_cfl = int(np.ceil(dt_seconds / dtb_cfl))
        return max(1, ndeltp_from_dt_max, ndeltp_cfl)

    for trial, (n_particles, max_speed, dt_max, cfl) in enumerate(
        [
            (900, 0.05, 30.0, 0.5),  # 半徑遠小於整個網格：只有部分元素會被查到
            (500, 0.9, 10.0, 0.3),   # 半徑接近／超過整個網格：幾乎全部元素都會被查到
            (150, 0.4, 60.0, 0.9),   # 跨過多個 chunk 邊界（chunk=32 時 150 顆跨 32/64/96/128 共 4 個邊界）
        ]
    ):
        u_field = np.full((nvrt, n_nodes), max_speed / (2.0 ** 0.5))
        v_field = np.full((nvrt, n_nodes), max_speed / (2.0 ** 0.5))
        active_x = rng.uniform(0.0, domain_extent, size=n_particles)
        active_y = rng.uniform(0.0, domain_extent, size=n_particles)

        ndeltp_reference = _unchunked_reference(
            u_field, v_field, active_x, active_y, dt_max_seconds=dt_max, cfl_number=cfl,
        )
        ndeltp_chunked = bd.estimate_substep_count_local_cfl(
            driver_grid, u_field, v_field,
            active_x=active_x, active_y=active_y, spatial_index=spatial_index,
            dt_max_seconds=dt_max, cfl_number=cfl,
        )
        assert ndeltp_chunked == ndeltp_reference, (
            f"trial {trial}: n_particles={n_particles} 分批查詢與不分批查詢的 ndeltp 不同 "
            f"({ndeltp_chunked} vs {ndeltp_reference})，違反 WP-34 位元相同性要求"
        )


def test_estimate_substep_count_local_cfl_peak_memory_bounded_by_chunk_not_particle_count(
    driver_grid_memory_probe,
) -> None:
    """WP-34 記憶體回歸測試：``estimate_substep_count_local_cfl`` 單次呼叫的
    ``tracemalloc`` 峰值配置量，門檻訂在「與 chunk 大小同量級的常數」，
    不隨活動粒子數線性增加——這正是 WP-34 修正前後的關鍵差異（修正前峰值
    與活動粒子數成正比，見 ``reports/WP-34.md`` 剖析章節；4,000 節點網格、
    2,000 顆活動粒子的量測峰值高達 602 MB，且隨粒子數線性增加）。

    門檻訂定依據：本測試網格（63x63＝3,969 節點、約 7,688 個元素）用大流速
    （2.0 m/s，半徑遠超網格對角線，逼近「全部元素都在查詢半徑內」的最壞
    情況）驅動查詢，實測修正後單次呼叫峰值在活動粒子數 500／2,000／
    6,000 顆之間幾乎不變（40.25／40.27／40.34 MB，見
    ``reports/WP-34.md`` 「回歸測試門檻訂定依據」一節的完整量測輸出）——
    粒子數增加 12 倍，峰值只增加約 0.2%，直接證明峰值已經與 chunk 大小
    （``_LOCAL_CFL_QUERY_CHUNK_SIZE=32``）同量級、不再隨活動粒子數線性
    增加。門檻訂為 60 MB（相對實測 40.3 MB 保留約 1.5 倍餘裕，吸收不同
    測試機器 tracemalloc 記帳開銷的雜訊），修正前同量級網格在 2,000 顆
    粒子就已經測到 602 MB（見 ``reports/WP-34.md`` 剖析章節），60 MB 門檻
    仍然足以攔住「又退化回一次查完全部粒子」這種回歸。刻意用「單次函式
    呼叫的峰值是否為常數」而不是「整個小時主軸批次跑完的總記憶體」當斷言
    對象，因為這個函式呼叫是 WP-34 唯一實際定位到的根因；用整個批次的
    總記憶體斷言，在本測試用的小型合成網格（元素數遠小於正式規模）上
    訊號會被其他正常、隨小時數增加的軌跡取樣記憶體稀釋，門檻會很難訂得
    有意義。
    """

    driver_grid = driver_grid_memory_probe
    n_nodes = _n_nodes(driver_grid)
    nvrt = int(driver_grid.schism_grid.nvrt)
    spatial_index = bd.build_local_cfl_spatial_index(driver_grid)

    # 刻意用大流速把查詢半徑推到遠超網格對角線，逼近「不管粒子在哪裡，
    # 幾乎全部元素都在查詢半徑內」的最壞情況——這是修正前峰值最大、
    # 最能凸顯「與活動粒子數成正比」問題的情境。
    u_field = np.full((nvrt, n_nodes), 2.0 / (2.0 ** 0.5))
    v_field = np.full((nvrt, n_nodes), 2.0 / (2.0 ** 0.5))
    domain_extent = 62 * 150.0
    rng = np.random.default_rng(9101)

    peak_per_n: dict[int, int] = {}
    for n_particles in (500, 2000, 6000):
        active_x = rng.uniform(0.0, domain_extent, size=n_particles)
        active_y = rng.uniform(0.0, domain_extent, size=n_particles)

        tracemalloc.stop()
        tracemalloc.start(5)
        bd.estimate_substep_count_local_cfl(
            driver_grid, u_field, v_field,
            active_x=active_x, active_y=active_y, spatial_index=spatial_index,
            dt_max_seconds=30.0, cfl_number=0.5,
        )
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_per_n[n_particles] = peak

    _PEAK_BYTES_LIMIT = 60 * 1024 * 1024  # 60 MB，訂定依據見本函式 docstring
    for n_particles, peak in peak_per_n.items():
        assert peak < _PEAK_BYTES_LIMIT, (
            f"n_particles={n_particles} 時 estimate_substep_count_local_cfl 單次呼叫峰值 "
            f"{peak / 1e6:.1f} MB 超過 {_PEAK_BYTES_LIMIT / 1e6:.0f} MB 門檻——peak 應該只跟 "
            "chunk 大小同量級，不該隨活動粒子數線性增加（WP-34 回歸）"
        )

    # 額外檢查方向性：6,000 顆粒子時的峰值不該比 500 顆時明顯放大（實測應
    # 該幾乎是 1.0，見本函式 docstring 的實測數字；門檻訂為 2.0 保留餘裕，
    # 但仍然遠遠攔得住「跟粒子數成正比放大數十倍」這種修正失效的訊號）。
    ratio = peak_per_n[6000] / max(peak_per_n[500], 1)
    assert ratio < 2.0, (
        f"peak(6000顆)/peak(500顆)={ratio:.2f}，比例過大，像是又退化回跟活動粒子數"
        "成正比的舊行為（WP-34 回歸）"
    )


# ══════════════════════════════════════════════════════════════════════════
# WP-16 任務 1：局部 CFL 讓 ndeltp 落在 dt_max 下限時，與更細的 ndeltp
# 相比，粒子最終位置是否仍然「統計等價」（RK4 對平滑流場的收斂性佐證）
# ══════════════════════════════════════════════════════════════════════════


def test_rk4_final_position_converged_at_dt_max_floor_for_smooth_field(driver_grid, handles) -> None:
    """局部 CFL 政策在活動粒子遠離極小元素時，通常會落在 ``ndeltp=120``
    （``dt_max_seconds=30`` 的下限）；本測試直接驗證：對一個空間平滑變化
    （非常數）的流場，``ndeltp=120`` 算出的最終位置，與遠超過 CFL 需要的
    ``ndeltp=4000``（對照全域版本在含極小元素網格上可能算出的量級，見
    ``test_estimate_substep_count_local_cfl_ignores_far_away_tiny_element``
    的實測比例）幾乎相同，差異遠小於粒子這一小時移動的距離——證明「局部
    CFL 讓 ndeltp 變小」不會犧牲可觀察到的軌跡精度，只是省掉「CFL 條件
    本來就用不到」的保守子步。
    """

    n_nodes = _n_nodes(driver_grid)
    nvrt = int(driver_grid.schism_grid.nvrt)
    grid_x = driver_grid.grid.grid_x
    grid_y = driver_grid.grid.grid_y

    # 空間平滑變化（非常數）流場：u/v 隨位置正弦變化，振幅與網格尺度相當，
    # 確保子步數真的會影響 RK4 對非線性軌跡的積分精度，不是平凡的等速直線。
    def _smooth_field(x, y):
        u = 0.05 + 0.03 * np.sin(2.0 * np.pi * x / _DOMAIN_X)
        v = 0.03 * np.cos(2.0 * np.pi * y / _DOMAIN_Y)
        return u, v

    u_node, v_node = _smooth_field(grid_x, grid_y)
    u_field = np.tile(u_node, (nvrt, 1))
    v_field = np.tile(v_node, (nvrt, 1))
    w_field = np.zeros((nvrt, n_nodes))
    eta = np.zeros(n_nodes)

    composed = bd.ComposedFieldPair(
        u_field1=u_field, v_field1=v_field, w_field1=w_field, eta_field1=eta,
        u_field2=u_field, v_field2=v_field, w_field2=w_field,
        node_is_dry=np.zeros(n_nodes, dtype=bool),
        elem_is_dry=np.zeros(driver_grid.grid.elnode.shape[0], dtype=bool),
        stokes_ok_fraction=1.0,
    )

    rng = np.random.default_rng(99)
    n_particles = 6
    x0 = rng.uniform(200.0, _DOMAIN_X - 200.0, size=n_particles)
    y0 = rng.uniform(200.0, _DOMAIN_Y - 200.0, size=n_particles)

    finals = {}
    for ndeltp in (120, 4000):
        pa = handles.particle_arrays_cls(n_particles)
        pa.x[:] = x0
        pa.y[:] = y0
        pa.z[:] = -5.0
        pa.status[:] = 0
        not_found = bd.locate_initial_elements(
            driver_grid, handles, pa,
            eta_m=eta, u_current_mps=u_field, v_current_mps=v_field, w_current_mps=w_field,
        )
        assert not np.any(not_found)
        dtb = 3600.0 / ndeltp
        bd.mirror_rk4_substep_loop(
            driver_grid, handles, pa,
            ibf=-1, ndeltp=ndeltp, dtb=dtb, composed=composed, enable_diffusion=False,
        )
        finals[ndeltp] = (pa.x.copy(), pa.y.copy())

    x_coarse, y_coarse = finals[120]
    x_fine, y_fine = finals[4000]
    displacement = np.hypot(x_coarse - x0, y_coarse - y0)
    position_diff = np.hypot(x_coarse - x_fine, y_coarse - y_fine)
    # RK4 是四階方法，dtb 相差超過 30 倍（30s vs 0.9s）時，離散化誤差差距
    # 應該是好幾個數量級；用「相對於這一小時實際位移量的比例」當門檻，
    # 避免與粒子移動速度、網格尺度綁死的絕對公尺數。
    assert np.all(position_diff < 1.0e-3 * displacement), (position_diff, displacement)


# ══════════════════════════════════════════════════════════════════════════
# WP-16 任務 2：substep_loop_impl（numba／python）
# ══════════════════════════════════════════════════════════════════════════


def test_hour_major_substep_loop_impl_python_matches_legacy_default_bit_identical(
    driver_grid, handles
) -> None:
    """``substep_loop_impl="python"`` 必須與 WP-16 新增這個欄位前的唯一行為
    （``mirror_rk4_substep_loop``）逐位元相同——「舊行為必須仍可用」的直接
    測試，含擴散啟用（涉及計數器型亂數）與關注區域判定，不只是最簡單的
    無擴散情境。
    """

    table, units, interest_regions, _t0 = _build_staggered_domain()
    resolver = _make_resolver(driver_grid, u=0.02, v=0.01, kh=2.0e-4)
    config = hm.HourMajorConfig(
        max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=True, master_seed=99,
        substep_loop_impl="python",
    )

    batch = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
    result = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=config, forcing_resolver=resolver,
    )

    for unit in units:
        standalone = _run_unit_standalone(
            driver_grid, handles, table=table, unit=unit, hm_config=config, resolver=resolver,
        )
        _assert_results_bit_identical(result.unit_results[unit.unit_id], standalone, label=unit.unit_id)


def test_hour_major_config_rejects_unknown_substep_loop_impl(driver_grid, handles) -> None:
    table, units, interest_regions, _t0 = _build_staggered_domain()
    resolver = _make_resolver(driver_grid, u=0.02, v=0.0)
    config = hm.HourMajorConfig(max_backtrack_seconds=1.0 * 3600.0, substep_loop_impl="quantum")
    batch = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
    with pytest.raises(hm.HourMajorError, match="substep_loop_impl"):
        hm.run_flow_domain_hour_major(driver_grid, handles, batch, config=config, forcing_resolver=resolver)


def test_hour_major_substep_loop_impl_numba_statistically_equivalent_to_python(driver_grid, handles) -> None:
    """WP-16 任務 2 的核心驗收：預設 ``substep_loop_impl="numba"`` 與
    ``"python"`` 在較大規模（較多粒子、較多子步，提高踩到 Box-Muller ulp
    差異的機率）下，最終粒子狀態、位置、departures 必須統計等價——不要求
    逐位元相同（見 ``substep_numba.py`` 模組文件差異第 2、3 點），但差異
    必須遠小於粒子實際移動量級，且離開關注區域的判定幾乎完全一致。
    """

    table, units, interest_regions, _t0 = _build_staggered_domain(n_per_unit=20)
    resolver = _make_resolver(driver_grid, u=0.025, v=-0.015, kh=3.0e-4)
    batch = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )

    results = {}
    for impl in ("python", "numba"):
        config = hm.HourMajorConfig(
            max_backtrack_seconds=6.0 * 3600.0, enable_diffusion=True, master_seed=2026,
            substep_loop_impl=impl, dt_max_seconds=5.0,  # 較小 dt_max → 較多子步，較容易踩到 ulp 差異
        )
        results[impl] = hm.run_flow_domain_hour_major(
            driver_grid, handles, batch, config=config, forcing_resolver=resolver,
        )

    for unit in units:
        py_result = results["python"].unit_results[unit.unit_id]
        nb_result = results["numba"].unit_results[unit.unit_id]
        assert py_result.hours_run == nb_result.hours_run, unit.unit_id
        assert np.array_equal(py_result.row_particle_index, nb_result.row_particle_index)

        displacement = np.hypot(
            py_result.final_x_m - py_result.trajectory_x_m[0, :],
            py_result.final_y_m - py_result.trajectory_y_m[0, :],
        )
        position_diff = np.hypot(
            py_result.final_x_m - nb_result.final_x_m, py_result.final_y_m - nb_result.final_y_m
        )
        assert np.all(position_diff < 5.0e-2 * np.maximum(displacement, 1.0)), (
            unit.unit_id, position_diff.max(), displacement.max(),
        )

        # 終止狀態碼（離開關注區域／回溯上限／數值失敗等）幾乎必須完全一致；
        # 只允許極少數（因 ulp 差異剛好落在邊界判定附近）不一致。
        status_mismatch = int(np.count_nonzero(py_result.final_status != nb_result.final_status))
        assert status_mismatch <= max(1, int(0.05 * py_result.final_status.shape[0])), (
            unit.unit_id, status_mismatch,
        )


# ══════════════════════════════════════════════════════════════════════════
# WP-28：HourMajorConfig.bed_no_deposition——把 WP-26 的敏感度案例接上小時
# 主軸合併執行路徑（正式 native-plan／native-run 實際使用的路徑）
# ══════════════════════════════════════════════════════════════════════════


def test_hour_major_bed_no_deposition_matches_standalone_and_records_events(
    driver_grid, handles
) -> None:
    """``HourMajorConfig.bed_no_deposition=True`` 在合併執行路徑必須與獨立呼叫
    ``run_backward_batch`` 位元相同，且真的關閉沉積終止判定（不是宣稱接了線、
    實際上合併路徑仍然照樣沉底）。

    情境與 ``tests/test_native_backward_driver.py::
    test_bed_no_deposition_default_bit_identical_and_sensitivity_case_never_deposits``
    相同（單一粒子、零水平流速、起點已貼近海床），但這裡刻意走
    ``run_flow_domain_hour_major`` 這條正式生產路徑會用到的合併執行路徑，
    而不是直接呼叫 ``run_backward_batch``——見 ``hour_major.py``
    ``_post_hour_checks`` 的 ``bed_no_deposition`` 分支（WP-28 新增，之前
    這個分支在合併路徑完全不存在，設定這個旗標不會有任何效果）。
    """

    arrival_utc = 6_000 * _HOUR_NS
    bed_z_m = -_DP_M

    resolver = _make_resolver(driver_grid, u=0.0, v=0.0)
    interest_regions = {"S": _BIG_POLYGON}

    x0 = np.array([_DOMAIN_X / 2.0])
    y0 = np.array([_DOMAIN_Y / 2.0])
    z0 = np.array([bed_z_m + 0.1])
    row_ids = np.arange(1, dtype=np.int64)
    table = _FakeIdentityTable(x0_m=x0, y0_m=y0, z0_m=z0, seed_hex=_seed_hex_for(row_ids))

    unit = _make_unit(
        unit_id="U", region="R", domain="D", site="S", arrival_utc=arrival_utc,
        settling=-0.00001, rows=row_ids,
    )

    hm_config = hm.HourMajorConfig(
        max_backtrack_seconds=5.0 * 3600.0, dt_max_seconds=30.0, enable_diffusion=False,
        deposition_consecutive_hours=3, deposition_tolerance_m=0.5, master_seed=11,
        bed_no_deposition=True,
    )
    batch = hm.build_hour_major_domain_batch(
        table, [unit], interest_regions=interest_regions, **_flow_only_kwargs(interest_regions),
    )
    merged_result_holder = hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=hm_config, forcing_resolver=resolver,
    )
    merged_result = merged_result_holder.unit_results["U"]

    standalone_result = _run_unit_standalone(
        driver_grid, handles, table=table, unit=unit, hm_config=hm_config, resolver=resolver,
    )
    _assert_results_bit_identical(merged_result, standalone_result, label="bed_no_deposition")

    # 真的關閉了沉積終止判定：最終狀態是回溯達上限，不是 DEPOSITED；
    # departures 完全沒有 DEPOSITED；有記到一筆不沉積事件。
    assert merged_result.final_status[0] == bd.NATIVE_STATUS_MAX_BACKTRACK
    assert all(d.status != bd.NATIVE_STATUS_DEPOSITED for d in merged_result.departures)
    assert len(merged_result.bed_no_deposition_events) == 1
    assert merged_result.bed_no_deposition_events[0].utc_ns == arrival_utc - 3 * _HOUR_NS

    # 對照組：省略 bed_no_deposition（預設 False）時，同一個合併路徑必須是
    # 既有基線行為（第 3 小時沉底），證明這個旗標真的是「開關」而不是恆定
    # 生效或恆定失效。
    baseline_config = dataclasses.replace(hm_config, bed_no_deposition=False)
    baseline_batch = hm.build_hour_major_domain_batch(
        table, [unit], interest_regions=interest_regions, **_flow_only_kwargs(interest_regions),
    )
    baseline_result = hm.run_flow_domain_hour_major(
        driver_grid, handles, baseline_batch, config=baseline_config, forcing_resolver=resolver,
    ).unit_results["U"]
    assert baseline_result.final_status[0] == bd.NATIVE_STATUS_DEPOSITED
    assert baseline_result.bed_no_deposition_events == ()


# ══════════════════════════════════════════════════════════════════════════
# WP-17 任務 2：native_runner.horizon_derivation——由既有較長結果截斷推導較短結果
# ══════════════════════════════════════════════════════════════════════════


def test_derive_horizon_matches_direct_run_bit_identical_under_global_cfl(driver_grid, handles) -> None:
    """H_src=6h 截斷推導出的 H_target=3h，必須與直接跑 H_target=3h 位元相同。

    只有 ``substep_policy="global_cfl"`` 才保證這個等價（``local_cfl`` 的子步數
    依當下整批哪些粒子仍在活動決定，只保證統計等價，見模組文件與
    ``hour_major.py`` 既有政策，不是本測試新增的限制）。用 SMALL／BIG 兩個測站
    分別涵蓋「截止前已終止（保留原狀）」與「截止時仍活動（改標回溯上限）」兩種
    分支。
    """

    t0 = 22_000 * _HOUR_NS
    n = 3
    rng = np.random.default_rng(7)
    x0 = _DOMAIN_X / 2.0 + rng.uniform(-30.0, 30.0, size=2 * n)
    y0 = _DOMAIN_Y / 2.0 + rng.uniform(-30.0, 30.0, size=2 * n)
    z0 = np.full(2 * n, -5.0)
    row_ids = np.arange(2 * n, dtype=np.int64)
    table = _FakeIdentityTable(x0_m=x0, y0_m=y0, z0_m=z0, seed_hex=_seed_hex_for(row_ids))

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

    source_config = hm.HourMajorConfig(
        max_backtrack_seconds=6.0 * 3600.0, enable_diffusion=False, master_seed=1,
        substep_policy="global_cfl",
    )
    source_batch = hm.build_hour_major_domain_batch(
        table, [unit_small, unit_big], interest_regions=interest_regions,
        **_flow_only_kwargs(interest_regions),
    )
    source_result = hm.run_flow_domain_hour_major(
        driver_grid, handles, source_batch, config=source_config, forcing_resolver=resolver,
    )
    # 確認場景確實同時涵蓋兩種分支：SMALL 在 3 小時前就已經離開，BIG 撐到 6 小時上限。
    assert np.all(source_result.unit_results["SMALL_UNIT"].final_utc_ns > t0 - 3 * _HOUR_NS)
    assert np.all(source_result.unit_results["BIG_UNIT"].final_status == bd.NATIVE_STATUS_MAX_BACKTRACK)

    resolved = ResolvedUnitHorizon(
        backtrack_mode="full_horizon_from_deposition",
        requested_horizon_days=3 / 24,
        effective_horizon_seconds=3.0 * 3600.0,
        effective_horizon_ns=3 * _HOUR_NS,
        pre_window_deposition=False,
        bed_residence_age_hours=None,
    )

    target_config = hm.HourMajorConfig(
        max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=False, master_seed=1,
        substep_policy="global_cfl",
    )
    unit_small_direct = dataclasses.replace(unit_small, resolved_max_backtrack_seconds=3.0 * 3600.0)
    unit_big_direct = dataclasses.replace(unit_big, resolved_max_backtrack_seconds=3.0 * 3600.0)
    direct_batch = hm.build_hour_major_domain_batch(
        table, [unit_small_direct, unit_big_direct], interest_regions=interest_regions,
        **_flow_only_kwargs(interest_regions),
    )
    direct_result = hm.run_flow_domain_hour_major(
        driver_grid, handles, direct_batch, config=target_config, forcing_resolver=resolver,
    )

    for unit_id in ("SMALL_UNIT", "BIG_UNIT"):
        derived, info = hd.derive_backward_driver_result(
            source_result.unit_results[unit_id],
            resolved=resolved,
            source_requested_horizon_days=6 / 24,
        )
        _assert_results_bit_identical(
            derived, direct_result.unit_results[unit_id], label=f"derive-vs-direct:{unit_id}"
        )
        assert info.hours_run == direct_result.unit_results[unit_id].hours_run
    # BIG 三顆粒子全部原本都撐到 6 小時上限，在 3 小時截止時仍活動，因此全部
    # 應該被重新標成「回溯達上限」；SMALL 則應該全部維持原狀（提早離開）。
    small_info = hd.derive_backward_driver_result(
        source_result.unit_results["SMALL_UNIT"], resolved=resolved, source_requested_horizon_days=6 / 24,
    )[1]
    big_info = hd.derive_backward_driver_result(
        source_result.unit_results["BIG_UNIT"], resolved=resolved, source_requested_horizon_days=6 / 24,
    )[1]
    assert small_info.newly_capped_particle_count == 0
    assert small_info.unchanged_particle_count == n
    assert big_info.newly_capped_particle_count == n
    assert big_info.unchanged_particle_count == 0


def test_derive_horizon_pre_window_shortcut(driver_grid, handles) -> None:
    """``pre_window_deposition=True`` 時完全不截斷計算，直接回到沉底起點。"""

    del driver_grid, handles
    t0 = 22_000 * _HOUR_NS
    n = 2
    row_ids = np.arange(n, dtype=np.int64)
    x0 = np.array([10.0, 20.0])
    y0 = np.array([-5.0, -6.0])
    z0 = np.array([-3.0, -4.0])
    trajectory_utc = np.array([t0, t0 - _HOUR_NS, t0 - 2 * _HOUR_NS], dtype=np.int64)
    trajectory_x = np.stack([x0, x0 + 1.0, x0 + 2.0])
    trajectory_y = np.stack([y0, y0 - 1.0, y0 - 2.0])
    trajectory_z = np.stack([z0, z0, z0])
    result = bd.BackwardDriverResult(
        row_particle_index=row_ids,
        final_status=np.full(n, bd.NATIVE_STATUS_MAX_BACKTRACK, dtype=np.int32),
        final_x_m=x0 + 2.0, final_y_m=y0 - 2.0, final_z_m=z0,
        final_utc_ns=np.full(n, t0 - 2 * _HOUR_NS, dtype=np.int64),
        trajectory_utc_ns=trajectory_utc,
        trajectory_x_m=trajectory_x, trajectory_y_m=trajectory_y, trajectory_z_m=trajectory_z,
        trajectory_status=np.full((3, n), bd.NATIVE_STATUS_ACTIVE, dtype=np.int32),
        departures=(), diagnostics=bd.SubstepDiagnostics(),
        overhead_diagnostics=bd.HourlyOverheadDiagnostics(),
        hours_run=2,
    )
    resolved = ResolvedUnitHorizon(
        backtrack_mode="fixed_calendar_window",
        requested_horizon_days=30.0,
        effective_horizon_seconds=0.0,
        effective_horizon_ns=0,
        pre_window_deposition=True,
        bed_residence_age_hours=45 * 24,
    )
    derived, info = hd.derive_backward_driver_result(
        result, resolved=resolved, source_requested_horizon_days=30.0
    )
    assert np.all(derived.final_status == bd.NATIVE_STATUS_PRE_WINDOW_DEPOSITION)
    assert np.array_equal(derived.final_x_m, x0)
    assert np.array_equal(derived.final_y_m, y0)
    assert derived.hours_run == 0
    assert info.pre_window_deposition is True
    assert info.cutoff_utc_ns == t0


def test_derive_horizon_rejects_target_longer_than_source() -> None:
    resolved = ResolvedUnitHorizon(
        backtrack_mode="full_horizon_from_deposition",
        requested_horizon_days=10.0,
        effective_horizon_seconds=10 * 86_400.0,
        effective_horizon_ns=10 * 86_400 * 1_000_000_000,
        pre_window_deposition=False,
        bed_residence_age_hours=None,
    )
    fake_result = bd.BackwardDriverResult(
        row_particle_index=np.array([0], dtype=np.int64),
        final_status=np.array([bd.NATIVE_STATUS_MAX_BACKTRACK], dtype=np.int32),
        final_x_m=np.array([0.0]), final_y_m=np.array([0.0]), final_z_m=np.array([0.0]),
        final_utc_ns=np.array([0], dtype=np.int64),
        trajectory_utc_ns=np.array([_HOUR_NS], dtype=np.int64),
        trajectory_x_m=np.zeros((1, 1)), trajectory_y_m=np.zeros((1, 1)), trajectory_z_m=np.zeros((1, 1)),
        trajectory_status=np.zeros((1, 1), dtype=np.int32),
        departures=(), diagnostics=bd.SubstepDiagnostics(),
        overhead_diagnostics=bd.HourlyOverheadDiagnostics(),
        hours_run=0,
    )
    with pytest.raises(hd.HorizonDerivationError):
        hd.derive_backward_driver_result(fake_result, resolved=resolved, source_requested_horizon_days=5.0)


# ── 子步數上限守門（WP-51 run05：重建流場發散使 ndeltp 爆量、單小時成本呈指數成長）──────


def test_substep_ceiling_fail_closed() -> None:
    import pytest

    from lagrangian_backtracking.native_runner.hour_major import (
        MAX_SUBSTEPS_PER_HOUR,
        HourMajorError,
        _check_substep_ceiling,
    )

    utc = 1_763_679_600_000_000_000  # 2025-11-20T23:00:00Z
    _check_substep_ceiling(10_363, utc)  # run05 實測的正常上界附近，不得觸發
    _check_substep_ceiling(MAX_SUBSTEPS_PER_HOUR, utc)
    with pytest.raises(HourMajorError, match="2025-11-20T23:00:00Z"):
        _check_substep_ceiling(MAX_SUBSTEPS_PER_HOUR + 1, utc)
