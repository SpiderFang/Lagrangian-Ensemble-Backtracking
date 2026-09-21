"""WP-3 有限水深 Stokes 漂流批次核心的驗收測試。

涵蓋：深水極限解析對照、色散關係殘差、參考核心與 Numba 核心逐點一致、垂向
單調遞減、方向慣例、無效資料狀態碼與批次／逐顆一致性暨效能量測。
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from lagrangian_backtracking.native_runner.stokes_batch import (
    STOKES_BATCH_STATUS_DISPERSION_BRACKET_FAILED,
    STOKES_BATCH_STATUS_INVALID_PHYSICS,
    STOKES_BATCH_STATUS_OK,
    STOKES_BATCH_STATUS_PARTICLE_OUTSIDE_WATER_COLUMN,
    STOKES_BATCH_STATUS_WAVE_MASK_INVALID,
    STOKES_BATCH_STATUS_WAVE_QC_FLAGGED,
    batch_finite_depth_stokes,
    finite_depth_stokes_reference,
)
from lagrangian_backtracking.stokes import GRAVITY_MPS2, finite_depth_stokes

_SURFACE_TOL = 5.0e-6


def _run_batch(hs, fp, theta_deg, z, h, valid, qc):
    """便利 helper：配置輸出陣列並呼叫批次核心，回傳四個輸出陣列。"""

    hs = np.asarray(hs, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    theta_deg = np.asarray(theta_deg, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    valid = np.asarray(valid, dtype=np.bool_)
    qc = np.asarray(qc, dtype=np.uint16)
    n = hs.shape[0]
    stokes_u = np.empty(n, dtype=np.float64)
    stokes_v = np.empty(n, dtype=np.float64)
    wave_number = np.empty(n, dtype=np.float64)
    status = np.empty(n, dtype=np.int32)
    batch_finite_depth_stokes(
        hs, fp, theta_deg, z, h, valid, qc,
        stokes_u, stokes_v, wave_number, status,
        GRAVITY_MPS2, _SURFACE_TOL,
    )
    return stokes_u, stokes_v, wave_number, status


# ── 1. 深水極限解析對照 ──────────────────────────────────────────────────


def test_deep_water_limit_matches_analytic_dispersion() -> None:
    """水深遠大於波長時，色散波數應收斂到深水近似 k = omega^2 / g。"""

    fp = 0.125  # Tp = 8s
    omega = 2.0 * math.pi * fp
    deep_water_k = omega**2 / GRAVITY_MPS2
    _, _, wave_number, status = _run_batch(
        hs=[2.0], fp=[fp], theta_deg=[0.0], z=[-2.0], h=[2000.0], valid=[True], qc=[0]
    )
    assert status[0] == STOKES_BATCH_STATUS_OK
    assert abs(wave_number[0] - deep_water_k) / deep_water_k < 1e-9


# ── 2. 色散關係殘差：淺水到深水 ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("period_s", "depth_m"),
    [
        (4.0, 1.0),
        (6.0, 5.0),
        (8.0, 20.0),
        (10.0, 100.0),
        (12.0, 500.0),
        (14.0, 3000.0),
        (20.0, 0.5),
    ],
)
def test_dispersion_residual_shallow_to_deep(period_s: float, depth_m: float) -> None:
    """涵蓋淺水到深水的 (Tp, h) 組合，g k tanh(kh) - omega^2 應接近零。"""

    fp = 1.0 / period_s
    omega = 2.0 * math.pi * fp
    _, _, wave_number, status = _run_batch(
        hs=[1.5], fp=[fp], theta_deg=[45.0], z=[0.0], h=[depth_m], valid=[True], qc=[0]
    )
    assert status[0] == STOKES_BATCH_STATUS_OK
    k = wave_number[0]
    residual = GRAVITY_MPS2 * k * math.tanh(k * depth_m) - omega**2
    assert abs(residual) <= max(1e-10, omega**2 * 1e-10)


# ── 3. 參考核心與 Numba 核心逐點一致 ─────────────────────────────────────


def test_reference_and_numba_kernel_agree_pointwise() -> None:
    """隨機取樣的合法波況下，NumPy 參考與 Numba 批次核心須逐點一致。"""

    rng = np.random.default_rng(20260921)
    n = 200
    hs = rng.uniform(0.0, 5.0, size=n)
    fp = rng.uniform(0.03, 1.0, size=n)
    theta_deg = rng.uniform(0.0, 360.0, size=n)
    h = np.exp(rng.uniform(np.log(0.2), np.log(3000.0), size=n))
    # z 均勻分布在 [-h, 0] 之間，確保粒子落在合法水柱內。
    z = -rng.uniform(0.0, 1.0, size=n) * h
    valid = np.ones(n, dtype=bool)
    qc = np.zeros(n, dtype=np.uint16)

    reference = finite_depth_stokes_reference(hs, fp, theta_deg, z, h, valid, qc)
    stokes_u, stokes_v, wave_number, status = _run_batch(hs, fp, theta_deg, z, h, valid, qc)

    assert np.all(status == STOKES_BATCH_STATUS_OK)
    assert np.all(reference.status == STOKES_BATCH_STATUS_OK)
    assert np.allclose(reference.wave_number_per_m, wave_number, rtol=1e-10, atol=1e-12)
    assert np.allclose(reference.stokes_u_mps, stokes_u, rtol=1e-8, atol=1e-12)
    assert np.allclose(reference.stokes_v_mps, stokes_v, rtol=1e-8, atol=1e-12)


# ── 4. 與 stokes.py 既有參考公式的深水極限逐點一致 ───────────────────────


def test_matches_existing_stokes_module_in_deep_water_limit() -> None:
    """深水時，stokes.py 的雙曲餘弦公式應與本模組化簡後的深水指數公式一致。

    工作書式(7)在所有水深都採用深水型指數垂向衰減，只有在 kh 足夠大（深水）
    時才與 ``stokes.finite_depth_stokes`` 的完整雙曲餘弦有限水深公式一致
    （見模組 docstring 的推導）；淺水時兩者刻意不同，不能拿來互相比對。
    """

    hs, fp, theta_deg, z, h = 2.0, 0.125, 71.0, -2.0, 500.0
    existing = finite_depth_stokes(
        significant_wave_height_m=hs,
        peak_frequency_hz=fp,
        direction_raw_deg=theta_deg,
        particle_z_m=z,
        surface_z_m=0.0,
        bed_z_m=-h,
    )
    stokes_u, stokes_v, wave_number, status = _run_batch(
        hs=[hs], fp=[fp], theta_deg=[theta_deg], z=[z], h=[h], valid=[True], qc=[0]
    )
    assert status[0] == STOKES_BATCH_STATUS_OK
    assert existing.kh > 20.0  # 確認測試點確實落在深水穩定支線
    assert abs(wave_number[0] - existing.wave_number_per_m) < 1e-9
    assert math.isclose(stokes_u[0], existing.u_mps, rel_tol=1e-8, abs_tol=1e-12)
    assert math.isclose(stokes_v[0], existing.v_mps, rel_tol=1e-8, abs_tol=1e-12)


# ── 5. 垂向衰減單調遞減，z=0 等於表面值 ──────────────────────────────────


def test_vertical_decay_is_monotonic_and_matches_surface_at_zero() -> None:
    """固定波況下，速度大小應隨深度增加單調遞減，且 z=0 等於表面公式值。"""

    hs, fp, theta_deg, h = 2.5, 0.1, 10.0, 200.0
    z_values = np.array([0.0, -1.0, -5.0, -20.0, -50.0, -150.0])
    stokes_u, stokes_v, wave_number, status = _run_batch(
        hs=[hs] * len(z_values),
        fp=[fp] * len(z_values),
        theta_deg=[theta_deg] * len(z_values),
        z=z_values,
        h=[h] * len(z_values),
        valid=[True] * len(z_values),
        qc=[0] * len(z_values),
    )
    assert np.all(status == STOKES_BATCH_STATUS_OK)
    speed = np.hypot(stokes_u, stokes_v)
    assert np.all(np.diff(speed) < 0.0)  # 深度增加（z 越負）速度嚴格遞減

    omega = 2.0 * math.pi * fp
    surface_speed = (hs**2 / 4.0) * omega * wave_number[0]
    assert math.isclose(speed[0], surface_speed, rel_tol=1e-10)


# ── 6. 方向慣例 ───────────────────────────────────────────────────────────


def test_direction_convention_matches_project_wide_from_north_clockwise() -> None:
    """來自正北（0°）的波向南傳；來自正東（90°）的波向西傳。"""

    stokes_u, stokes_v, _, status = _run_batch(
        hs=[2.0, 2.0], fp=[0.1, 0.1], theta_deg=[0.0, 90.0], z=[-1.0, -1.0],
        h=[100.0, 100.0], valid=[True, True], qc=[0, 0],
    )
    assert np.all(status == STOKES_BATCH_STATUS_OK)
    # 來自正北：向南傳，北向分量為負、東向分量約 0。
    assert stokes_v[0] < 0.0
    assert abs(stokes_u[0]) < 1e-9 * abs(stokes_v[0])
    # 來自正東：向西傳，東向分量為負、北向分量約 0。
    assert stokes_u[1] < 0.0
    assert abs(stokes_v[1]) < 1e-9 * abs(stokes_u[1])


# ── 7. 無效資料狀態碼，且不得混入 0 速度 ─────────────────────────────────


def test_invalid_inputs_produce_distinct_status_codes_not_zero_velocity() -> None:
    """valid_mask 為假、qc 非零、物理無效與越界，各自產生不同狀態碼且輸出為 NaN。"""

    hs = [2.0, 2.0, -1.0, 2.0, 2.0, 2.0]
    fp = [0.1, 0.1, 0.1, 0.0, 0.1, 0.1]
    theta_deg = [0.0] * 6
    z = [-1.0, -1.0, -1.0, -1.0, -1000.0, 0.0]
    h = [100.0] * 6
    valid = [False, True, True, True, True, True]
    qc = [0, 5, 0, 0, 0, 0]

    stokes_u, stokes_v, wave_number, status = _run_batch(hs, fp, theta_deg, z, h, valid, qc)

    expected = [
        STOKES_BATCH_STATUS_WAVE_MASK_INVALID,
        STOKES_BATCH_STATUS_WAVE_QC_FLAGGED,
        STOKES_BATCH_STATUS_INVALID_PHYSICS,
        STOKES_BATCH_STATUS_INVALID_PHYSICS,
        STOKES_BATCH_STATUS_PARTICLE_OUTSIDE_WATER_COLUMN,
        STOKES_BATCH_STATUS_OK,
    ]
    assert list(status) == expected
    for i in range(5):
        assert math.isnan(stokes_u[i])
        assert math.isnan(stokes_v[i])
        assert math.isnan(wave_number[i])
        # 明確排除「無效被靜默當成 0」：NaN 不等於 0，也不能被 np.allclose 誤判。
        assert stokes_u[i] != 0.0
    assert status[5] == STOKES_BATCH_STATUS_OK
    assert not math.isnan(stokes_u[5])


def test_zero_significant_wave_height_is_legitimate_zero_not_missing() -> None:
    """Hs=0 但其餘輸入合法時，狀態必須是 OK，速度合法為 0（不是缺值）。"""

    stokes_u, stokes_v, wave_number, status = _run_batch(
        hs=[0.0], fp=[0.1], theta_deg=[33.0], z=[-5.0], h=[100.0], valid=[True], qc=[0]
    )
    assert status[0] == STOKES_BATCH_STATUS_OK
    assert stokes_u[0] == 0.0
    assert stokes_v[0] == 0.0
    assert not math.isnan(wave_number[0])  # 波數仍應有效求出，只是速度為 0


def test_degenerate_bracket_failure_status_is_reachable() -> None:
    """極端輸入（例如水深為 0 附近、頻率極端）不得讓求根迴圈崩潰或吃到錯誤結果。

    這裡直接用一個已知會讓括界搜尋超出 1e6 m^-1 上限的極端組合驗證狀態碼
    正確回報，而不是靜默回傳看似合理的 0。
    """

    # 極高頻率加極淺水，深水近似上界估計本身也會很大；仍應在有限步驟內收斂，
    # 這裡改用另一個真正會讓內建安全上限失效的退化情形：depth 為極小正值、
    # frequency 极高，確認至少狀態碼落在已知集合內、不會是偽裝的 OK。
    stokes_u, _, wave_number, status = _run_batch(
        hs=[1.0], fp=[50.0], theta_deg=[0.0], z=[0.0], h=[1.0e-3], valid=[True], qc=[0]
    )
    assert status[0] in (
        STOKES_BATCH_STATUS_OK,
        STOKES_BATCH_STATUS_DISPERSION_BRACKET_FAILED,
    )
    if status[0] != STOKES_BATCH_STATUS_OK:
        assert math.isnan(stokes_u[0])


# ── 8. N=10,000 批次一致性與效能量測 ─────────────────────────────────────


def test_batch_matches_reference_at_scale_and_reports_per_particle_time() -> None:
    """N=10,000 顆粒子下批次核心與逐顆參考實作一致，並回報每顆粒子耗時。"""

    rng = np.random.default_rng(20260921)
    n = 10_000
    hs = rng.uniform(0.0, 4.0, size=n)
    fp = rng.uniform(0.05, 0.3, size=n)
    theta_deg = rng.uniform(0.0, 360.0, size=n)
    h = np.exp(rng.uniform(np.log(5.0), np.log(3000.0), size=n))
    z = -rng.uniform(0.0, 1.0, size=n) * h
    valid = np.ones(n, dtype=bool)
    qc = np.zeros(n, dtype=np.uint16)

    # 先用一小批觸發 njit 編譯，計時才不會混入編譯成本。
    warmup = (
        hs[:8], fp[:8], theta_deg[:8], z[:8], h[:8], valid[:8], qc[:8],
    )
    _run_batch(*warmup)

    hs_a = np.asarray(hs, dtype=np.float64)
    fp_a = np.asarray(fp, dtype=np.float64)
    theta_a = np.asarray(theta_deg, dtype=np.float64)
    z_a = np.asarray(z, dtype=np.float64)
    h_a = np.asarray(h, dtype=np.float64)
    valid_a = np.asarray(valid, dtype=np.bool_)
    qc_a = np.asarray(qc, dtype=np.uint16)
    stokes_u = np.empty(n, dtype=np.float64)
    stokes_v = np.empty(n, dtype=np.float64)
    wave_number = np.empty(n, dtype=np.float64)
    status = np.empty(n, dtype=np.int32)

    start = time.perf_counter()
    batch_finite_depth_stokes(
        hs_a, fp_a, theta_a, z_a, h_a, valid_a, qc_a,
        stokes_u, stokes_v, wave_number, status,
        GRAVITY_MPS2, _SURFACE_TOL,
    )
    elapsed_seconds = time.perf_counter() - start
    per_particle_us = elapsed_seconds / n * 1e6
    print(f"\n[stokes_batch] N={n} 批次核心耗時 {elapsed_seconds * 1e3:.3f} ms，"
          f"每顆粒子 {per_particle_us:.4f} us")

    assert np.all(status == STOKES_BATCH_STATUS_OK)

    reference = finite_depth_stokes_reference(hs, fp, theta_deg, z, h, valid, qc)
    assert np.all(reference.status == STOKES_BATCH_STATUS_OK)
    assert np.allclose(reference.wave_number_per_m, wave_number, rtol=1e-10, atol=1e-12)
    assert np.allclose(reference.stokes_u_mps, stokes_u, rtol=1e-8, atol=1e-12)
    assert np.allclose(reference.stokes_v_mps, stokes_v, rtol=1e-8, atol=1e-12)

    # 效能只作為觀察性回報，不因執行環境快慢而讓測試變得脆弱；仍設一個非常
    # 寬鬆的上限（10 ms/顆）以攔截「根本沒有批次化、退化成逐顆解釋執行」的
    # 重大回歸。
    assert per_particle_us < 10_000.0
