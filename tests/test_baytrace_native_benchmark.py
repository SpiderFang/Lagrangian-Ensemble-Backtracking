"""Q2 合成 benchmark 的資料對齊、比較契約與 evidence 保護測試。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK_PATH = _REPO_ROOT / "scripts" / "benchmark_baytrace_native_cpu.py"
_DEFAULT_BAYTRACE_ROOT = Path(
    "/Users/mustlab/Workspace/Lagrangian-Ensemble-Backtracking/BayTrace/Baytech_BayTrace_v4.7.2"
)


def _benchmark_module():
    """以檔案路徑載入 benchmark，避免把 scripts 目錄變成正式套件 API。"""

    spec = importlib.util.spec_from_file_location("q2_benchmark_under_test", _BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"無法載入 benchmark script: {_BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    # dataclass 需要在裝載期間由 sys.modules 取得 module namespace；這裡仍維持
    # scripts 非套件的 lazy import，不讓測試偷偷改變正式 import path。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _baytrace_root_or_skip() -> Path:
    """只在測試環境明確提供 PI artifact 時執行實源對齊測試。"""

    root = Path(os.environ.get("LBT_BAYTRACE_ROOT", _DEFAULT_BAYTRACE_ROOT))
    if not root.is_dir():
        pytest.skip("未提供 PI BayTrace artifact；略過需要原始 Numba dispatcher 的對齊測試")
    return root


def test_synthetic_ocm_axes_and_nonuniform_sampling_match_baytrace(tmp_path: Path) -> None:
    """逐節點／逐層欄位與非均勻場取樣必須和原始 BayTrace k1 一致。"""

    benchmark = _benchmark_module()
    root = _baytrace_root_or_skip()
    bridge, kernels = benchmark._load_bridge(root, tmp_path / "numba-cache")
    case = benchmark._build_case(bridge, kernels, 4, 0.25)

    # 這些 assert 直接鎖住曾造成假等價的軸向錯置：BayTrace 是 (layer,node)，
    # OCMNativeMonth 是 (time,node,layer,component)，且 u/v/w 必須逐欄對齊。
    np.testing.assert_array_equal(case.month.hvel[0, :, :, 0], case.fields.u1.T)
    np.testing.assert_array_equal(case.month.hvel[0, :, :, 1], case.fields.v1.T)
    np.testing.assert_array_equal(case.month.vertical_velocity[0], case.fields.w1.T)
    np.testing.assert_array_equal(case.month.hvel[0], case.month.hvel[1])
    np.testing.assert_array_equal(case.month.vertical_velocity[0], case.month.vertical_velocity[1])
    assert np.ptp(case.fields.u1[0]) > 0.0
    assert np.ptp(case.fields.v1[0]) > 0.0
    assert np.ptp(case.fields.w1[0]) > 0.0

    # 使用原始 batch k1 與同一 initial position 逐粒子比對；不以均勻流場掩蓋
    # 節點插值／mesh locate 的錯誤，也不把 production lifecycle 放進此等價檢查。
    particles = case.new_particles()
    benchmark._advance_raw(case, particles, 1, 1)
    raw_k1 = np.column_stack((particles.u, particles.v, particles.w))
    lbt_k1 = np.asarray(
        [
            (
                sample.u_mps,
                sample.v_mps,
                sample.w_mps,
            )
            for x, y, z in zip(case.x, case.y, case.z, strict=True)
            for sample in [case.month.sample(float(x), float(y), float(z), benchmark.INITIAL_TIME_NS)]
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(raw_k1, lbt_k1, rtol=0.0, atol=1.0e-12)


def test_benchmark_rejects_existing_evidence_without_loading_source(tmp_path: Path) -> None:
    """既有 JSON 不可被 benchmark CLI 靜默覆寫。"""

    benchmark = _benchmark_module()
    output = tmp_path / "existing.json"
    output.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="拒絕覆蓋"):
        benchmark.run_benchmark(
            baytrace_root=tmp_path / "missing-baytrace",
            output=output,
            particles=4,
            steps=2,
            repeats=1,
        )
    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_hint_and_k1_reuse_counts_are_measured_for_core_and_production(tmp_path: Path) -> None:
    """核心 direct RK4 應實測 4 次取樣；Production 額外查詢只能記錄實測值。"""

    benchmark = _benchmark_module()
    root = _baytrace_root_or_skip()
    bridge, kernels = benchmark._load_bridge(root, tmp_path / "numba-cache")
    case = benchmark._build_case(bridge, kernels, 2, 0.25)

    sources, providers = benchmark._new_lbt_providers(case)
    assert len(sources) == len(providers) == case.n
    assert all(provider.step_start_sample_reuse_safe is True for provider in providers)
    direct_metric, _ = benchmark._run_path(case, "lbt_numba_passive_rk4", -1, 1, 1)
    assert direct_metric["velocity_sample_calls"]["median"] == 4 * case.n
    assert direct_metric["triangle_hint_non_null_calls"]["median"] == 4 * case.n
    assert direct_metric["hint_enabled_provider"] is True

    production_metric, _ = benchmark._run_path(
        case,
        "lbt_production_batch_lifecycle",
        -1,
        1,
        1,
    )
    observed_calls = production_metric["velocity_sample_calls"]["median"]
    assert observed_calls >= 4 * case.n
    assert production_metric["velocity_sample_calls_per_particle_step"] == observed_calls / case.n
    assert production_metric["hint_enabled_provider"] is True


def test_benchmark_contract_keeps_stationary_tolerance_and_direction_scope() -> None:
    """固定數值容差與 ProductionBatch backward-only 邊界，避免測試放寬比較條件。"""

    benchmark = _benchmark_module()
    assert benchmark.TOLERANCE_M == 1.0e-9
    assert benchmark.TRAT == 0.5
    assert benchmark.DEFAULT_PARTICLES == 128
    assert benchmark.DEFAULT_STEPS == 16
    assert benchmark.DEFAULT_REPEATS == 3
