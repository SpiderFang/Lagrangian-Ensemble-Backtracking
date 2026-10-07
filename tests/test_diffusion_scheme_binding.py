"""WP-51 整合：擴散組態綁定（checkpoint／計畫檔）與設定檔 Kz 唯一來源的測試。

涵蓋：

1. ``physics_case.resolve_diffusion_scheme``：預設值、設定檔 Kz 為唯一來源（環境變數只能補缺、
   不能與設定檔衝突）、無效值 fail closed、``as_declaration`` 與不一致比對。
2. ``hour_major`` checkpoint：meta 記錄擴散組態；以不同雜訊型態、垂向擴散、Kz 或 Visser 修正
   續跑一律明確報錯；缺簽章的舊 checkpoint 視為舊版高斯水平擴散（以新預設續跑會被擋下）；
   組態相同時續跑與未中斷執行位元相同。
3. 正式設定檔 YAML 的 ``constant_kz_m2ps`` 等於程式實際採用的預設值（設定＝實際），
   ``hour_major_provider`` 確實讀它。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

import test_native_hour_major as thm
from lagrangian_backtracking.config import load_config
from lagrangian_backtracking.native_runner import backward_driver as bd
from lagrangian_backtracking.native_runner import build_native_grid_arrays
from lagrangian_backtracking.native_runner import hour_major as hm
from lagrangian_backtracking.native_runner import hour_major_provider as hmp
from lagrangian_backtracking.native_runner import physics_case as pc

ROOT = Path(__file__).resolve().parents[1]
FORMAL_YAML = ROOT / "configs" / "lagrangian_backtracking.formal_h30_h60_h90_m10.yaml"


# ══════════════════════════════════════════════════════════════════════════
# 1. 解析函式
# ══════════════════════════════════════════════════════════════════════════


def test_resolver_defaults_match_library_and_provider_defaults() -> None:
    scheme = pc.resolve_diffusion_scheme(env={})
    assert (scheme.diffusion_noise, scheme.vertical_diffusion, scheme.visser_correction) == (
        "uniform",
        "constant",
        False,
    )
    assert scheme.kz_constant_m2ps == pc.DEFAULT_KZ_CONSTANT_M2PS == hm.HourMajorConfig().kz_constant_m2ps
    assert scheme.kz_source == "default"
    assert scheme.as_declaration() == {
        "diffusion_noise": "uniform",
        "vertical_diffusion": "constant",
        "kz_constant_m2ps": 3.0e-4,
        "visser_correction": False,
        "kz_source": "default",
    }


def test_config_value_is_the_single_source_of_kz() -> None:
    scheme = pc.resolve_diffusion_scheme(env={}, config_kz_constant_m2ps=7.0e-4)
    assert scheme.kz_constant_m2ps == 7.0e-4 and scheme.kz_source == "config"
    # 環境變數與設定檔相同不算衝突
    same = pc.resolve_diffusion_scheme(env={pc.KZ_CONSTANT_ENV_VAR: "0.0007"}, config_kz_constant_m2ps=7.0e-4)
    assert same.kz_constant_m2ps == 7.0e-4 and same.kz_source == "config"
    # 環境變數與設定檔不同：fail closed，不可悄悄蓋掉設定檔
    with pytest.raises(pc.PhysicsCaseError, match="設定檔是 Kz 的唯一來源"):
        pc.resolve_diffusion_scheme(env={pc.KZ_CONSTANT_ENV_VAR: "5e-4"}, config_kz_constant_m2ps=7.0e-4)
    # 設定檔沒有值時才讀環境變數
    only_env = pc.resolve_diffusion_scheme(env={pc.KZ_CONSTANT_ENV_VAR: "5e-4"})
    assert only_env.kz_constant_m2ps == 5.0e-4 and only_env.kz_source == "environment"


@pytest.mark.parametrize(
    "env",
    [
        {pc.DIFFUSION_NOISE_ENV_VAR: "laplace"},
        {pc.VERTICAL_DIFFUSION_ENV_VAR: "maybe"},
        {pc.KZ_CONSTANT_ENV_VAR: "abc"},
        {pc.KZ_CONSTANT_ENV_VAR: "-1"},
        {pc.KZ_CONSTANT_ENV_VAR: "nan"},
    ],
)
def test_resolver_rejects_invalid_values(env) -> None:
    with pytest.raises(pc.PhysicsCaseError):
        pc.resolve_diffusion_scheme(env=env)


def test_resolver_rejects_nonpositive_config_kz() -> None:
    with pytest.raises(pc.PhysicsCaseError, match="有限的正值"):
        pc.resolve_diffusion_scheme(env={}, config_kz_constant_m2ps=0.0)


def test_declaration_ignores_kz_when_vertical_diffusion_is_off_and_mismatch_listing() -> None:
    off = pc.resolve_diffusion_scheme(env={pc.VERTICAL_DIFFUSION_ENV_VAR: "off"}, config_kz_constant_m2ps=7.0e-4)
    assert off.as_declaration()["kz_constant_m2ps"] is None
    on = pc.resolve_diffusion_scheme(env={}, config_kz_constant_m2ps=7.0e-4).as_declaration()
    assert pc.diffusion_scheme_mismatches(on, dict(on)) == []
    changed = dict(on, kz_source="environment")  # 只供稽核，不算不一致
    assert pc.diffusion_scheme_mismatches(on, changed) == []
    problems = pc.diffusion_scheme_mismatches(on, off.as_declaration())
    assert any("vertical_diffusion" in item for item in problems)
    assert any("kz_constant_m2ps" in item for item in problems)


def test_config_kz_reader_handles_missing_null_and_bad_values() -> None:
    class _Config:
        def __init__(self, physics):
            self.physics = physics

    assert pc.config_kz_constant_m2ps(_Config({})) is None
    assert pc.config_kz_constant_m2ps(_Config({"vertical_diffusion": {"constant_kz_m2ps": None}})) is None
    assert pc.config_kz_constant_m2ps(_Config({"vertical_diffusion": {"constant_kz_m2ps": 2}})) == 2.0
    with pytest.raises(pc.PhysicsCaseError):
        pc.config_kz_constant_m2ps(_Config({"vertical_diffusion": {"constant_kz_m2ps": "0.1"}}))


# ══════════════════════════════════════════════════════════════════════════
# 2. checkpoint 綁定
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def handles() -> bd.BayTraceModuleHandles:
    root = os.environ.get("LBT_BAYTRACE_ROOT")
    if not root:
        pytest.skip("需要 LBT_BAYTRACE_ROOT 指向 BayTrace 原始碼根目錄才能執行本測試")
    return bd.load_baytrace_modules(root)


@pytest.fixture(scope="module")
def grid2(handles):
    mesh = thm._build_mesh()
    n_nodes = mesh.node_xy.shape[0]
    zcor = np.zeros((n_nodes, thm._NVRT), dtype=np.float64)
    zcor[:, 0] = -thm._DP_M
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor, eta_reference=np.zeros(n_nodes))
    built = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)
    assert built.ic3_cross_check_ok
    return built


def _config(**overrides) -> hm.HourMajorConfig:
    params = dict(max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=True, master_seed=42)
    params.update(overrides)
    return hm.HourMajorConfig(**params)


def _uniform(**overrides) -> hm.HourMajorConfig:
    params = dict(diffusion_noise="uniform", vertical_diffusion="constant", kz_constant_m2ps=3.0e-4)
    params.update(overrides)
    return _config(**params)


def _run(grid2, handles, config, checkpoint_dir=None, *, max_hours=None):
    table, units, interest_regions, _t0 = thm._build_staggered_domain()
    resolver = thm._make_resolver(grid2, u=0.02, v=-0.015, kh=1.5e-4)
    batch = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **thm._flow_only_kwargs(interest_regions)
    )
    return hm.run_flow_domain_hour_major(
        grid2,
        handles,
        batch,
        config=config,
        forcing_resolver=resolver,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_hours=1,
        _max_master_hours=max_hours,
    )


def test_checkpoint_meta_records_diffusion_scheme(grid2, handles, tmp_path) -> None:
    _run(grid2, handles, _uniform(kz_constant_m2ps=2.0e-3, visser_correction=True), tmp_path / "ck", max_hours=1)
    meta = json.loads((tmp_path / "ck" / "meta.json").read_text(encoding="utf-8"))
    assert meta["diffusion_scheme"] == {
        "diffusion_noise": "uniform",
        "vertical_diffusion": "constant",
        "kz_constant_m2ps": 2.0e-3,
        "visser_correction": True,
        "enable_diffusion": True,
    }


@pytest.mark.parametrize(
    ("written", "resumed", "expected_key"),
    [
        (_uniform(), _config(), "diffusion_noise"),  # 新預設的 checkpoint 以舊版高斯／無垂向擴散續跑
        (_config(), _uniform(), "diffusion_noise"),  # 舊版 checkpoint 以新預設續跑
        (_uniform(), _uniform(kz_constant_m2ps=1.0e-3), "kz_constant_m2ps"),
        (_uniform(), _uniform(visser_correction=True), "visser_correction"),
        (_uniform(), _uniform(vertical_diffusion="off"), "vertical_diffusion"),
        (_uniform(), _uniform(enable_diffusion=False, vertical_diffusion="off"), "vertical_diffusion"),
    ],
)
def test_resume_with_different_diffusion_scheme_is_rejected(
    grid2, handles, tmp_path, written, resumed, expected_key
) -> None:
    checkpoint = tmp_path / "ck"
    _run(grid2, handles, written, checkpoint, max_hours=1)
    with pytest.raises(hm.HourMajorError, match="拒絕續跑") as excinfo:
        _run(grid2, handles, resumed, checkpoint)
    assert expected_key in str(excinfo.value)


def test_legacy_checkpoint_without_signature_is_treated_as_old_gaussian_scheme(grid2, handles, tmp_path) -> None:
    checkpoint = tmp_path / "ck"
    _run(grid2, handles, _config(), checkpoint, max_hours=1)
    meta_path = checkpoint / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["diffusion_scheme"]  # 模擬本欄位新增之前寫出的 checkpoint
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(hm.HourMajorError, match="沒有擴散簽章"):
        _run(grid2, handles, _uniform(), checkpoint)
    # 以舊版（高斯、無垂向擴散）組態續跑仍可進行
    resumed = _run(grid2, handles, _config(), checkpoint)
    full = _run(grid2, handles, _config())
    assert set(resumed.unit_results) == set(full.unit_results)


def test_resume_with_same_new_scheme_is_bit_identical(grid2, handles, tmp_path) -> None:
    checkpoint = tmp_path / "ck"
    config = _uniform()
    full = _run(grid2, handles, config)
    _run(grid2, handles, config, checkpoint, max_hours=1)
    resumed = _run(grid2, handles, config, checkpoint)
    for unit_id in full.unit_results:
        thm._assert_results_bit_identical(
            full.unit_results[unit_id], resumed.unit_results[unit_id], label=unit_id
        )


# ══════════════════════════════════════════════════════════════════════════
# 3. 設定＝實際
# ══════════════════════════════════════════════════════════════════════════


def test_formal_yaml_kz_equals_actual_default() -> None:
    payload = yaml.safe_load(FORMAL_YAML.read_text(encoding="utf-8"))
    kz = payload["physics"]["vertical_diffusion"]["constant_kz_m2ps"]
    assert kz == pc.DEFAULT_KZ_CONSTANT_M2PS == hm.HourMajorConfig().kz_constant_m2ps


def _provider_env(config_path: Path) -> dict[str, str]:
    return {
        "LBT_NATIVE_CONFIG": str(config_path),
        "LBT_NATIVE_PLAN": "plan.json",
        "LBT_NATIVE_IDENTITY_TABLE": "identity.npz",
        "LBT_NATIVE_RESULTS_DIR": "results",
    }


def test_provider_reads_kz_from_config_and_rejects_conflicting_env(tmp_path) -> None:
    payload = yaml.safe_load(FORMAL_YAML.read_text(encoding="utf-8"))
    payload["physics"]["vertical_diffusion"]["constant_kz_m2ps"] = 7.0e-4
    config_path = tmp_path / "kz7.yaml"
    config_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    assert load_config(str(config_path), formal_release=False) is not None

    params = hmp._resolve_params(_provider_env(config_path))
    assert params.hour_major_config.kz_constant_m2ps == 7.0e-4  # 設定檔值，不是 3.0e-4 預設

    # 與設定檔不同的環境變數：啟動時 fail closed
    env = dict(_provider_env(config_path), LBT_NATIVE_KZ_CONSTANT_M2PS="3.0e-4")
    with pytest.raises(hmp.HourMajorProviderError, match="唯一來源"):
        hmp._resolve_params(env)

    # 正式 YAML 本身：實際採用的就是 YAML 的 3.0e-4
    formal = hmp._resolve_params(_provider_env(FORMAL_YAML))
    assert formal.hour_major_config.kz_constant_m2ps == 3.0e-4
    assert formal.hour_major_config.diffusion_noise == "uniform"
    assert formal.hour_major_config.vertical_diffusion == "constant"
