"""WP-18：``native_runner.physics_case`` 的 CLI／環境變數解析與 fail-closed 行為。"""

from __future__ import annotations

import pytest

from lagrangian_backtracking.native_runner import physics_case as pc


def test_default_case_is_finite_depth_stokes_when_omitted() -> None:
    """省略 CLI 旗標與環境變數時，維持 WP-18 之前「永遠嘗試 Stokes」的既有行為。"""

    resolved = pc.resolve_physics_case(cli_experiment_case_id=None, env={})
    assert resolved.experiment_case_id == "finite_depth_stokes"
    assert resolved.include_stokes is True
    assert resolved.diffusion_method == pc.DIFFUSION_METHOD_DECLARATION


def test_no_stokes_case_resolves_include_stokes_false() -> None:
    resolved = pc.resolve_physics_case(cli_experiment_case_id="no_stokes", env={})
    assert resolved.experiment_case_id == "no_stokes"
    assert resolved.include_stokes is False
    assert resolved.diffusion_method == pc.DIFFUSION_METHOD_DECLARATION


def test_env_var_selects_case_when_cli_omitted() -> None:
    case_id = pc.resolve_experiment_case_id(
        cli_experiment_case_id=None, env={pc.EXPERIMENT_CASE_ENV_VAR: "no_stokes"}
    )
    assert case_id == "no_stokes"


def test_cli_takes_precedence_when_both_agree() -> None:
    case_id = pc.resolve_experiment_case_id(
        cli_experiment_case_id="no_stokes",
        env={pc.EXPERIMENT_CASE_ENV_VAR: "no_stokes"},
    )
    assert case_id == "no_stokes"


def test_cli_env_mismatch_fails_closed() -> None:
    with pytest.raises(pc.PhysicsCaseError):
        pc.resolve_experiment_case_id(
            cli_experiment_case_id="no_stokes",
            env={pc.EXPERIMENT_CASE_ENV_VAR: "finite_depth_stokes"},
        )


def test_unknown_cli_case_rejected() -> None:
    with pytest.raises(pc.PhysicsCaseError):
        pc.resolve_experiment_case_id(cli_experiment_case_id="not_a_real_case", env={})


def test_unknown_env_case_rejected() -> None:
    with pytest.raises(pc.PhysicsCaseError):
        pc.resolve_experiment_case_id(
            cli_experiment_case_id=None, env={pc.EXPERIMENT_CASE_ENV_VAR: "not_a_real_case"}
        )


def test_smagorinsky_sensitivity_case_still_includes_stokes() -> None:
    """``smagorinsky_cs_*`` 三個敏感度案例的 ``include_stokes`` 沿用
    ``runtime.EXPERIMENT_CASE_SPECS`` 唯一 registry，本模組不重複維護第二份清單。"""

    resolved = pc.resolve_physics_case(cli_experiment_case_id="smagorinsky_cs_010", env={})
    assert resolved.include_stokes is True
    # 擴散仍固定聲明為工作書式 (10)，與 EXPERIMENT_CASE_SPECS 的 coefficient_cs 無關
    # （native 路徑本來就只有一種擴散實作，見 physics_case 模組文件）。
    assert resolved.diffusion_method == pc.DIFFUSION_METHOD_DECLARATION
