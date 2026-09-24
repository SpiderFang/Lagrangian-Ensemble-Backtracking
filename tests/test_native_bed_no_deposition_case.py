"""WP-26／WP-28：``native_runner.bed_no_deposition_case`` 的 CLI／環境變數解析
與 fail-closed 行為（介面與 ``physics_case.py``／``backtrack_horizon.py`` 同款，
見該模組文件）。"""

from __future__ import annotations

import pytest

from lagrangian_backtracking.native_runner import bed_no_deposition_case as bnd


def test_default_is_false_when_both_omitted() -> None:
    """CLI 旗標與環境變數都省略時，維持加入這個敏感度案例之前的既有行為
    （粒子連續停留海床即視為沉積並終止）。"""

    assert bnd.resolve_bed_no_deposition(cli_bed_no_deposition=None, env={}) is False
    assert bnd.DEFAULT_BED_NO_DEPOSITION is False


def test_cli_true_enables_when_env_omitted() -> None:
    assert bnd.resolve_bed_no_deposition(cli_bed_no_deposition=True, env={}) is True


def test_env_var_enables_when_cli_omitted() -> None:
    resolved = bnd.resolve_bed_no_deposition(
        cli_bed_no_deposition=None, env={bnd.BED_NO_DEPOSITION_ENV_VAR: "1"}
    )
    assert resolved is True


@pytest.mark.parametrize("falsy_value", ["0", "false", "False", ""])
def test_env_var_falsy_values_resolve_to_false(falsy_value: str) -> None:
    resolved = bnd.resolve_bed_no_deposition(
        cli_bed_no_deposition=None, env={bnd.BED_NO_DEPOSITION_ENV_VAR: falsy_value}
    )
    assert resolved is False


def test_cli_takes_precedence_when_both_agree() -> None:
    resolved = bnd.resolve_bed_no_deposition(
        cli_bed_no_deposition=True, env={bnd.BED_NO_DEPOSITION_ENV_VAR: "1"}
    )
    assert resolved is True


def test_cli_env_mismatch_fails_closed() -> None:
    with pytest.raises(bnd.BedNoDepositionCaseError):
        bnd.resolve_bed_no_deposition(
            cli_bed_no_deposition=True, env={bnd.BED_NO_DEPOSITION_ENV_VAR: "0"}
        )


def test_cli_false_and_env_true_also_fails_closed() -> None:
    """不一致的檢查是對稱的：CLI 明確為 ``False``、環境變數明確為 ``True`` 也要
    fail closed，不能只檢查「CLI 為 True」這一種方向。"""

    with pytest.raises(bnd.BedNoDepositionCaseError):
        bnd.resolve_bed_no_deposition(
            cli_bed_no_deposition=False, env={bnd.BED_NO_DEPOSITION_ENV_VAR: "1"}
        )
