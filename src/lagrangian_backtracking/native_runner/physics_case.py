"""native 路徑的實驗案例（experiment case）明確化：「只有流」與「流＋波浪」。

## 背景

``runtime.EXPERIMENT_CASE_SPECS``（見該模組文件）是 runtime physics case 的唯一
registry，登錄 ``no_stokes``（不含 Stokes 漂移）、``finite_depth_stokes``（含有限
水深 Stokes 漂移）與三個 ``smagorinsky_cs_*`` 敏感度案例。舊引擎的 ``run-create``
（``cli.py`` 的 ``--experiment-case``）已經接到這份 registry；但 native 路徑
（``native-plan``／``native-run``）在 WP-18 之前**沒有任何案例選擇**，永遠只要
NWW3 資料存在就套用 Stokes，完全沒有「只有流」的開關（見
``reports/WP-18.md`` 的完整調查記錄）。

## 擴散與舊登錄表的差異（PI 已裁定，見 ``reports/WP-18.md``）

``EXPERIMENT_CASE_SPECS`` 的 ``diffusion_kind`` 欄位（``constant``／
``smagorinsky``）是**舊引擎**的擴散選擇；native 路徑從一開始就只有 Smagorinsky
一種擴散實作（工作書式 (10)，``horcon=0.2``，見
``native_runner.hour_major.HourMajorConfig``／``native_runner.smagorinsky_batch``），
沒有常數擴散（``constant_kh_m2ps``）的程式碼路徑。本模組解析出的實驗案例**只**
決定 ``include_stokes``（是否含 Stokes 漂移、是否讀取波浪資料），**不**採用
``ExperimentCaseSpec.diffusion_kind``／``coefficient_cs``——native 的 ``no_stokes``
與 ``finite_depth_stokes`` 兩案例都固定使用 Smagorinsky 工作書式 (10)，這是與舊
登錄表語意不同之處，呼叫端不得把 native 的 ``no_stokes``／``finite_depth_stokes``
結果與舊引擎同名案例（常數擴散）混用或直接比較擴散貢獻。

## 本模組的責任

只做「決定並記錄這次 native 執行實際採用哪個實驗案例（是否含 Stokes）」，介面
仿照 ``backtrack_horizon.py``（CLI 旗標優先於環境變數、兩者不一致 fail closed、
省略時使用預設值）。案例合法性直接查詢 ``runtime.EXPERIMENT_CASE_SPECS`` 這個
唯一 registry，不自行維護第二份案例清單。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from ..runtime import EXPERIMENT_CASE_SPECS, ExperimentCaseSpec

__all__ = [
    "EXPERIMENT_CASE_ENV_VAR",
    "DEFAULT_NATIVE_EXPERIMENT_CASE",
    "DIFFUSION_METHOD_DECLARATION",
    "PhysicsCaseError",
    "ResolvedPhysicsCase",
    "resolve_experiment_case_id",
    "resolve_physics_case",
]

# 預設值：WP-18 之前 native 路徑永遠嘗試 Stokes（資料存在就用），為了不默默改變
# 既有計畫檔／結果的行為，省略 CLI 旗標與環境變數時維持這個既有行為，對應
# ``finite_depth_stokes`` 案例。
DEFAULT_NATIVE_EXPERIMENT_CASE: Final[str] = "finite_depth_stokes"
EXPERIMENT_CASE_ENV_VAR: Final[str] = "LBT_NATIVE_EXPERIMENT_CASE"

# native 路徑擴散方法的固定聲明文字：兩種案例（no_stokes／finite_depth_stokes）
# 都採此法，與 ``ExperimentCaseSpec.diffusion_kind`` 的舊常數擴散選項無關。
DIFFUSION_METHOD_DECLARATION: Final[str] = "smagorinsky_workbook_eq10"


class PhysicsCaseError(ValueError):
    """實驗案例解析失敗或 CLI／環境變數不一致時的 fail-closed 例外。"""


def resolve_experiment_case_id(
    *,
    cli_experiment_case_id: str | None,
    env: dict[str, str] | None = None,
) -> str:
    """依 CLI 旗標／環境變數決定「實際採用」的實驗案例，兩者都省略時使用預設值。

    合法性只依 ``runtime.EXPERIMENT_CASE_SPECS`` 判斷；CLI 優先於環境變數，兩者
    都提供且不一致時 fail closed（理由與 ``backtrack_horizon.resolve_backtrack_
    mode`` 相同：避免使用者以為改了 CLI 旗標就生效）。
    """

    env = os.environ if env is None else env
    env_case = env.get(EXPERIMENT_CASE_ENV_VAR)
    if env_case is not None and env_case not in EXPERIMENT_CASE_SPECS:
        raise PhysicsCaseError(
            f"{EXPERIMENT_CASE_ENV_VAR}={env_case!r} 不是已登錄的 experiment_case_id，"
            f"必須是 {tuple(EXPERIMENT_CASE_SPECS)} 其中之一"
        )
    if cli_experiment_case_id is not None and cli_experiment_case_id not in EXPERIMENT_CASE_SPECS:
        raise PhysicsCaseError(
            f"--experiment-case={cli_experiment_case_id!r} 不是已登錄的 experiment_case_id，"
            f"必須是 {tuple(EXPERIMENT_CASE_SPECS)} 其中之一"
        )
    if (
        cli_experiment_case_id is not None
        and env_case is not None
        and cli_experiment_case_id != env_case
    ):
        raise PhysicsCaseError(
            f"--experiment-case={cli_experiment_case_id!r} 與 {EXPERIMENT_CASE_ENV_VAR}={env_case!r} "
            "不一致，請只用其中一種方式指定，或讓兩者相同"
        )
    if cli_experiment_case_id is not None:
        return cli_experiment_case_id
    if env_case is not None:
        return env_case
    return DEFAULT_NATIVE_EXPERIMENT_CASE


@dataclass(frozen=True, slots=True)
class ResolvedPhysicsCase:
    """單次 native 執行實際採用的實驗案例契約；plan／run／postprocess 三階段共用。

    ``diffusion_method`` 固定為 ``DIFFUSION_METHOD_DECLARATION``——native 路徑
    沒有第二種擴散實作，這裡明確寫入是為了讓輸出 manifest／結果摘要不必讓讀者
    回頭查程式碼就知道用的是哪種擴散，也避免與舊引擎常數擴散案例的結果混用。
    """

    experiment_case_id: str
    include_stokes: bool
    diffusion_method: str


def resolve_physics_case(
    *,
    cli_experiment_case_id: str | None,
    env: dict[str, str] | None = None,
) -> ResolvedPhysicsCase:
    """解析並回傳本次執行實際採用的實驗案例契約（見 :class:`ResolvedPhysicsCase`）。"""

    experiment_case_id = resolve_experiment_case_id(
        cli_experiment_case_id=cli_experiment_case_id, env=env
    )
    spec: ExperimentCaseSpec = EXPERIMENT_CASE_SPECS[experiment_case_id]
    return ResolvedPhysicsCase(
        experiment_case_id=experiment_case_id,
        include_stokes=spec.include_stokes,
        diffusion_method=DIFFUSION_METHOD_DECLARATION,
    )
