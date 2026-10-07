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

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from ..runtime import EXPERIMENT_CASE_SPECS, ExperimentCaseSpec

__all__ = [
    "EXPERIMENT_CASE_ENV_VAR",
    "DEFAULT_NATIVE_EXPERIMENT_CASE",
    "DIFFUSION_METHOD_DECLARATION",
    "PhysicsCaseError",
    "ResolvedPhysicsCase",
    "resolve_experiment_case_id",
    "resolve_physics_case",
    "DIFFUSION_NOISE_ENV_VAR",
    "VERTICAL_DIFFUSION_ENV_VAR",
    "KZ_CONSTANT_ENV_VAR",
    "VISSER_ENV_VAR",
    "DEFAULT_DIFFUSION_NOISE",
    "DEFAULT_VERTICAL_DIFFUSION",
    "DEFAULT_KZ_CONSTANT_M2PS",
    "ResolvedDiffusionScheme",
    "resolve_diffusion_scheme",
    "config_kz_constant_m2ps",
    "diffusion_scheme_mismatches",
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


# ══════════════════════════════════════════════════════════════════════════
# 擴散組態（WP-51 整合）：雜訊型態、垂向擴散、常數 Kz、Visser 修正
# ══════════════════════════════════════════════════════════════════════════
#
# 為什麼放在這裡：計畫檔（``native-plan``）、續跑檢查點（checkpoint）與每個工作行程
# （``hour_major_provider``）都必須對「這次實際用哪一組擴散設定」有同一個答案，否則
# 中途改了環境變數的續跑會把兩種擴散混在同一批粒子裡。這個函式是唯一的解析來源。
#
# 常數垂向擴散係數（Kz，單位 m²/s）的來源順序，**設定檔是唯一來源**：
#
# 1. 設定檔 ``physics.vertical_diffusion.constant_kz_m2ps``（有值就用它）；
# 2. 設定檔沒有該值（缺欄位或 ``null``）時，才讀環境變數 ``LBT_NATIVE_KZ_CONSTANT_M2PS``；
# 3. 兩者都沒有時用預設 3.0e-4（BayTrace ``kv_const`` 預設）。
#
# 設定檔有值又設了環境變數且兩者不同時 fail closed：不允許環境變數悄悄蓋掉設定檔，
# 否則論文或報告引用的設定檔數值會與實際執行不一致。

DIFFUSION_NOISE_ENV_VAR: Final[str] = "LBT_NATIVE_DIFFUSION_NOISE"
VERTICAL_DIFFUSION_ENV_VAR: Final[str] = "LBT_NATIVE_VERTICAL_DIFFUSION"
KZ_CONSTANT_ENV_VAR: Final[str] = "LBT_NATIVE_KZ_CONSTANT_M2PS"
VISSER_ENV_VAR: Final[str] = "LBT_NATIVE_VISSER"
DEFAULT_DIFFUSION_NOISE: Final[str] = "uniform"
DEFAULT_VERTICAL_DIFFUSION: Final[str] = "constant"
DEFAULT_KZ_CONSTANT_M2PS: Final[float] = 3.0e-4


@dataclass(frozen=True, slots=True)
class ResolvedDiffusionScheme:
    """單次 native 執行實際採用的擴散組態；plan／run／checkpoint 三處共用。

    ``kz_constant_m2ps`` 只在 ``vertical_diffusion == "constant"`` 時有意義；其他情況
    宣告值為 ``None``，避免「關閉垂向擴散」的案例因為預設 Kz 不同而被誤判為不一致。
    ``kz_source`` 記錄 Kz 取自 ``config``、``environment`` 或 ``default``，只供稽核，
    不參與一致性比對。
    """

    diffusion_noise: str
    vertical_diffusion: str
    kz_constant_m2ps: float
    visser_correction: bool
    kz_source: str

    def as_declaration(self) -> dict[str, Any]:
        """計畫檔／checkpoint 的宣告內容（不含只供稽核的 ``kz_source`` 以外的衍生值）。"""

        return {
            "diffusion_noise": self.diffusion_noise,
            "vertical_diffusion": self.vertical_diffusion,
            "kz_constant_m2ps": (
                float(self.kz_constant_m2ps) if self.vertical_diffusion == "constant" else None
            ),
            "visser_correction": bool(self.visser_correction),
            "kz_source": self.kz_source,
        }


def config_kz_constant_m2ps(config: Any) -> float | None:
    """讀設定檔 ``physics.vertical_diffusion.constant_kz_m2ps``；缺欄位或 ``null`` 回傳 ``None``。"""

    physics = getattr(config, "physics", None)
    vertical = physics.get("vertical_diffusion") if isinstance(physics, Mapping) else None
    value = vertical.get("constant_kz_m2ps") if isinstance(vertical, Mapping) else None
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicsCaseError(
            f"physics.vertical_diffusion.constant_kz_m2ps 必須是數值或 null，收到 {value!r}"
        )
    return float(value)


def _env_text(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def resolve_diffusion_scheme(
    *,
    env: Mapping[str, str] | None = None,
    config_kz_constant_m2ps: float | None = None,
) -> ResolvedDiffusionScheme:
    """解析本次執行實際採用的擴散組態；無效值或設定檔與環境變數衝突時 fail closed。

    ``config_kz_constant_m2ps`` 是設定檔 ``physics.vertical_diffusion.constant_kz_m2ps``
    （``None`` 表示設定檔沒有該值）。環境變數：``LBT_NATIVE_DIFFUSION_NOISE``
    （``uniform``／``gaussian``，預設 ``uniform``）、``LBT_NATIVE_VERTICAL_DIFFUSION``
    （``constant``／``off``／``ocm_field``，預設 ``constant``；``ocm_field`` 的合法性由
    ``hour_major.validate_diffusion_scheme`` 判定）、``LBT_NATIVE_KZ_CONSTANT_M2PS``、
    ``LBT_NATIVE_VISSER``（``0``／``1``，預設 ``0``）。
    """

    env = os.environ if env is None else env
    noise = (_env_text(env, DIFFUSION_NOISE_ENV_VAR) or DEFAULT_DIFFUSION_NOISE).lower()
    if noise not in ("uniform", "gaussian"):
        raise PhysicsCaseError(f"{DIFFUSION_NOISE_ENV_VAR} 只接受 uniform／gaussian，收到 {noise!r}")
    vertical = (_env_text(env, VERTICAL_DIFFUSION_ENV_VAR) or DEFAULT_VERTICAL_DIFFUSION).lower()
    if vertical not in ("constant", "off", "ocm_field"):
        raise PhysicsCaseError(
            f"{VERTICAL_DIFFUSION_ENV_VAR} 只接受 constant／off／ocm_field，收到 {vertical!r}"
        )
    visser_text = _env_text(env, VISSER_ENV_VAR)
    visser = visser_text is not None and visser_text not in ("0", "false", "False")

    env_kz_text = _env_text(env, KZ_CONSTANT_ENV_VAR)
    env_kz: float | None = None
    if env_kz_text is not None:
        try:
            env_kz = float(env_kz_text)
        except ValueError as error:
            raise PhysicsCaseError(f"{KZ_CONSTANT_ENV_VAR} 不是數值：{env_kz_text!r}") from error
    for label, value in (("設定檔 constant_kz_m2ps", config_kz_constant_m2ps), (KZ_CONSTANT_ENV_VAR, env_kz)):
        if value is not None and not (math.isfinite(float(value)) and float(value) > 0.0):
            raise PhysicsCaseError(f"{label} 必須是有限的正值，收到 {value!r}")
    if config_kz_constant_m2ps is not None:
        if env_kz is not None and float(env_kz) != float(config_kz_constant_m2ps):
            raise PhysicsCaseError(
                f"{KZ_CONSTANT_ENV_VAR}={env_kz!r} 與設定檔 physics.vertical_diffusion."
                f"constant_kz_m2ps={float(config_kz_constant_m2ps)!r} 不一致：設定檔是 Kz 的唯一來源，"
                "請修改設定檔，或移除該環境變數"
            )
        kz, source = float(config_kz_constant_m2ps), "config"
    elif env_kz is not None:
        kz, source = float(env_kz), "environment"
    else:
        kz, source = DEFAULT_KZ_CONSTANT_M2PS, "default"
    return ResolvedDiffusionScheme(
        diffusion_noise=noise,
        vertical_diffusion=vertical,
        kz_constant_m2ps=kz,
        visser_correction=visser,
        kz_source=source,
    )


_DIFFUSION_SCHEME_COMPARED_KEYS: Final[tuple[str, ...]] = (
    "diffusion_noise",
    "vertical_diffusion",
    "kz_constant_m2ps",
    "visser_correction",
)


def diffusion_scheme_mismatches(
    declared: Mapping[str, Any], actual: Mapping[str, Any]
) -> list[str]:
    """比較兩份擴散組態宣告（計畫檔 vs 本次執行），回傳人看得懂的不一致項目清單。

    只比較 ``_DIFFUSION_SCHEME_COMPARED_KEYS`` 四項；``kz_source`` 只供稽核不比較。
    空清單代表一致。
    """

    problems: list[str] = []
    for key in _DIFFUSION_SCHEME_COMPARED_KEYS:
        if declared.get(key) != actual.get(key):
            problems.append(f"{key}：計畫檔={declared.get(key)!r}，本次執行={actual.get(key)!r}")
    return problems
