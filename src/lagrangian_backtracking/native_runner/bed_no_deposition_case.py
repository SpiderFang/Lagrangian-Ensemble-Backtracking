"""``bed_no_deposition`` 敏感度案例的 CLI／環境變數解析（WP-28 補完接線）。

## 背景

``backward_driver.BackwardDriverConfig.bed_no_deposition``（WP-26 新增、WP-28
正名，見該模組文件「WP-26／WP-28」一節與 ``reports/WP-28.md``）是一個**研究用
敏感度開關**：預設 ``False`` 時行為與加入這個欄位之前逐位元相同；設為 ``True``
時，粒子接觸海床達連續沉積門檻**不視為沉積、不終止**，由既有平流＋擴散＋
``batch_clamp_z`` 垂向夾擠繼續驅動——這不是鏡射反射，也不是再懸浮動力學，見
``backward_driver.py`` 模組文件的完整正名說明。

這個開關**刻意不進入** ``config.ProjectConfig`` schema（不是已核准的正式情境
設定，而是像 ``--experiment-case``／``--backtrack-mode`` 一樣的「這次執行要
不要套用」旋鈕），因此介面比照 ``physics_case.py``／``backtrack_horizon.py``
既有的「CLI 旗標優先於環境變數、兩者都提供且不一致時 fail closed、都省略時
用預設值」慣例，而不是走設定檔驗證那條路。

## 為什麼不用 ``argparse`` 的 ``store_true`` 預設值

``bed_no_deposition`` 是一個布林旋鈕，但仍需要「CLI 是否有明確指定」與「CLI
明確指定為 False」兩種狀態可以分辨，才能做「CLI 與環境變數不一致時 fail
closed」的檢查（否則使用者用環境變數打開這個案例、CLI 完全沒提這個旗標時，
會被誤判成「CLI 說 False、環境變數說 True，不一致」）。呼叫端建立
``argparse`` 選項時必須把 ``default`` 明確設成 ``None``（而不是讓
``store_true`` 動作自己預設 ``False``），才能把「使用者完全沒有動這個旗標」
與「使用者明確要求啟用」分開，見 ``cli.py`` 的 ``--bed-no-deposition`` 定義。
"""

from __future__ import annotations

import os
from typing import Final

__all__ = [
    "BED_NO_DEPOSITION_ENV_VAR",
    "DEFAULT_BED_NO_DEPOSITION",
    "BedNoDepositionCaseError",
    "resolve_bed_no_deposition",
]

# 預設值：省略 CLI 旗標與環境變數時，維持加入這個敏感度案例之前的既有行為
# （粒子連續停留海床即視為沉積並終止），不默默改變任何既有計畫檔／結果。
DEFAULT_BED_NO_DEPOSITION: Final[bool] = False
BED_NO_DEPOSITION_ENV_VAR: Final[str] = "LBT_NATIVE_BED_NO_DEPOSITION"

_FALSY_ENV_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "False"})


class BedNoDepositionCaseError(ValueError):
    """CLI 旗標與環境變數不一致、或解析失敗時的 fail-closed 例外。"""


def _env_flag_or_none(raw: str | None) -> bool | None:
    """把環境變數字串解析成布林；空字串或未設定回傳 ``None``（代表「沒有明示」）。"""

    if raw is None or not raw.strip():
        return None
    return raw.strip() not in _FALSY_ENV_VALUES


def resolve_bed_no_deposition(
    *,
    cli_bed_no_deposition: bool | None,
    env: dict[str, str] | None = None,
) -> bool:
    """依 CLI 旗標／環境變數決定本次是否啟用 ``bed_no_deposition`` 敏感度案例。

    介面與 ``physics_case.resolve_experiment_case_id``／``backtrack_horizon.
    resolve_backtrack_mode`` 完全同款：CLI 優先於環境變數；兩者都提供且不一致
    時 fail closed（避免使用者以為改了其中一種方式就生效，卻被另一種方式的
    既有值蓋掉）；兩者都省略時回傳 :data:`DEFAULT_BED_NO_DEPOSITION`
    （``False``，與加入本開關之前的既有行為逐位元相同）。

    Args:
        cli_bed_no_deposition: 呼叫端從 ``argparse`` 解析出的值；``None`` 代表
            使用者完全沒有指定這個旗標（見模組文件「為什麼不用 store_true 預設值」），
            ``True``／``False`` 代表明確指定。
        env: 供測試注入的環境變數字典；省略時讀取 ``os.environ``。
    """

    env = os.environ if env is None else env
    env_value = _env_flag_or_none(env.get(BED_NO_DEPOSITION_ENV_VAR))

    if cli_bed_no_deposition is not None and env_value is not None and cli_bed_no_deposition != env_value:
        raise BedNoDepositionCaseError(
            f"--bed-no-deposition={cli_bed_no_deposition!r} 與 "
            f"{BED_NO_DEPOSITION_ENV_VAR}={env.get(BED_NO_DEPOSITION_ENV_VAR)!r} 不一致，"
            "請只用其中一種方式指定，或讓兩者相同"
        )
    if cli_bed_no_deposition is not None:
        return cli_bed_no_deposition
    if env_value is not None:
        return env_value
    return DEFAULT_BED_NO_DEPOSITION
