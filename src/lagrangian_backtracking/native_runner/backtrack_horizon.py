"""native 路徑的回溯模式明確化：讓 plan／run／postprocess 都知道「實際採用的回溯模式與 H」。

## 背景

``bed_residence.py`` 定義了兩種回溯模式（見該模組與
``docs/operations/14_input_derivation_and_release_contract.md`` 第 3.1 節）：

- ``full_horizon_from_deposition``：從沉底 UTC 起完整回溯 H 天。這是 PI 原始需求
  （「不能假設瓶子在沉底後仍隨流漂移」）對應的正式模式，也是 native_runner 目前
  （WP-17 之前）**唯一實際執行**的行為——只是先前設計把設定檔／`backtrack_mode`
  欄位誤標成 ``fixed_calendar_window``，本模組把這個現況改成明確、可稽核的宣告。
- ``fixed_calendar_window``：固定日曆終點（觀測日）往前回溯 H 天，有效漂流期間是
  ``H - bed_residence_age_hours``；沉底早於這個研究窗的成員（年齡 ≥ H）標記
  ``pre_window_deposition``，完全不進入運算。這是前一位開發者設計、PI 原話未要求
  的替代敏感度分析模式，本模組保留它但不再讓它被默默當成預設。

## 本模組的責任

只做「決定並記錄這次 native 執行實際採用哪個模式、每個工作單元（＝每個到達時刻）
的有效回溯時間是多少」，不重寫 ``bed_residence.resolve_bed_residence_timing`` 的科學
邏輯（該函式已經是 fixed_calendar_window 模式的唯一權威實作，這裡只呼叫它）。

``full_horizon_from_deposition`` 不需要沉底 metadata 就能決定有效回溯時間（現行行為：
H 天原樣套用到每個到達時刻），因此呼叫端即使沒有 ``ArrivalTime`` 物件（例如 legacy
非沉底母體）也能使用這個模式；只有明確要求 ``fixed_calendar_window`` 時才需要提供
對應的 ``ArrivalTime``（含 ``bed_residence_*`` metadata），缺少時 fail closed。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from ..bed_residence import (
    BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
    BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
    resolve_bed_residence_timing,
)
from ..scenarios import ArrivalTime

__all__ = [
    "BACKTRACK_MODE_ENV_VAR",
    "DEFAULT_NATIVE_BACKTRACK_MODE",
    "SUPPORTED_BACKTRACK_MODES",
    "BacktrackHorizonError",
    "ResolvedUnitHorizon",
    "resolve_backtrack_mode",
    "require_mode_acknowledged_if_mismatched",
    "resolve_unit_backtrack_horizon",
]

# 預設值：依主代理判讀 PI 原話，正式回溯必須從沉底日起算完整 H 天，見模組文件。
DEFAULT_NATIVE_BACKTRACK_MODE: Final[str] = BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION
SUPPORTED_BACKTRACK_MODES: Final[tuple[str, ...]] = (
    BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
    BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
)
BACKTRACK_MODE_ENV_VAR: Final[str] = "LBT_NATIVE_BACKTRACK_MODE"

_NANOSECONDS_PER_SECOND: Final[int] = 1_000_000_000


class BacktrackHorizonError(ValueError):
    """回溯模式解析或設定／實際模式不一致時的 fail-closed 例外。"""


def resolve_backtrack_mode(
    *,
    cli_backtrack_mode: str | None,
    env: dict[str, str] | None = None,
) -> str:
    """依 CLI 旗標／環境變數決定「實際採用」的回溯模式，兩者都省略時使用預設值。

    CLI 旗標優先於環境變數；兩者都提供且不一致時 fail closed（避免使用者以為改了
    CLI 旗標就生效，實際上環境變數才是實際套用值的常見誤用）。回傳值只保證是
    ``SUPPORTED_BACKTRACK_MODES`` 其中之一，不在這裡比對設定檔（設定檔比對見
    ``require_mode_acknowledged_if_mismatched``，該函式需要另外讀取 config，兩者
    分開是為了讓不需要比對設定檔的呼叫端（例如純粹想解析 CLI/env）不必連帶載入
    ``config.py``）。
    """

    env = os.environ if env is None else env
    env_mode = env.get(BACKTRACK_MODE_ENV_VAR)
    if env_mode is not None and env_mode not in SUPPORTED_BACKTRACK_MODES:
        raise BacktrackHorizonError(
            f"{BACKTRACK_MODE_ENV_VAR}={env_mode!r} 不是支援的回溯模式，"
            f"必須是 {SUPPORTED_BACKTRACK_MODES} 其中之一"
        )
    if cli_backtrack_mode is not None and cli_backtrack_mode not in SUPPORTED_BACKTRACK_MODES:
        raise BacktrackHorizonError(
            f"--backtrack-mode={cli_backtrack_mode!r} 不是支援的回溯模式，"
            f"必須是 {SUPPORTED_BACKTRACK_MODES} 其中之一"
        )
    if cli_backtrack_mode is not None and env_mode is not None and cli_backtrack_mode != env_mode:
        raise BacktrackHorizonError(
            f"--backtrack-mode={cli_backtrack_mode!r} 與 {BACKTRACK_MODE_ENV_VAR}={env_mode!r} 不一致，"
            "請只用其中一種方式指定，或讓兩者相同"
        )
    if cli_backtrack_mode is not None:
        return cli_backtrack_mode
    if env_mode is not None:
        return env_mode
    return DEFAULT_NATIVE_BACKTRACK_MODE


def require_mode_acknowledged_if_mismatched(
    *,
    effective_backtrack_mode: str,
    config_declared_backtrack_mode: str | None,
    acknowledge_config_mode_mismatch: bool,
) -> None:
    """設定檔的 ``bed_residence_time.backtrack_mode`` 與實際採用模式不同時，強制使用者明確承認。

    ``config_declared_backtrack_mode`` 為 ``None`` 表示設定檔根本沒有啟用隨機沉底
    （legacy generic 母體），此時沒有「設定檔宣稱的模式」可比對，不視為不一致——
    這種設定本來就只支援 ``full_horizon_from_deposition`` 語意下的固定回溯窗
    （``boundaries.max_backtrack_days``），``fixed_calendar_window`` 需要沉底年齡
    才有意義，呼叫端應在更早的階段就拒絕對這種設定要求 ``fixed_calendar_window``。
    """

    if config_declared_backtrack_mode is None:
        return
    if config_declared_backtrack_mode == effective_backtrack_mode:
        return
    if acknowledge_config_mode_mismatch:
        return
    raise BacktrackHorizonError(
        f"設定檔 bed_residence_time.backtrack_mode={config_declared_backtrack_mode!r} 與實際採用的 "
        f"--backtrack-mode/{BACKTRACK_MODE_ENV_VAR}={effective_backtrack_mode!r} 不一致，"
        "拒絕默默執行。請明確加上 --acknowledge-config-mode-mismatch 承認此差異，"
        "或改用與設定檔一致的模式"
    )


@dataclass(frozen=True, slots=True)
class ResolvedUnitHorizon:
    """單一工作單元（一個到達＝沉底 UTC）在指定回溯模式下的實際運算時間契約。

    這是 native plan／run／postprocess 三個階段共用的最小事實集合：計畫檔、每個
    結果摘要與後處理 manifest 都應該原樣保存這幾個欄位，讓任何人事後都能不重跑
    地核對「這個單元當初到底算了多久、算到哪一年齡的沉底粒子就被排除」。
    """

    backtrack_mode: str
    requested_horizon_days: float
    effective_horizon_seconds: float
    effective_horizon_ns: int
    pre_window_deposition: bool
    bed_residence_age_hours: int | None


def resolve_unit_backtrack_horizon(
    *,
    backtrack_mode: str,
    requested_horizon_days: float,
    arrival: ArrivalTime | None = None,
    maximum_age_days: int | None = None,
    sampling_seed: int | None = None,
) -> ResolvedUnitHorizon:
    """決定單一工作單元（一個到達時刻）的有效回溯時間與 pre-window 判定。

    ``full_horizon_from_deposition``：維持現行行為，H 天原樣套用，不需要 ``arrival``
    （呼叫端甚至不需要有沉底 metadata，因為這個模式本來就不看年齡）。

    ``fixed_calendar_window``：必須提供 ``arrival``（含完整沉底 metadata）、
    ``maximum_age_days`` 與 ``sampling_seed``，直接轉呼叫
    ``bed_residence.resolve_bed_residence_timing``——本函式不重寫其驗證或有效期間
    計算，只是把回傳的 ``BedResidenceTiming`` 轉成本模組的共用資料形狀，供 native
    路徑的三個階段一致使用。
    """

    if backtrack_mode not in SUPPORTED_BACKTRACK_MODES:
        raise BacktrackHorizonError(f"不支援的回溯模式：{backtrack_mode!r}")

    if backtrack_mode == BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION:
        horizon_seconds = float(requested_horizon_days) * 86400.0
        if horizon_seconds <= 0.0:
            raise BacktrackHorizonError("requested_horizon_days 必須是正值")
        horizon_ns = int(round(horizon_seconds * _NANOSECONDS_PER_SECOND))
        age_hours: int | None = None
        if arrival is not None and isinstance(arrival.metadata, dict):
            raw_age = arrival.metadata.get("bed_residence_age_hours")
            if raw_age is not None:
                age_hours = int(raw_age)
        return ResolvedUnitHorizon(
            backtrack_mode=backtrack_mode,
            requested_horizon_days=float(requested_horizon_days),
            effective_horizon_seconds=horizon_seconds,
            effective_horizon_ns=horizon_ns,
            pre_window_deposition=False,
            bed_residence_age_hours=age_hours,
        )

    # fixed_calendar_window：需要完整沉底 metadata 才能算出「H - age」。
    if arrival is None or maximum_age_days is None or sampling_seed is None:
        raise BacktrackHorizonError(
            "fixed_calendar_window 模式必須提供 arrival（含沉底 metadata）、"
            "maximum_age_days 與 sampling_seed，才能呼叫 resolve_bed_residence_timing"
        )
    timing = resolve_bed_residence_timing(
        arrival,
        backtrack_mode=backtrack_mode,
        requested_horizon_days=requested_horizon_days,
        maximum_age_days=maximum_age_days,
        sampling_seed=sampling_seed,
    )
    age_hours = int(arrival.metadata["bed_residence_age_hours"])  # type: ignore[index]
    return ResolvedUnitHorizon(
        backtrack_mode=backtrack_mode,
        requested_horizon_days=float(requested_horizon_days),
        effective_horizon_seconds=timing.effective_horizon_seconds,
        effective_horizon_ns=timing.effective_horizon_ns,
        pre_window_deposition=timing.pre_window_deposition,
        bed_residence_age_hours=age_hours,
    )
