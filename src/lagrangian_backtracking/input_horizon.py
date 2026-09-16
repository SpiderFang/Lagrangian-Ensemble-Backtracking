"""共同到達時刻回溯母體的時間窗與 coverage 驗證。

本模組只處理「一個時間錨點往前需要多少個逐時資料節點」以及這些節點是否真的存在。
未啟用隨機沉底時，``inputs.backtrack_support_days`` 是共同輸入母體需證明的 runtime
支援上限；啟用時，該欄位改代表 observation selector 的保守 selection envelope，需至少
等於 ``runtime_horizon_support_days + maximum_age_days``，而沉底後逐筆驗證只使用
``scenarios.bed_residence_time.runtime_horizon_support_days``。``boundaries.max_backtrack_days``
仍表示單一 release 實際要求的回溯長度。這些值分開保存，避免把觀測選時前置包絡誤當成
粒子的回溯日數，也讓同一套共同母體支援多個 horizon release。

這裡刻意不建立 NumPy 的完整時間軸，也不把 7、30、60 寫成選項。先用 Python 整數檢查
UTC nanoseconds 的範圍、資料期邊界與預期節點數，再以 lazy ``range`` 逐時檢查；因此
錯誤的超長支援日數會在讀取大型 forcing 或配置巨量陣列前被拒絕。實際 forcing 的值、
空間遮罩與來源檔案 hash 仍由 ``input_derivation`` 及既有 source validator 負責，本模組
不讀取 NetCDF，也不以 metadata 的 ``approved`` 字樣取代來源驗證。
"""

from __future__ import annotations

import math
from calendar import monthrange
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from numbers import Integral, Real
from typing import Any

UTC_HOUR_NS = 3_600_000_000_000
"""逐時時間軸的一小時 nanoseconds；所有算術均使用整數避免浮點日期誤差。"""

LEGACY_INPUT_SCHEMA_VERSION = "1.0.0"
"""未啟用隨機沉底政策的既有輸入文件版本，供舊 artifact 維持精確相容。"""

BED_RESIDENCE_INPUT_SCHEMA_VERSION = "1.1.0"
"""隨機沉底輸入文件版本，只用於新增沉底中繼資料的 arrival、gap 與 release binding。"""

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

GENERIC_HORIZON_POLICY_ID = "observed_hourly_shared_arrival_horizon_v2"
"""共同 hourly arrival 母體的版本化資料政策識別碼。"""

GENERIC_HORIZON_METHOD_ID = "server_v3_ocm_shared_arrival_horizon_v2"
"""共同 hourly arrival 母體的建置方法識別碼。"""

BED_RESIDENCE_HORIZON_POLICY_ID = "observed_hourly_bed_deposition_horizon_v1"
"""先以觀測錨點篩選、再以沉底時刻核對執行支援窗的版本化政策。"""

BED_RESIDENCE_HORIZON_METHOD_ID = "server_v3_ocm_bed_deposition_horizon_v1"
"""含隨機沉底時間的逐沉底時刻 gap-safe 支援窗建置方法。"""

LEGACY_HORIZON_POLICY_ID = "7_day_gap_safe_baseline"
LEGACY_HORIZON_METHOD_ID = "server_v3_ocm_gap_safe_arrival_horizon_v1"


class HorizonContractError(ValueError):
    """回溯母體的設定、時間軸或 gap 證據不符合資料契約。"""


@dataclass(frozen=True, slots=True)
class HorizonSettings:
    """解析後的 runtime 回溯設定與共同母體政策。

    ``requested_days`` 代表目前執行要求，尚未定案時可為 ``None``；只有 ``is_generic=True`` 時，
    ``support_days`` 才代表設定明示的共同母體支援值。一般 generic 是 runtime support；
    bed-residence generic 則是 observation selection envelope，真正的沉底後 runtime 上限
    另存 ``runtime_support_days``。legacy 設定的 ``support_days`` 保留 ``None``，避免把未宣告
    新欄位的舊 YAML 假裝成已通過 generic 支援驗收。
    """

    requested_days: float | None
    support_days: int | None
    is_generic: bool
    policy_id: str
    method_id: str
    runtime_support_days: int | None = None
    bed_residence_enabled: bool = False

    @property
    def selection_days(self) -> float:
        """回傳 arrival selector 與 gap payload 應使用的日數。

        generic 母體必須以 support 上限篩選所有 arrival，不能讓較短的 requested horizon
        先選出一批在長窗中會失敗的時間；legacy 則保留既有 boundaries 語意。
        """

        if self.is_generic:
            assert self.support_days is not None
            return float(self.support_days)
        assert self.requested_days is not None
        return self.requested_days

    @property
    def selection_support_days(self) -> int | None:
        """回傳 observation selector 必須完整通過的保守時間包絡日數。"""

        return self.support_days


@dataclass(frozen=True, slots=True)
class HorizonWindow:
    """一筆 arrival 的 inclusive 逐時回溯窗口。

    ``expected_step_count`` 包含 arrival 本身，所以支援 ``N`` 日時固定為 ``N*24+1``。
    所有欄位皆為 epoch nanoseconds 或 Python 整數；未配置 NumPy 陣列，避免超長輸入在
    前置檢查階段突然要求大量連續記憶體。
    """

    arrival_time_ns: int
    start_time_ns: int
    end_time_ns: int
    support_days: int
    expected_step_count: int


@dataclass(frozen=True, slots=True)
class HorizonCoverage:
    """依 expected period 與 canonical OCM gap 重算出的窗口支援結果。"""

    window: HorizonWindow
    missing_time_ns: tuple[int, ...]

    @property
    def supported_step_count(self) -> int:
        """回傳實際存在的逐時節點數。"""

        return self.window.expected_step_count - len(self.missing_time_ns)

    @property
    def crossed_gap(self) -> bool:
        """只要窗口有任何資料期外或 canonical gap 節點即視為不可跨越。"""

        return bool(self.missing_time_ns)


@dataclass(frozen=True, slots=True)
class HorizonValidationResult:
    """generic gap manifest 的錯誤、診斷警告與可供 summary 使用的計數。"""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: Mapping[str, Any]

    @property
    def valid(self) -> bool:
        """只有沒有語意錯誤時才算通過；warning 可保留 generated 診斷。"""

        return not self.errors


def validate_support_days(value: Any, *, field_name: str = "backtrack_support_days") -> int:
    """驗證共同母體支援日數為可安全轉成逐時節點的正整數。

    來源通常是 Pydantic 的嚴格整數，但此函式也作為 JSON／測試／外部 caller 的第二道
    防線，因此拒絕 bool、字串、浮點數、零與負數。上限由 int64 epoch 算術決定，不用
    固定 7／30／60 清單；真正的資料期是否容納窗口會在 ``build_horizon_window`` 再檢查。
    """

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise HorizonContractError(f"{field_name} 必須是正整數日")
    days = int(value)
    if days <= 0:
        raise HorizonContractError(f"{field_name} 必須是正整數日")
    if days > (INT64_MAX - INT64_MIN) // UTC_HOUR_NS:
        raise HorizonContractError(f"{field_name} 會造成 UTC int64 算術溢位")
    return days


def _validate_requested_days(value: Any, *, field_name: str = "max_backtrack_days") -> float:
    """驗證 runtime 要求為有限正數；整日限制由需要逐時窗口的 caller 執行。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise HorizonContractError(f"{field_name} 必須是有限正數")
    days = float(value)
    if not math.isfinite(days) or days <= 0.0:
        raise HorizonContractError(f"{field_name} 必須是有限正數")
    return days


def _configured_support(config: Any) -> Any:
    """讀取 config 明示的新欄位，不把舊 YAML 的 Pydantic 預設值當成 generic。"""

    inputs = getattr(config, "inputs", None)
    if inputs is None:
        raise HorizonContractError("config 缺少 inputs")
    return getattr(inputs, "backtrack_support_days", None)


def resolve_configured_horizon(
    config: Any,
    *,
    legacy_default_days: float = 7.0,
) -> HorizonSettings:
    """將 ProjectConfig 解析成 legacy 或 generic 的共同 horizon 設定。

    只有 YAML 明示 ``inputs.backtrack_support_days`` 且值非 ``None`` 才啟用 generic。這
    個區分保留舊設定的 canonical hash 與 pilot 語意；明示 support 時，requested 若已定案
    必須不超過 support，並且 selector 一律先以 support 重新篩選 arrival。generic config
    若尚未指定 ``boundaries.max_backtrack_days``，保留 ``None`` 供 runtime guard 阻擋未定案
    執行；只有未明示 support 的 legacy config 才套用既有 7 日預設。
    """

    boundaries = getattr(config, "boundaries", None)
    if boundaries is None:
        raise HorizonContractError("config 缺少 boundaries")
    support_raw = _configured_support(config)
    inputs = getattr(config, "inputs", None)
    support_explicit = bool(
        inputs is not None and "backtrack_support_days" in getattr(inputs, "model_fields_set", set())
    )
    if support_explicit and support_raw is None:
        raise HorizonContractError(
            "inputs.backtrack_support_days 明示為 null，代表共同母體尚未定案，不能降級為 legacy"
        )
    requested_raw = getattr(boundaries, "max_backtrack_days", None)
    scenarios = getattr(config, "scenarios", None)
    bed_residence = getattr(scenarios, "bed_residence_time", None) if scenarios is not None else None
    bed_enabled = bed_residence is not None
    if support_raw is None:
        requested = _validate_requested_days(
            legacy_default_days if requested_raw is None else requested_raw,
            field_name="boundaries.max_backtrack_days",
        )
        return HorizonSettings(
            requested_days=requested,
            support_days=None,
            is_generic=False,
            policy_id=LEGACY_HORIZON_POLICY_ID,
            method_id=LEGACY_HORIZON_METHOD_ID,
            runtime_support_days=None,
            bed_residence_enabled=False,
        )
    support = validate_support_days(support_raw)
    requested = (
        None
        if requested_raw is None
        else _validate_requested_days(
            requested_raw,
            field_name="boundaries.max_backtrack_days",
        )
    )
    runtime_support: int | None
    policy_id = GENERIC_HORIZON_POLICY_ID
    method_id = GENERIC_HORIZON_METHOD_ID
    if bed_enabled:
        # 含隨機沉底年齡時，inputs 支援窗涵蓋「觀測選時最長回溯期 + 最老沉底年齡」；
        # 沉底後逐筆實際 gap 檢查只使用 runtime horizon。來源 template 可將 runtime
        # 支援設為 null，這時沿用已明示 requested horizon，讓 suite 能在產生 common
        # config 時填入共同 H，而不必增加另一份來源 template。
        maximum_age = getattr(bed_residence, "maximum_age_days", None)
        if type(maximum_age) is not int or maximum_age <= 0:
            raise HorizonContractError("scenarios.bed_residence_time.maximum_age_days 必須是正整數")
        raw_runtime_support = getattr(bed_residence, "runtime_horizon_support_days", None)
        if raw_runtime_support is None:
            runtime_support = (
                int(requested)
                if requested is not None and requested.is_integer()
                else support - maximum_age
            )
        else:
            runtime_support = validate_support_days(
                raw_runtime_support,
                field_name="scenarios.bed_residence_time.runtime_horizon_support_days",
            )
        if runtime_support <= 0:
            raise HorizonContractError("bed residence runtime support 必須是正整數日")
        if support < runtime_support + maximum_age:
            raise HorizonContractError(
                "inputs.backtrack_support_days 必須至少等於 runtime horizon + maximum bed age"
            )
        if requested is not None and requested > float(runtime_support):
            raise HorizonContractError(
                "boundaries.max_backtrack_days 不得超過 bed residence runtime_horizon_support_days"
            )
        policy_id = BED_RESIDENCE_HORIZON_POLICY_ID
        method_id = BED_RESIDENCE_HORIZON_METHOD_ID
    else:
        runtime_support = support
        if requested is not None and requested > float(support):
            raise HorizonContractError(
                "boundaries.max_backtrack_days 不得超過 inputs.backtrack_support_days"
            )
    return HorizonSettings(
        requested_days=requested,
        support_days=support,
        is_generic=True,
        policy_id=policy_id,
        method_id=method_id,
        runtime_support_days=runtime_support,
        bed_residence_enabled=bed_enabled,
    )


# 這個別名讓外部檢查器可以用「解析共同母體」的名稱呼叫同一契約；不複製另一套邏輯。
resolve_horizon_settings = resolve_configured_horizon


def _int64_time(value: Any, *, field_name: str) -> int:
    """把 UTC epoch nanoseconds 驗證為 Python int64 範圍內的整數。"""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise HorizonContractError(f"{field_name} 必須是 UTC epoch nanoseconds 整數")
    result = int(value)
    if result < INT64_MIN or result > INT64_MAX:
        raise HorizonContractError(f"{field_name} 超出 int64 UTC nanoseconds 範圍")
    return result


def build_horizon_window(
    arrival_time_ns: Any,
    support_days: Any,
    *,
    expected_start_ns: Any | None = None,
    expected_end_ns: Any | None = None,
    context: str = "arrival",
) -> HorizonWindow:
    """以整數算術建立 inclusive hourly window，並先拒絕資料期外或溢位。

    ``expected_start_ns``／``expected_end_ns`` 應來自 forcing inventory 的 expected period。
    傳入時會先確認它本身是連續整點，再確認 ``arrival-support`` 未超出資料期；因此不會
    先建立一個巨大的 ``np.arange`` 才在後面發現資料不足。未傳資料期時仍只作 int64 與
    exact-hour 算術，供單元測試或 caller 之後自行提供實際資料範圍。
    """

    arrival = _int64_time(arrival_time_ns, field_name=f"{context} UTC")
    support = validate_support_days(support_days, field_name=f"{context} support_days")
    if arrival % UTC_HOUR_NS != 0:
        raise HorizonContractError(f"{context} UTC 必須落在 exact-hour")
    if (expected_start_ns is None) != (expected_end_ns is None):
        raise HorizonContractError(f"{context} expected period 必須同時提供起點與終點")
    if expected_start_ns is not None and expected_end_ns is not None:
        period_start = _int64_time(expected_start_ns, field_name="expected period 起點")
        period_end = _int64_time(expected_end_ns, field_name="expected period 終點")
        if period_start % UTC_HOUR_NS or period_end % UTC_HOUR_NS:
            raise HorizonContractError("expected period 必須是 exact-hour UTC")
        if period_end < period_start or (period_end - period_start) % UTC_HOUR_NS:
            raise HorizonContractError("expected period 必須是連續逐時 UTC 範圍")
    else:
        period_start = period_end = None

    span_ns = support * 24 * UTC_HOUR_NS
    start = arrival - span_ns
    if start < INT64_MIN or start > INT64_MAX:
        raise HorizonContractError(f"{context} support window 造成 UTC int64 underflow/overflow")
    if period_start is not None and (start < period_start or arrival > period_end):
        raise HorizonContractError(
            f"{context} 的 {support} 日 support window 超出 expected forcing period"
        )
    return HorizonWindow(
        arrival_time_ns=arrival,
        start_time_ns=start,
        end_time_ns=arrival,
        support_days=support,
        expected_step_count=support * 24 + 1,
    )


def validate_support_against_period(
    support_days: Any,
    *,
    expected_start_ns: Any,
    expected_end_ns: Any,
) -> int:
    """在建立 arrival 或讀取 forcing 前，確認 support 不可能大於整段資料期。

    這是 metadata-only 的早期 gate：它不需要知道任何 arrival，也不配置逐時陣列；只要
    ``support*24+1`` 已超過 expected period 的節點數，就直接拒絕，避免後續 NumPy
    時間窗運算因不切實際的正整日而配置或溢位。每筆 arrival 的實際起點仍由
    ``build_horizon_window`` 再檢查，因此本函式通過不代表所有 arrival 都已支援。
    """

    support = validate_support_days(support_days)
    start = _int64_time(expected_start_ns, field_name="expected period 起點")
    end = _int64_time(expected_end_ns, field_name="expected period 終點")
    if (
        start % UTC_HOUR_NS
        or end % UTC_HOUR_NS
        or end < start
        or (end - start) % UTC_HOUR_NS
    ):
        raise HorizonContractError("expected period 必須是連續 exact-hour UTC 範圍")
    period_steps = (end - start) // UTC_HOUR_NS + 1
    if support * 24 + 1 > period_steps:
        raise HorizonContractError("support_days 超過 expected forcing period")
    return support


def iter_horizon_utc_ns(window: HorizonWindow) -> Iterable[int]:
    """以 lazy Python iterator 產生窗口節點，不配置整段 NumPy 陣列。"""

    for index in range(window.expected_step_count):
        yield window.start_time_ns + index * UTC_HOUR_NS


def _item_value(item: Any, name: str) -> Any:
    """同時讀取 ``GapInterval`` dataclass 與 JSON mapping 的欄位。"""

    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _normalise_gap_intervals(
    canonical_gaps: Iterable[Any],
    *,
    canonical_start_ns: int,
    canonical_end_ns: int,
) -> tuple[tuple[int, int, int], ...]:
    """驗證 canonical gap 的端點與 missing count，回傳排序後的純整數資料。"""

    normalised: list[tuple[int, int, int]] = []
    for index, item in enumerate(canonical_gaps):
        before_raw = _item_value(item, "before_utc_ns")
        if before_raw is None:
            before_raw = _item_value(item, "before_utc")
        after_raw = _item_value(item, "after_utc_ns")
        if after_raw is None:
            after_raw = _item_value(item, "after_utc")
        count_raw = _item_value(item, "missing_step_count")
        try:
            if isinstance(before_raw, str):
                before = _utc_ns_from_string(
                    before_raw,
                    field_name=f"canonical gap[{index}] before",
                )
            else:
                before = _int64_time(before_raw, field_name=f"canonical gap[{index}] before")
            if isinstance(after_raw, str):
                after = _utc_ns_from_string(
                    after_raw,
                    field_name=f"canonical gap[{index}] after",
                )
            else:
                after = _int64_time(after_raw, field_name=f"canonical gap[{index}] after")
        except HorizonContractError:
            raise
        if isinstance(count_raw, bool) or not isinstance(count_raw, Integral):
            raise HorizonContractError(f"canonical gap[{index}] missing_step_count 必須是整數")
        count = int(count_raw)
        gap_hours_raw = _item_value(item, "gap_hours")
        span = after - before
        if (
            before < canonical_start_ns
            or after > canonical_end_ns
            or before >= after
            or before % UTC_HOUR_NS
            or after % UTC_HOUR_NS
            or span % UTC_HOUR_NS
            or count != span // UTC_HOUR_NS - 1
            or count < 1
        ):
            raise HorizonContractError(f"canonical gap[{index}] 端點或 missing_step_count 不一致")
        if gap_hours_raw is not None and (
            isinstance(gap_hours_raw, bool)
            or not isinstance(gap_hours_raw, Real)
            or not math.isclose(
                float(gap_hours_raw),
                span / UTC_HOUR_NS,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise HorizonContractError(f"canonical gap[{index}] gap_hours 與端點不一致")
        normalised.append((before, after, count))
    normalised.sort()
    for previous, current in zip(normalised, normalised[1:], strict=False):
        # 兩個缺口可以共享同一個觀測端點，例如 00→02 與 02→04；真正重疊才是
        # metadata 互相矛盾。共享端點不會讓兩組 missing hourly nodes 重複計數。
        if previous[1] > current[0]:
            raise HorizonContractError("canonical gaps 重疊或未按時間分離")
    return tuple(normalised)


def compute_horizon_coverage(
    window: HorizonWindow,
    *,
    expected_start_ns: Any,
    expected_end_ns: Any,
    canonical_start_ns: Any,
    canonical_end_ns: Any,
    canonical_gaps: Iterable[Any] = (),
) -> HorizonCoverage:
    """由 expected period 與 canonical gap 重新計算窗口的 missing 節點。

    這個計算只依 forcing inventory 的期別／canonical bounds／gap 事實，完全忽略 gap
    manifest 自己宣稱的 ``missing_utc``、``supported_step_count`` 或 ``approved`` 狀態。
    因此篡改後重新計算 hash 也不能用偽造的空 missing 清單掩蓋真實缺口。窗口已先以
    ``build_horizon_window`` 限制資料期，逐時檢查最多只會迭代該窗口的實際節點數。
    """

    expected_start = _int64_time(expected_start_ns, field_name="expected period 起點")
    expected_end = _int64_time(expected_end_ns, field_name="expected period 終點")
    canonical_start = _int64_time(canonical_start_ns, field_name="canonical 起點")
    canonical_end = _int64_time(canonical_end_ns, field_name="canonical 終點")
    expected_window = build_horizon_window(
        window.arrival_time_ns,
        window.support_days,
        expected_start_ns=expected_start,
        expected_end_ns=expected_end,
        context="arrival",
    )
    if expected_window != window:
        raise HorizonContractError("傳入的 horizon window 與 support／arrival 不一致")
    if (
        canonical_start % UTC_HOUR_NS
        or canonical_end % UTC_HOUR_NS
        or canonical_end < canonical_start
        or (canonical_end - canonical_start) % UTC_HOUR_NS
    ):
        raise HorizonContractError("canonical UTC bounds 必須是連續 exact-hour 範圍")
    gaps = _normalise_gap_intervals(
        canonical_gaps,
        canonical_start_ns=canonical_start,
        canonical_end_ns=canonical_end,
    )
    missing: list[int] = []
    gap_index = 0
    for value in iter_horizon_utc_ns(window):
        while gap_index < len(gaps) and gaps[gap_index][1] <= value:
            gap_index += 1
        outside = value < canonical_start or value > canonical_end
        inside_gap = bool(
            gap_index < len(gaps)
            and gaps[gap_index][0] < value < gaps[gap_index][1]
            and (value - gaps[gap_index][0]) % UTC_HOUR_NS == 0
        )
        if outside or inside_gap:
            missing.append(value)
    return HorizonCoverage(window=window, missing_time_ns=tuple(missing))


def _utc_ns_from_string(value: Any, *, field_name: str) -> int:
    """解析 manifest 固定 UTC Z 字串，避免用當地時區或浮點 timestamp。"""

    if not isinstance(value, str) or not value.endswith("Z"):
        raise HorizonContractError(f"{field_name} 必須是 UTC Z 字串")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise HorizonContractError(f"{field_name} UTC 格式無效") from exc
    if parsed.utcoffset() != timedelta(0) or parsed.microsecond or parsed.minute or parsed.second:
        raise HorizonContractError(f"{field_name} 必須是 exact-hour UTC")
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = parsed - epoch
    return _int64_time(
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000,
        field_name=field_name,
    )


def utc_string(time_ns: Any) -> str:
    """把整數 epoch nanoseconds 轉成 manifest 使用的固定 UTC 字串。"""

    value = _int64_time(time_ns, field_name="UTC")
    if value % 1_000_000_000:
        raise HorizonContractError("UTC nanoseconds 不是整秒，無法寫入固定 manifest 字串")
    return datetime.fromtimestamp(value // 1_000_000_000, tz=UTC).isoformat().replace("+00:00", "Z")


def _numeric_equal_int(value: Any, expected: int) -> bool:
    """判斷 JSON 數字是否恰好代表指定整數，不接受 bool 或非有限浮點。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    number = float(value)
    return math.isfinite(number) and number == float(expected)


def _strict_json_int_equal(value: Any, expected: int) -> bool:
    """generic manifest 的日數／節點數必須保留 JSON 整數型別。"""

    return type(value) is int and value == expected


def _parse_inventory_period(inventory: Mapping[str, Any]) -> tuple[int, int, int]:
    """解析 forcing inventory 的 expected period 並驗證宣稱筆數。"""

    period = inventory.get("expected_period")
    if not isinstance(period, Mapping):
        raise HorizonContractError("forcing_inventory.expected_period 缺少")
    start = _utc_ns_from_string(period.get("start_utc"), field_name="expected_period.start_utc")
    end = _utc_ns_from_string(period.get("end_utc"), field_name="expected_period.end_utc")
    count = period.get("hourly_step_count")
    if isinstance(count, bool) or not isinstance(count, Integral) or int(count) < 1:
        raise HorizonContractError("expected_period.hourly_step_count 必須是正整數")
    expected_count = (end - start) // UTC_HOUR_NS + 1
    if end < start or (end - start) % UTC_HOUR_NS or int(count) != expected_count:
        raise HorizonContractError("expected_period 的 UTC bounds 與 hourly_step_count 不一致")
    return start, end, int(count)


def _config_support_and_requested(config: Any | None) -> tuple[int | None, float | None]:
    """取出 config 的明示 support／requested，並把設定錯誤轉成驗證錯誤。"""

    if config is None:
        return None, None
    settings = resolve_configured_horizon(config)
    return settings.support_days, settings.requested_days


def _config_expected_period(config: Any) -> tuple[int, int, int]:
    """由 config 的年份與 hourly contract 重建 expected period，拒絕縮短 inventory 自述。"""

    inputs = getattr(config, "inputs", None)
    years = getattr(inputs, "years", None)
    if not isinstance(years, Sequence) or not years:
        raise HorizonContractError("config.inputs.years 必須是非空清單")
    normalised_years: list[int] = []
    for year in years:
        if isinstance(year, bool) or not isinstance(year, Integral):
            raise HorizonContractError("config.inputs.years 必須是整數")
        normalised_years.append(int(year))
    if len(set(normalised_years)) != len(normalised_years):
        raise HorizonContractError("config.inputs.years 不可重複")
    time_contract = getattr(inputs, "time_axis_contract", None)
    step_hours = time_contract.get("expected_timestep_hours") if isinstance(time_contract, Mapping) else None
    if isinstance(step_hours, bool) or not isinstance(step_hours, Real):
        raise HorizonContractError("config expected_timestep_hours 缺少")
    if not math.isclose(float(step_hours), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise HorizonContractError("generic shared horizon 只接受逐時 config expected period")
    first_year = min(normalised_years)
    last_year = max(normalised_years)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    start_dt = datetime(first_year, 1, 1, tzinfo=UTC)
    end_dt = datetime(last_year, 12, 31, 23, tzinfo=UTC)
    start = int((start_dt - epoch).total_seconds()) * 1_000_000_000
    end = int((end_dt - epoch).total_seconds()) * 1_000_000_000
    count = sum(monthrange(year, month)[1] for year in normalised_years for month in range(1, 13)) * 24
    if end < start or (end - start) // UTC_HOUR_NS + 1 != count:
        raise HorizonContractError("config expected period 的年份不是連續 hourly 範圍")
    return start, end, count


def _config_observation_years(config: Any | None) -> tuple[int, ...] | None:
    """取得新版 arrival anchor 年份，避免 gap validator 把 forcing 年份當 observation。

    ``inputs.years`` 描述 forcing 產品需要載入的完整聯集；新版 typed selection 位於
    ``ProjectConfig.arrival_time_selection``，過渡期也可能暫存在 ``scenarios``。若沒有
    observation block，回傳 ``None`` 以保留 legacy generic 的原有行為；若有 block 但
    未明示 observation years，則以 forcing years 作為舊政策 fallback。這個 helper 只
    讀設定，不判定 forcing 內容是否真的存在，實際時間軸仍由下方 coverage 重建。
    """

    if config is None:
        return None
    selection = getattr(config, "arrival_time_selection", None)
    if selection is None:
        scenarios = getattr(config, "scenarios", None)
        selection = getattr(scenarios, "arrival_time_selection", None)
    if selection is None:
        return None
    years = getattr(selection, "observation_years", None)
    if years is None and isinstance(selection, Mapping):
        years = selection.get("observation_years")
    if years is None:
        inputs = getattr(config, "inputs", None)
        years = getattr(inputs, "years", None)
    if not isinstance(years, Sequence) or isinstance(years, (str, bytes, bytearray)):
        raise HorizonContractError("arrival_time_selection.observation_years 必須是年份序列")
    result = tuple(int(value) for value in years)
    if not result or len(set(result)) != len(result):
        raise HorizonContractError("arrival_time_selection.observation_years 不得為空或重複")
    return result


def validate_generic_gap_payload(
    gap_payload: Mapping[str, Any],
    arrival_payload: Mapping[str, Any],
    forcing_inventory: Mapping[str, Any],
    *,
    config: Any | None = None,
    strict: bool = False,
) -> HorizonValidationResult:
    """逐 arrival 驗證 generic shared horizon 的 identity、時間窗與實際 missing。

    驗證順序先檢查版本化政策、設定 support 與 forcing inventory 的 canonical 期別，之後
    才逐筆對照 arrival／gap records。每筆 gap 的 site、region、flow、UTC、起終點、
    ``support*24+1`` 節點數與 missing 清單都由 arrival／inventory 重算；strict 模式遇到
    任一缺口會失敗。非 strict generated artifact 可保留缺口診斷 warning，但只要政策或
    來源 metadata 自相矛盾仍會失敗，不能退回 legacy 驗證。
    """

    errors: list[str] = []
    warnings: list[str] = []
    checked = 0
    crossed = 0
    try:
        configured_support, requested_days = _config_support_and_requested(config)
    except Exception as exc:
        errors.append(f"generic_horizon_config_invalid:{type(exc).__name__}")
        configured_support = requested_days = None

    bed_residence = None
    if config is not None:
        scenarios = getattr(config, "scenarios", None)
        bed_residence = (
            getattr(scenarios, "bed_residence_time", None) if scenarios is not None else None
        )
    bed_enabled = bed_residence is not None
    expected_schema_version = (
        BED_RESIDENCE_INPUT_SCHEMA_VERSION
        if bed_enabled
        else LEGACY_INPUT_SCHEMA_VERSION
    )
    if gap_payload.get("schema_version") != expected_schema_version:
        errors.append("generic_horizon_schema_version_invalid")
    policy = gap_payload.get("policy")
    method = (gap_payload.get("provenance") or {}).get("method_id")
    root_support = gap_payload.get("support_days")
    expected_policy = BED_RESIDENCE_HORIZON_POLICY_ID if bed_enabled else GENERIC_HORIZON_POLICY_ID
    expected_method = BED_RESIDENCE_HORIZON_METHOD_ID if bed_enabled else GENERIC_HORIZON_METHOD_ID
    if policy != expected_policy or method != expected_method:
        errors.append("generic_horizon_policy_or_method_invalid")
    try:
        support = validate_support_days(
            root_support if root_support is not None else gap_payload.get("max_backtrack_days"),
            field_name="gap.root.support_days",
        )
    except HorizonContractError as exc:
        errors.append(f"generic_horizon_support_invalid:{exc}")
        support = None
    runtime_support: int | None = None
    if bed_enabled:
        raw_runtime_support = getattr(bed_residence, "runtime_horizon_support_days", None)
        maximum_age = getattr(bed_residence, "maximum_age_days", None)
        if raw_runtime_support is None:
            runtime_support = (
                int(requested_days)
                if requested_days is not None and requested_days.is_integer()
                else (
                    configured_support - maximum_age
                    if configured_support is not None and type(maximum_age) is int
                    else None
                )
            )
        else:
            try:
                runtime_support = validate_support_days(
                    raw_runtime_support,
                    field_name="scenarios.bed_residence_time.runtime_horizon_support_days",
                )
            except HorizonContractError as exc:
                errors.append(f"generic_horizon_runtime_support_invalid:{exc}")
        if configured_support is not None and not _strict_json_int_equal(
            gap_payload.get("selection_support_days"), configured_support
        ):
            errors.append("generic_horizon_selection_support_config_mismatch")
        if runtime_support is not None and support != runtime_support:
            errors.append("generic_horizon_runtime_support_config_mismatch")
        if runtime_support is not None and requested_days is not None and requested_days > runtime_support:
            errors.append("generic_horizon_requested_exceeds_runtime_support")
        if (
            configured_support is not None
            and type(maximum_age) is int
            and runtime_support is not None
            and configured_support < runtime_support + maximum_age
        ):
            errors.append("generic_horizon_selection_envelope_too_short")
        if runtime_support is not None and not _strict_json_int_equal(
            gap_payload.get("runtime_support_days"), runtime_support
        ):
            errors.append("generic_horizon_root_runtime_support_mismatch")
    elif configured_support is not None and support != configured_support:
        errors.append("generic_horizon_support_config_mismatch")
    if requested_days is not None and support is not None and requested_days > float(support):
        errors.append("generic_horizon_requested_exceeds_support")
    if support is not None and not _numeric_equal_int(gap_payload.get("max_backtrack_days"), support):
        errors.append("generic_horizon_root_max_backtrack_days_mismatch")

    try:
        expected_start, expected_end, expected_count = _parse_inventory_period(forcing_inventory)
    except HorizonContractError as exc:
        errors.append(f"generic_horizon_expected_period_invalid:{exc}")
        expected_start = expected_end = expected_count = None
    if config is not None and expected_start is not None:
        try:
            config_start, config_end, config_count = _config_expected_period(config)
            if (expected_start, expected_end, expected_count) != (config_start, config_end, config_count):
                errors.append("generic_horizon_expected_period_config_mismatch")
        except HorizonContractError as exc:
            errors.append(f"generic_horizon_config_expected_period_invalid:{exc}")

    canonical_by_region: dict[str, Mapping[str, Any]] = {}
    flow_by_region: dict[str, str] = {}
    products = forcing_inventory.get("products")
    if not isinstance(products, list):
        errors.append("generic_horizon_forcing_products_missing")
    else:
        for product in products:
            if not isinstance(product, Mapping) or product.get("product") != "ocm_native":
                continue
            region = product.get("analysis_region_id")
            canonical = product.get("canonical_time")
            if not isinstance(region, str) or not isinstance(canonical, Mapping):
                errors.append("generic_horizon_ocm_canonical_metadata_invalid")
                continue
            if region in canonical_by_region:
                errors.append(f"generic_horizon_duplicate_ocm_region:{region}")
            canonical_by_region[region] = canonical
            flow_domain_id = product.get("flow_domain_id")
            if not isinstance(flow_domain_id, str) or not flow_domain_id:
                errors.append(f"generic_horizon_ocm_flow_invalid:{region}")
            else:
                flow_by_region[region] = flow_domain_id

    arrival_rows = arrival_payload.get("records")
    gap_rows = gap_payload.get("records")
    if not isinstance(arrival_rows, list):
        errors.append("generic_horizon_arrival_records_missing")
        arrival_rows = []
    if not isinstance(gap_rows, list):
        errors.append("generic_horizon_gap_records_missing")
        gap_rows = []
    arrivals_by_id: dict[str, Mapping[str, Any]] = {}
    for row in arrival_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("arrival_time_id"), str):
            errors.append("generic_horizon_arrival_identity_invalid")
            continue
        arrival_id = row["arrival_time_id"]
        if arrival_id in arrivals_by_id:
            errors.append(f"generic_horizon_duplicate_arrival_id:{arrival_id}")
        else:
            arrivals_by_id[arrival_id] = row
    gaps_by_id: dict[str, Mapping[str, Any]] = {}
    for row in gap_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("arrival_time_id"), str):
            errors.append("generic_horizon_gap_identity_invalid")
            continue
        arrival_id = row["arrival_time_id"]
        if arrival_id in gaps_by_id:
            errors.append(f"generic_horizon_duplicate_gap_id:{arrival_id}")
        else:
            gaps_by_id[arrival_id] = row
    if set(arrivals_by_id) != set(gaps_by_id):
        errors.append("generic_horizon_arrival_gap_one_to_one_invalid")

    configured_site_regions: dict[str, str] = {}
    observation_years = _config_observation_years(config)
    if config is not None:
        configured_sites = getattr(config, "study_sites", ())
        if isinstance(configured_sites, Sequence):
            for configured_site in configured_sites:
                site_id = getattr(configured_site, "study_site_id", None)
                region_id = getattr(configured_site, "analysis_region_id", None)
                if isinstance(site_id, str) and isinstance(region_id, str):
                    configured_site_regions[site_id] = region_id
    require_site_region_config = config is None and any(
        isinstance(row, Mapping)
        and isinstance(row.get("study_site_id"), str)
        and not isinstance(row.get("analysis_region_id"), str)
        for row in arrival_rows
    )
    if require_site_region_config:
        # 現有 arrival schema 只保存 study_site_id，region binding 位於 ProjectConfig；
        # 離線 caller 若沒有 config，不能把 gap row 自己宣稱的 region 當成獨立證據。
        errors.append("generic_horizon_config_required_for_site_region_binding")

    for arrival_id, arrival in arrivals_by_id.items():
        gap = gaps_by_id.get(arrival_id)
        if gap is None:
            continue
        checked += 1
        site = arrival.get("study_site_id")
        declared_region = arrival.get("analysis_region_id")
        region = declared_region if isinstance(declared_region, str) else configured_site_regions.get(site)
        arrival_ns = arrival.get("time_utc_ns")
        if require_site_region_config and not isinstance(declared_region, str):
            continue
        if not isinstance(region, str) or not isinstance(site, str):
            errors.append(f"generic_horizon_identity_invalid:{arrival_id}")
            continue
        try:
            arrival_ns_int = _int64_time(arrival_ns, field_name=f"arrival[{arrival_id}].time_utc_ns")
            arrival_utc = utc_string(arrival_ns_int)
        except HorizonContractError as exc:
            errors.append(f"generic_horizon_arrival_utc_invalid:{arrival_id}:{exc}")
            continue
        if observation_years is not None and bed_enabled is False:
            arrival_year = datetime.fromtimestamp(arrival_ns_int // 1_000_000_000, tz=UTC).year
            if arrival_year not in observation_years:
                errors.append(f"generic_horizon_observation_year_invalid:{arrival_id}")
        if (
            gap.get("study_site_id") != site
            or gap.get("analysis_region_id") != region
            or (isinstance(declared_region, str) and declared_region != region)
        ):
            errors.append(f"generic_horizon_site_region_mismatch:{arrival_id}")
        expected_flow = flow_by_region.get(region)
        if expected_flow is None or gap.get("flow_domain_id") != expected_flow:
            errors.append(f"generic_horizon_flow_mismatch:{arrival_id}")
        if gap.get("flow_domain_id") is None:
            errors.append(f"generic_horizon_flow_missing:{arrival_id}")
        if gap.get("arrival_time_utc") != arrival_utc:
            errors.append(f"generic_horizon_arrival_utc_mismatch:{arrival_id}")
        if support is None or expected_start is None or expected_end is None:
            continue
        canonical = canonical_by_region.get(region)
        if canonical is None:
            errors.append(f"generic_horizon_ocm_region_missing:{region}")
            continue
        try:
            canonical_start = _utc_ns_from_string(
                canonical.get("time_start_utc"),
                field_name=f"canonical[{region}].time_start_utc",
            )
            canonical_end = _utc_ns_from_string(
                canonical.get("time_end_utc"),
                field_name=f"canonical[{region}].time_end_utc",
            )
            canonical_count = canonical.get("canonical_time_count")
            if isinstance(canonical_count, bool) or not isinstance(canonical_count, Integral):
                raise HorizonContractError("canonical_time_count 必須是整數")
            expected_timestep = canonical.get("expected_timestep_hours", 1.0)
            if (
                isinstance(expected_timestep, bool)
                or not isinstance(expected_timestep, Real)
                or not math.isclose(float(expected_timestep), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
            ):
                raise HorizonContractError("generic shared horizon 只接受逐時 OCM canonical 軸")
            gap_metadata = canonical.get("gaps")
            if not isinstance(gap_metadata, list):
                raise HorizonContractError("canonical gaps 必須是 list")
            gap_counts: list[int] = []
            for gap_index, gap_item in enumerate(gap_metadata):
                raw_count = _item_value(gap_item, "missing_step_count")
                if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
                    raise HorizonContractError(
                        f"canonical gap[{gap_index}] missing_step_count 必須是整數"
                    )
                gap_counts.append(int(raw_count))
            canonical_span_count = (canonical_end - canonical_start) // UTC_HOUR_NS + 1
            if (
                canonical_end < canonical_start
                or (canonical_end - canonical_start) % UTC_HOUR_NS
                or int(canonical_count) != canonical_span_count - sum(gap_counts)
            ):
                raise HorizonContractError("canonical bounds、gaps 與 canonical_time_count 不一致")
            window = build_horizon_window(
                arrival_ns_int,
                support,
                expected_start_ns=expected_start,
                expected_end_ns=expected_end,
                context=f"arrival[{arrival_id}]",
            )
            coverage = compute_horizon_coverage(
                window,
                expected_start_ns=expected_start,
                expected_end_ns=expected_end,
                canonical_start_ns=canonical_start,
                canonical_end_ns=canonical_end,
                canonical_gaps=gap_metadata,
            )
        except HorizonContractError as exc:
            errors.append(f"generic_horizon_window_invalid:{arrival_id}:{exc}")
            continue
        missing_strings = [utc_string(value) for value in coverage.missing_time_ns]
        if gap.get("horizon_start_utc") != utc_string(window.start_time_ns):
            errors.append(f"generic_horizon_start_mismatch:{arrival_id}")
        if gap.get("horizon_end_utc") != arrival_utc:
            errors.append(f"generic_horizon_end_mismatch:{arrival_id}")
        if not _strict_json_int_equal(gap.get("support_days"), support):
            errors.append(f"generic_horizon_record_support_mismatch:{arrival_id}")
        if not _strict_json_int_equal(gap.get("max_backtrack_days"), support):
            errors.append(f"generic_horizon_record_max_days_mismatch:{arrival_id}")
        if not _strict_json_int_equal(gap.get("expected_step_count"), window.expected_step_count):
            errors.append(f"generic_horizon_expected_steps_mismatch:{arrival_id}")
        if not _strict_json_int_equal(gap.get("supported_step_count"), coverage.supported_step_count):
            errors.append(f"generic_horizon_supported_steps_mismatch:{arrival_id}")
        if gap.get("missing_utc") != missing_strings:
            errors.append(f"generic_horizon_missing_nodes_mismatch:{arrival_id}")
        if gap.get("crossed_gap") is not coverage.crossed_gap:
            errors.append(f"generic_horizon_crossed_gap_mismatch:{arrival_id}")
        expected_record_policy = (
            BED_RESIDENCE_HORIZON_POLICY_ID if bed_enabled else GENERIC_HORIZON_POLICY_ID
        )
        if gap.get("time_support_policy") != expected_record_policy:
            errors.append(f"generic_horizon_record_policy_mismatch:{arrival_id}")
        if bed_enabled:
            metadata = arrival.get("metadata")
            observation_ns = (
                metadata.get("observation_time_utc_ns")
                if isinstance(metadata, Mapping)
                else None
            )
            try:
                observation_time = _int64_time(
                    observation_ns,
                    field_name=f"arrival[{arrival_id}].metadata.observation_time_utc_ns",
                )
                if observation_years is not None:
                    observation_year = datetime.fromtimestamp(
                        observation_time // 1_000_000_000, tz=UTC
                    ).year
                    if observation_year not in observation_years:
                        errors.append(f"generic_horizon_observation_year_invalid:{arrival_id}")
                if configured_support is None:
                    raise HorizonContractError("bed residence 缺少 selection support")
                selection_window = build_horizon_window(
                    observation_time,
                    configured_support,
                    expected_start_ns=expected_start,
                    expected_end_ns=expected_end,
                    context=f"observation[{arrival_id}]",
                )
                selection_coverage = compute_horizon_coverage(
                    selection_window,
                    expected_start_ns=expected_start,
                    expected_end_ns=expected_end,
                    canonical_start_ns=canonical_start,
                    canonical_end_ns=canonical_end,
                    canonical_gaps=gap_metadata,
                )
                selection_missing = [utc_string(value) for value in selection_coverage.missing_time_ns]
                if gap.get("observation_time_utc") != utc_string(observation_time):
                    errors.append(f"generic_horizon_observation_utc_mismatch:{arrival_id}")
                if gap.get("selection_horizon_start_utc") != utc_string(selection_window.start_time_ns):
                    errors.append(f"generic_horizon_selection_start_mismatch:{arrival_id}")
                if not _strict_json_int_equal(gap.get("selection_support_days"), configured_support):
                    errors.append(f"generic_horizon_selection_record_support_mismatch:{arrival_id}")
                if gap.get("selection_horizon_end_utc") != utc_string(observation_time):
                    errors.append(f"generic_horizon_selection_end_mismatch:{arrival_id}")
                if gap.get("selection_expected_step_count") != selection_window.expected_step_count:
                    errors.append(f"generic_horizon_selection_expected_steps_mismatch:{arrival_id}")
                if gap.get("selection_supported_step_count") != selection_coverage.supported_step_count:
                    errors.append(f"generic_horizon_selection_supported_steps_mismatch:{arrival_id}")
                if gap.get("selection_missing_utc") != selection_missing:
                    errors.append(f"generic_horizon_selection_missing_mismatch:{arrival_id}")
                if gap.get("selection_crossed_gap") is not selection_coverage.crossed_gap:
                    errors.append(f"generic_horizon_selection_crossed_gap_mismatch:{arrival_id}")
                if selection_coverage.crossed_gap:
                    crossed += 1
                    if strict:
                        errors.append(f"generic_horizon_selection_crosses_gap:{arrival_id}")
                    else:
                        warnings.append(f"generic_horizon_selection_crosses_gap:{arrival_id}")
            except HorizonContractError as exc:
                errors.append(f"generic_horizon_selection_window_invalid:{arrival_id}:{exc}")
        if coverage.crossed_gap:
            crossed += 1
            if strict:
                errors.append(f"generic_horizon_crosses_gap:{arrival_id}")
            else:
                warnings.append(f"generic_horizon_crosses_gap:{arrival_id}")

    if expected_count is not None and expected_count < 1:
        errors.append("generic_horizon_expected_period_empty")
    return HorizonValidationResult(
        errors=tuple(errors),
        warnings=tuple(warnings),
        summary={
            "policy": policy,
            "method_id": method,
            "support_days": support,
            "selection_support_days": configured_support if bed_enabled else support,
            "runtime_support_days": runtime_support if bed_enabled else support,
            "arrival_records_checked": checked,
            "arrival_records_crossing_gap": crossed,
        },
    )


# 語意較完整的別名供 input_derivation 與外部 validator 使用；兩者共享同一結果格式。
validate_shared_horizon_payload = validate_generic_gap_payload


__all__ = [
    "BED_RESIDENCE_INPUT_SCHEMA_VERSION",
    "GENERIC_HORIZON_METHOD_ID",
    "GENERIC_HORIZON_POLICY_ID",
    "HorizonContractError",
    "HorizonCoverage",
    "HorizonSettings",
    "HorizonValidationResult",
    "HorizonWindow",
    "LEGACY_HORIZON_METHOD_ID",
    "LEGACY_HORIZON_POLICY_ID",
    "LEGACY_INPUT_SCHEMA_VERSION",
    "UTC_HOUR_NS",
    "build_horizon_window",
    "compute_horizon_coverage",
    "iter_horizon_utc_ns",
    "resolve_configured_horizon",
    "resolve_horizon_settings",
    "utc_string",
    "validate_support_against_period",
    "validate_generic_gap_payload",
    "validate_shared_horizon_payload",
    "validate_support_days",
]
