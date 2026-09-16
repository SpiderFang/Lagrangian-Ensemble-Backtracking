"""隨機沉底時間的可重現抽樣、到達時刻轉換與回溯時間解析。

本模組把觀測錨點轉為候選沉底 UTC，並保留能稽核兩個時間點、停留年齡及抽樣算法的
metadata。五站共用同一組離散小時年齡；各站先按觀測 UTC 與原始 ID 穩定排序，再將
年齡逐一配對，因此站點資料列輸入順序不會改變抽樣對應。運算時間解析只接受完整且
相互一致的 metadata，不會替缺欄位、試跑資料或不同抽樣設定做猜測。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Final, Literal

import numpy as np

from .gap_policy import (
    REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID,
)
from .scenarios import ArrivalTime, stable_identifier

__all__ = [
    "BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW",
    "BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION",
    "BED_RESIDENCE_POLICY_ID",
    "BED_RESIDENCE_SAMPLING_METHOD_ID",
    "BED_RESIDENCE_AVAILABILITY_CONDITIONED_SAMPLING_METHOD_ID",
    "BED_RESIDENCE_SAMPLING_POLICY_ID",
    "BedResidenceAvailabilitySampling",
    "BedResidenceStratumRejectionAudit",
    "BedResidenceMode",
    "BedResidenceTiming",
    "apply_bed_residence_sampling",
    "resolve_bed_residence_timing",
    "sample_bed_residence_age_hours",
    "sample_bed_residence_age_hours_conditioned_on_availability",
    "sample_bed_residence_age_hours_gap_aware",
]

# 模式字串會進入設定、run 身分與輸出，採具名值以避免裸數字 1/2 在不同介面有歧義。
BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW: Final[str] = "fixed_calendar_window"
BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION: Final[str] = (
    "full_horizon_from_deposition"
)
BedResidenceMode = Literal["fixed_calendar_window", "full_horizon_from_deposition"]

# policy 表示科學抽樣契約；sampling method 則鎖定亂數來源及抽樣後排序方式。修改任一
# 算法都必須升版，避免舊 run 的 metadata 被誤認為使用新算法。
BED_RESIDENCE_POLICY_ID: Final[str] = "random_bed_residence_age_v1"
BED_RESIDENCE_SAMPLING_POLICY_ID: Final[str] = (
    "discrete_hourly_stratified_uniform_v1"
)
BED_RESIDENCE_SAMPLING_METHOD_ID: Final[str] = (
    "numpy_pcg64dxsm_one_per_stratum_then_permutation_v1"
)
# 新版條件式抽樣將第 n 個 age stratum 固定配對到各站排序後的第 n 個
# observation rank，再使用同一個 PCG64DXSM generator 在該 stratum 內逐一嘗試候選。
# 此識別碼不可沿用 legacy method，否則 loader 無法區分「缺口拒絕抽樣」與舊向量。
BED_RESIDENCE_AVAILABILITY_CONDITIONED_SAMPLING_METHOD_ID: Final[str] = (
    "numpy_pcg64dxsm_stratum_rank_first_available_deposition_v1"
)

_HOURS_PER_DAY: Final[int] = 24
_NANOSECONDS_PER_SECOND: Final[int] = 1_000_000_000
_NANOSECONDS_PER_HOUR: Final[int] = 3_600 * _NANOSECONDS_PER_SECOND
_NANOSECONDS_PER_DAY: Final[int] = 86_400 * _NANOSECONDS_PER_SECOND
_INT64_MIN: Final[int] = -(1 << 63)
_INT64_MAX: Final[int] = (1 << 63) - 1
_SAMPLE_COUNT: Final[int] = 50


@dataclass(frozen=True, slots=True)
class BedResidenceStratumRejectionAudit:
    """一個沉底年齡分層的候選拒絕稽核。

    ``observation_rank`` 是每一站按 observation UTC、arrival ID 穩定排序後的共同
    rank；``rejected_by_site`` 逐候選保存哪些站點的 deposition exact-hour 不在
    canonical available-time set。這些欄位只描述可重現的候選篩選，不代表資料缺口
    已被補齊；完全沒有可用候選時，抽樣器會直接丟出例外。
    """

    stratum_index: int
    observation_rank: int
    candidate_count: int
    rejected_age_hours: tuple[int, ...]
    rejected_by_site: tuple[tuple[int, tuple[str, ...]], ...]
    selected_age_hours: int

    def as_metadata(self) -> dict[str, object]:
        """轉為 JSON manifest 可保存的純 Python mapping。"""

        return {
            "stratum_index": self.stratum_index,
            "observation_rank": self.observation_rank,
            "candidate_count": self.candidate_count,
            "rejected_age_hours": list(self.rejected_age_hours),
            "rejected_by_site": [
                {"age_hours": age, "missing_site_ids": list(site_ids)}
                for age, site_ids in self.rejected_by_site
            ],
            "selected_age_hours": self.selected_age_hours,
            "rejection_count": len(self.rejected_age_hours),
        }


@dataclass(frozen=True, slots=True)
class BedResidenceAvailabilitySampling:
    """缺口條件式沉底年齡母體及逐分層拒絕稽核。

    ``age_hours`` 已按排序後 observation rank 排列，五站必須共用同一向量。因為
    可用性判斷要求每一個候選沉底時刻都出現在五站各自 canonical time set，返回
    向量不會含 nearest、linear、zero-fill 或跨缺口外插的時間。``rejection_audit``
    可直接放入 arrival provenance；其順序固定為 stratum index 0 到 49。
    """

    age_hours: tuple[int, ...]
    rejection_audit: tuple[BedResidenceStratumRejectionAudit, ...]
    sampling_method_id: str = BED_RESIDENCE_AVAILABILITY_CONDITIONED_SAMPLING_METHOD_ID
    availability_conditioning_policy: str = (
        REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID
    )


@dataclass(frozen=True, slots=True)
class BedResidenceTiming:
    """單一沉底錨點的有效回溯時間與 forcing 起點。

    秒數欄位供既有引擎使用；奈秒欄位供 runtime 精確比較。固定日曆模式中，沉底早於
    研究窗的成員以零有效期間及 pre-window 標記表示，不能送入 forcing 積分。完整沉底
    回溯模式則始終保留設定的正回溯長度。
    """

    effective_horizon_seconds: float
    effective_horizon_ns: int
    earliest_forcing_time_utc_ns: int
    pre_window_deposition: bool


def _strict_int(value: object, *, label: str, minimum: int | None = None) -> int:
    """只接受非 bool 的原生 Python 整數，避免時間或 seed 被隱式轉型。"""

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 int，且不可是 bool 或 NumPy scalar")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} 不可小於 {minimum}")
    return value


def _nonempty_text(value: object, *, label: str) -> str:
    """驗證進入 identity 或 provenance 的文字欄位不為空且沒有首尾空白。"""

    if type(value) is not str:
        raise TypeError(f"{label} 必須是原生 str")
    if not value or value != value.strip():
        raise ValueError(f"{label} 必須是非空且沒有首尾空白的文字")
    return value


def _stratum_bounds(
    *, maximum_age_days: int, sample_count: int, stratum_index: int
) -> tuple[int, int]:
    """回傳一個連續整點小時分層的半開範圍。

    可抽小時由 0 到 maximum_age_days 乘 24，兩端都包含，因此總格點數多一個端點。
    以整數比例切分可讓分層長度只相差一小時，且沒有重疊或漏掉的格點。
    """

    total_hours = maximum_age_days * _HOURS_PER_DAY + 1
    return (
        (stratum_index * total_hours) // sample_count,
        ((stratum_index + 1) * total_hours) // sample_count,
    )


def _stratum_index_for_age(
    age_hours: int, *, maximum_age_days: int, sample_count: int
) -> int:
    """找出年齡小時所屬的唯一連續分層，並拒絕範圍外的值。"""

    for index in range(sample_count):
        start, stop = _stratum_bounds(
            maximum_age_days=maximum_age_days,
            sample_count=sample_count,
            stratum_index=index,
        )
        if start <= age_hours < stop:
            return index
    raise ValueError("bed residence age 不在指定整點小時分層範圍內")


def sample_bed_residence_age_hours(
    *, maximum_age_days: int, sample_count: int, seed: int
) -> tuple[int, ...]:
    """依固定分層抽出唯一沉底年齡小時，並以明示 seed 重現結果。

    maximum_age_days 定義含端點的離散整點小時範圍；正式設計由 config 鎖定為 90 天及
    每站 50 筆。函式將範圍切成 sample_count 個連續分層，每層用 PCG64DXSM 抽一個整數
    小時，再以同一 Generator 做 deterministic permutation。各層互不重疊，因此輸出必
    為唯一值；五站必須共用同一回傳向量。
    """

    maximum_age_days = _strict_int(
        maximum_age_days, label="maximum_age_days", minimum=1
    )
    sample_count = _strict_int(sample_count, label="sample_count", minimum=1)
    seed = _strict_int(seed, label="seed", minimum=0)
    available_hour_count = maximum_age_days * _HOURS_PER_DAY + 1
    if sample_count > available_hour_count:
        raise ValueError("sample_count 不可大於可抽取的唯一整點小時數")

    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    stratified: list[int] = []
    for stratum_index in range(sample_count):
        start, stop = _stratum_bounds(
            maximum_age_days=maximum_age_days,
            sample_count=sample_count,
            stratum_index=stratum_index,
        )
        if stop <= start:
            raise ValueError("抽樣分層不可為空")
        stratified.append(int(generator.integers(start, stop)))

    permutation = generator.permutation(sample_count)
    output = tuple(stratified[int(index)] for index in permutation)
    if len(set(output)) != sample_count:
        raise RuntimeError("分層抽樣產生重複年齡，拒絕輸出不完整母體")
    return output


def _available_hour_set(
    values: Iterable[int], *, site_id: str, field_name: str
) -> frozenset[int]:
    """把 canonical available-time set 正規化為 exact-hour 整數集合。

    available set 必須來自已驗收 OCM canonical time axis；此函式只檢查交換介面的
    時間格式，不讀取檔案，也不把相鄰時刻推算成可用節點。任何非整點、bool、字串或
    int64 範圍外值都直接拒絕，避免條件式抽樣在缺口附近產生隱含補值。
    """

    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{field_name}[{site_id}] 必須是 exact-hour UTC nanoseconds 序列") from error
    normalised: set[int] = set()
    for index, value in enumerate(raw_values):
        if type(value) is not int:
            raise TypeError(f"{field_name}[{site_id}][{index}] 必須是原生 int UTC nanoseconds")
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise ValueError(f"{field_name}[{site_id}][{index}] 超出有號 64 位範圍")
        if value % _NANOSECONDS_PER_HOUR:
            raise ValueError(f"{field_name}[{site_id}][{index}] 必須落在 exact-hour")
        normalised.add(value)
    return frozenset(normalised)


def _sorted_observation_rank_times(
    observation_times_by_site: Mapping[str, Sequence[int]], *, sample_count: int
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    """依 UTC 與原始索引建立五站共同 observation rank。

    年齡抽樣的科學語意是「同一 rank 使用同一 age」，不是把不同站點的第 n 筆
    輸入列碰巧配在一起。因此這裡先對每站排序，再要求各站有相同 rank 數量與唯一
    exact-hour；輸入順序可改變，但排序後的 rank 與候選結果不會改變。
    """

    if not isinstance(observation_times_by_site, Mapping) or not observation_times_by_site:
        raise ValueError("observation_times_by_site 必須是非空站點 mapping")
    site_ids = tuple(sorted(observation_times_by_site))
    if any(type(site_id) is not str or not site_id.strip() for site_id in site_ids):
        raise TypeError("observation_times_by_site 的站點 ID 必須是非空 str")
    sorted_times: list[tuple[int, ...]] = []
    for site_id in site_ids:
        values = observation_times_by_site[site_id]
        if isinstance(values, (str, bytes, bytearray)):
            raise TypeError(f"observation_times_by_site[{site_id}] 不可為文字")
        try:
            converted = tuple(values)
        except TypeError as error:
            raise TypeError(f"observation_times_by_site[{site_id}] 必須是時間序列") from error
        if len(converted) != sample_count:
            raise ValueError(
                f"observation_times_by_site[{site_id}] 必須恰有 {sample_count} 筆 observation"
            )
        checked: list[int] = []
        for index, value in enumerate(converted):
            if type(value) is not int:
                raise TypeError(
                    f"observation_times_by_site[{site_id}][{index}] 必須是原生 int UTC nanoseconds"
                )
            if not _INT64_MIN <= value <= _INT64_MAX:
                raise ValueError(f"observation_times_by_site[{site_id}][{index}] 超出有號 64 位範圍")
            if value % _NANOSECONDS_PER_HOUR:
                raise ValueError(f"observation_times_by_site[{site_id}][{index}] 必須落在 exact-hour")
            checked.append(value)
        ordered = tuple(sorted(checked))
        if len(set(ordered)) != sample_count:
            raise ValueError(f"observation_times_by_site[{site_id}] UTC 不可重複")
        sorted_times.append(ordered)
    if len({len(values) for values in sorted_times}) != 1:
        raise ValueError("五站 observation rank 數量必須一致")
    return site_ids, tuple(sorted_times)


def sample_bed_residence_age_hours_conditioned_on_availability(
    *,
    observation_times_by_site: Mapping[str, Sequence[int]],
    available_time_ns_by_site: Mapping[str, Iterable[int]],
    maximum_age_days: int,
    sample_count: int,
    seed: int,
) -> BedResidenceAvailabilitySampling:
    """依五站 canonical available-time set 條件式抽樣沉底年齡。

    先把每站 observation UTC 以 ``(time_utc_ns, 原始輸入順序)`` 的穩定規則排序，將
    stratum ``i`` 固定配對至 rank ``i``。每層再由同一個 NumPy PCG64DXSM generator
    對該層所有整點候選建立無放回順序，選出第一個使「每一站該 rank observation
    UTC 減 age」均存在於該站 canonical available-time set 的年齡。被拒絕的候選與
    缺少該 deposition hour 的站點逐項保存於 audit；某層無候選時 fail closed。

    ``age_hours`` 的順序是排序後 observation rank 順序，不是 legacy sampler 的
    額外 permutation 順序。這是有意的新版 method identity；呼叫端不得以舊
    ``BED_RESIDENCE_SAMPLING_METHOD_ID`` 寫入新版 manifest。函式只做 exact-hour
    membership，不允許 nearest、linear、zero fill 或跨 gap 推估。
    """

    maximum_age_days = _strict_int(maximum_age_days, label="maximum_age_days", minimum=1)
    sample_count = _strict_int(sample_count, label="sample_count", minimum=1)
    seed = _strict_int(seed, label="seed", minimum=0)
    if sample_count != _SAMPLE_COUNT:
        raise ValueError("正式缺口條件式 bed residence 抽樣必須使用 50 個分層")
    available_hour_count = maximum_age_days * _HOURS_PER_DAY + 1
    if sample_count > available_hour_count:
        raise ValueError("sample_count 不可大於可抽取的唯一整點小時數")
    site_ids, observation_ranks = _sorted_observation_rank_times(
        observation_times_by_site, sample_count=sample_count
    )
    if set(available_time_ns_by_site) != set(site_ids):
        raise ValueError("available_time_ns_by_site 必須與 observation_times_by_site 站點集合 exact 相同")
    available_by_site = {
        site_id: _available_hour_set(
            available_time_ns_by_site[site_id],
            site_id=site_id,
            field_name="available_time_ns_by_site",
        )
        for site_id in site_ids
    }

    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    selected_ages: list[int] = []
    audits: list[BedResidenceStratumRejectionAudit] = []
    for stratum_index in range(sample_count):
        start, stop = _stratum_bounds(
            maximum_age_days=maximum_age_days,
            sample_count=sample_count,
            stratum_index=stratum_index,
        )
        if stop <= start:
            raise ValueError(f"bed residence stratum {stratum_index} 不可為空")
        # permutation 產生每層無放回候選順序；使用同一 generator 並保留層順序，
        # 使 seed、站點集合與 canonical axis 一起決定唯一的 rejection audit。
        candidates = tuple(
            int(value)
            for value in generator.permutation(np.arange(start, stop, dtype=np.int64))
        )
        rejected: list[int] = []
        rejected_by_site: list[tuple[int, tuple[str, ...]]] = []
        selected: int | None = None
        rank_observation = tuple(times[stratum_index] for times in observation_ranks)
        for age in candidates:
            missing_sites = tuple(
                site_id
                for site_id, observation_ns in zip(site_ids, rank_observation, strict=True)
                if observation_ns - age * _NANOSECONDS_PER_HOUR
                not in available_by_site[site_id]
            )
            if missing_sites:
                rejected.append(age)
                rejected_by_site.append((age, missing_sites))
                continue
            selected = age
            break
        if selected is None:
            raise ValueError(
                "bed residence 缺口條件式抽樣無可用候選："
                f"stratum={stratum_index}, observation_rank={stratum_index}, "
                f"candidate_count={len(candidates)}"
            )
        selected_ages.append(selected)
        audits.append(
            BedResidenceStratumRejectionAudit(
                stratum_index=stratum_index,
                observation_rank=stratum_index,
                candidate_count=len(candidates),
                rejected_age_hours=tuple(rejected),
                rejected_by_site=tuple(rejected_by_site),
                selected_age_hours=selected,
            )
        )

    age_vector = _validated_age_vector(selected_ages, maximum_age_days=maximum_age_days)
    return BedResidenceAvailabilitySampling(
        age_hours=age_vector,
        rejection_audit=tuple(audits),
    )


# 這個較短 alias 提供 input builder 使用；它與完整名稱共用同一實作，不另外產生
# 可能漂移的抽樣算法。對外文件應優先使用完整名稱，避免把「條件式」誤讀成舊抽樣。
sample_bed_residence_age_hours_gap_aware = (
    sample_bed_residence_age_hours_conditioned_on_availability
)


def _format_utc_ns(value: int) -> str:
    """將有號 64 位 UTC 奈秒轉為保留奈秒精度的 ISO-8601 Z 文字。"""

    seconds, nanoseconds = divmod(value, _NANOSECONDS_PER_SECOND)
    try:
        timestamp = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    except OverflowError as error:
        raise ValueError("UTC 奈秒無法表示為 ISO-8601 時間") from error
    base = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    fraction = f".{nanoseconds:09d}" if nanoseconds else ""
    return f"{base}{fraction}Z"


def _source_metadata(metadata: object) -> dict[str, object]:
    """複製純量 metadata，拒絕 pilot provenance 或重複套用沉底轉換。"""

    if not isinstance(metadata, dict):
        raise TypeError("ArrivalTime.metadata 必須是 dict")
    copied = dict(metadata)
    if any("pilot" in str(key).casefold() for key in copied) or any(
        isinstance(value, str) and "pilot" in value.casefold()
        for value in copied.values()
    ):
        raise ValueError("pilot metadata 不可混入正式沉底時間抽樣")
    generated = {
        "observation_time_utc_ns",
        "observation_time_utc",
        "deposition_time_utc_ns",
        "deposition_time_utc",
        "bed_residence_age_hours",
        "bed_residence_stratum_index",
        "bed_residence_policy_id",
        "bed_residence_sampling_policy_id",
        "bed_residence_sampling_method_id",
        "bed_residence_availability_conditioning_policy",
        "bed_residence_availability_rejection_audit",
        "bed_residence_sampling_seed",
        "observation_arrival_time_id",
    }
    if generated.intersection(copied):
        raise ValueError("ArrivalTime 已含沉底抽樣 metadata，禁止再次轉換")
    return copied


def _validated_age_vector(
    age_hours: Sequence[int], *, maximum_age_days: int
) -> tuple[int, ...]:
    """驗證跨站共用的 50 筆年齡向量完整涵蓋每個分層且沒有重複。"""

    if isinstance(age_hours, (str, bytes, bytearray)):
        raise TypeError("age_hours 必須是 50 個整數小時的序列")
    try:
        ages = tuple(age_hours)
    except TypeError as error:
        raise TypeError("age_hours 必須是 50 個整數小時的序列") from error
    if len(ages) != _SAMPLE_COUNT:
        raise ValueError("正式 bed residence age 向量必須恰有 50 個值")
    canonical = tuple(
        _strict_int(age, label=f"age_hours[{index}]", minimum=0)
        for index, age in enumerate(ages)
    )
    if len(set(canonical)) != _SAMPLE_COUNT:
        raise ValueError("bed residence age 向量必須包含 50 個唯一整數小時")
    if any(age > maximum_age_days * _HOURS_PER_DAY for age in canonical):
        raise ValueError("bed residence age 超過 maximum_age_days")
    strata = {
        _stratum_index_for_age(
            age,
            maximum_age_days=maximum_age_days,
            sample_count=_SAMPLE_COUNT,
        )
        for age in canonical
    }
    if strata != set(range(_SAMPLE_COUNT)):
        raise ValueError("bed residence age 向量必須每個分層恰有一個值")
    return canonical


def _season_for_month(month: int) -> str:
    """依沉底 UTC 月份重算季節，避免沿用觀測錨點原有季節。"""

    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    raise ValueError(f"UTC month 不合法：{month}")


def apply_bed_residence_sampling(
    arrivals: Sequence[ArrivalTime],
    *,
    age_hours: Sequence[int],
    sampling_seed: int,
    maximum_age_days: int,
    design_version: str,
    sampling_method_id: str = BED_RESIDENCE_SAMPLING_METHOD_ID,
    availability_conditioning_policy: str | None = None,
    availability_rejection_audit: Sequence[BedResidenceStratumRejectionAudit] | None = None,
) -> tuple[ArrivalTime, ...]:
    """把單站觀測錨點依穩定順序轉為沉底 UTC 到達紀錄。

    每站輸入必須恰有 50 筆未轉換的 observation-anchor ArrivalTime，age_hours 是五站
    共用的同一向量。配對順序固定以 (time_utc_ns, arrival_time_id) 排序；沉底時間等於
    observation UTC 減去整數小時年齡。year/season 改依沉底 UTC 重算，潮況及事件標籤
    保持原 observation strata，並另存原始 year/season 供追溯。metadata 保存兩個 UTC
    時間、年齡、分層、算法版本、seed、設計版本及原始 arrival ID。新版條件式抽樣
    可以另帶 availability policy 與每分層 rejection audit；這些欄位只作 provenance，
    不會替缺少的 forcing 節點補值。legacy method 仍拒絕站內重複沉底 UTC；新版
    條件式 method 則以 observation identity 區分資料列，只要求候選沉底整點在五站
    canonical available-time set 中存在，不額外加入未登錄的唯一性限制。兩者都會拒絕
    pilot provenance、已轉換紀錄或不完整分層向量。
    """

    maximum_age_days = _strict_int(
        maximum_age_days, label="maximum_age_days", minimum=1
    )
    sampling_seed = _strict_int(sampling_seed, label="sampling_seed", minimum=0)
    design_version = _nonempty_text(design_version, label="design_version")
    ages = _validated_age_vector(age_hours, maximum_age_days=maximum_age_days)
    sampling_method_id = _nonempty_text(sampling_method_id, label="sampling_method_id")
    if sampling_method_id not in {
        BED_RESIDENCE_SAMPLING_METHOD_ID,
        BED_RESIDENCE_AVAILABILITY_CONDITIONED_SAMPLING_METHOD_ID,
    }:
        raise ValueError(f"不支援的 bed residence sampling method：{sampling_method_id!r}")
    if sampling_method_id == BED_RESIDENCE_AVAILABILITY_CONDITIONED_SAMPLING_METHOD_ID:
        if availability_conditioning_policy != REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID:
            raise ValueError(
                "新版條件式 bed residence method 必須 exact 綁定 availability_conditioning_policy"
            )
        if availability_rejection_audit is None:
            raise ValueError("新版條件式 bed residence method 必須保存 rejection audit")
        audit_records = tuple(availability_rejection_audit)
        if len(audit_records) != _SAMPLE_COUNT:
            raise ValueError("新版條件式 bed residence method 必須保存 50 筆 rejection audit")
        if tuple(item.stratum_index for item in audit_records) != tuple(range(_SAMPLE_COUNT)):
            raise ValueError("rejection audit 必須依 stratum index 0..49 排列")
        if tuple(item.selected_age_hours for item in audit_records) != ages:
            raise ValueError("rejection audit 的 selected age 與 age_hours 不一致")
    elif availability_conditioning_policy is not None or availability_rejection_audit is not None:
        raise ValueError("legacy bed residence method 不得攜帶新版 availability provenance")
    if isinstance(arrivals, (str, bytes, bytearray)):
        raise TypeError("arrivals 必須是 ArrivalTime 序列")
    records = tuple(arrivals)
    if len(records) != _SAMPLE_COUNT:
        raise ValueError("每站 observation-anchor ArrivalTime 必須恰有 50 筆")
    if any(type(item) is not ArrivalTime for item in records):
        raise TypeError("arrivals 的每筆資料都必須是 exact ArrivalTime")
    if len({item.study_site_id for item in records}) != 1:
        raise ValueError("apply_bed_residence_sampling 一次只能處理單一站點")
    original_ids = [
        _nonempty_text(item.arrival_time_id, label="arrival_time_id")
        for item in records
    ]
    if len(set(original_ids)) != _SAMPLE_COUNT:
        raise ValueError("原始 observation arrival ID 必須唯一")

    indexed = sorted(
        enumerate(records),
        key=lambda pair: (pair[1].time_utc_ns, pair[1].arrival_time_id),
    )
    by_original_index: dict[int, tuple[int, int, dict[str, object]]] = {}
    deposition_values: list[int] = []
    for sorted_index, (original_index, arrival) in enumerate(indexed):
        observation_ns = _strict_int(
            arrival.time_utc_ns, label="ArrivalTime.time_utc_ns"
        )
        if not _INT64_MIN <= observation_ns <= _INT64_MAX:
            raise ValueError("observation UTC 奈秒超出有號 64 位範圍")
        _strict_int(arrival.year, label="ArrivalTime.year", minimum=1)
        _nonempty_text(arrival.season, label="ArrivalTime.season")
        _nonempty_text(arrival.tide_class, label="ArrivalTime.tide_class")
        _nonempty_text(arrival.phase_or_event, label="ArrivalTime.phase_or_event")
        metadata = _source_metadata(arrival.metadata)
        age = ages[sorted_index]
        deposition_ns = observation_ns - age * _NANOSECONDS_PER_HOUR
        if not _INT64_MIN <= deposition_ns <= _INT64_MAX:
            raise ValueError("deposition UTC 奈秒超出有號 64 位範圍")
        by_original_index[original_index] = (observation_ns, age, metadata)
        deposition_values.append(deposition_ns)
    if (
        len(set(deposition_values)) != _SAMPLE_COUNT
        and sampling_method_id == BED_RESIDENCE_SAMPLING_METHOD_ID
    ):
        # legacy arrival identity 尚未把 observation rank 納入完整 identity，維持舊版
        # 對重複沉底 UTC 的拒絕；新版條件式 identity 已綁定 observation arrival ID，
        # 且候選規則只要求五站 exact-hour available，不額外偷偷增加全站唯一時間限制。
        raise ValueError("50 筆沉底時間出現重複 UTC，拒絕建立模糊 arrival identity")

    design_hash = hashlib.sha256(design_version.encode("utf-8")).hexdigest()
    output: list[ArrivalTime | None] = [None] * _SAMPLE_COUNT
    for sorted_index, (original_index, arrival) in enumerate(indexed):
        observation_ns, age, metadata = by_original_index[original_index]
        deposition_ns = deposition_values[sorted_index]
        seconds, subsecond_ns = divmod(deposition_ns, _NANOSECONDS_PER_SECOND)
        try:
            deposition_dt = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
                seconds=seconds,
                microseconds=subsecond_ns // 1_000,
            )
        except OverflowError as error:
            raise ValueError("deposition UTC 無法表示為曆法時間") from error
        # season/year 描述真正的沉底時間；原 observation strata 另存於 metadata，潮況與
        # 事件欄位則保留在 ArrivalTime 頂層，供既有情境分層契約使用。
        metadata.update(
            {
                "observation_year": arrival.year,
                "observation_season": arrival.season,
                "observation_tide_class": arrival.tide_class,
                "observation_phase_or_event": arrival.phase_or_event,
                "observation_time_utc_ns": observation_ns,
                "observation_time_utc": _format_utc_ns(observation_ns),
                "deposition_time_utc_ns": deposition_ns,
                "deposition_time_utc": _format_utc_ns(deposition_ns),
                "bed_residence_age_hours": age,
                "bed_residence_stratum_index": _stratum_index_for_age(
                    age,
                    maximum_age_days=maximum_age_days,
                    sample_count=_SAMPLE_COUNT,
                ),
                "bed_residence_policy_id": BED_RESIDENCE_POLICY_ID,
                "bed_residence_sampling_policy_id": BED_RESIDENCE_SAMPLING_POLICY_ID,
                "bed_residence_sampling_method_id": sampling_method_id,
                "bed_residence_sampling_seed": sampling_seed,
                "bed_residence_maximum_age_days": maximum_age_days,
                "bed_residence_design_version": design_version,
                "bed_residence_design_hash": design_hash,
                "observation_arrival_time_id": arrival.arrival_time_id,
            }
        )
        if sampling_method_id == BED_RESIDENCE_AVAILABILITY_CONDITIONED_SAMPLING_METHOD_ID:
            assert availability_rejection_audit is not None
            stratum_index = _stratum_index_for_age(
                age,
                maximum_age_days=maximum_age_days,
                sample_count=_SAMPLE_COUNT,
            )
            metadata.update(
                {
                    "bed_residence_availability_conditioning_policy": availability_conditioning_policy,
                    "bed_residence_availability_rejection_audit": (
                        availability_rejection_audit[stratum_index].as_metadata()
                    ),
                }
            )
        identity_fields = [
            arrival.arrival_time_id,
            str(deposition_ns),
            BED_RESIDENCE_POLICY_ID,
            BED_RESIDENCE_SAMPLING_POLICY_ID,
        ]
        # 舊 method 的 identity 欄位不可改動，否則同一份 legacy config 會因新增
        # optional schema 欄位產生 hash／arrival ID 漂移。只有新版 method 需要把
        # method 與 availability policy 納入 identity，避免兩種抽樣結果混用。
        if sampling_method_id == BED_RESIDENCE_AVAILABILITY_CONDITIONED_SAMPLING_METHOD_ID:
            identity_fields.extend([sampling_method_id, availability_conditioning_policy])
        identity_fields.extend([str(sampling_seed), design_hash])
        new_id = stable_identifier("arrival_bed", identity_fields)
        output[original_index] = ArrivalTime(
            arrival_time_id=new_id,
            study_site_id=arrival.study_site_id,
            time_utc_ns=deposition_ns,
            year=deposition_dt.year,
            season=_season_for_month(deposition_dt.month),
            tide_class=arrival.tide_class,
            phase_or_event=arrival.phase_or_event,
            metadata=metadata,  # type: ignore[arg-type]
        )
    if any(item is None for item in output):
        raise RuntimeError("沉底 arrival 轉換未產生完整 50 筆輸出")
    return tuple(item for item in output if item is not None)


def _days_to_nanoseconds(value: object, *, label: str) -> int:
    """將有限正天數精確換成整數奈秒，拒絕不可精確表達的次奈秒設定。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} 必須是有限正數")
    days = float(value)
    if not math.isfinite(days) or days <= 0.0:
        raise ValueError(f"{label} 必須是有限正數")
    try:
        exact_ns = Decimal(str(value)) * Decimal(_NANOSECONDS_PER_DAY)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} 無法轉為整數奈秒") from error
    integral_ns = exact_ns.to_integral_value()
    if exact_ns != integral_ns:
        raise ValueError(f"{label} 無法精確表示為整數奈秒")
    horizon_ns = int(integral_ns)
    if not 0 < horizon_ns <= _INT64_MAX:
        raise ValueError(f"{label} 超出有號 64 位奈秒範圍")
    seconds = horizon_ns / _NANOSECONDS_PER_SECOND
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{label} 換算後不是有限正秒數")
    return horizon_ns


def _metadata_int(
    metadata: dict[str, object],
    key: str,
    *,
    minimum: int | None = 0,
) -> int:
    """從沉底 metadata 讀取嚴格整數欄位，不對缺值做預設。"""

    if key not in metadata:
        raise ValueError(f"ArrivalTime.metadata 缺少 {key}")
    return _strict_int(
        metadata[key],
        label=f"ArrivalTime.metadata.{key}",
        minimum=minimum,
    )


def resolve_bed_residence_timing(
    arrival: ArrivalTime,
    *,
    backtrack_mode: str,
    requested_horizon_days: int | float,
    maximum_age_days: int,
    sampling_seed: int,
) -> BedResidenceTiming:
    """以已驗證設定及沉底 metadata 決定單一成員實際逆向運算時間。

    函式確認 observation 與 deposition UTC 的奈秒差恰等於 metadata 年齡、arrival 起點
    等於沉底時間，並核對抽樣政策、方法、seed、年齡上限及分層索引。固定日曆模式的
    有效期間是 H-age；結果小於或等於零時回傳零期間及 pre-window 標記。完整沉底回溯
    模式始終由沉底 UTC 向前使用完整 H。時間軸先以整數奈秒計算，輸出秒數有限性及最早
    UTC 有號 64 位範圍也會逐項檢查。
    """

    if type(arrival) is not ArrivalTime:
        raise TypeError("arrival 必須是 exact ArrivalTime")
    if backtrack_mode not in {
        BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
        BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
    }:
        raise ValueError(f"不支援 bed residence 回溯模式：{backtrack_mode!r}")
    maximum_age_days = _strict_int(
        maximum_age_days, label="maximum_age_days", minimum=1
    )
    sampling_seed = _strict_int(sampling_seed, label="sampling_seed", minimum=0)
    if not isinstance(arrival.metadata, dict):
        raise TypeError("ArrivalTime.metadata 必須是 dict")
    metadata = arrival.metadata
    if any("pilot" in str(key).casefold() for key in metadata) or any(
        isinstance(value, str) and "pilot" in value.casefold()
        for value in metadata.values()
    ):
        raise ValueError("pilot metadata 不可進入正式 runtime")

    observation_ns = _metadata_int(
        metadata, "observation_time_utc_ns", minimum=None
    )
    deposition_ns = _metadata_int(
        metadata, "deposition_time_utc_ns", minimum=None
    )
    age_hours = _metadata_int(metadata, "bed_residence_age_hours")
    stratum_index = _metadata_int(metadata, "bed_residence_stratum_index")
    metadata_seed = _metadata_int(metadata, "bed_residence_sampling_seed")
    metadata_max_age = _metadata_int(metadata, "bed_residence_maximum_age_days")
    if not _INT64_MIN <= observation_ns <= _INT64_MAX:
        raise ValueError("metadata observation UTC 奈秒超出有號 64 位範圍")
    if not _INT64_MIN <= deposition_ns <= _INT64_MAX:
        raise ValueError("metadata deposition UTC 奈秒超出有號 64 位範圍")
    if arrival.time_utc_ns != deposition_ns:
        raise ValueError("ArrivalTime.time_utc_ns 必須等於 metadata deposition UTC")
    if observation_ns - deposition_ns != age_hours * _NANOSECONDS_PER_HOUR:
        raise ValueError("observation UTC 減 deposition UTC 必須等於 bed residence age")
    if age_hours > maximum_age_days * _HOURS_PER_DAY:
        raise ValueError("metadata bed residence age 超過 config 最大年齡")
    if metadata_max_age != maximum_age_days:
        raise ValueError("arrival metadata 最大年齡與 config 不一致")
    if metadata_seed != sampling_seed:
        raise ValueError("arrival metadata sampling seed 與 config 不一致")
    if metadata.get("bed_residence_policy_id") != BED_RESIDENCE_POLICY_ID:
        raise ValueError("arrival metadata bed residence policy 不一致")
    if metadata.get("bed_residence_sampling_policy_id") != BED_RESIDENCE_SAMPLING_POLICY_ID:
        raise ValueError("arrival metadata sampling policy 不一致")
    sampling_method = metadata.get("bed_residence_sampling_method_id")
    if sampling_method not in {
        BED_RESIDENCE_SAMPLING_METHOD_ID,
        BED_RESIDENCE_AVAILABILITY_CONDITIONED_SAMPLING_METHOD_ID,
    }:
        raise ValueError("arrival metadata sampling method 不一致或未登錄")
    if sampling_method == BED_RESIDENCE_AVAILABILITY_CONDITIONED_SAMPLING_METHOD_ID:
        if metadata.get("bed_residence_availability_conditioning_policy") != (
            REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID
        ):
            raise ValueError("arrival metadata availability conditioning policy 不一致")
        if not isinstance(metadata.get("bed_residence_availability_rejection_audit"), dict):
            raise ValueError("arrival metadata 缺少 availability rejection audit")
    elif any(
        key in metadata
        for key in (
            "bed_residence_availability_conditioning_policy",
            "bed_residence_availability_rejection_audit",
        )
    ):
        raise ValueError("legacy arrival metadata 不得帶 availability provenance")
    _nonempty_text(
        metadata.get("observation_arrival_time_id"),
        label="ArrivalTime.metadata.observation_arrival_time_id",
    )
    if metadata.get("observation_time_utc") != _format_utc_ns(observation_ns):
        raise ValueError("metadata observation_time_utc 與整數奈秒不一致")
    if metadata.get("deposition_time_utc") != _format_utc_ns(deposition_ns):
        raise ValueError("metadata deposition_time_utc 與整數奈秒不一致")
    expected_stratum = _stratum_index_for_age(
        age_hours,
        maximum_age_days=maximum_age_days,
        sample_count=_SAMPLE_COUNT,
    )
    if stratum_index != expected_stratum:
        raise ValueError("metadata stratum index 與 age_hours 不一致")

    horizon_ns = _days_to_nanoseconds(
        requested_horizon_days, label="requested_horizon_days"
    )
    if backtrack_mode == BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW:
        effective_ns = horizon_ns - age_hours * _NANOSECONDS_PER_HOUR
        pre_window = effective_ns <= 0
        effective_ns = max(effective_ns, 0)
    else:
        effective_ns = horizon_ns
        pre_window = False
    earliest_ns = deposition_ns - effective_ns
    if not _INT64_MIN <= earliest_ns <= _INT64_MAX:
        raise ValueError("earliest forcing UTC 超出有號 64 位奈秒範圍")
    effective_seconds = effective_ns / _NANOSECONDS_PER_SECOND
    if not math.isfinite(effective_seconds) or effective_seconds < 0.0:
        raise ValueError("effective horizon seconds 必須是有限非負值")
    if effective_ns > 0 and effective_seconds <= 0.0:
        raise ValueError("正有效期間無法表示為有限正秒數")
    return BedResidenceTiming(
        effective_horizon_seconds=effective_seconds,
        effective_horizon_ns=effective_ns,
        earliest_forcing_time_utc_ns=earliest_ns,
        pre_window_deposition=pre_window,
    )
