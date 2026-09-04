"""建立來源段—受體的條件式比例、event share 與旅行年齡產品。

本模組只接受已驗證的 source-receptor raw counts、對應秒制 travel-age histograms、
receptor 有效成員分母，或直接接受 exact ``AggregateReleasePayload``。它不讀取檔案、
不重新開啟 trajectory shard，也不依目前結果補建缺少的來源段 key。raw count 與
histogram 必須一一對齊；零事件 key 仍保留，因為「拓撲上有此來源段但本批為零」
與「資料缺列」不是同一種語意。

每一來源段列保存兩種不同比例：條件式通過比例的分母是同一
``study_site_id × receptor_id`` 的有效 member 數；event share 的分母是同一
``study_site_id × receptor_id × boundary_kind`` 下所有 segment raw event count
合計。前者回答有效成員中有多少來自該段，後者回答該受體／邊界分類的事件中有多少
落在該段，兩者不能混稱為同一個機率。

旅行年齡以秒（s）保存，histogram 軸是 ``(age_bin,)``；內部重用既有
``build_histogram_quantiles`` 的 age-bin midpoint 規則，因此輸出的分位數是固定箱
解析度下的中點近似，不是未分箱資料的連續精確分位數。所有 NumPy 陣列、mapping、
records 與比例都在產品邊界 defensive-copy／唯讀化；缺值使用 ``None`` 加明確
``EstimateStatus``，不得以零值冒充不可用。結果只表示條件式來源足跡／相對來源權重，
不是絕對來源機率或因果歸因。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np

from .aggregate_release_payload import AggregateReleasePayload
from .event_aggregation import ReceptorAggregateKey, SourceReceptorAggregateKey
from .report_ratio_statistics import (
    CountRatio,
    EstimateStatus,
    build_count_ratio,
    build_histogram_quantiles,
)
from .report_spec import ReportSpec, validate_report_spec_against_aggregate_spec

__all__ = [
    "DEFAULT_TRAVEL_AGE_QUANTILES",
    "SourceReceptorRecord",
    "SourceReceptorStatistic",
    "SourceReceptorStatistics",
    "SourceReceptorStatisticsProduct",
    "SourceReceptorStatisticsResult",
    "TravelAgeStatistics",
    "TravelAgeStatisticsProduct",
    "build_source_receptor_statistics",
]


_INT64_MAX: Final[int] = int(np.iinfo(np.int64).max)
# ReportSpec 的正式 travel-age quantiles 固定為此五個值；純 mapping 單元入口也
# 只能使用明示 quantiles 或這個版本化常數，facade 則優先從 exact ReportSpec 讀取。
DEFAULT_TRAVEL_AGE_QUANTILES: Final[tuple[float, ...]] = (
    0.05,
    0.25,
    0.5,
    0.75,
    0.95,
)


def _require_native_nonnegative_int(value: object, *, label: str) -> int:
    """驗證 raw count／分母是可安全保存的非負 signed-int64 原生整數。"""

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 Python int，不接受 bool 或 NumPy scalar")
    if value < 0:
        raise ValueError(f"{label} 不可為負數")
    if value > _INT64_MAX:
        raise ValueError(f"{label} 不得超過 signed int64 上限")
    return value


def _require_positive_native_int(value: object, *, label: str) -> int:
    """驗證低樣本門檻是大於零的原生 Python ``int``。"""

    number = _require_native_nonnegative_int(value, label=label)
    if number == 0:
        raise ValueError(f"{label} 必須大於 0")
    return number


def _snapshot_text_tuple(value: object, *, label: str) -> tuple[str, ...]:
    """複製並驗證非空、唯一且沒有首尾空白的文字 tuple。"""

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} 必須是非字串、非 mapping sequence")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} 必須是可 materialize 的 sequence") from error
    if not items:
        raise ValueError(f"{label} 不可為空")
    normalized: list[str] = []
    for index, item in enumerate(items):
        if type(item) is not str:
            raise TypeError(f"{label}[{index}] 必須是原生 str")
        if not item or item != item.strip():
            raise ValueError(f"{label}[{index}] 必須是非空且沒有首尾空白的文字")
        if item in normalized:
            raise ValueError(f"{label} 不可有重複文字")
        normalized.append(item)
    return tuple(normalized)


def _snapshot_quantiles(value: object) -> tuple[float, ...]:
    """複製並驗證 age quantile 的原生 float 序列。

    quantile 的順序會進入 renderer 欄位與 age 近似 mapping；不排序、不去重，也不
    接受 NumPy scalar，避免同一份 report spec 在不同 I/O 邊界得到不同欄位順序。
    """

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("quantiles 必須是非字串、非 mapping sequence")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError("quantiles 必須是可 materialize 的 sequence") from error
    if not items:
        raise ValueError("quantiles 不可為空")
    normalized: list[float] = []
    for index, item in enumerate(items):
        if type(item) is not float:
            raise TypeError(f"quantiles[{index}] 必須是原生 Python float")
        if not math.isfinite(item) or not 0.0 < item < 1.0:
            raise ValueError("quantiles 必須是介於 0 與 1 之間的有限值")
        if item in normalized:
            raise ValueError("quantiles 不可重複")
        normalized.append(item)
    return tuple(normalized)


def _snapshot_age_edges(value: object) -> np.ndarray:
    """複製共同 age 軸，要求從 0 秒開始的有限嚴格遞增格線。"""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("age_bin_edges_seconds 必須是一維數值陣列") from error
    if raw.ndim != 1 or raw.size < 2 or raw.dtype.kind not in "iuf":
        raise ValueError("age_bin_edges_seconds 必須是至少兩點的一維數值陣列")
    copied = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(copied)):
        raise ValueError("age_bin_edges_seconds 必須全部為有限值")
    if copied[0] != 0.0:
        raise ValueError("age_bin_edges_seconds 第一個邊界必須精確為 0 秒")
    if not np.all(copied[1:] > copied[:-1]):
        raise ValueError("age_bin_edges_seconds 必須嚴格遞增")
    copied.setflags(write=False)
    return copied


def _snapshot_histogram(value: object, *, label: str) -> np.ndarray:
    """複製一維非負 age-bin raw count，防止 uint64 超界轉型繞回。"""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 必須是一維整數陣列") from error
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise TypeError(f"{label} 必須是一維整數陣列，不接受 bool 或浮點")
    for element in raw.flat:
        _require_native_nonnegative_int(int(element), label=label)
    copied = np.array(raw, dtype=np.int64, copy=True)
    copied.setflags(write=False)
    return copied


def _snapshot_source_raw_mapping(
    value: object,
) -> Mapping[SourceReceptorAggregateKey, int]:
    """封存來源段 key 到 raw scalar count 的 mapping。"""

    if not isinstance(value, Mapping):
        raise TypeError("source_receptor_raw_count 必須是 mapping")
    copied: dict[SourceReceptorAggregateKey, int] = {}
    for key, raw_count in value.items():
        if type(key) is not SourceReceptorAggregateKey:
            raise TypeError("source_receptor_raw_count 的 key 必須是 exact SourceReceptorAggregateKey")
        if key in copied:
            raise ValueError("source_receptor_raw_count 不可有重複 key")
        copied[key] = _require_native_nonnegative_int(
            raw_count,
            label=f"source_receptor_raw_count[{key!r}]",
        )
    return MappingProxyType(copied)


def _snapshot_source_histogram_mapping(
    value: object,
) -> Mapping[SourceReceptorAggregateKey, np.ndarray]:
    """封存來源段到一維秒制旅行年齡 histogram 的 mapping。"""

    if not isinstance(value, Mapping):
        raise TypeError("source_receptor_travel_age_histogram 必須是 mapping")
    copied: dict[SourceReceptorAggregateKey, np.ndarray] = {}
    for key, histogram in value.items():
        if type(key) is not SourceReceptorAggregateKey:
            raise TypeError(
                "source_receptor_travel_age_histogram 的 key 必須是 exact SourceReceptorAggregateKey"
            )
        if key in copied:
            raise ValueError("source_receptor_travel_age_histogram 不可有重複 key")
        copied[key] = _snapshot_histogram(
            histogram,
            label=f"source_receptor_travel_age_histogram[{key!r}]",
        )
    return MappingProxyType(copied)


def _snapshot_receptor_denominators(
    value: object,
) -> Mapping[ReceptorAggregateKey, int]:
    """封存 receptor 有效成員分母；允許額外的零來源 receptor 拓撲列。"""

    if not isinstance(value, Mapping):
        raise TypeError("valid_member_denominator_by_receptor 必須是 mapping")
    copied: dict[ReceptorAggregateKey, int] = {}
    for key, denominator in value.items():
        if type(key) is not ReceptorAggregateKey:
            raise TypeError(
                "valid_member_denominator_by_receptor 的 key 必須是 exact ReceptorAggregateKey"
            )
        if key in copied:
            raise ValueError("valid_member_denominator_by_receptor 不可有重複 key")
        copied[key] = _require_native_nonnegative_int(
            denominator,
            label=f"valid_member_denominator_by_receptor[{key!r}]",
        )
    return MappingProxyType(copied)


def _midpoints_and_widths(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """依既有分箱定義建立秒制中點與實際 bin 寬度的唯讀副本。"""

    midpoints = edges[:-1] * 0.5 + edges[1:] * 0.5
    widths = np.diff(edges)
    if not np.all(np.isfinite(midpoints)) or not np.all(np.isfinite(widths)):
        raise ValueError("age-bin midpoint／width 必須全部為有限秒數")
    midpoints = np.array(midpoints, dtype=np.float64, copy=True)
    widths = np.array(widths, dtype=np.float64, copy=True)
    midpoints.setflags(write=False)
    widths.setflags(write=False)
    return midpoints, widths


def _age_quantile_values(
    histogram: np.ndarray,
    *,
    edges: np.ndarray,
    quantiles: tuple[float, ...],
    minimum_count: int,
) -> tuple[EstimateStatus, Mapping[float, float | None]]:
    """重用既有三維 histogram quantile core，取出單一來源段的 scalar 結果。"""

    # 既有 aggregate source-receptor histogram 是一維 (age_bin)，而共用的報告
    # quantile core 固定以 (y_cell, x_cell, age_bin) 運算；包成 1×1×age 只是在
    # 邊界適配軸形狀，不會改變計數、bin 包含規則或中點定義。
    product = build_histogram_quantiles(
        histogram.reshape(1, 1, histogram.size),
        age_bin_edges_seconds=edges,
        quantiles=quantiles,
        minimum_count=minimum_count,
    )
    status = product.status[0, 0]
    values: dict[float, float | None] = {}
    for quantile in quantiles:
        value = product.quantile_values_seconds[quantile][0, 0]
        values[quantile] = None if status is EstimateStatus.zero_denominator else float(value)
    return status, MappingProxyType(values)


@dataclass(frozen=True, slots=True)
class TravelAgeStatistics:
    """保存單一來源段的一維 travel-age histogram 與分箱分位數。

    ``raw_count`` 的軸是 ``(age_bin,)``，計數代表該來源段通過事件的 member raw
    count；``age_bin_edges_seconds``、``bin_midpoints_seconds`` 與
    ``bin_widths_seconds`` 全部使用秒（s）。分位數值由既有
    ``first_passage_quantiles`` 規則以 midpoint 近似，不能被誤標為未分箱精確 age。
    raw count 為零時 quantile value 使用 ``None``，而 ``status`` 明確保存
    ``zero_denominator``；正數但低於門檻時仍保存有限中點近似與 ``low_count``。
    """

    raw_count: np.ndarray
    age_bin_edges_seconds: np.ndarray
    quantiles: tuple[float, ...]
    quantile_values_seconds: Mapping[float, float | None]
    minimum_count: int = 1
    status: EstimateStatus | None = None
    _cell_count: int = field(init=False, repr=False)
    _bin_midpoints_seconds: np.ndarray = field(init=False, repr=False)
    _bin_widths_seconds: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """驗證 histogram、分箱幾何、quantile midpoint 與狀態的一致性。"""

        raw_count = _snapshot_histogram(self.raw_count, label="TravelAgeStatistics.raw_count")
        edges = _snapshot_age_edges(self.age_bin_edges_seconds)
        if raw_count.size != edges.size - 1:
            raise ValueError("raw_count 的 age-bin 軸長度必須等於 edges.size - 1")
        quantiles = _snapshot_quantiles(self.quantiles)
        minimum_count = _require_positive_native_int(
            self.minimum_count,
            label="TravelAgeStatistics.minimum_count",
        )
        cell_count = sum(int(element) for element in raw_count.flat)
        if cell_count > _INT64_MAX:
            raise ValueError("TravelAgeStatistics raw count 合計不得超過 signed int64 上限")
        expected_status = (
            EstimateStatus.zero_denominator
            if cell_count == 0
            else EstimateStatus.low_count
            if cell_count < minimum_count
            else EstimateStatus.available
        )
        if self.status is not None and type(self.status) is not EstimateStatus:
            raise TypeError("TravelAgeStatistics.status 必須是 EstimateStatus 或 None")
        if self.status is not None and self.status is not expected_status:
            raise ValueError("TravelAgeStatistics.status 必須與 raw count 及門檻一致")
        status = expected_status

        midpoints, widths = _midpoints_and_widths(edges)
        expected_status_from_core, expected_values = _age_quantile_values(
            raw_count,
            edges=edges,
            quantiles=quantiles,
            minimum_count=minimum_count,
        )
        if expected_status_from_core is not status:
            raise RuntimeError("既有 age quantile core 與 raw count 狀態不一致")
        if not isinstance(self.quantile_values_seconds, Mapping):
            raise TypeError("quantile_values_seconds 必須是 mapping")
        try:
            items = tuple(self.quantile_values_seconds.items())
        except Exception as error:
            raise ValueError("quantile_values_seconds 無法穩定讀取") from error
        if tuple(key for key, _ in items) != quantiles:
            raise ValueError("quantile_values_seconds key 順序必須與 quantiles 完全一致")
        provided_values: dict[float, float | None] = {}
        for quantile, value in items:
            if value is None:
                normalized_value = None
            else:
                if type(value) is not float:
                    raise TypeError(
                        f"quantile_values_seconds[{quantile}] 必須是原生 float 或 None"
                    )
                if not math.isfinite(value):
                    raise ValueError("quantile_values_seconds 的正樣本值必須是有限秒數")
                normalized_value = value
            expected_value = expected_values[quantile]
            if normalized_value != expected_value:
                raise ValueError(
                    f"quantile_values_seconds[{quantile}] 必須精確符合 age-bin midpoint 近似"
                )
            provided_values[quantile] = normalized_value

        object.__setattr__(self, "raw_count", raw_count)
        object.__setattr__(self, "age_bin_edges_seconds", edges)
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "quantile_values_seconds", MappingProxyType(provided_values))
        object.__setattr__(self, "minimum_count", minimum_count)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "_cell_count", cell_count)
        object.__setattr__(self, "_bin_midpoints_seconds", midpoints)
        object.__setattr__(self, "_bin_widths_seconds", widths)

    @property
    def raw_histogram(self) -> np.ndarray:
        """raw travel-age histogram 的簡短別名。"""

        return self.raw_count

    @property
    def raw_counts(self) -> np.ndarray:
        """raw travel-age histogram 的複數欄位別名。"""

        return self.raw_count

    @property
    def cell_count(self) -> int:
        """回傳此來源段 travel-age histogram 的 Python int 總數。"""

        return self._cell_count

    @property
    def quantile_values(self) -> Mapping[float, float | None]:
        """travel-age quantile mapping 的簡短欄位別名。"""

        return self.quantile_values_seconds

    @property
    def values(self) -> Mapping[float, float | None]:
        """供 renderer/table adapter 使用的 quantile mapping 別名。"""

        return self.quantile_values_seconds

    @property
    def bin_midpoints_seconds(self) -> np.ndarray:
        """回傳固定 age-bin midpoint 秒數近似的唯讀向量。"""

        return self._bin_midpoints_seconds

    @property
    def age_bin_midpoints_seconds(self) -> np.ndarray:
        """``bin_midpoints_seconds`` 的欄位別名。"""

        return self._bin_midpoints_seconds

    @property
    def bin_widths_seconds(self) -> np.ndarray:
        """回傳實際 age-bin 寬度秒數，作為分箱解析度 provenance。"""

        return self._bin_widths_seconds

    @property
    def bin_resolution_seconds(self) -> np.ndarray:
        """``bin_widths_seconds`` 的 renderer-facing 別名。"""

        return self._bin_widths_seconds

    @property
    def status_code(self) -> str:
        """回傳明確的字串狀態碼，供 JSON／表格交換層使用。"""

        return self.status.value


@dataclass(frozen=True, slots=True)
class SourceReceptorStatistic:
    """一個 source segment 到 receptor 的 immutable 統計列。

    ``raw_numerator`` 是 ``SourceReceptorAggregateKey`` 對應的 raw member count；
    ``valid_receptor_denominator`` 用於 ``conditional_ratio``，
    ``event_share_denominator`` 用於 ``event_share_ratio``。兩個 ``CountRatio``
    都保留自己的 raw denominator 與 low/zero status，因此下游不能把 event share
    誤當作有效成員條件式比例。``travel_age`` 則保存同一 key 的秒制分箱 histogram
    與 midpoint quantiles。
    """

    key: SourceReceptorAggregateKey
    raw_numerator: int
    valid_receptor_denominator: int
    event_share_denominator: int
    travel_age: TravelAgeStatistics
    minimum_count: int = 1
    _conditional_ratio: CountRatio = field(init=False, repr=False)
    _event_share_ratio: CountRatio = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """驗證來源 key、兩組分母、raw count 上限與 immutable ratio。"""

        if type(self.key) is not SourceReceptorAggregateKey:
            raise TypeError("key 必須是 exact SourceReceptorAggregateKey")
        raw_numerator = _require_native_nonnegative_int(
            self.raw_numerator,
            label="raw_numerator",
        )
        valid_denominator = _require_native_nonnegative_int(
            self.valid_receptor_denominator,
            label="valid_receptor_denominator",
        )
        event_denominator = _require_native_nonnegative_int(
            self.event_share_denominator,
            label="event_share_denominator",
        )
        if raw_numerator > valid_denominator:
            raise ValueError("raw_numerator 不可大於 valid_receptor_denominator")
        if raw_numerator > event_denominator:
            raise ValueError("raw_numerator 不可大於 event_share_denominator")
        if type(self.travel_age) is not TravelAgeStatistics:
            raise TypeError("travel_age 必須是 exact TravelAgeStatistics")
        minimum_count = _require_positive_native_int(
            self.minimum_count,
            label="SourceReceptorStatistic.minimum_count",
        )
        if self.travel_age.minimum_count != minimum_count:
            raise ValueError("travel_age.minimum_count 必須與 source-receptor minimum_count 相同")
        if self.travel_age.cell_count != raw_numerator:
            raise ValueError(
                "travel_age.cell_count 必須等於 source-receptor raw_numerator"
            )
        conditional = build_count_ratio(
            raw_numerator,
            valid_denominator,
            minimum_count=minimum_count,
        )
        event_share = build_count_ratio(
            raw_numerator,
            event_denominator,
            minimum_count=minimum_count,
        )
        object.__setattr__(self, "raw_numerator", raw_numerator)
        object.__setattr__(self, "valid_receptor_denominator", valid_denominator)
        object.__setattr__(self, "event_share_denominator", event_denominator)
        object.__setattr__(self, "minimum_count", minimum_count)
        object.__setattr__(self, "_conditional_ratio", conditional)
        object.__setattr__(self, "_event_share_ratio", event_share)

    @property
    def study_site_id(self) -> str:
        """來源站點識別碼。"""

        return self.key.study_site_id

    @property
    def receptor_id(self) -> str:
        """受體識別碼。"""

        return self.key.receptor_id

    @property
    def boundary_kind(self) -> str:
        """邊界分類（``local`` 或 ``outer``）。"""

        return self.key.boundary_kind

    @property
    def boundary_segment_id(self) -> str:
        """來源邊界段識別碼。"""

        return self.key.boundary_segment_id

    @property
    def raw_count(self) -> int:
        """raw source-receptor numerator 的欄位別名。"""

        return self.raw_numerator

    @property
    def numerator(self) -> int:
        """``raw_numerator`` 的通用 ratio 欄位別名。"""

        return self.raw_numerator

    @property
    def denominator(self) -> int:
        """條件式比例所用有效 receptor 分母的通用欄位別名。"""

        return self.valid_receptor_denominator

    @property
    def conditional_ratio(self) -> CountRatio:
        """回傳以有效 receptor member 為分母的條件式通過比例。"""

        return self._conditional_ratio

    @property
    def conditional_fraction(self) -> float | None:
        """條件式通過比例的 scalar 別名；零分母時為 ``None``。"""

        return self._conditional_ratio.ratio

    @property
    def conditional_proportion(self) -> float | None:
        """``conditional_fraction`` 的語意別名。"""

        return self._conditional_ratio.ratio

    @property
    def ratio(self) -> CountRatio:
        """預設 ratio 介面，明確指向條件式通過比例。"""

        return self._conditional_ratio

    @property
    def event_share_ratio(self) -> CountRatio:
        """回傳同 receptor／boundary kind 內部 event share。"""

        return self._event_share_ratio

    @property
    def event_share(self) -> float | None:
        """event share 的 scalar 別名；零事件分母時為 ``None``。"""

        return self._event_share_ratio.ratio

    @property
    def event_fraction(self) -> float | None:
        """``event_share`` 的比例欄位別名。"""

        return self._event_share_ratio.ratio

    @property
    def event_share_fraction(self) -> float | None:
        """event share 的完整欄位名稱別名；零事件分母時仍為 ``None``。"""

        return self._event_share_ratio.ratio

    @property
    def travel_age_histogram(self) -> np.ndarray:
        """回傳此來源段的秒制 travel-age raw histogram。"""

        return self.travel_age.raw_count

    @property
    def age_bin_edges_seconds(self) -> np.ndarray:
        """回傳此來源段共同 age 軸的唯讀秒數格線。"""

        return self.travel_age.age_bin_edges_seconds

    @property
    def travel_age_quantiles(self) -> Mapping[float, float | None]:
        """回傳 age-bin midpoint quantiles 的 mapping。"""

        return self.travel_age.quantile_values_seconds

    @property
    def travel_age_quantile_values_seconds(self) -> Mapping[float, float | None]:
        """``travel_age_quantiles`` 的完整欄位別名。"""

        return self.travel_age.quantile_values_seconds


@dataclass(frozen=True, slots=True)
class SourceReceptorStatistics:
    """保存完整 source-receptor typed rows 的 immutable collection。

    ``records`` 依站點、受體、邊界分類與來源段 ID 的 canonical tuple 排序；每個
    aggregate key 只能有一列。產品層另外保存共同的秒制 age edges、quantile 順序
    與 raw low-sample 門檻，確保 renderer 不必為每列自行挑 quantile 或分母。即使
    某來源段 raw count 為零，也會保留該列與其 ``zero_denominator`` 狀態，不把零值
    當成缺列或可估計的零比例。
    """

    records: tuple[SourceReceptorStatistic, ...]
    age_bin_edges_seconds: np.ndarray | None = None
    quantiles: tuple[float, ...] = ()
    minimum_count: int = 1
    _by_key: Mapping[SourceReceptorAggregateKey, SourceReceptorStatistic] = field(
        init=False,
        repr=False,
    )
    _site_ids: tuple[str, ...] = field(init=False, repr=False)
    _receptor_ids: tuple[tuple[str, str], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """驗證 rows 的 key 唯一性、共同 age 軸與 nested mapping 封存。"""

        if isinstance(self.records, (str, bytes, bytearray, Mapping)):
            raise TypeError("records 必須是 SourceReceptorStatistic iterable")
        try:
            records = tuple(self.records)
        except (TypeError, ValueError) as error:
            raise TypeError("records 必須是 SourceReceptorStatistic iterable") from error
        if any(type(record) is not SourceReceptorStatistic for record in records):
            raise TypeError("records 每一列必須是 exact SourceReceptorStatistic")
        by_key: dict[SourceReceptorAggregateKey, SourceReceptorStatistic] = {}
        for record in records:
            if record.key in by_key:
                raise ValueError("records 的 source-receptor key 不可重複")
            by_key[record.key] = record

        # 即使每一列各自沒有超過 receptor 分母，多個 segment 合計仍可能不合法；
        # event share 的 group 母體必須與條件式比例共用同一批有效 member。此處在
        # collection 邊界再核對一次，避免 caller 以手動 row 組合繞過 builder gate。
        grouped_raw: dict[tuple[str, str, str], int] = {}
        grouped_denominator: dict[tuple[str, str, str], int] = {}
        for record in records:
            group = (
                record.study_site_id,
                record.receptor_id,
                record.boundary_kind,
            )
            grouped_raw[group] = grouped_raw.get(group, 0) + record.raw_numerator
            existing_denominator = grouped_denominator.setdefault(
                group,
                record.valid_receptor_denominator,
            )
            if existing_denominator != record.valid_receptor_denominator:
                raise ValueError("同一 receptor／boundary kind 的 valid denominator 必須一致")
        for group, raw_total in grouped_raw.items():
            if raw_total > grouped_denominator[group]:
                raise ValueError(
                    f"source-receptor {group!r} 的 raw count 合計不可大於 valid receptor denominator"
                )

        if self.age_bin_edges_seconds is None:
            if not records:
                raise ValueError("空 records 必須明示 age_bin_edges_seconds")
            edges = _snapshot_age_edges(records[0].travel_age.age_bin_edges_seconds)
        else:
            edges = _snapshot_age_edges(self.age_bin_edges_seconds)
        if self.quantiles:
            quantiles = _snapshot_quantiles(self.quantiles)
        elif records:
            quantiles = tuple(records[0].travel_age.quantiles)
        else:
            raise ValueError("空 records 必須明示 quantiles")
        minimum_count = _require_positive_native_int(
            self.minimum_count,
            label="SourceReceptorStatistics.minimum_count",
        )
        for record in records:
            if record.minimum_count != minimum_count:
                raise ValueError("所有 records 必須使用相同 minimum_count")
            if not np.array_equal(record.travel_age.age_bin_edges_seconds, edges):
                raise ValueError("所有 records 必須共享相同 age_bin_edges_seconds")
            if record.travel_age.quantiles != quantiles:
                raise ValueError("所有 records 必須共享相同 quantiles 順序")

        ordered_keys = tuple(
            sorted(
                by_key,
                key=lambda key: (
                    key.study_site_id,
                    key.receptor_id,
                    key.boundary_kind,
                    key.boundary_segment_id,
                ),
            )
        )
        ordered_records = tuple(by_key[key] for key in ordered_keys)
        ordered_mapping = {key: by_key[key] for key in ordered_keys}
        site_ids = tuple(sorted({key.study_site_id for key in ordered_keys}))
        receptor_ids = tuple(
            sorted({(key.study_site_id, key.receptor_id) for key in ordered_keys})
        )

        object.__setattr__(self, "records", ordered_records)
        object.__setattr__(self, "age_bin_edges_seconds", edges)
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "minimum_count", minimum_count)
        object.__setattr__(self, "_by_key", MappingProxyType(ordered_mapping))
        object.__setattr__(self, "_site_ids", site_ids)
        object.__setattr__(self, "_receptor_ids", receptor_ids)

    @property
    def by_key(self) -> Mapping[SourceReceptorAggregateKey, SourceReceptorStatistic]:
        """回傳 source-receptor key 到 immutable row 的 mapping。"""

        return self._by_key

    @property
    def statistics(self) -> Mapping[SourceReceptorAggregateKey, SourceReceptorStatistic]:
        """``by_key`` 的 renderer-facing 別名。"""

        return self._by_key

    @property
    def statistics_by_key(self) -> Mapping[SourceReceptorAggregateKey, SourceReceptorStatistic]:
        """``by_key`` 的完整欄位別名。"""

        return self._by_key

    @property
    def source_receptor_records(self) -> tuple[SourceReceptorStatistic, ...]:
        """``records`` 的語意別名。"""

        return self.records

    @property
    def site_ids(self) -> tuple[str, ...]:
        """回傳產品涵蓋的排序後站點 ID。"""

        return self._site_ids

    @property
    def receptor_ids(self) -> tuple[tuple[str, str], ...]:
        """回傳排序後的 ``(study_site_id, receptor_id)`` join key。"""

        return self._receptor_ids

    def __getitem__(self, key: SourceReceptorAggregateKey) -> SourceReceptorStatistic:
        """依 immutable aggregate key 讀取單一來源段列。"""

        return self._by_key[key]

    def __iter__(self):
        """依 canonical key 順序迭代來源段 key。"""

        return iter(self._by_key)

    def __len__(self) -> int:
        """回傳來源段統計列數。"""

        return len(self._by_key)


# 與既有 MaterialStatisticsProduct 的命名習慣相容；兩者是同一個 exact collection
# class，不複製資料，也不建立第二套 schema。
SourceReceptorStatisticsProduct = SourceReceptorStatistics
SourceReceptorStatisticsResult = SourceReceptorStatistics
SourceReceptorRecord = SourceReceptorStatistic
TravelAgeStatisticsProduct = TravelAgeStatistics


def _resolve_payload_or_mapping(
    first: object | None,
    *,
    aggregate_payload: AggregateReleasePayload | None,
    payload: AggregateReleasePayload | None,
) -> tuple[AggregateReleasePayload | None, object | None]:
    """統一 source builder 的 positional／keyword payload 入口。"""

    candidates = [candidate for candidate in (aggregate_payload, payload) if candidate is not None]
    if len(candidates) > 1:
        raise ValueError("aggregate_payload 與 payload 不可同時提供")
    resolved_payload = candidates[0] if candidates else None
    if resolved_payload is not None:
        if type(resolved_payload) is not AggregateReleasePayload:
            raise TypeError("aggregate_payload 必須是 exact AggregateReleasePayload")
        if first is not None:
            raise ValueError("payload 已由 keyword 提供時不可再指定 positional data")
        return resolved_payload, None
    if type(first) is AggregateReleasePayload:
        return first, None
    return None, first


def _resolve_report_policy(
    *,
    payload: AggregateReleasePayload | None,
    report_spec: ReportSpec | None,
    quantiles: object | None,
    minimum_count: object | None,
) -> tuple[tuple[float, ...], int]:
    """解析固定 report policy，並拒絕與 ReportSpec 不一致的 renderer 參數。"""

    if report_spec is not None and type(report_spec) is not ReportSpec:
        raise TypeError("report_spec 必須是 exact ReportSpec")
    if report_spec is not None and payload is not None:
        # payload 本身已把 AggregateSpec 鎖在 event／pathway；此處再次執行公開
        # binding gate，避免 caller 以另一份 report spec 的 quantile 偷換產品。
        validate_report_spec_against_aggregate_spec(report_spec, payload.aggregate_spec)
    expected_quantiles = (
        tuple(report_spec.travel_age_quantiles)
        if report_spec is not None
        else DEFAULT_TRAVEL_AGE_QUANTILES
    )
    if quantiles is None:
        resolved_quantiles = expected_quantiles
    else:
        resolved_quantiles = _snapshot_quantiles(quantiles)
        if report_spec is not None and resolved_quantiles != expected_quantiles:
            raise ValueError("quantiles 必須精確等於 report_spec.travel_age_quantiles")
    if minimum_count is None:
        resolved_minimum = (
            report_spec.low_sample_min_member_count if report_spec is not None else 1
        )
    else:
        resolved_minimum = _require_positive_native_int(
            minimum_count,
            label="minimum_count",
        )
        if report_spec is not None and resolved_minimum != report_spec.low_sample_min_member_count:
            raise ValueError("minimum_count 必須精確等於 report_spec.low_sample_min_member_count")
    return tuple(resolved_quantiles), resolved_minimum


def build_source_receptor_statistics(
    source_receptor_raw_count: AggregateReleasePayload
    | Mapping[SourceReceptorAggregateKey, int]
    | None = None,
    source_receptor_travel_age_histogram: Mapping[SourceReceptorAggregateKey, np.ndarray]
    | None = None,
    valid_member_denominator_by_receptor: Mapping[ReceptorAggregateKey, int] | None = None,
    *,
    age_bin_edges_seconds: Sequence[float] | np.ndarray | None = None,
    quantiles: Sequence[float] | None = None,
    minimum_count: int | None = None,
    aggregate_payload: AggregateReleasePayload | None = None,
    payload: AggregateReleasePayload | None = None,
    report_spec: ReportSpec | None = None,
) -> SourceReceptorStatistics:
    """由 source-receptor aggregate 建立條件式比例與旅行年齡產品。

    Args:
        source_receptor_raw_count: exact ``AggregateReleasePayload``，或
            ``SourceReceptorAggregateKey`` 到 raw member count 的 mapping。mapping
            入口不能省略已驗證拓撲中的零來源段。
        source_receptor_travel_age_histogram: 每個來源段對應的一維 age-bin raw count；
            軸是 ``(age_bin,)``，秒軸由 ``age_bin_edges_seconds`` 定義。
        valid_member_denominator_by_receptor: ``ReceptorAggregateKey`` 到有效 member
            分母的 mapping；失敗 member 已由既有 aggregate denominator policy 排除。
        age_bin_edges_seconds: 純 mapping 入口的共同秒制格線。payload 入口固定重用
            event aggregate／AggregateSpec 的同一組 edges。
        quantiles: 純 mapping 入口可明示 quantile；省略時使用版本化五分位數常數。
            若提供 exact ``ReportSpec``，必須精確等於其 ``travel_age_quantiles``，
            不允許 renderer 傳入臨時 quantile。
        minimum_count: raw numerator／age histogram 的低樣本門檻；有 ReportSpec 時
            必須精確等於 ``low_sample_min_member_count``。
        report_spec: optional exact ReportSpec。payload 入口若提供，會再次執行
            AggregateSpec binding；完整 facade 會使用此入口固定 report policy。

    Returns:
        frozen ``SourceReceptorStatistics`` collection；每一列都保存 raw count、
        有效 receptor 分母、條件式比例、同 receptor/kind 內部 event share、秒制
        histogram、中點近似與明確 status。

    Raises:
        TypeError／ValueError: key set、histogram、age 軸、分母、比例或 report policy
            不符合契約時。
    """

    resolved_payload, raw_counts = _resolve_payload_or_mapping(
        source_receptor_raw_count,
        aggregate_payload=aggregate_payload,
        payload=payload,
    )
    if resolved_payload is not None:
        if any(
            value is not None
            for value in (
                source_receptor_travel_age_histogram,
                valid_member_denominator_by_receptor,
                age_bin_edges_seconds,
            )
        ):
            raise ValueError("payload 入口不可混用手動 source mapping 或 age edges")
        event = resolved_payload.event_aggregate
        raw_counts = event.source_receptor_raw_count
        travel_histograms = event.source_receptor_travel_age_histogram
        receptor_denominators = event.valid_member_denominator_by_receptor
        resolved_edges = _snapshot_age_edges(event.age_bin_edges_seconds)
    else:
        if raw_counts is None:
            raise TypeError("必須提供 AggregateReleasePayload 或 source-receptor raw mapping")
        if source_receptor_travel_age_histogram is None:
            raise TypeError("純 mapping 入口必須提供 source-receptor travel-age histogram")
        if valid_member_denominator_by_receptor is None:
            raise TypeError("純 mapping 入口必須提供 receptor valid denominator")
        if age_bin_edges_seconds is None:
            raise TypeError("純 mapping 入口必須提供 age_bin_edges_seconds")
        travel_histograms = source_receptor_travel_age_histogram
        receptor_denominators = valid_member_denominator_by_receptor
        resolved_edges = _snapshot_age_edges(age_bin_edges_seconds)

    resolved_quantiles, resolved_minimum = _resolve_report_policy(
        payload=resolved_payload,
        report_spec=report_spec,
        quantiles=quantiles,
        minimum_count=minimum_count,
    )
    raw_mapping = _snapshot_source_raw_mapping(raw_counts)
    histogram_mapping = _snapshot_source_histogram_mapping(travel_histograms)
    denominator_mapping = _snapshot_receptor_denominators(receptor_denominators)
    if set(raw_mapping) != set(histogram_mapping):
        raise ValueError("source-receptor raw count 與 travel-age histogram 的 key set 必須相同")

    # 同一 receptor／boundary kind 的 event share 分母跨越 segment 累加；累加器使用
    # Python int，直到通過 int64 上限檢查後才交給 CountRatio，避免 NumPy sum 繞回。
    event_denominators: dict[tuple[str, str, str], int] = {}
    for key, count in raw_mapping.items():
        receptor_key = ReceptorAggregateKey(key.study_site_id, key.receptor_id)
        if receptor_key not in denominator_mapping:
            raise ValueError(f"source-receptor key {key!r} 缺少對應 receptor valid denominator")
        group_key = (key.study_site_id, key.receptor_id, key.boundary_kind)
        total = event_denominators.get(group_key, 0) + count
        if total > _INT64_MAX:
            raise ValueError("event_share_denominator 不得超過 signed int64 上限")
        event_denominators[group_key] = total
    for group_key, event_denominator in event_denominators.items():
        receptor_key = ReceptorAggregateKey(group_key[0], group_key[1])
        if event_denominator > denominator_mapping[receptor_key]:
            raise ValueError(
                f"source-receptor {group_key!r} 的 raw count 合計不可大於 valid receptor denominator"
            )

    records: list[SourceReceptorStatistic] = []
    ordered_keys = sorted(
        raw_mapping,
        key=lambda key: (
            key.study_site_id,
            key.receptor_id,
            key.boundary_kind,
            key.boundary_segment_id,
        ),
    )
    for key in ordered_keys:
        histogram = histogram_mapping[key]
        raw_count = raw_mapping[key]
        histogram_total = sum(int(element) for element in histogram.flat)
        if histogram_total != raw_count:
            raise ValueError(
                f"source-receptor key {key!r} 的 raw count 必須等於 travel-age histogram 合計"
            )
        receptor_key = ReceptorAggregateKey(key.study_site_id, key.receptor_id)
        travel_age_status, travel_age_values = _age_quantile_values(
            histogram,
            edges=resolved_edges,
            quantiles=resolved_quantiles,
            minimum_count=resolved_minimum,
        )
        travel_age = TravelAgeStatistics(
            raw_count=histogram,
            age_bin_edges_seconds=resolved_edges,
            quantiles=resolved_quantiles,
            quantile_values_seconds=travel_age_values,
            minimum_count=resolved_minimum,
            status=travel_age_status,
        )
        group_key = (key.study_site_id, key.receptor_id, key.boundary_kind)
        records.append(
            SourceReceptorStatistic(
                key=key,
                raw_numerator=raw_count,
                valid_receptor_denominator=denominator_mapping[receptor_key],
                event_share_denominator=event_denominators[group_key],
                travel_age=travel_age,
                minimum_count=resolved_minimum,
            )
        )

    return SourceReceptorStatistics(
        records=tuple(records),
        age_bin_edges_seconds=resolved_edges,
        quantiles=resolved_quantiles,
        minimum_count=resolved_minimum,
    )
