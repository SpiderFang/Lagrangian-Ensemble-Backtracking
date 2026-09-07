"""以一次遍歷建立軌跡環境、材質、代表案例與有效 pathway 統計。

本模組包含兩個層次。既有的 ``EnvironmentCompletenessAccumulator`` 逐筆消費已完成的
``ParticleResult``，不保存完整結果或觀測序列，只留下每站的整數計數、固定垂向分箱
陣列及 duplicate identity set；新增的 ``TrajectoryReportAccumulator`` 則把同一條
trajectory stream 同步送入環境、材質、代表軌跡與 pathway 四個 bounded reducer。
正式 pathway 必須由這條 stream 重新計算，因為 aggregate pipeline 的 pathway 可能包含
``DATA_GAP``／``NUMERICAL_FAILURE``；本模組只把有效成員交給
``stream_pathway_first_passage``，使 pathway numerator 與 valid denominator 來自同一母體。
每個 shard 的有效 ``ParticleResult`` 只在當前 shard 暫存，建立 chunk 後立即釋放，不保存
已處理 shard，也不重讀 aggregate pipeline 的 private helper。

Observation 的環境狀態只使用既有 ``environment_sample_status``。只有明確標成
``VALID`` 且帶有有限 ``eta_m``、``bed_z_m``、``z_m`` 的觀測，才會計入 positive-down
深度與離底高度分箱；``INVALID`` 與 ``NOT_SAMPLED`` 不以零值代替，也不會猜測其缺失
上下界。深度定義為 ``eta_m - z_m``，離底高度定義為 ``z_m - bed_z_m``，兩者單位均為
公尺（m）。分箱採左閉右開，最後一箱右端點包含在內；第一個邊界以下與最後一個邊界
以上分別保存為 underflow／overflow。

這些 raw count 與 pathway 只能描述指定 trajectory stream 的環境資料可追溯程度及
條件式來源足跡，不能被解讀成絕對來源機率、因果歸因或已完成的 OCM／NWW3 科學驗證。
正式流程仍須由上游已驗收的結果與完整 provenance 建立，且缺值狀態必須維持原始語意。
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np

from .aggregate_release_records import ScenarioStratum
from .aggregate_spec import AggregateSpec
from .engine import EnvironmentSampleStatus, Observation, ParticleResult
from .models import (
    SURFACE_BOUNDARY_TOLERANCE_M,
    VERTICAL_BOUNDARY_TOLERANCE_M,
    ParticleState,
    ParticleStatus,
)
from .report_material_statistics import MaterialStatisticsAccumulator, MaterialStatisticsProduct
from .report_pathway_statistics import PathwayGridStatistics, build_pathway_grid_statistics
from .report_spec import ReportSpec, validate_report_spec_against_aggregate_spec
from .report_trajectory_identity import is_valid_report_member
from .report_trajectory_selection import RepresentativeSelection, RepresentativeTrajectorySelector
from .streaming_aggregation import StreamingPathwayAccumulator, stream_pathway_first_passage

__all__ = [
    "EnvironmentCompletenessAccumulator",
    "EnvironmentCompletenessProduct",
    "EnvironmentCompletenessResult",
    "EnvironmentCompletenessSiteStatistics",
    "EnvironmentCompletenessStatistics",
    "build_environment_completeness_statistics",
    "reduce_environment_completeness",
    "TrajectoryStreamStatistics",
    "TrajectoryStreamStatisticsProduct",
    "TrajectoryStreamStatisticsResult",
    "TrajectoryReportAccumulator",
    "TrajectoryReportStreamAccumulator",
    "build_trajectory_stream_statistics",
]


_MAX_COUNT: Final[int] = 2**63 - 1
_MEMBER_STATUS_KEYS: Final[tuple[str, ...]] = (
    "valid",
    ParticleStatus.DATA_GAP.value,
    ParticleStatus.NUMERICAL_FAILURE.value,
)
_SAMPLE_STATUS_KEYS: Final[tuple[str, ...]] = tuple(
    status.value for status in EnvironmentSampleStatus
)
_YYYYMM_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9]{6}$")


def _require_text(value: object, *, label: str) -> str:
    """驗證站點與 identity 欄位為非空、無首尾空白的原生文字。

    這些欄位會成為跨 shard 的去重鍵；不自動 trim 或轉型，才能避免上游資料把同一
    站點或粒子悄悄拆成兩組統計。
    """

    if type(value) is not str:
        raise TypeError(f"{label} 必須是原生 str")
    if not value or value != value.strip():
        raise ValueError(f"{label} 必須是非空且沒有首尾空白的文字")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    """驗證報告計數與 member 編號是非負原生整數。

    拒絕 ``bool`` 與 NumPy scalar 可讓結果在 JSON、Arrow 與不同執行器之間維持相同的
    整數資料契約；上限使用 signed int64，與輸出的 NumPy raw count dtype 對齊。
    """

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 int，且不可是 bool 或 NumPy scalar")
    if value < 0:
        raise ValueError(f"{label} 不可為負數")
    if value > _MAX_COUNT:
        raise ValueError(f"{label} 不可超過 signed int64 上限")
    return value


def _require_finite_number(value: object, *, label: str) -> float:
    """把位置／環境高度驗證成有限公尺制 ``float``，但不以轉型填補缺值。"""

    if type(value) not in (int, float):
        raise TypeError(f"{label} 必須是原生 int 或 float")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} 必須是有限值")
    return normalized


def _snapshot_site_ids(value: object, *, allow_empty: bool) -> tuple[str, ...]:
    """複製站點 topology，固定排序並拒絕重複或模糊的字串容器。"""

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("site_ids 必須是非字串、非 mapping iterable")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError("site_ids 必須是 iterable") from error
    if not items and not allow_empty:
        raise ValueError("site_ids 不可為空")
    normalized = tuple(
        _require_text(item, label=f"site_ids[{index}]") for index, item in enumerate(items)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("site_ids 不可重複")
    return tuple(sorted(normalized))


def _snapshot_vertical_edges(value: object) -> tuple[float, ...]:
    """複製並驗證 ReportSpec 的 positive-down 垂向分箱邊界。

    第一個 edge 必須是海面基準 ``0.0 m``，後續 edge 必須嚴格遞增。此處不把最後
    edge 外的資料裁切進最後一箱，因為 reducer 必須另外保存 overflow raw count。
    """

    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        raise TypeError("vertical_depth_bin_edges_m 必須是非字串、非 mapping sequence")
    copied = tuple(value)
    if len(copied) < 2:
        raise ValueError("vertical_depth_bin_edges_m 至少需要兩個邊界")
    normalized: list[float] = []
    for index, edge in enumerate(copied):
        if type(edge) not in (int, float):
            raise TypeError(
                f"vertical_depth_bin_edges_m[{index}] 必須是原生 int 或 float，且不可是 bool"
            )
        normalized_edge = float(edge)
        if not math.isfinite(normalized_edge):
            raise ValueError(f"vertical_depth_bin_edges_m[{index}] 必須是有限值")
        normalized.append(normalized_edge)
    if normalized[0] != 0.0:
        raise ValueError("vertical_depth_bin_edges_m 第一個邊界必須精確為 0.0 m")
    normalized[0] = 0.0
    if not all(left < right for left, right in zip(normalized, normalized[1:], strict=False)):
        raise ValueError("vertical_depth_bin_edges_m 必須嚴格遞增")
    return tuple(normalized)


def _readonly_count_array(value: object, *, expected_size: int, label: str) -> np.ndarray:
    """建立 defensive-copy 的 signed-int64 raw count 陣列並鎖成唯讀。"""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 必須是一維整數陣列") from error
    if raw.ndim != 1 or raw.shape != (expected_size,):
        raise ValueError(f"{label} shape 必須是 ({expected_size},)")
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{label} 必須是整數 dtype")
    for index, item in enumerate(raw):
        try:
            count = int(item)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"{label}[{index}] 必須是非負整數") from error
        if count < 0 or count > _MAX_COUNT:
            raise ValueError(f"{label}[{index}] 超出 signed int64 raw count 範圍")
    copied = np.array(raw, dtype=np.int64, copy=True)
    copied.setflags(write=False)
    return copied


def _count_sum(value: np.ndarray) -> int:
    """以 Python 整數逐項相加，避免 NumPy int64 sum 在極端值時繞回。"""

    total = sum(int(item) for item in value)
    if total > _MAX_COUNT:
        raise ValueError("raw count 總和超過 signed int64 上限")
    return total


def _snapshot_flat_counts(
    value: object | None,
    *,
    defaults: Mapping[str, int],
    label: str,
) -> Mapping[str, int]:
    """複製固定狀態計數 mapping，保存完整 key topology 並拒絕未知狀態。"""

    source: object = defaults if value is None else value
    if not isinstance(source, Mapping):
        raise TypeError(f"{label} 必須是 mapping")
    if set(source) != set(defaults):
        raise ValueError(f"{label} 的 key 必須精確符合固定狀態集合")
    copied = {
        key: _require_nonnegative_int(source[key], label=f"{label}[{key!r}]")
        for key in defaults
    }
    return MappingProxyType(copied)


def _snapshot_nested_counts(
    value: object | None,
    *,
    defaults: Mapping[str, Mapping[str, int]],
    label: str,
) -> Mapping[str, Mapping[str, int]]:
    """複製 member class×environment status 的雙層唯讀 raw count mapping。"""

    source: object = defaults if value is None else value
    if not isinstance(source, Mapping):
        raise TypeError(f"{label} 必須是 mapping")
    if set(source) != set(defaults):
        raise ValueError(f"{label} 的外層 key 必須精確符合固定 member 狀態集合")
    copied: dict[str, Mapping[str, int]] = {}
    for member_status in _MEMBER_STATUS_KEYS:
        member_counts = source[member_status]
        if not isinstance(member_counts, Mapping):
            raise TypeError(f"{label}[{member_status!r}] 必須是 mapping")
        expected = defaults[member_status]
        if set(member_counts) != set(expected):
            raise ValueError(
                f"{label}[{member_status!r}] 的 key 必須精確符合環境狀態集合"
            )
        copied_inner = {
            sample_status: _require_nonnegative_int(
                member_counts[sample_status],
                label=f"{label}[{member_status!r}][{sample_status!r}]",
            )
            for sample_status in _SAMPLE_STATUS_KEYS
        }
        copied[member_status] = MappingProxyType(copied_inner)
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class EnvironmentCompletenessSiteStatistics:
    """單一研究站點的 immutable 環境完整性 raw count。

    ``valid_member_count`` 是依 ``is_valid_report_member`` 留在報告分母的 member 數；
    ``data_gap_member_count`` 與 ``numerical_failure_member_count`` 是分開保存的失敗
    exposure，三者加總為 ``total_member_count``。``observation_count`` 及其三個
    ``valid_observation_count``／``not_sampled_observation_count``／
    ``invalid_observation_count`` 只統計有效 report member 的 Observation；失敗 member
    的 Observation 狀態仍可由 ``observation_counts_by_member_status`` 查核，但不會污染
    有效分母或垂向 histogram。

    ``depth_bin_counts`` 是相對瞬時海面的 positive-down 深度（``eta_m - z_m``），
    ``height_above_bed_bin_counts`` 是離底高度（``z_m - bed_z_m``），兩者單位都是
    公尺（m）。陣列長度是 ``len(vertical_depth_bin_edges_m) - 1``；精確落在內部 edge
    的值進入該 edge 起始的箱，精確落在最後 edge 的值進入最後一箱，範圍外則只增加
    對應 underflow／overflow。所有計數都是 raw count，不在零樣本時以零猜測比例或
    缺值，也不保存任何 ``ParticleResult``／``Observation``。
    """

    study_site_id: str
    vertical_depth_bin_edges_m: tuple[float, ...]
    total_member_count: int
    valid_member_count: int
    data_gap_member_count: int
    numerical_failure_member_count: int
    observation_count: int
    valid_observation_count: int
    not_sampled_observation_count: int
    invalid_observation_count: int
    terminal_not_sampled_count: int
    depth_bin_counts: np.ndarray
    depth_underflow_count: int
    depth_overflow_count: int
    height_above_bed_bin_counts: np.ndarray
    height_above_bed_underflow_count: int
    height_above_bed_overflow_count: int
    member_status_counts: Mapping[str, int] | None = None
    observation_counts_by_member_status: Mapping[str, Mapping[str, int]] | None = None
    terminal_not_sampled_count_by_member_status: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        """驗證 member／observation／histogram 守恆並建立所有 defensive snapshots。"""

        site_id = _require_text(self.study_site_id, label="study_site_id")
        edges = _snapshot_vertical_edges(self.vertical_depth_bin_edges_m)
        total_members = _require_nonnegative_int(self.total_member_count, label="total_member_count")
        valid_members = _require_nonnegative_int(self.valid_member_count, label="valid_member_count")
        data_gap_members = _require_nonnegative_int(
            self.data_gap_member_count,
            label="data_gap_member_count",
        )
        numerical_members = _require_nonnegative_int(
            self.numerical_failure_member_count,
            label="numerical_failure_member_count",
        )
        if total_members != valid_members + data_gap_members + numerical_members:
            raise ValueError("member raw count 必須由 valid、data_gap、numerical_failure 精確加總")

        observation_count = _require_nonnegative_int(self.observation_count, label="observation_count")
        valid_observations = _require_nonnegative_int(
            self.valid_observation_count,
            label="valid_observation_count",
        )
        not_sampled_observations = _require_nonnegative_int(
            self.not_sampled_observation_count,
            label="not_sampled_observation_count",
        )
        invalid_observations = _require_nonnegative_int(
            self.invalid_observation_count,
            label="invalid_observation_count",
        )
        if observation_count != (
            valid_observations + not_sampled_observations + invalid_observations
        ):
            raise ValueError("有效 member 的 observation raw count 必須由三種環境狀態加總")
        terminal_not_sampled = _require_nonnegative_int(
            self.terminal_not_sampled_count,
            label="terminal_not_sampled_count",
        )
        if terminal_not_sampled > not_sampled_observations:
            raise ValueError("terminal_not_sampled_count 不可大於 not_sampled_observation_count")

        member_defaults = {
            "valid": valid_members,
            ParticleStatus.DATA_GAP.value: data_gap_members,
            ParticleStatus.NUMERICAL_FAILURE.value: numerical_members,
        }
        member_counts = _snapshot_flat_counts(
            self.member_status_counts,
            defaults=member_defaults,
            label="member_status_counts",
        )
        if dict(member_counts) != member_defaults:
            raise ValueError("member_status_counts 必須精確符合具名 member raw count")

        observation_defaults = {
            member_status: {sample_status: 0 for sample_status in _SAMPLE_STATUS_KEYS}
            for member_status in _MEMBER_STATUS_KEYS
        }
        observation_defaults["valid"] = {
            EnvironmentSampleStatus.VALID.value: valid_observations,
            EnvironmentSampleStatus.NOT_SAMPLED.value: not_sampled_observations,
            EnvironmentSampleStatus.INVALID.value: invalid_observations,
        }
        observation_counts = _snapshot_nested_counts(
            self.observation_counts_by_member_status,
            defaults=observation_defaults,
            label="observation_counts_by_member_status",
        )
        if dict(observation_counts["valid"]) != observation_defaults["valid"]:
            raise ValueError("有效 member 的 observation status mapping 與具名 raw count 不一致")

        terminal_defaults = {
            member_status: 0 for member_status in _MEMBER_STATUS_KEYS
        }
        terminal_defaults["valid"] = terminal_not_sampled
        terminal_counts = _snapshot_flat_counts(
            self.terminal_not_sampled_count_by_member_status,
            defaults=terminal_defaults,
            label="terminal_not_sampled_count_by_member_status",
        )
        if dict(terminal_counts) != terminal_defaults:
            raise ValueError("terminal_not_sampled mapping 與具名 raw count 不一致")

        bin_count = len(edges) - 1
        depth_counts = _readonly_count_array(
            self.depth_bin_counts,
            expected_size=bin_count,
            label="depth_bin_counts",
        )
        height_counts = _readonly_count_array(
            self.height_above_bed_bin_counts,
            expected_size=bin_count,
            label="height_above_bed_bin_counts",
        )
        depth_underflow = _require_nonnegative_int(
            self.depth_underflow_count,
            label="depth_underflow_count",
        )
        depth_overflow = _require_nonnegative_int(
            self.depth_overflow_count,
            label="depth_overflow_count",
        )
        height_underflow = _require_nonnegative_int(
            self.height_above_bed_underflow_count,
            label="height_above_bed_underflow_count",
        )
        height_overflow = _require_nonnegative_int(
            self.height_above_bed_overflow_count,
            label="height_above_bed_overflow_count",
        )
        if _count_sum(depth_counts) + depth_underflow + depth_overflow != valid_observations:
            raise ValueError("depth histogram 必須精確覆蓋 valid environment observation raw count")
        if _count_sum(height_counts) + height_underflow + height_overflow != valid_observations:
            raise ValueError(
                "height-above-bed histogram 必須精確覆蓋 valid environment observation raw count"
            )

        object.__setattr__(self, "study_site_id", site_id)
        object.__setattr__(self, "vertical_depth_bin_edges_m", edges)
        object.__setattr__(self, "member_status_counts", member_counts)
        object.__setattr__(self, "observation_counts_by_member_status", observation_counts)
        object.__setattr__(self, "terminal_not_sampled_count_by_member_status", terminal_counts)
        object.__setattr__(self, "depth_bin_counts", depth_counts)
        object.__setattr__(self, "height_above_bed_bin_counts", height_counts)

    @property
    def valid_member_denominator(self) -> int:
        """有效 report member 分母的相容別名。"""

        return self.valid_member_count

    @property
    def excluded_data_gap_member_count(self) -> int:
        """``DATA_GAP`` member exposure 的語意別名。"""

        return self.data_gap_member_count

    @property
    def excluded_numerical_failure_member_count(self) -> int:
        """``NUMERICAL_FAILURE`` member exposure 的語意別名。"""

        return self.numerical_failure_member_count

    @property
    def environment_sample_status_counts(self) -> Mapping[str, int]:
        """回傳只屬於有效 member 的 valid/not_sampled/invalid 唯讀 mapping。"""

        return self.observation_counts_by_member_status["valid"]

    @property
    def sample_status_counts(self) -> Mapping[str, int]:
        """``environment_sample_status_counts`` 的簡短別名。"""

        return self.environment_sample_status_counts

    @property
    def depth_counts(self) -> np.ndarray:
        """positive-down depth raw bin count 的簡短別名。"""

        return self.depth_bin_counts

    @property
    def height_above_bed_counts(self) -> np.ndarray:
        """離底高度 raw bin count 的簡短別名。"""

        return self.height_above_bed_bin_counts

    @property
    def total_observation_count(self) -> int:
        """包含失敗 member exposure 的所有 observation raw count。"""

        return sum(
            sum(member_counts.values())
            for member_counts in self.observation_counts_by_member_status.values()
        )


@dataclass(frozen=True, slots=True)
class EnvironmentCompletenessStatistics:
    """完整 study-site topology 的 immutable 環境完整性統計產品。

    ``statistics_by_site`` 以 study site ID 對應一列
    ``EnvironmentCompletenessSiteStatistics``，外層與內層 mapping 都是唯讀 snapshot；
    ``records`` 則依 site ID 排序提供穩定的 table-like 迭代順序。每站即使沒有任何結果，
    只要 caller 在 accumulator 宣告該站，仍會保留零 member、零 observation 的 raw count
    列；這個零是「沒有收到結果」的計數，不是用零猜測缺失環境值。
    """

    site_ids: tuple[str, ...]
    statistics_by_site: Mapping[str, EnvironmentCompletenessSiteStatistics]
    vertical_depth_bin_edges_m: tuple[float, ...]
    _records: tuple[EnvironmentCompletenessSiteStatistics, ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """驗證 site closure、row edge binding 並建立唯讀 mapping／record tuple。"""

        edges = _snapshot_vertical_edges(self.vertical_depth_bin_edges_m)
        normalized_site_ids = _snapshot_site_ids(self.site_ids, allow_empty=True)
        if not isinstance(self.statistics_by_site, Mapping):
            raise TypeError("statistics_by_site 必須是 mapping")
        copied: dict[str, EnvironmentCompletenessSiteStatistics] = {}
        for site_id, row in self.statistics_by_site.items():
            normalized_site = _require_text(site_id, label="statistics_by_site site key")
            if type(row) is not EnvironmentCompletenessSiteStatistics:
                raise TypeError(
                    "statistics_by_site 的 value 必須是 exact EnvironmentCompletenessSiteStatistics"
                )
            if normalized_site in copied:
                raise ValueError("statistics_by_site 不可有重複 site key")
            if row.study_site_id != normalized_site:
                raise ValueError("統計列 study_site_id 必須與 mapping key 一致")
            if row.vertical_depth_bin_edges_m != edges:
                raise ValueError("所有統計列必須使用同一份 vertical_depth_bin_edges_m")
            copied[normalized_site] = row

        if not normalized_site_ids:
            normalized_site_ids = tuple(sorted(copied))
        if set(copied) != set(normalized_site_ids):
            raise ValueError("statistics_by_site 的 key 必須精確覆蓋 site_ids")
        ordered = {site_id: copied[site_id] for site_id in normalized_site_ids}
        object.__setattr__(self, "site_ids", normalized_site_ids)
        object.__setattr__(self, "statistics_by_site", MappingProxyType(ordered))
        object.__setattr__(self, "vertical_depth_bin_edges_m", edges)
        object.__setattr__(self, "_records", tuple(ordered.values()))

    @property
    def records(self) -> tuple[EnvironmentCompletenessSiteStatistics, ...]:
        """依固定 site axis 回傳 immutable 統計列 tuple。"""

        return self._records

    @property
    def by_site(self) -> Mapping[str, EnvironmentCompletenessSiteStatistics]:
        """``statistics_by_site`` 的簡短別名。"""

        return self.statistics_by_site

    @property
    def stats_by_site(self) -> Mapping[str, EnvironmentCompletenessSiteStatistics]:
        """供 renderer／表格使用的 site mapping 別名。"""

        return self.statistics_by_site

    @property
    def statistics(self) -> Mapping[str, EnvironmentCompletenessSiteStatistics]:
        """一般化的統計 mapping 別名。"""

        return self.statistics_by_site

    def __getitem__(self, site_id: str) -> EnvironmentCompletenessSiteStatistics:
        """以 study site ID 讀取單站 immutable 統計列。"""

        return self.statistics_by_site[site_id]

    def __iter__(self):
        """依固定 site ID 順序迭代。"""

        return iter(self.statistics_by_site)

    def __len__(self) -> int:
        """回傳統計站點數。"""

        return len(self.statistics_by_site)


# 這些 alias 保持與既有 report product 命名慣例相容，不建立第二份資料型別。
EnvironmentCompletenessProduct = EnvironmentCompletenessStatistics
EnvironmentCompletenessResult = EnvironmentCompletenessStatistics


@dataclass(slots=True)
class _MutableSiteCounts:
    """reducer 內部只保存計數陣列與固定狀態 mapping，不保存 trajectory payload。"""

    total_member_count: int
    valid_member_count: int
    data_gap_member_count: int
    numerical_failure_member_count: int
    observation_counts_by_member_status: dict[str, dict[str, int]]
    terminal_not_sampled_count_by_member_status: dict[str, int]
    depth_bin_counts: np.ndarray
    depth_underflow_count: int
    depth_overflow_count: int
    height_above_bed_bin_counts: np.ndarray
    height_above_bed_underflow_count: int
    height_above_bed_overflow_count: int


def _new_site_counts(bin_count: int) -> _MutableSiteCounts:
    """建立一列固定 bin topology 的空內部計數。"""

    return _MutableSiteCounts(
        total_member_count=0,
        valid_member_count=0,
        data_gap_member_count=0,
        numerical_failure_member_count=0,
        observation_counts_by_member_status={
            member_status: {sample_status: 0 for sample_status in _SAMPLE_STATUS_KEYS}
            for member_status in _MEMBER_STATUS_KEYS
        },
        terminal_not_sampled_count_by_member_status={
            member_status: 0 for member_status in _MEMBER_STATUS_KEYS
        },
        depth_bin_counts=np.zeros(bin_count, dtype=np.int64),
        depth_underflow_count=0,
        depth_overflow_count=0,
        height_above_bed_bin_counts=np.zeros(bin_count, dtype=np.int64),
        height_above_bed_underflow_count=0,
        height_above_bed_overflow_count=0,
    )


def _validate_forcing_month(value: object, *, label: str) -> str:
    """驗證 valid environment context 的 UTC forcing 月份識別。"""

    if type(value) is not str or _YYYYMM_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} 必須是合法 YYYYMM")
    year = int(value[:4])
    month = int(value[4:])
    if not 1 <= year <= 9999 or not 1 <= month <= 12:
        raise ValueError(f"{label} 必須是合法 YYYYMM")
    return value


def _validate_environment_context(
    observation: Observation,
    *,
    label: str,
) -> tuple[float | None, float | None, float]:
    """驗證 observation 的環境狀態，回傳只供 VALID histogram 使用的 ``eta/bed/z``。

    ``NOT_SAMPLED`` 必須保留四個 context 缺值，``INVALID`` 可以保存有限的部分上下文
    但品質旗標必須非零；兩者都不會進入垂向數值。只有 ``VALID`` 才要求完整上下界與
    forcing 月份，並在 geometry tolerance 內確認 ``bed_z_m <= z_m <= eta_m``。
    """

    status = observation.environment_sample_status
    if type(status) is not EnvironmentSampleStatus:
        raise TypeError(f"{label}.environment_sample_status 必須是 exact EnvironmentSampleStatus")
    if status is EnvironmentSampleStatus.NOT_SAMPLED:
        if any(
            value is not None
            for value in (
                observation.eta_m,
                observation.bed_z_m,
                observation.forcing_month_id,
                observation.environment_qc_flags,
            )
        ):
            raise ValueError(f"{label} 的 not_sampled context 必須全部是 None")
        return None, None, _require_finite_number(observation.z_m, label=f"{label}.z_m")

    if status is EnvironmentSampleStatus.INVALID:
        qc_flags = observation.environment_qc_flags
        if type(qc_flags) is not int or not 1 <= qc_flags <= (1 << 32) - 1:
            raise ValueError(f"{label}.environment_qc_flags 必須是 1 到 2^32-1")
        for field_name, value in (
            ("eta_m", observation.eta_m),
            ("bed_z_m", observation.bed_z_m),
        ):
            if value is not None:
                _require_finite_number(value, label=f"{label}.{field_name}")
        if observation.forcing_month_id is not None:
            _validate_forcing_month(observation.forcing_month_id, label=f"{label}.forcing_month_id")
        return None, None, _require_finite_number(observation.z_m, label=f"{label}.z_m")

    eta_m = _require_finite_number(observation.eta_m, label=f"{label}.eta_m")
    bed_z_m = _require_finite_number(observation.bed_z_m, label=f"{label}.bed_z_m")
    z_m = _require_finite_number(observation.z_m, label=f"{label}.z_m")
    _validate_forcing_month(observation.forcing_month_id, label=f"{label}.forcing_month_id")
    if observation.environment_qc_flags != 0 or type(observation.environment_qc_flags) is not int:
        raise ValueError(f"{label}.environment_qc_flags 必須是原生整數 0")
    if (
        bed_z_m > z_m + VERTICAL_BOUNDARY_TOLERANCE_M
        or z_m > eta_m + SURFACE_BOUNDARY_TOLERANCE_M
        or bed_z_m > eta_m + VERTICAL_BOUNDARY_TOLERANCE_M
    ):
        raise ValueError(f"{label} 的 bed_z_m、z_m、eta_m 垂向範圍不相容")
    return eta_m, bed_z_m, z_m


def _bin_value(value: float, edges: tuple[float, ...]) -> tuple[str, int | None]:
    """依左閉右開、最後 edge 包含的規則回傳 bin／underflow／overflow。"""

    if value < edges[0]:
        return "underflow", None
    if value > edges[-1]:
        return "overflow", None
    # bisect_right 對最後 edge 會回傳 len(edges)；明確夾到最後一個 bin，讓
    # [edge[-2], edge[-1]] 成為唯一包含最右端點的箱，而不會產生越界索引。
    return "bin", min(bisect_right(edges, value) - 1, len(edges) - 2)


class EnvironmentCompletenessAccumulator:
    """逐筆吸收 ``ParticleResult`` 的 bounded-payload environment reducer。

    Args:
        report_spec: exact ``ReportSpec``；只讀取並 defensive-copy
            ``vertical_depth_bin_edges_m``，垂向軸單位是公尺（m）。
        site_ids: 可選的完整 study-site topology。提供時會保留沒有結果的零計數站點，且
            未登錄站點會 fail-closed；省略時站點由輸入結果逐筆發現。輸入清單會複製且
            排序，呼叫端後續修改不會污染 reducer。

    ``add_result`` 每次只遍歷目前一筆結果的 observations，將環境狀態與 valid member
    的兩組 histogram 合併到固定計數器，之後不保存該 ``ParticleResult``／Observation。
    reducer 只另外保存去重所需的 ``(scenario_id, member_id)`` 與 ``particle_id`` set，
    因此記憶體不隨單條軌跡的 observation 長度成長；identity set 會隨已接受 member
    數成長，這是 fail-closed duplicate policy 的必要成本。任何驗證錯誤會永久關閉
    reducer，避免 caller 使用只吸收半筆資料的結果。成功 ``finalize`` 後不可再加，重複
    ``finalize`` 會回傳同一份 immutable product。
    """

    __slots__ = (
        "_vertical_depth_bin_edges_m",
        "_bin_count",
        "_declared_site_ids",
        "_site_ids",
        "_counts",
        "_seen_member_keys",
        "_seen_particle_ids",
        "_state",
        "_finalized_result",
    )

    _OPEN: Final[str] = "open"
    _FINALIZED: Final[str] = "finalized"
    _FAILED: Final[str] = "failed"

    def __init__(self, report_spec: ReportSpec, site_ids: Iterable[str] | None = None) -> None:
        """建立空的固定垂向 topology 與 per-site counters。"""

        if type(report_spec) is not ReportSpec:
            raise TypeError("report_spec 必須是 exact ReportSpec")
        edges = _snapshot_vertical_edges(report_spec.vertical_depth_bin_edges_m)
        declared_site_ids = (
            None if site_ids is None else _snapshot_site_ids(site_ids, allow_empty=False)
        )
        self._vertical_depth_bin_edges_m = edges
        self._bin_count = len(edges) - 1
        self._declared_site_ids = None if declared_site_ids is None else frozenset(declared_site_ids)
        self._site_ids = set() if declared_site_ids is None else set(declared_site_ids)
        self._counts = {
            site_id: _new_site_counts(self._bin_count) for site_id in self._site_ids
        }
        self._seen_member_keys: set[tuple[str, int]] = set()
        self._seen_particle_ids: set[str] = set()
        self._state = self._OPEN
        self._finalized_result: EnvironmentCompletenessStatistics | None = None

    @property
    def site_ids(self) -> tuple[str, ...]:
        """回傳目前已知且排序後的 study-site ID tuple。"""

        return tuple(sorted(self._site_ids))

    @property
    def vertical_depth_bin_edges_m(self) -> tuple[float, ...]:
        """回傳 defensive-copied positive-down 深度 edge tuple，單位為公尺（m）。"""

        return self._vertical_depth_bin_edges_m

    @property
    def member_count(self) -> int:
        """回傳已接受且 identity 唯一的 member 數，包含兩類失敗 member。"""

        return len(self._seen_member_keys)

    @property
    def particle_count(self) -> int:
        """回傳已接受且 identity 唯一的 particle 數。"""

        return len(self._seen_particle_ids)

    @property
    def state(self) -> str:
        """回傳 reducer 的 ``open``／``finalized``／``failed`` 生命週期狀態。"""

        return self._state

    def _ensure_open(self) -> None:
        """拒絕在 finalized 或 failed 狀態繼續使用 reducer。"""

        if self._state == self._FINALIZED:
            raise RuntimeError("EnvironmentCompletenessAccumulator finalize 後不可再加")
        if self._state != self._OPEN:
            raise ValueError("EnvironmentCompletenessAccumulator 已關閉")

    def _validate_result_identity(
        self,
        result: ParticleResult,
        *,
        valid_member: bool,
    ) -> tuple[ParticleState, str, str, int, str]:
        """驗證結果 identity、站點 topology 與 member 分流，不複製軌跡內容。"""

        if type(result.final_state) is not ParticleState:
            raise TypeError("result.final_state 必須是 exact ParticleState")
        state = result.final_state
        scenario_id = _require_text(state.scenario_id, label="result.final_state.scenario_id")
        particle_id = _require_text(state.particle_id, label="result.final_state.particle_id")
        site_id = _require_text(state.study_site_id, label="result.final_state.study_site_id")
        member_id = _require_nonnegative_int(state.member_id, label="result.final_state.member_id")
        if self._declared_site_ids is not None and site_id not in self._declared_site_ids:
            raise ValueError("ParticleResult 的 study_site_id 未在 reducer 登錄")
        status = state.status
        if type(status) is not ParticleStatus:
            raise TypeError("result.final_state.status 必須是 exact ParticleStatus")
        expected_category = "valid" if valid_member else status.value
        if expected_category not in _MEMBER_STATUS_KEYS:
            raise ValueError("is_valid_report_member 回傳了不受支援的 member 分類")
        member_key = (scenario_id, member_id)
        if member_key in self._seen_member_keys:
            raise ValueError("同一 scenario_id × member_id 不可重複輸入")
        if particle_id in self._seen_particle_ids:
            raise ValueError("同一 particle_id 不可重複輸入")
        return state, site_id, particle_id, member_id, expected_category

    def _pending_counts(
        self,
        result: ParticleResult,
        *,
        particle_id: str,
        member_category: str,
    ) -> _MutableSiteCounts:
        """只以固定大小暫存單筆結果的增量，避免部分 observation 直接污染 reducer。"""

        pending = _new_site_counts(self._bin_count)
        try:
            observations = iter(result.observations)
        except TypeError as error:
            raise TypeError("result.observations 必須是 Observation iterable") from error

        for index, observation in enumerate(observations):
            label = f"observations[{index}]"
            if type(observation) is not Observation:
                raise TypeError(f"{label} 必須是 exact Observation")
            if observation.particle_id != particle_id:
                raise ValueError("Observation.particle_id 必須與 final_state.particle_id 一致")
            if type(observation.status) is not ParticleStatus:
                raise TypeError(f"{label}.status 必須是 exact ParticleStatus")
            eta_m, bed_z_m, z_m = _validate_environment_context(observation, label=label)
            sample_status = observation.environment_sample_status.value
            pending.observation_counts_by_member_status[member_category][sample_status] += 1
            # terminal_not_sampled_count 的正式欄位只描述有效 report member 的終止
            # observation；失敗 member 的完整 sample status 仍保留在
            # observation_counts_by_member_status，避免把 failure exposure 誤當成有效
            # member 的垂向／環境完整性分母。這個限制也使 immutable product 的
            # terminal_not_sampled_count 與 terminal mapping 保持精確一致。
            if (
                member_category == "valid"
                and observation.status is not ParticleStatus.ACTIVE
                and sample_status == EnvironmentSampleStatus.NOT_SAMPLED.value
            ):
                pending.terminal_not_sampled_count_by_member_status[member_category] += 1

            # 失敗 member 的 environment context 只作 exposure 診斷；既有 report policy
            # 已將它們排除，因此只有 valid report member 的 VALID observation 能進 histogram。
            if member_category != "valid" or sample_status != EnvironmentSampleStatus.VALID.value:
                continue
            if eta_m is None or bed_z_m is None:
                raise ValueError(f"{label} 的 VALID environment context 不可缺少 eta_m／bed_z_m")
            depth_m = eta_m - z_m
            height_above_bed_m = z_m - bed_z_m
            if not math.isfinite(depth_m) or not math.isfinite(height_above_bed_m):
                raise ValueError(f"{label} 的 depth／height-above-bed 計算結果必須是有限值")
            depth_kind, depth_index = _bin_value(depth_m, self._vertical_depth_bin_edges_m)
            if depth_kind == "underflow":
                pending.depth_underflow_count += 1
            elif depth_kind == "overflow":
                pending.depth_overflow_count += 1
            else:
                assert depth_index is not None
                pending.depth_bin_counts[depth_index] += 1
            height_kind, height_index = _bin_value(
                height_above_bed_m,
                self._vertical_depth_bin_edges_m,
            )
            if height_kind == "underflow":
                pending.height_above_bed_underflow_count += 1
            elif height_kind == "overflow":
                pending.height_above_bed_overflow_count += 1
            else:
                assert height_index is not None
                pending.height_above_bed_bin_counts[height_index] += 1
        return pending

    def _merge_pending(
        self,
        counts: _MutableSiteCounts,
        pending: _MutableSiteCounts,
        *,
        member_category: str,
    ) -> None:
        """把一筆已完整驗證的增量合併到該站固定 counters。"""

        counts.total_member_count += 1
        if member_category == "valid":
            counts.valid_member_count += 1
        elif member_category == ParticleStatus.DATA_GAP.value:
            counts.data_gap_member_count += 1
        else:
            counts.numerical_failure_member_count += 1
        for sample_status in _SAMPLE_STATUS_KEYS:
            counts.observation_counts_by_member_status[member_category][sample_status] += (
                pending.observation_counts_by_member_status[member_category][sample_status]
            )
        counts.terminal_not_sampled_count_by_member_status[member_category] += (
            pending.terminal_not_sampled_count_by_member_status[member_category]
        )
        counts.depth_bin_counts += pending.depth_bin_counts
        counts.depth_underflow_count += pending.depth_underflow_count
        counts.depth_overflow_count += pending.depth_overflow_count
        counts.height_above_bed_bin_counts += pending.height_above_bed_bin_counts
        counts.height_above_bed_underflow_count += pending.height_above_bed_underflow_count
        counts.height_above_bed_overflow_count += pending.height_above_bed_overflow_count

    def add_result(self, result: ParticleResult) -> None:
        """逐一吸收一個 ``ParticleResult``，完成後立即釋放其 payload。

        ``is_valid_report_member`` 先判定 final status；``DATA_GAP``／
        ``NUMERICAL_FAILURE`` 只進入各自的 failure exposure，其他正式終止狀態進入有效
        member 分母。有效 member 的三種 environment sample status 與 terminal not sampled
        會逐 observation 累加；只有 VALID context 才計算公尺制 depth／height bins。任一
        identity 或資料型別錯誤都會令 reducer 永久 failed，且不回傳部分產品。
        """

        self._ensure_open()
        try:
            if type(result) is not ParticleResult:
                raise TypeError("result 必須是 exact ParticleResult")
            valid_member = is_valid_report_member(result)
            state, site_id, particle_id, member_id, member_category = self._validate_result_identity(
                result,
                valid_member=valid_member,
            )
            del state
            pending = self._pending_counts(
                result,
                particle_id=particle_id,
                member_category=member_category,
            )
            if site_id not in self._counts:
                self._counts[site_id] = _new_site_counts(self._bin_count)
                self._site_ids.add(site_id)
            self._merge_pending(
                self._counts[site_id],
                pending,
                member_category=member_category,
            )
            self._seen_member_keys.add((result.final_state.scenario_id, member_id))
            self._seen_particle_ids.add(particle_id)
        except Exception:
            self._state = self._FAILED
            raise

    def add(self, result: ParticleResult) -> None:
        """``add_result`` 的相容別名，保留既有 report reducer 呼叫習慣。"""

        self.add_result(result)

    def add_many(self, results: Iterable[ParticleResult]) -> None:
        """一次遍歷 iterable 串流加入結果，不先 materialize 全量結果。"""

        self._ensure_open()
        if isinstance(results, (str, bytes, bytearray, Mapping)):
            self._state = self._FAILED
            raise TypeError("results 必須是非字串、非 mapping iterable")
        try:
            iterator = iter(results)
        except TypeError as error:
            self._state = self._FAILED
            raise TypeError("results 必須是 iterable") from error
        try:
            for result in iterator:
                self.add_result(result)
        except Exception:
            self._state = self._FAILED
            raise

    def finalize(self) -> EnvironmentCompletenessStatistics:
        """封存每站統計並回傳 immutable product；重複呼叫回傳同一物件。"""

        if self._state == self._FINALIZED:
            assert self._finalized_result is not None
            return self._finalized_result
        self._ensure_open()
        try:
            rows: dict[str, EnvironmentCompletenessSiteStatistics] = {}
            for site_id in self.site_ids:
                counts = self._counts[site_id]
                valid_status_counts = counts.observation_counts_by_member_status["valid"]
                rows[site_id] = EnvironmentCompletenessSiteStatistics(
                    study_site_id=site_id,
                    vertical_depth_bin_edges_m=self._vertical_depth_bin_edges_m,
                    total_member_count=counts.total_member_count,
                    valid_member_count=counts.valid_member_count,
                    data_gap_member_count=counts.data_gap_member_count,
                    numerical_failure_member_count=counts.numerical_failure_member_count,
                    observation_count=sum(valid_status_counts.values()),
                    valid_observation_count=valid_status_counts[EnvironmentSampleStatus.VALID.value],
                    not_sampled_observation_count=valid_status_counts[
                        EnvironmentSampleStatus.NOT_SAMPLED.value
                    ],
                    invalid_observation_count=valid_status_counts[EnvironmentSampleStatus.INVALID.value],
                    terminal_not_sampled_count=(
                        counts.terminal_not_sampled_count_by_member_status["valid"]
                    ),
                    depth_bin_counts=counts.depth_bin_counts,
                    depth_underflow_count=counts.depth_underflow_count,
                    depth_overflow_count=counts.depth_overflow_count,
                    height_above_bed_bin_counts=counts.height_above_bed_bin_counts,
                    height_above_bed_underflow_count=counts.height_above_bed_underflow_count,
                    height_above_bed_overflow_count=counts.height_above_bed_overflow_count,
                    member_status_counts={
                        "valid": counts.valid_member_count,
                        ParticleStatus.DATA_GAP.value: counts.data_gap_member_count,
                        ParticleStatus.NUMERICAL_FAILURE.value: counts.numerical_failure_member_count,
                    },
                    observation_counts_by_member_status=counts.observation_counts_by_member_status,
                    terminal_not_sampled_count_by_member_status=(
                        counts.terminal_not_sampled_count_by_member_status
                    ),
                )
            product = EnvironmentCompletenessStatistics(
                site_ids=self.site_ids,
                statistics_by_site=rows,
                vertical_depth_bin_edges_m=self._vertical_depth_bin_edges_m,
            )
        except Exception:
            self._state = self._FAILED
            raise
        self._finalized_result = product
        self._state = self._FINALIZED
        return product


def build_environment_completeness_statistics(
    results: Iterable[ParticleResult],
    report_spec: ReportSpec,
    *,
    site_ids: Iterable[str] | None = None,
) -> EnvironmentCompletenessStatistics:
    """以一次遍歷建立環境完整性 product。

    Args:
        results: ``ParticleResult`` 的一次性 iterable；函式不會將完整結果或觀測 materialize
            成第二份集合。
        report_spec: exact ``ReportSpec``，其垂向邊界以公尺 positive-down 語意使用。
        site_ids: 可選完整站點 topology；未提供時由結果中的 study site 動態建立。

    Returns:
        具 immutable site mapping、唯讀 raw count arrays 與明確缺值狀態的統計產品。
    """

    accumulator = EnvironmentCompletenessAccumulator(report_spec, site_ids)
    accumulator.add_many(results)
    return accumulator.finalize()


# 以語意化名稱提供相同的一次遍歷 reducer，不複製或改變統計型別。
reduce_environment_completeness = build_environment_completeness_statistics


_TRAJECTORY_INVALID_MEMBER_STATUSES: Final[frozenset[ParticleStatus]] = frozenset(
    {
        ParticleStatus.DATA_GAP,
        ParticleStatus.NUMERICAL_FAILURE,
    }
)
_GRID_ALIGNMENT_REL_TOLERANCE: Final[float] = 1.0e-9
_GRID_ALIGNMENT_ABS_TOLERANCE: Final[float] = 1.0e-9


def _snapshot_topology_ids(value: object, *, label: str) -> tuple[str, ...]:
    """複製並驗證完整的站點或材質拓撲識別碼。

    拓撲會決定 reducer 是否保留零計數列，因此不能接受字串本身、mapping key 的
    模糊推斷、重複值或帶首尾空白的識別碼。輸入 iterable 只在這個建構邊界消費一次，
    後續 reducer 使用排序後的 tuple，避免呼叫端修改清單或改變插入順序而影響輸出。
    """

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} 必須是非字串、非 mapping iterable")
    try:
        identifiers = tuple(value)  # type: ignore[arg-type]
    except Exception as error:
        raise TypeError(f"{label} 必須是可迭代的識別碼集合") from error
    if not identifiers:
        raise ValueError(f"{label} 不可為空")

    normalized = tuple(
        _require_text(identifier, label=f"{label}[{index}]")
        for index, identifier in enumerate(identifiers)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} 不可包含重複識別碼")
    return tuple(sorted(normalized))


def _snapshot_scenario_strata(
    value: Iterable[ScenarioStratum] | Mapping[str, ScenarioStratum],
) -> Mapping[str, ScenarioStratum]:
    """建立完整 ``ScenarioStratum`` 索引的防禦性快照。

    ``ParticleResult`` 只帶有 scenario、站點、分析區域與受體 identity，材質、到達時間、
    季節與潮況必須由這份已驗證分層資料提供；因此本協調器只接受 exact
    ``ScenarioStratum``，不接受較寬鬆的 ``Scenario`` 或 duck-typed 物件。iterable 版本
    會在建構時完整消費一次，mapping 版本則要求 key 與每列 ``scenario_id`` 完全相等；
    任一缺列、重複列、錯誤 key 或輸入 iterator 例外都在 reducer 建立前 fail-closed。
    """

    if isinstance(value, Mapping):
        try:
            raw_items = tuple(value.items())
        except Exception as error:
            raise TypeError("scenario_strata mapping 無法穩定讀取") from error
    else:
        if isinstance(value, (str, bytes, bytearray)):
            raise TypeError("scenario_strata 必須是 ScenarioStratum iterable 或 mapping")
        try:
            raw_strata = tuple(value)
        except Exception as error:
            raise TypeError("scenario_strata 必須是 ScenarioStratum iterable") from error
        raw_items = []
        for index, stratum in enumerate(raw_strata):
            if type(stratum) is not ScenarioStratum:
                raise TypeError(f"scenario_strata[{index}] 必須是 exact ScenarioStratum")
            raw_items.append((stratum.scenario_id, stratum))

    if not raw_items:
        raise ValueError("scenario_strata 不可為空")

    normalized: dict[str, ScenarioStratum] = {}
    for index, item in enumerate(raw_items):
        try:
            mapping_key, stratum = item
        except (TypeError, ValueError) as error:
            raise TypeError(f"scenario_strata mapping item[{index}] 必須是二元素 pair") from error
        if type(stratum) is not ScenarioStratum:
            raise TypeError(f"scenario_strata[{index}] 必須是 exact ScenarioStratum")
        key = _require_text(mapping_key, label=f"scenario_strata key[{index}]")
        scenario_id = _require_text(
            stratum.scenario_id,
            label=f"scenario_strata[{index}].scenario_id",
        )
        if key != scenario_id:
            raise ValueError("scenario_strata mapping key 必須與 scenario_id 完全一致")
        if scenario_id in normalized:
            raise ValueError("scenario_strata 不可包含重複 scenario_id")

        # 這些欄位是 parent router 與三個子 reducer 共用的 join key；即使 caller 以
        # object.__setattr__ 竄改 frozen dataclass，也要在進入結果 stream 前重新檢查，
        # 避免把一列不完整 metadata 當成可用的正式 provenance。
        for field_name in (
            "study_site_id",
            "analysis_region_id",
            "material_id",
            "receptor_id",
            "arrival_time_id",
            "season",
            "tide_class",
        ):
            _require_text(
                getattr(stratum, field_name),
                label=f"scenario_strata[{index}].{field_name}",
            )
        if type(stratum.arrival_time_utc_ns) is not int:
            raise TypeError(
                f"scenario_strata[{index}].arrival_time_utc_ns 必須是原生 int"
            )
        normalized[scenario_id] = stratum

    ordered = {scenario_id: normalized[scenario_id] for scenario_id in sorted(normalized)}
    return MappingProxyType(ordered)


def _aggregate_age_edges(aggregate_spec: AggregateSpec) -> np.ndarray:
    """由 ``AggregateSpec`` 公開欄位建立共同的唯讀 age 秒數格線。

    pathway 與其他 travel-age 產品必須共用同一個秒制分箱，不能依當前軌跡範圍推導。
    此 helper 只讀取 ``AggregateSpec.age_bin_edges_seconds``，重新建立 float64 副本並
    驗證從 0 開始、有限且嚴格遞增；不依賴 aggregate pipeline 的 private helper，讓本
    模組能獨立測試並維持公開資料契約。
    """

    try:
        raw = np.asarray(aggregate_spec.age_bin_edges_seconds)
        edges = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("aggregate_spec.age_bin_edges_seconds 必須是可安全轉換的數值格線") from error
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("aggregate_spec.age_bin_edges_seconds 必須是至少兩點的一維格線")
    if not np.all(np.isfinite(edges)) or edges[0] != 0.0:
        raise ValueError("aggregate_spec.age_bin_edges_seconds 必須有限且從 0 秒開始")
    if not np.all(edges[1:] > edges[:-1]):
        raise ValueError("aggregate_spec.age_bin_edges_seconds 必須嚴格遞增")
    edges.setflags(write=False)
    return edges


def _native_metric_number(value: object, *, label: str) -> float:
    """將公開規格的公尺欄位驗證並 canonical 成有限 Python ``float``。

    x/y 格線是公尺制計算座標；只接受 AggregateSpec 已定義的原生 int／float，拒絕
    bool、NumPy scalar、NaN、無限值與無法轉成有限 float 的極大值，避免格線 shape 或
    端點在不同 NumPy backend 產生漂移。
    """

    if type(value) not in (int, float):
        raise TypeError(f"{label} 必須是原生 int 或 float")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} 必須可安全轉成有限公尺數值") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{label} 必須是有限公尺數值")
    return normalized


def _metric_edges_for_site(
    aggregate_spec: AggregateSpec,
    *,
    site_id: str,
    axis: str,
) -> np.ndarray:
    """依單站公開矩形與 cell size 重建對齊的公尺制 x/y edges。

    ``AggregateSpec`` 已驗證矩形寬高是接近整數個 cell，但本協調器仍在自身邊界重做
    count 與 finite／strictly-increasing gate。格線以 ``minimum + index × cell_size``
    建立，最後一點明確固定為公開 ``maximum``，使兩端與 spec 完全對齊；這裡只產生
    pathway 的格線，不重新計算任何 pathway numerator 或物理路徑。
    """

    if axis not in {"x", "y"}:
        raise ValueError("axis 必須是 x 或 y")
    try:
        grid = aggregate_spec.site_grids[site_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"site {site_id!r} 不存在於 aggregate_spec.site_grids") from error

    minimum_raw = grid.x_min_m if axis == "x" else grid.y_min_m
    maximum_raw = grid.x_max_m if axis == "x" else grid.y_max_m
    minimum = _native_metric_number(minimum_raw, label=f"site {site_id} {axis}_min_m")
    maximum = _native_metric_number(maximum_raw, label=f"site {site_id} {axis}_max_m")
    cell_size = _native_metric_number(
        aggregate_spec.grid_cell_size_m,
        label="aggregate_spec.grid_cell_size_m",
    )
    if maximum <= minimum or cell_size <= 0.0:
        raise ValueError(f"site {site_id} 的 {axis} 公尺範圍或 cell size 不合法")

    width = maximum - minimum
    ratio = width / cell_size
    if not math.isfinite(ratio):
        raise ValueError(f"site {site_id} 的 {axis} 公尺範圍無法建立有限 cell count")
    cell_count = int(round(ratio))
    if cell_count <= 0 or not math.isclose(
        ratio,
        cell_count,
        rel_tol=_GRID_ALIGNMENT_REL_TOLERANCE,
        abs_tol=_GRID_ALIGNMENT_ABS_TOLERANCE,
    ):
        raise ValueError(f"site {site_id} 的 {axis} 公尺範圍未與 AggregateSpec cell size 對齊")

    try:
        edges = minimum + np.arange(cell_count + 1, dtype=np.float64) * cell_size
    except (MemoryError, OverflowError, ValueError) as error:
        raise ValueError(f"site {site_id} 的 {axis} 公尺格線無法配置") from error
    if edges.shape != (cell_count + 1,):
        raise ValueError(f"site {site_id} 的 {axis} 公尺格線 shape 不符合 cell count")
    # 浮點乘法可能使尾端與公開 maximum 有最後一位差異；固定尾端不改變任何 cell
    # 數或中間邊界，只讓所有 chunk 使用同一個可重建的外框端點。
    edges[-1] = maximum
    if not np.all(np.isfinite(edges)) or not np.all(edges[1:] > edges[:-1]):
        raise ValueError(f"site {site_id} 的 {axis} 公尺格線必須有限且嚴格遞增")
    edges.setflags(write=False)
    return edges


def _snapshot_pathway_mapping(
    value: object,
    *,
    environment: EnvironmentCompletenessStatistics,
) -> Mapping[str, PathwayGridStatistics]:
    """驗證 pathway-by-site 產品 closure，並建立 immutable mapping snapshot。

    pathway product 只能由目前環境產品的完整 site 軸組成；每列 denominator 必須等於
    同站環境 reducer 的 valid member count。mapping 本身與輸入 key 不共用，產品內的
    陣列則由 ``PathwayGridStatistics`` 自身負責 defensive-copy／唯讀，因此組合產品不會
    暴露 parent 的可變 reducer 狀態。
    """

    if not isinstance(value, Mapping):
        raise TypeError("pathway_by_site 必須是 site 到 PathwayGridStatistics 的 mapping")
    copied: dict[str, PathwayGridStatistics] = {}
    for raw_site_id, product in value.items():
        site_id = _require_text(raw_site_id, label="pathway_by_site site key")
        if site_id in copied:
            raise ValueError("pathway_by_site 不可包含重複 site key")
        if type(product) is not PathwayGridStatistics:
            raise TypeError(
                f"pathway_by_site[{site_id!r}] 必須是 exact PathwayGridStatistics"
            )
        if site_id not in environment.statistics_by_site:
            raise ValueError("pathway_by_site 不可包含未登錄環境 site")
        expected_denominator = environment[site_id].valid_member_count
        if product.valid_member_denominator != expected_denominator:
            raise ValueError("pathway denominator 必須等於同站 environment valid_member_count")
        copied[site_id] = product
    if set(copied) != set(environment.site_ids):
        raise ValueError("pathway_by_site 的 site key 必須完整覆蓋 environment site axis")
    return MappingProxyType({site_id: copied[site_id] for site_id in environment.site_ids})


@dataclass(frozen=True, slots=True)
class TrajectoryStreamStatistics:
    """一次 trajectory stream 的 immutable 四類報告產品組合。

    ``environment_completeness``、``material_statistics`` 與
    ``representative_selection`` 必須分別是 exact ``EnvironmentCompletenessStatistics``、
    ``MaterialStatisticsProduct`` 與 ``RepresentativeSelection``；它們都保存各自的
    nested defensive snapshot，這裡再檢查 site／分母 closure。``pathway_by_site`` 是由
    同一有效 member stream 重算的 ``PathwayGridStatistics`` mapping，不能使用 aggregate
    pipeline 可能含失敗成員的 pathway numerator 取代。

    pathway 的 x/y 格線採公尺（m），平面陣列軸為 ``(y_cell, x_cell)``；首次進入年齡
    還原使用 AggregateSpec 的共同秒（s）分箱。這些產品只能表示指定條件下的條件式
    來源足跡／相對來源權重，不是絕對來源機率或因果歸因。
    """

    environment_completeness: EnvironmentCompletenessStatistics
    material_statistics: MaterialStatisticsProduct
    representative_selection: RepresentativeSelection
    pathway_by_site: Mapping[str, PathwayGridStatistics]

    def __post_init__(self) -> None:
        """以 exact type 與 site／分母 join gate 建立組合產品的唯讀 snapshot。"""

        if type(self.environment_completeness) is not EnvironmentCompletenessStatistics:
            raise TypeError(
                "environment_completeness 必須是 exact EnvironmentCompletenessStatistics"
            )
        if type(self.material_statistics) is not MaterialStatisticsProduct:
            raise TypeError("material_statistics 必須是 exact MaterialStatisticsProduct")
        if type(self.representative_selection) is not RepresentativeSelection:
            raise TypeError("representative_selection 必須是 exact RepresentativeSelection")

        environment_site_ids = self.environment_completeness.site_ids
        if self.material_statistics.site_ids != environment_site_ids:
            raise ValueError("material_statistics 的 site axis 必須與 environment 完全一致")
        if self.representative_selection.site_ids != environment_site_ids:
            raise ValueError("representative_selection 的 site axis 必須與 environment 完全一致")
        pathway = _snapshot_pathway_mapping(
            self.pathway_by_site,
            environment=self.environment_completeness,
        )

        object.__setattr__(self, "pathway_by_site", pathway)

    @property
    def environment(self) -> EnvironmentCompletenessStatistics:
        """回傳 environment completeness product 的簡短別名。"""

        return self.environment_completeness

    @property
    def environment_statistics(self) -> EnvironmentCompletenessStatistics:
        """回傳 environment completeness product 的語意化別名。"""

        return self.environment_completeness

    @property
    def material(self) -> MaterialStatisticsProduct:
        """回傳材質統計 product 的簡短別名。"""

        return self.material_statistics

    @property
    def selection(self) -> RepresentativeSelection:
        """回傳代表軌跡選樣 product 的簡短別名。"""

        return self.representative_selection

    @property
    def pathway_statistics_by_site(self) -> Mapping[str, PathwayGridStatistics]:
        """回傳 pathway-by-site product 的既有 renderer-facing 別名。"""

        return self.pathway_by_site


# 兩個名稱讓不同報告入口可引用同一份 exact immutable 組合型別，不複製產品資料。
TrajectoryStreamStatisticsProduct = TrajectoryStreamStatistics
TrajectoryStreamStatisticsResult = TrajectoryStreamStatistics


class TrajectoryReportAccumulator:
    """同步累加環境、材質、代表軌跡與有效 pathway 的單次串流協調器。

    Args:
        aggregate_spec: exact ``AggregateSpec``。它提供每站公尺制 x/y 矩形、cell size
            與共用 age 秒數邊界；本類別會依公開欄位建立獨立唯讀 edges，不依賴
            aggregate pipeline 的 private helper，也不從 trajectory extent 自動縮放。
        report_spec: exact ``ReportSpec``，先與 AggregateSpec 做公開 spec binding；其
            pathway quantiles、低樣本有效成員門檻、代表軌跡容量／policy／seed 都由此
            規格固定，不能由結果數量推測或降級。
        scenario_strata: 完整 ``ScenarioStratum`` iterable 或以 scenario ID 為 key 的
            mapping。每列提供 material、arrival、season、tide 與 site／region／receptor
            identity；建構時會完整 defensive snapshot，拒絕 ``Scenario`` 或不一致 key。
        site_ids: 可選的完整站點拓撲。省略時採 AggregateSpec 的全部 site_grids；提供時
            可選其子集合，但所有 scenario stratum 必須落在選定站點與 AggregateSpec 內。
        material_ids: 可選的完整材質拓撲。省略時由 strata 中的 material_id 建立；提供
            時可保留沒有事件的 zero row，但所有 strata material 必須已登錄。

    ``add_shard`` 是正式更新入口：它只逐次讀取當前 shard，先讓每筆結果同步通過
    selector、environment 與 material reducer，再將該 shard 的有效成員按 site 暫存，呼叫
    ``stream_pathway_first_passage`` 產生 bounded chunk 後立即釋放暫存。失敗 member
    （``DATA_GAP``／``NUMERICAL_FAILURE``）永不進 pathway；其早期 observation 仍可留在
    environment 的 failure exposure 診斷，但不會進有效 pathway numerator 或 denominator。

    reducer 沒有跨子 reducer 的 rollback。若 selector 已吸收而 environment/material 或
    pathway 後續失敗，parent 會依交易邊界永久標記 ``failed``，不允許 finalize 或再加資料；
    因此即使內部留下子 reducer 的部分計數，也絕不回傳部分組合產品。成功 finalize 後
    會封存同一個 immutable ``TrajectoryStreamStatistics``，重複 finalize 回傳同一物件。
    """

    _OPEN: Final[str] = "open"
    _FINALIZED: Final[str] = "finalized"
    _FAILED: Final[str] = "failed"

    __slots__ = (
        "_aggregate_spec",
        "_report_spec",
        "_scenario_strata",
        "_site_ids",
        "_material_ids",
        "_age_edges",
        "_pathway_edges_by_site",
        "_environment",
        "_material",
        "_selector",
        "_pathway_by_site_accumulator",
        "_state",
        "_result",
    )

    def __init__(
        self,
        aggregate_spec: AggregateSpec,
        report_spec: ReportSpec,
        scenario_strata: Iterable[ScenarioStratum] | Mapping[str, ScenarioStratum],
        *,
        site_ids: Iterable[str] | None = None,
        material_ids: Iterable[str] | None = None,
    ) -> None:
        """先綁定兩份 exact spec，再建立四個固定拓撲 reducer。"""

        if type(aggregate_spec) is not AggregateSpec:
            raise TypeError("aggregate_spec 必須是 exact AggregateSpec")
        if type(report_spec) is not ReportSpec:
            raise TypeError("report_spec 必須是 exact ReportSpec")
        # binding 必須先於 scenario／reducer 建立；如此任何 run、aggregate canonical hash
        # 或 primary KDE policy 不一致，都不會先配置可被誤用的報告狀態。
        validate_report_spec_against_aggregate_spec(report_spec, aggregate_spec)

        normalized_strata = _snapshot_scenario_strata(scenario_strata)
        aggregate_site_ids = _snapshot_topology_ids(
            aggregate_spec.site_grids.keys(),
            label="aggregate_spec.site_grids site_ids",
        )
        resolved_site_ids = (
            aggregate_site_ids
            if site_ids is None
            else _snapshot_topology_ids(site_ids, label="site_ids")
        )
        aggregate_site_set = frozenset(aggregate_site_ids)
        resolved_site_set = frozenset(resolved_site_ids)
        if not resolved_site_set <= aggregate_site_set:
            raise ValueError("site_ids 必須是 aggregate_spec.site_grids 的子集合")

        derived_material_ids = tuple(
            sorted({stratum.material_id for stratum in normalized_strata.values()})
        )
        resolved_material_ids = (
            derived_material_ids
            if material_ids is None
            else _snapshot_topology_ids(material_ids, label="material_ids")
        )
        resolved_material_set = frozenset(resolved_material_ids)
        for stratum in normalized_strata.values():
            if stratum.study_site_id not in resolved_site_set:
                raise ValueError("scenario_strata 的 study_site_id 不在選定 site topology")
            if stratum.material_id not in resolved_material_set:
                raise ValueError("scenario_strata 的 material_id 不在選定 material topology")

        age_edges = _aggregate_age_edges(aggregate_spec)
        pathway_edges: dict[str, Mapping[str, np.ndarray]] = {}
        for site_id in resolved_site_ids:
            pathway_edges[site_id] = MappingProxyType(
                {
                    "x_edges_m": _metric_edges_for_site(
                        aggregate_spec,
                        site_id=site_id,
                        axis="x",
                    ),
                    "y_edges_m": _metric_edges_for_site(
                        aggregate_spec,
                        site_id=site_id,
                        axis="y",
                    ),
                }
            )

        # 子 reducer 仍各自執行完整的資料契約驗證；parent 的 scenario snapshot 與 topology
        # 只作共同路由真相，不把任一 reducer 的寬鬆輸入模式帶入正式協調器。
        environment = EnvironmentCompletenessAccumulator(report_spec, resolved_site_ids)
        material = MaterialStatisticsAccumulator(
            scenario_strata=normalized_strata,
            site_ids=resolved_site_ids,
            material_ids=resolved_material_ids,
        )
        selector = RepresentativeTrajectorySelector(report_spec, resolved_site_ids)

        self._aggregate_spec = aggregate_spec
        self._report_spec = report_spec
        self._scenario_strata = normalized_strata
        self._site_ids = resolved_site_ids
        self._material_ids = resolved_material_ids
        self._age_edges = age_edges
        self._pathway_edges_by_site = MappingProxyType(pathway_edges)
        self._environment = environment
        self._material = material
        self._selector = selector
        self._pathway_by_site_accumulator = {
            site_id: StreamingPathwayAccumulator() for site_id in resolved_site_ids
        }
        self._state = self._OPEN
        self._result: TrajectoryStreamStatistics | None = None

    @property
    def aggregate_spec(self) -> AggregateSpec:
        """回傳已綁定的 exact AggregateSpec。"""

        return self._aggregate_spec

    @property
    def report_spec(self) -> ReportSpec:
        """回傳已綁定的 exact ReportSpec。"""

        return self._report_spec

    @property
    def scenario_strata(self) -> Mapping[str, ScenarioStratum]:
        """回傳 scenario ID 到 immutable ScenarioStratum 的唯讀索引。"""

        return self._scenario_strata

    @property
    def site_ids(self) -> tuple[str, ...]:
        """回傳 pathway／environment／selector 共用的排序 site axis。"""

        return self._site_ids

    @property
    def material_ids(self) -> tuple[str, ...]:
        """回傳 material reducer 使用的排序 material axis。"""

        return self._material_ids

    @property
    def state(self) -> str:
        """回傳 parent 的 ``open``／``finalized``／``failed`` 狀態。"""

        return self._state

    @property
    def environment_accumulator(self) -> EnvironmentCompletenessAccumulator:
        """回傳環境子 reducer，供唯讀檢查其計數摘要；更新仍應走 parent。"""

        return self._environment

    @property
    def material_accumulator(self) -> MaterialStatisticsAccumulator:
        """回傳材質子 reducer，供唯讀檢查其拓撲與計數摘要；更新仍應走 parent。"""

        return self._material

    @property
    def representative_selector(self) -> RepresentativeTrajectorySelector:
        """回傳固定容量代表案例子 reducer，更新仍應走 parent。"""

        return self._selector

    @property
    def pathway_chunk_count_by_site(self) -> Mapping[str, int]:
        """回傳每站成功 pathway chunk 數的 defensive mapping。"""

        return MappingProxyType(
            {
                site_id: reducer.chunk_count
                for site_id, reducer in self._pathway_by_site_accumulator.items()
            }
        )

    def _ensure_open(self) -> None:
        """拒絕 finalized／failed parent，避免 caller 忽略交易邊界。"""

        if self._state == self._FINALIZED:
            raise RuntimeError("TrajectoryReportAccumulator finalize 後不可再加")
        if self._state != self._OPEN:
            raise ValueError("TrajectoryReportAccumulator 已關閉")

    def _result_route(
        self,
        result: object,
    ) -> tuple[ScenarioStratum, str, bool]:
        """驗證結果 scenario 與三組 site／region／receptor identity，回傳路由資訊。

        這個 parent gate 先於任何子 reducer 執行，確保 material／arrival／season／tide
        來自同一列 scenario metadata。``bool`` 只依既有有效成員政策表示是否可進 pathway；
        不是把資料缺口或數值失敗轉成零值，失敗 member 仍由 environment/material reducer
        保存對應 exposure。
        """

        if type(result) is not ParticleResult:
            raise TypeError("result 必須是 exact ParticleResult")
        if type(result.final_state) is not ParticleState:
            raise TypeError("result.final_state 必須是 exact ParticleState")
        state = result.final_state
        scenario_id = _require_text(state.scenario_id, label="result.final_state.scenario_id")
        stratum = self._scenario_strata.get(scenario_id)
        if stratum is None:
            raise ValueError("ParticleResult 的 scenario_id 不存在於 scenario_strata")
        result_site_id = _require_text(
            state.study_site_id,
            label="result.final_state.study_site_id",
        )
        result_region_id = _require_text(
            state.analysis_region_id,
            label="result.final_state.analysis_region_id",
        )
        result_receptor_id = _require_text(
            state.receptor_id,
            label="result.final_state.receptor_id",
        )
        if (
            result_site_id != stratum.study_site_id
            or result_region_id != stratum.analysis_region_id
            or result_receptor_id != stratum.receptor_id
        ):
            raise ValueError("ParticleResult 的 scenario/site/region/receptor identity 不一致")
        if result_site_id not in self._site_ids:
            raise ValueError("ParticleResult 的 study_site_id 未在協調器 site topology 登錄")
        if type(state.status) is not ParticleStatus:
            raise TypeError("result.final_state.status 必須是 exact ParticleStatus")
        if state.status is ParticleStatus.ACTIVE:
            raise ValueError("ACTIVE ParticleResult 尚未終止，不可進入報告")
        return stratum, result_site_id, state.status not in _TRAJECTORY_INVALID_MEMBER_STATUSES

    def _accept_result_to_reducers(
        self,
        result: object,
    ) -> tuple[str, bool]:
        """將單筆結果依同一 stratum metadata 餵入三個子 reducer。

        selector 先執行，接著才是 environment 與 material；這個固定順序不是 rollback
        交易，而是明示「同一筆資料的三個 side effect 可能只完成前兩個」的邊界。外層
        ``add_result``／``add_shard`` 一旦任一步失敗就封閉 parent，故部分吸收不會形成
        可 finalize 的科學產品。
        """

        stratum, site_id, valid_member = self._result_route(result)
        self._selector.add(
            result,  # type: ignore[arg-type]
            material_id=stratum.material_id,
            arrival_time_id=stratum.arrival_time_id,
            season=stratum.season,
            tide_class=stratum.tide_class,
        )
        self._environment.add_result(result)  # type: ignore[arg-type]
        self._material.add_result(result)  # type: ignore[arg-type]
        return site_id, valid_member

    def _add_pathway_chunk(
        self,
        site_id: str,
        results: Iterable[ParticleResult],
    ) -> None:
        """以 AggregateSpec 固定軸建立單站有效 pathway chunk 並立即合併。"""

        edges = self._pathway_edges_by_site[site_id]
        chunk = stream_pathway_first_passage(
            results,
            x_edges_m=edges["x_edges_m"],
            y_edges_m=edges["y_edges_m"],
            age_bin_edges_seconds=self._age_edges,
        )
        self._pathway_by_site_accumulator[site_id].add(chunk)

    def add_result(self, result: ParticleResult) -> None:
        """低階逐筆入口，同步更新三個報告 reducer 與單筆有效 pathway chunk。

        正式管線應優先使用 ``add_shard``，讓同一 shard 的有效結果可一次建立 bounded
        pathway chunk；本入口保留給小型串流或互動 caller。資料錯誤會永久關閉 parent，
        且不回傳任何 partial product。有效成員才會呼叫 pathway；兩類失敗 member 只由
        environment/material 留存其 exposure，不會進 pathway。
        """

        self._ensure_open()
        try:
            site_id, valid_member = self._accept_result_to_reducers(result)
            if valid_member:
                self._add_pathway_chunk(site_id, (result,))
        except Exception:
            self._state = self._FAILED
            raise

    def add(self, result: ParticleResult) -> None:
        """``add_result`` 的相容別名。"""

        self.add_result(result)

    def add_shard(self, shard: Iterable[ParticleResult]) -> None:
        """一次遍歷單一 shard，按站點重算有效 pathway 並合併至固定 reducer。

        ``shard`` 可以是 list、tuple 或一次性 generator，但不可是 mapping／字串。迭代
        過程只保留當前 shard 的有效 ``ParticleResult`` reference，分站建立 chunk 後在
        ``finally`` 清空 mapping；不保存舊 shard 或全案結果。每筆結果仍先同步餵入
        environment、material、selector，故任何 duplicate identity、metadata、環境或
        事件錯誤都會讓 parent fail-closed。
        """

        self._ensure_open()
        current_valid_results: dict[str, list[ParticleResult]] = {}
        try:
            if isinstance(shard, (str, bytes, bytearray, Mapping)):
                raise TypeError("shard 必須是非字串、非 mapping 的 ParticleResult iterable")
            try:
                iterator = iter(shard)
            except Exception as error:
                raise TypeError("shard 必須是 ParticleResult iterable") from error

            # 只建立當前 shard 的 per-site list；失敗 member 在此處永遠不會被放入 pathway
            # 暫存，因此後續 stream_pathway_first_passage 看不到它的任何 observation。
            for result in iterator:
                site_id, valid_member = self._accept_result_to_reducers(result)
                if valid_member:
                    current_valid_results.setdefault(site_id, []).append(result)

            for site_id in self._site_ids:
                site_results = current_valid_results.get(site_id)
                if site_results:
                    self._add_pathway_chunk(site_id, site_results)
        except Exception:
            self._state = self._FAILED
            raise
        finally:
            # pathway chunk 已由 accumulator defensive-copy／合併；這裡釋放當前 shard
            # reference，避免下一 shard 與舊資料同時留在 parent 記憶體。
            current_valid_results.clear()

    def add_many(self, results: Iterable[ParticleResult]) -> None:
        """將單一結果 iterable 視為一個 shard 的相容入口，不 materialize 全量結果。"""

        self.add_shard(results)

    def finalize(self) -> TrajectoryStreamStatistics:
        """核對 pathway、八層代表選樣與 material/environment closure 後封存組合產品。

        每個 site 必須至少有一個有效 pathway chunk；零有效 site 或只有失敗 member 的
        shard 不會以空 chunk、aggregate all-member pathway 或 pooled strata 補值。pathway
        builder 會使用 report spec 的 quantiles／低樣本門檻，並再次核對其輸入粒子數等於
        environment valid member count。任一 gate 失敗都將 parent 永久標成 failed，不建立
        可被誤用的部分組合產品。
        """

        if self._state == self._FINALIZED:
            assert self._result is not None
            return self._result
        self._ensure_open()
        try:
            environment = self._environment.finalize()
            material = self._material.finalize()
            selection = self._selector.finalize()

            pathway_by_site: dict[str, PathwayGridStatistics] = {}
            for site_id in self._site_ids:
                pathway_accumulator = self._pathway_by_site_accumulator[site_id]
                if pathway_accumulator.chunk_count < 1:
                    raise ValueError(
                        f"site {site_id!r} 不可在零有效 pathway chunk 時 finalize"
                    )
                pathway_aggregate = pathway_accumulator.finalize()
                pathway = build_pathway_grid_statistics(
                    pathway_aggregate,
                    environment[site_id].valid_member_count,
                    self._report_spec.pathway_first_passage_quantiles,
                    self._report_spec.low_sample_min_member_count,
                )
                expected_edges = self._pathway_edges_by_site[site_id]
                if not np.array_equal(pathway.x_edges_m, expected_edges["x_edges_m"]):
                    raise ValueError("pathway x_edges_m 必須精確來自 AggregateSpec")
                if not np.array_equal(pathway.y_edges_m, expected_edges["y_edges_m"]):
                    raise ValueError("pathway y_edges_m 必須精確來自 AggregateSpec")
                if not np.array_equal(pathway.age_bin_edges_seconds, self._age_edges):
                    raise ValueError("pathway age_bin_edges_seconds 必須精確來自 AggregateSpec")
                pathway_by_site[site_id] = pathway

            product = TrajectoryStreamStatistics(
                environment_completeness=environment,
                material_statistics=material,
                representative_selection=selection,
                pathway_by_site=pathway_by_site,
            )
        except Exception:
            # 子 reducer 沒有共同 rollback；只要其中一個 finalizer 或 cross-product gate
            # 失敗，parent 就永久 failed。即使 environment/material/selector 中已有部分
            # finalized 狀態，也不讓 caller 取得或重試一份不完整的科學產品。
            self._state = self._FAILED
            raise
        self._result = product
        self._state = self._FINALIZED
        return product


# 前一版需求使用的名稱保留為同一個 exact class alias，避免呼叫端因命名調整而分裂 reducer。
TrajectoryReportStreamAccumulator = TrajectoryReportAccumulator


def build_trajectory_stream_statistics(
    results: Iterable[ParticleResult],
    aggregate_spec: AggregateSpec,
    report_spec: ReportSpec,
    scenario_strata: Iterable[ScenarioStratum] | Mapping[str, ScenarioStratum],
    *,
    site_ids: Iterable[str] | None = None,
    material_ids: Iterable[str] | None = None,
) -> TrajectoryStreamStatistics:
    """以一次遍歷把 ``results`` 視為一個 shard，建立完整 trajectory stream product。

    這個 builder 不把結果 materialize 成全案 list；它只呼叫一次
    ``TrajectoryReportAccumulator.add_shard``，所以有效 pathway、環境完整性、材質分母
    與代表軌跡都來自相同的一次結果輸入。``AggregateSpec`` 先固定每站公尺格線與共同
    age 秒軸，``ReportSpec`` 先完成 canonical binding，再由 reducer 產生 quantile、低樣本
    與八層容量 gate。若輸入只是多 shard 的外層 iterable，應改由 caller 逐項呼叫
    ``add_shard``，避免把 shard iterable 與單筆結果混淆。

    Raises:
        TypeError／ValueError／RuntimeError: spec、scenario metadata、結果 identity、
            pathway 軸、有效分母、事件／環境資料或固定八層容量不符合契約時；不回傳部分
            product。
    """

    accumulator = TrajectoryReportAccumulator(
        aggregate_spec,
        report_spec,
        scenario_strata,
        site_ids=site_ids,
        material_ids=material_ids,
    )
    accumulator.add_shard(results)
    return accumulator.finalize()
