"""F11/T05 比較統計的純計算與精確相容性（exact compatibility）邊界。

本模組只接受兩份已完成來源驗證的報告統計集合（ReportStatistics）。它不讀取檔案、
不呼叫聚合管線、不繪圖，也不把基準組自身複製成零差異結果。兩份輸入必須各自帶有
來源聚合資料（AggregateReleasePayload）與報告規格（ReportSpec）；本模組會先比對
站點、情境、受體、到達時間、材料識別、幾何、年齡軸、核密度估計（KDE）／
高密度區域（HDR）軸、路徑格網軸與報告政策，通過後才計算比較組（comparison）
減基準組（baseline）的差值。

目前來源綁定比較尚不支援材質差值：聚合資料不足以核對逐粒子材質統計來源，
帶來源的 ReportStatistics 因此拒絕材質產品。無來源身分的純產品入口雖保留材質
用法，卻不符合本模組的輸入要求；保留的材質差值欄位不代表該功能已可公開使用。

輸出描述同一研究設計下、單一明示物理參數差異的敏感度，不是因果歸因。每個比例
差值都同時保留兩側原始分子、原始分母與估計狀態（EstimateStatus）；零分母的
比例維持 None，陣列型產品則以非數值標記（NaN）搭配狀態／分母保存不可估計語意，
絕不以 0 代替不可用狀態。所有輸出陣列與巢狀對照表都在此邊界重新複製並
設為唯讀。
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from .aggregate_release_payload import AggregateReleasePayload
from .aggregate_release_records import ScenarioStratum
from .aggregate_spec import AggregateSpec
from .event_aggregation import SourceReceptorAggregateKey
from .report_kde_statistics import KDEEstimateStatus
from .report_ratio_statistics import CountRatio, EstimateStatus
from .report_spec import ReportSpec, validate_report_spec_against_aggregate_spec
from .report_statistics import ReportStatistics

__all__ = [
    "ComparisonHDRStatus",
    "ComparisonParameterDifference",
    "ComparisonRatioDifference",
    "ComparisonScalarDifference",
    "ReportComparisonStatistics",
    "build_report_comparison_statistics",
]


def _require_text(value: object, *, label: str) -> str:
    """要求 join key 或物理參數說明是非空、沒有首尾空白的原生字串。"""

    if type(value) is not str:
        raise TypeError(f"{label} 必須是原生 str")
    if not value or value != value.strip():
        raise ValueError(f"{label} 必須是非空且沒有首尾空白的文字")
    return value


def _require_finite_number(value: object, *, label: str) -> int | float:
    """要求物理參數值是有限原生 int 或 float，拒絕 bool 與 NumPy scalar。"""

    if type(value) not in (int, float):
        raise TypeError(f"{label} 必須是原生 int 或 float")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{label} 必須是有限值")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} 無法表示為有限 float") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{label} 必須是有限值")
    return value


def _require_nonnegative_count(value: object, *, label: str) -> int:
    """要求差值產品保存的 raw count 是非負原生 Python 整數。"""

    if type(value) is not int:
        raise TypeError(f"{label} 必須是非負原生 int")
    if value < 0:
        raise ValueError(f"{label} 不可為負值")
    return value


def _snapshot_site_ids(value: object) -> tuple[str, ...]:
    """複製有順序的 site axis；矩陣產品不能在此處排序或轉置。"""

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("site_ids 必須是非字串、非 mapping iterable")
    try:
        copied = tuple(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError("site_ids 必須是可重複迭代的 iterable") from error
    if not copied:
        raise ValueError("site_ids 不可為空")
    result = tuple(_require_text(item, label="site_ids item") for item in copied)
    if len(set(result)) != len(result):
        raise ValueError("site_ids 不可重複")
    return result


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype[Any],
    ndim: int,
    label: str,
) -> np.ndarray:
    """建立獨立唯讀陣列快照，讓 caller 無法透過原始 buffer 回寫產品。"""

    try:
        array = np.array(value, dtype=dtype, copy=True)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} 必須可建立 {dtype} 陣列") from error
    if array.ndim != ndim:
        raise ValueError(f"{label} 必須是 {ndim} 維陣列")
    array.setflags(write=False)
    return array


def _readonly_edges(value: object, *, label: str) -> np.ndarray:
    """封存嚴格遞增的公尺或秒軸，並拒絕 NaN、Infinity 及錯誤維度。"""

    edges = _readonly_array(
        value,
        dtype=np.dtype(np.float64),
        ndim=1,
        label=label,
    )
    if edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError(f"{label} 必須至少含兩個有限邊界")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError(f"{label} 必須嚴格遞增")
    return edges


def _deep_equal(left: object, right: object) -> bool:
    """比較只由 scalar、list、tuple、mapping 組成的 canonical spec snapshot。"""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if tuple(left.keys()) != tuple(right.keys()):
            return False
        return all(_deep_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _deep_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=False)
        )
    return bool(left == right)


def _mapping_proxy(
    value: Mapping[Any, Any],
    *,
    label: str,
) -> Mapping[Any, Any]:
    """以普通 dict 中介後包成唯讀 mapping，避免保留 caller 的 dict alias。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必須是 mapping")
    try:
        return MappingProxyType(dict(value))
    except Exception as error:
        raise TypeError(f"{label} 無法建立 defensive mapping") from error


@dataclass(frozen=True, slots=True)
class ComparisonParameterDifference:
    """明示 baseline 與 comparison 間唯一物理參數差異。

    name 是參數的穩定識別碼，兩個 value 必須是有限的原生數值，unit 是資料交換
    與 caption 使用的單位文字。差值方向固定為 comparison value 減 baseline value；
    這個欄位只記錄實驗設計，不替呼叫端推論因果關係，也不接受把缺值寫成零值。
    """

    name: str
    baseline_value: int | float
    comparison_value: int | float
    unit: str

    def __post_init__(self) -> None:
        """封存參數說明並拒絕沒有實際差異或非有限數值。"""

        name = _require_text(self.name, label="parameter name")
        unit = _require_text(self.unit, label="parameter unit")
        baseline = _require_finite_number(
            self.baseline_value,
            label="baseline_value",
        )
        comparison = _require_finite_number(
            self.comparison_value,
            label="comparison_value",
        )
        if float(baseline) == float(comparison):
            raise ValueError("baseline_value 與 comparison_value 必須構成明示差異")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "baseline_value", baseline)
        object.__setattr__(self, "comparison_value", comparison)
        object.__setattr__(self, "unit", unit)

    @property
    def delta(self) -> float:
        """回傳固定方向的 comparison 減 baseline 參數差值。"""

        return float(self.comparison_value) - float(self.baseline_value)

    @property
    def parameter_name(self) -> str:
        """name 的語意別名，方便表格 adapter 使用。"""

        return self.name


def _validate_optional_ratio_value(
    value: object,
    *,
    status: EstimateStatus | None,
    label: str,
) -> float | None:
    """核對 ratio 與 status 的 None／有限性關係，不把不可估計誤當成零。"""

    if value is None:
        if status is not None and status is not EstimateStatus.zero_denominator:
            raise ValueError(f"{label} 正分母狀態不可保存 None")
        return None
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{label} 必須是有限原生 float 或 None")
    if status is EstimateStatus.zero_denominator:
        raise ValueError(f"{label} 的 zero_denominator 狀態必須保存 None")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} 必須位於 [0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class ComparisonRatioDifference:
    """保存一個比例的兩側 raw 計數、狀態與有方向差值。

    difference 永遠是 comparison ratio 減 baseline ratio。只有兩側 ratio 都可估計
    時才會有有限差值；任一側為 zero_denominator／None，difference 就維持 None。
    low_count 不會抹掉正分母的比例，仍可計算差值，但兩側 status 會原樣保留供報告
    顯示樣本警示。
    """

    baseline_numerator: int
    baseline_denominator: int
    baseline_ratio: float | None
    baseline_status: EstimateStatus
    comparison_numerator: int
    comparison_denominator: int
    comparison_ratio: float | None
    comparison_status: EstimateStatus
    difference: float | None

    def __post_init__(self) -> None:
        """驗證兩側計數、比例與狀態的一致性。"""

        for name in ("baseline_numerator", "baseline_denominator"):
            _require_nonnegative_count(getattr(self, name), label=name)
        for name in ("comparison_numerator", "comparison_denominator"):
            _require_nonnegative_count(getattr(self, name), label=name)
        if self.baseline_numerator > self.baseline_denominator:
            raise ValueError("baseline numerator 不可大於 denominator")
        if self.comparison_numerator > self.comparison_denominator:
            raise ValueError("comparison numerator 不可大於 denominator")
        if type(self.baseline_status) is not EstimateStatus:
            raise TypeError("baseline_status 必須是 EstimateStatus")
        if type(self.comparison_status) is not EstimateStatus:
            raise TypeError("comparison_status 必須是 EstimateStatus")
        baseline_ratio = _validate_optional_ratio_value(
            self.baseline_ratio,
            status=self.baseline_status,
            label="baseline_ratio",
        )
        comparison_ratio = _validate_optional_ratio_value(
            self.comparison_ratio,
            status=self.comparison_status,
            label="comparison_ratio",
        )
        expected_difference = (
            None
            if baseline_ratio is None or comparison_ratio is None
            else float(comparison_ratio - baseline_ratio)
        )
        if self.difference != expected_difference:
            raise ValueError("difference 必須是 comparison_ratio - baseline_ratio，或不可估計時為 None")
        object.__setattr__(self, "baseline_ratio", baseline_ratio)
        object.__setattr__(self, "comparison_ratio", comparison_ratio)
        object.__setattr__(self, "difference", expected_difference)

    @property
    def delta(self) -> float | None:
        """difference 的方向性欄位別名。"""

        return self.difference

    @property
    def baseline_raw_numerator(self) -> int:
        """baseline_numerator 的報告欄位別名。"""

        return self.baseline_numerator

    @property
    def baseline_raw_denominator(self) -> int:
        """baseline_denominator 的報告欄位別名。"""

        return self.baseline_denominator

    @property
    def comparison_raw_numerator(self) -> int:
        """comparison_numerator 的報告欄位別名。"""

        return self.comparison_numerator

    @property
    def comparison_raw_denominator(self) -> int:
        """comparison_denominator 的報告欄位別名。"""

        return self.comparison_denominator


@dataclass(frozen=True, slots=True)
class ComparisonScalarDifference:
    """保存兩側有限 scalar 或 None 的有方向差值與狀態。

    此產品主要承載來源—受體旅行年齡中位數，單位由欄位語境明示為秒。正分母但
    低樣本的中點近似仍保留；只有缺值／零樣本時才以 None 表示，不以零替代。
    """

    baseline_value: float | None
    baseline_status: EstimateStatus
    comparison_value: float | None
    comparison_status: EstimateStatus
    difference: float | None

    def __post_init__(self) -> None:
        """驗證 scalar 與 EstimateStatus 的可估計關係。"""

        if type(self.baseline_status) is not EstimateStatus:
            raise TypeError("baseline_status 必須是 EstimateStatus")
        if type(self.comparison_status) is not EstimateStatus:
            raise TypeError("comparison_status 必須是 EstimateStatus")
        baseline = _validate_optional_scalar(
            self.baseline_value,
            status=self.baseline_status,
            label="baseline_value",
        )
        comparison = _validate_optional_scalar(
            self.comparison_value,
            status=self.comparison_status,
            label="comparison_value",
        )
        expected_difference = (
            None
            if baseline is None or comparison is None
            else float(comparison - baseline)
        )
        if self.difference != expected_difference:
            raise ValueError("scalar difference 方向或 None 語意不一致")
        object.__setattr__(self, "baseline_value", baseline)
        object.__setattr__(self, "comparison_value", comparison)
        object.__setattr__(self, "difference", expected_difference)

    @property
    def delta(self) -> float | None:
        """difference 的方向性欄位別名。"""

        return self.difference


def _validate_optional_scalar(
    value: object,
    *,
    status: EstimateStatus,
    label: str,
) -> float | None:
    """核對 travel-age scalar 的有限性與 zero-denominator 狀態。"""

    if value is None:
        if status is not EstimateStatus.zero_denominator:
            raise ValueError(f"{label} 只有 zero_denominator 才可為 None")
        return None
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{label} 必須是有限原生 float 或 None")
    if status is EstimateStatus.zero_denominator:
        raise ValueError(f"{label} 的 zero_denominator 狀態必須是 None")
    return value


@dataclass(frozen=True, slots=True)
class ComparisonHDRStatus:
    """保留 primary bandwidth 的兩側 HDR 狀態，不把不可估計當成零重疊。"""

    baseline_status: KDEEstimateStatus
    comparison_status: KDEEstimateStatus

    def __post_init__(self) -> None:
        """要求兩側狀態都是既有 KDEEstimateStatus 成員。"""

        if type(self.baseline_status) is not KDEEstimateStatus:
            raise TypeError("baseline_status 必須是 KDEEstimateStatus")
        if type(self.comparison_status) is not KDEEstimateStatus:
            raise TypeError("comparison_status 必須是 KDEEstimateStatus")

    @property
    def available(self) -> bool:
        """回傳兩側 primary layer 是否都具有可比較的 KDE grid。"""

        return (
            self.baseline_status is KDEEstimateStatus.available
            and self.comparison_status is KDEEstimateStatus.available
        )


def _comparison_ratio(
    baseline: CountRatio,
    comparison: CountRatio,
) -> ComparisonRatioDifference:
    """由既有 CountRatio 建立 comparison - baseline 的 immutable 差值。"""

    if type(baseline) is not CountRatio or type(comparison) is not CountRatio:
        raise TypeError("ratio input 必須是 exact CountRatio")
    difference = (
        None
        if baseline.ratio is None or comparison.ratio is None
        else float(comparison.ratio - baseline.ratio)
    )
    return ComparisonRatioDifference(
        baseline_numerator=baseline.raw_numerator,
        baseline_denominator=baseline.raw_denominator,
        baseline_ratio=baseline.ratio,
        baseline_status=baseline.status,
        comparison_numerator=comparison.raw_numerator,
        comparison_denominator=comparison.raw_denominator,
        comparison_ratio=comparison.ratio,
        comparison_status=comparison.status,
        difference=difference,
    )


def _material_ratio(
    numerator: int,
    denominator: int,
) -> CountRatio:
    """將 material typed product 的有效分母轉為既有三態 CountRatio。

    MaterialStatistics 沒有另存 minimum_count；其 raw fraction 仍可由同一個 member
    count 建立完整的 zero／low／available status。這裡只建立狀態 view，不改寫原產品
    的 count 或 fraction，也不把 zero denominator 變成 0.0。
    """

    from .report_ratio_statistics import build_count_ratio

    return build_count_ratio(numerator, denominator, minimum_count=1)


def _aggregate_design_signature(spec: AggregateSpec) -> dict[str, object]:
    """取出不含 run-bound hash 的 AggregateSpec 語意軸 snapshot。"""

    payload = spec.to_dict()
    for field_name in ("run_id", "source_sha256", "canonical_sha256"):
        payload.pop(field_name, None)
    return payload


def _report_policy_signature(spec: ReportSpec) -> dict[str, object]:
    """取出不含 run／aggregate binding hash 的 ReportSpec policy snapshot。"""

    payload = spec.to_dict()
    for field_name in (
        "run_id",
        "aggregate_spec_canonical_sha256",
        "source_sha256",
        "canonical_sha256",
    ):
        payload.pop(field_name, None)
    return payload


def _stratum_axis_signature(
    stratum: ScenarioStratum,
) -> tuple[tuple[str, object], ...]:
    """保留 scenario／site／receptor／arrival／material axis 的完整 typed identity。

    settling_velocity_mps 是目前允許由明示單一物理參數改變的 scenario scalar，因此
    不列入 topology signature；其餘欄位若不同，代表比較不再是同一研究設計，必須拒絕。
    """

    excluded = {"settling_velocity_mps"}
    return tuple(
        (field.name, getattr(stratum, field.name))
        for field in dataclasses.fields(ScenarioStratum)
        if field.name not in excluded
    )


def _validate_input_and_compatibility(
    baseline: object,
    comparison: object,
    parameter_difference: object,
) -> tuple[
    ReportStatistics,
    ReportStatistics,
    AggregateReleasePayload,
    AggregateReleasePayload,
    ReportSpec,
    ReportSpec,
]:
    """先完成 exact input、payload/spec binding 與所有語意軸 compatibility gate。"""

    if type(baseline) is not ReportStatistics:
        raise TypeError("baseline 必須是 exact ReportStatistics")
    if type(comparison) is not ReportStatistics:
        raise TypeError("comparison 必須是 exact ReportStatistics")
    if type(parameter_difference) is not ComparisonParameterDifference:
        raise TypeError("parameter_difference 必須是 exact ComparisonParameterDifference")

    baseline_payload = baseline.aggregate_payload
    comparison_payload = comparison.aggregate_payload
    baseline_spec = baseline.report_spec
    comparison_spec = comparison.report_spec
    if type(baseline_payload) is not AggregateReleasePayload:
        raise ValueError("baseline 必須帶有 exact AggregateReleasePayload")
    if type(comparison_payload) is not AggregateReleasePayload:
        raise ValueError("comparison 必須帶有 exact AggregateReleasePayload")
    if type(baseline_spec) is not ReportSpec:
        raise ValueError("baseline 必須帶有 exact ReportSpec")
    if type(comparison_spec) is not ReportSpec:
        raise ValueError("comparison 必須帶有 exact ReportSpec")

    validate_report_spec_against_aggregate_spec(
        baseline_spec,
        baseline_payload.aggregate_spec,
    )
    validate_report_spec_against_aggregate_spec(
        comparison_spec,
        comparison_payload.aggregate_spec,
    )
    if baseline_payload.run_id == comparison_payload.run_id:
        raise ValueError("baseline 與 comparison 的 run_id 必須不同")
    if baseline_payload.run_kind != comparison_payload.run_kind:
        raise ValueError("baseline 與 comparison 的 run_kind 必須相同")
    if baseline_payload.members_per_scenario != comparison_payload.members_per_scenario:
        raise ValueError("members_per_scenario 必須相同")

    baseline_aggregate = baseline_payload.aggregate_spec
    comparison_aggregate = comparison_payload.aggregate_spec
    if not _deep_equal(
        _aggregate_design_signature(baseline_aggregate),
        _aggregate_design_signature(comparison_aggregate),
    ):
        raise ValueError("AggregateSpec 的 site／geometry／age／KDE 軸或統計政策不相容")
    if not _deep_equal(
        _report_policy_signature(baseline_spec),
        _report_policy_signature(comparison_spec),
    ):
        raise ValueError("ReportSpec 的報告政策不相容")

    # shard 的 digest 與 count 屬於各 run 的結果／發布 provenance，不能拿來代替 axis
    # compatibility；但 scenario range、shard identity 與相對路徑仍必須相同，避免把
    # 不同拓撲的兩組 aggregate 誤當成單一參數比較。
    baseline_shard_axis = tuple(
        (
            shard.shard_id,
            shard.scenario_start_index,
            shard.scenario_stop_index,
            shard.output_relative_path,
        )
        for shard in baseline_payload.shard_bindings
    )
    comparison_shard_axis = tuple(
        (
            shard.shard_id,
            shard.scenario_start_index,
            shard.scenario_stop_index,
            shard.output_relative_path,
        )
        for shard in comparison_payload.shard_bindings
    )
    if baseline_shard_axis != comparison_shard_axis:
        raise ValueError("shard scenario range／identity 不相容")

    baseline_strata = tuple(
        (
            stratum.scenario_id,
            _stratum_axis_signature(stratum),
        )
        for stratum in baseline_payload.scenario_strata
    )
    comparison_strata = tuple(
        (
            stratum.scenario_id,
            _stratum_axis_signature(stratum),
        )
        for stratum in comparison_payload.scenario_strata
    )
    if baseline_strata != comparison_strata:
        raise ValueError("scenario／site／region／receptor／arrival／material identity 不相容")

    # 若 scenario strata 直接保存 settling velocity，該差異必須與 caller 明示的
    # 物理參數一致；不允許藉由未命名的第二個情境欄位偷偷改變比較設計。
    baseline_by_id = {item.scenario_id: item for item in baseline_payload.scenario_strata}
    comparison_by_id = {item.scenario_id: item for item in comparison_payload.scenario_strata}
    settling_pairs = {
        (
            base.settling_velocity_mps,
            comp.settling_velocity_mps,
        )
        for scenario_id, base in baseline_by_id.items()
        for comp in (comparison_by_id[scenario_id],)
        if base.settling_velocity_mps != comp.settling_velocity_mps
    }
    if settling_pairs:
        allowed_names = {
            "settling_velocity_mps",
            "settling_velocity",
            "沉降速度",
        }
        if parameter_difference.name not in allowed_names:
            raise ValueError("scenario settling_velocity 的差異未由明示參數命名")
        if settling_pairs != {
            (
                float(parameter_difference.baseline_value),
                float(parameter_difference.comparison_value),
            )
        }:
            raise ValueError("scenario settling_velocity 差異與明示參數值不一致")

    _validate_product_axes(baseline, comparison)
    return (
        baseline,
        comparison,
        baseline_payload,
        comparison_payload,
        baseline_spec,
        comparison_spec,
    )


def _validate_product_axes(
    baseline: ReportStatistics,
    comparison: ReportStatistics,
) -> None:
    """核對各 typed product 的 semantic axis，不只比較矩陣 shape。"""

    baseline_outcome = baseline.outcome_statistics
    comparison_outcome = comparison.outcome_statistics
    if baseline_outcome.site_ids != comparison_outcome.site_ids:
        raise ValueError("OutcomeStatistics site axis 不相容")
    for site_id in baseline_outcome.site_ids:
        baseline_keys = set(baseline_outcome.outcome_count_by_site[site_id])
        comparison_keys = set(comparison_outcome.outcome_count_by_site[site_id])
        if baseline_keys != comparison_keys:
            raise ValueError(f"site {site_id!r} 的 outcome status axis 不相容")

    baseline_connectivity = baseline.connectivity_statistics
    comparison_connectivity = comparison.connectivity_statistics
    if baseline_connectivity.site_ids != comparison_connectivity.site_ids:
        raise ValueError("ConnectivityStatistics source／target site axis 不相容")
    if baseline_connectivity.raw_matrix.shape != comparison_connectivity.raw_matrix.shape:
        raise ValueError("connectivity raw matrix shape 不相容")

    baseline_source = baseline.source_receptor_statistics
    comparison_source = comparison.source_receptor_statistics
    if baseline_source.age_bin_edges_seconds.shape != comparison_source.age_bin_edges_seconds.shape:
        raise ValueError("source-receptor age axis shape 不相容")
    if not np.array_equal(
        baseline_source.age_bin_edges_seconds,
        comparison_source.age_bin_edges_seconds,
    ):
        raise ValueError("source-receptor age axis 不相容")
    if baseline_source.quantiles != comparison_source.quantiles:
        raise ValueError("source-receptor quantile policy 不相容")
    if tuple(baseline_source.by_key) != tuple(comparison_source.by_key):
        raise ValueError("source-receptor site／receptor／boundary axis 不相容")

    baseline_pathway = baseline.pathway_by_site
    comparison_pathway = comparison.pathway_by_site
    site_ids = baseline_outcome.site_ids
    if set(baseline_pathway) != set(site_ids) or set(comparison_pathway) != set(site_ids):
        raise ValueError("兩側 pathway 必須完整覆蓋 outcome site axis")
    for site_id in site_ids:
        left = baseline_pathway[site_id]
        right = comparison_pathway[site_id]
        for left_axis, right_axis, label in (
            (left.x_edges_m, right.x_edges_m, "pathway x"),
            (left.y_edges_m, right.y_edges_m, "pathway y"),
            (left.age_bin_edges_seconds, right.age_bin_edges_seconds, "pathway age"),
        ):
            if not np.array_equal(left_axis, right_axis):
                raise ValueError(f"site {site_id!r} 的 {label} axis 不相容")
        if left.first_passage_quantiles.keys() != right.first_passage_quantiles.keys():
            raise ValueError(f"site {site_id!r} 的 pathway quantile axis 不相容")
        if left.low_sample_min_member_count != right.low_sample_min_member_count:
            raise ValueError(f"site {site_id!r} 的 pathway low-sample policy 不相容")

    baseline_kde = baseline.kde_by_site
    comparison_kde = comparison.kde_by_site
    if set(baseline_kde) != set(site_ids) or set(comparison_kde) != set(site_ids):
        raise ValueError("兩側 KDE 必須完整覆蓋 outcome site axis")
    for site_id in site_ids:
        left = baseline_kde[site_id]
        right = comparison_kde[site_id]
        if left.site_id != site_id or right.site_id != site_id:
            raise ValueError(f"site {site_id!r} 的 KDE site identity 不相容")
        for left_axis, right_axis, label in (
            (left.x_edges_m, right.x_edges_m, "KDE x"),
            (left.y_edges_m, right.y_edges_m, "KDE y"),
        ):
            if not np.array_equal(left_axis, right_axis):
                raise ValueError(f"site {site_id!r} 的 {label} axis 不相容")
        if (
            left.bandwidths_m != right.bandwidths_m
            or left.hdr_levels != right.hdr_levels
            or left.primary_bandwidth_m != right.primary_bandwidth_m
            or left.minimum_raw_count != right.minimum_raw_count
        ):
            raise ValueError(f"site {site_id!r} 的 KDE／HDR policy 不相容")

    baseline_material = baseline.material_statistics
    comparison_material = comparison.material_statistics
    if (baseline_material is None) != (comparison_material is None):
        raise ValueError("material statistics 必須兩側同時存在或同時缺失")
    if baseline_material is not None and comparison_material is not None and (
        baseline_material.site_ids != comparison_material.site_ids
        or baseline_material.material_ids != comparison_material.material_ids
        or tuple(baseline_material.statistics) != tuple(comparison_material.statistics)
    ):
        raise ValueError("material site／material axis 不相容")


def _cell_status(
    *,
    denominator: int,
    low_sample: bool,
) -> EstimateStatus:
    """由 pathway cell 的有效分母與既有 low-sample mask 重建三態狀態。"""

    if denominator == 0:
        return EstimateStatus.zero_denominator
    if low_sample:
        return EstimateStatus.low_count
    return EstimateStatus.available


def _status_pair_array(
    baseline: np.ndarray,
    comparison: np.ndarray,
    *,
    diagonal_none: bool = False,
) -> np.ndarray:
    """複製 object status pair 陣列並鎖成唯讀，避免 nested tuple 被 caller 改寫。"""

    if baseline.shape != comparison.shape:
        raise ValueError("status pair arrays shape 不一致")
    result = np.empty(baseline.shape, dtype=object)
    for index in np.ndindex(result.shape):
        if diagonal_none and len(index) == 2 and index[0] == index[1]:
            result[index] = None
        else:
            result[index] = (baseline[index], comparison[index])
    result.setflags(write=False)
    return result


def _pathway_status_pairs(
    baseline: Any,
    comparison: Any,
) -> np.ndarray:
    """建立 pathway cell 的 baseline／comparison status pair 陣列。"""

    if baseline.visit_fraction.shape != comparison.visit_fraction.shape:
        raise ValueError("pathway visit fraction shape 不相容")
    result = np.empty(baseline.visit_fraction.shape, dtype=object)
    for index in np.ndindex(result.shape):
        yx = index
        result[index] = (
            _cell_status(
                denominator=baseline.valid_member_denominator,
                low_sample=bool(baseline.low_sample_mask[yx]),
            ),
            _cell_status(
                denominator=comparison.valid_member_denominator,
                low_sample=bool(comparison.low_sample_mask[yx]),
            ),
        )
    result.setflags(write=False)
    return result


def _difference_array(
    baseline: np.ndarray,
    comparison: np.ndarray,
    *,
    unavailable: np.ndarray,
) -> np.ndarray:
    """計算兩側有限陣列的 comparison - baseline，unavailable 保持 NaN。"""

    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(comparison, dtype=np.float64)
    if left.shape != right.shape or left.shape != unavailable.shape:
        raise ValueError("difference array shape 不相容")
    result = np.full(left.shape, np.nan, dtype=np.float64)
    available = (~unavailable) & np.isfinite(left) & np.isfinite(right)
    result[available] = right[available] - left[available]
    result.setflags(write=False)
    return result


def _pathway_mean_residence(
    pathway: Any,
) -> np.ndarray:
    """以有效 member 分母計算每格平均 residence seconds；零分母保持 NaN。"""

    result = np.full(pathway.residence_time_seconds.shape, np.nan, dtype=np.float64)
    if pathway.valid_member_denominator > 0:
        result[:] = (
            pathway.residence_time_seconds / pathway.valid_member_denominator
        )
    result.setflags(write=False)
    return result


def _hdr_overlap(
    baseline_layer: Any,
    comparison_layer: Any,
    *,
    levels: tuple[float, ...],
) -> tuple[Mapping[float, float | None], ComparisonHDRStatus]:
    """計算同一 primary bandwidth 各 HDR level 的 Jaccard overlap。

    兩側 layer 都 unavailable 時回傳 None；兩側 available 且兩張 mask 都為空時，
    將空集合與空集合的 Jaccard 定義為 1.0，表示兩側支援集合完全相同，而不是
    以零表示不可估計。
    """

    status = ComparisonHDRStatus(
        baseline_status=baseline_layer.status,
        comparison_status=comparison_layer.status,
    )
    values: dict[float, float | None] = {}
    for level in levels:
        if not status.available:
            values[level] = None
            continue
        if baseline_layer.grid is None or comparison_layer.grid is None:
            raise RuntimeError("available KDE layer 必須具有 grid")
        baseline_mask = np.asarray(baseline_layer.grid.hdr_masks[level], dtype=bool)
        comparison_mask = np.asarray(comparison_layer.grid.hdr_masks[level], dtype=bool)
        if baseline_mask.shape != comparison_mask.shape:
            raise ValueError("HDR mask shape 不相容")
        union = np.logical_or(baseline_mask, comparison_mask)
        union_count = int(np.count_nonzero(union))
        if union_count == 0:
            values[level] = 1.0
        else:
            intersection_count = int(
                np.count_nonzero(np.logical_and(baseline_mask, comparison_mask))
            )
            values[level] = float(intersection_count / union_count)
    return MappingProxyType(values), status


@dataclass(frozen=True, slots=True)
class ReportComparisonStatistics:
    """封存 F11/T05 不可變的比較統計產品及兩側可追溯資訊。

    比例對照表以站點、來源段—受體或站點×材質識別碼為索引；跨站與路徑格網差值
    保留原產品矩陣軸，軸順序是來源站×目標站或水平格網的 (y_cell, x_cell)。
    公尺軸與秒軸也隨產品保存，避免差值脫離原始幾何與年齡解析度而被誤讀。
    陣列中的非數值標記（NaN）只表示該格任一側不可估計或結構性不適用，
    對應的兩側狀態與有效分母仍可追溯。

    材質（material）差值欄位目前由公開建立函式回傳空對照表；材質來源核對尚未完成，
    帶來源的統計輸入不接受材質產品，因此目前不提供材質差值。所有差值方向均為比較組
    （comparison）減基準組（baseline），產品本身只
    表達同一設計下的單一參數敏感度，不是因果效果估計。
    """

    baseline_run_id: str
    comparison_run_id: str
    parameter_difference: ComparisonParameterDifference
    site_ids: tuple[str, ...]
    age_bin_edges_seconds: np.ndarray
    kde_primary_bandwidth_m: float
    hdr_levels: tuple[float, ...]
    outcome_ratio_differences_by_site: Mapping[str, Mapping[str, ComparisonRatioDifference]]
    connectivity_visit_fraction_difference: np.ndarray
    connectivity_visit_fraction_status: np.ndarray
    connectivity_valid_member_denominator_by_source_site: Mapping[str, tuple[int, int]]
    source_receptor_conditional_ratio_differences_by_key: Mapping[
        SourceReceptorAggregateKey,
        ComparisonRatioDifference,
    ]
    source_receptor_travel_age_median_differences_by_key: Mapping[
        SourceReceptorAggregateKey,
        ComparisonScalarDifference,
    ]
    pathway_visit_fraction_difference_by_site: Mapping[str, np.ndarray]
    pathway_mean_residence_seconds_difference_by_site: Mapping[str, np.ndarray]
    pathway_valid_member_denominator_by_site: Mapping[str, tuple[int, int]]
    pathway_visit_fraction_status_by_site: Mapping[str, np.ndarray]
    pathway_x_edges_m_by_site: Mapping[str, np.ndarray]
    pathway_y_edges_m_by_site: Mapping[str, np.ndarray]
    kde_primary_hdr_jaccard_by_site: Mapping[str, Mapping[float, float | None]]
    kde_primary_hdr_status_by_site: Mapping[str, Mapping[float, ComparisonHDRStatus]]
    kde_primary_raw_point_count_by_site: Mapping[str, tuple[int, int]]
    kde_x_edges_m_by_site: Mapping[str, np.ndarray]
    kde_y_edges_m_by_site: Mapping[str, np.ndarray]
    material_bed_contact_ratio_differences_by_site_material: Mapping[
        tuple[str, str],
        ComparisonRatioDifference,
    ] = field(default_factory=dict)
    material_deposited_ratio_differences_by_site_material: Mapping[
        tuple[str, str],
        ComparisonRatioDifference,
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """重新驗證並 defensive-copy 公開組合產品的 mapping 與陣列。"""

        baseline_run_id = _require_text(self.baseline_run_id, label="baseline_run_id")
        comparison_run_id = _require_text(
            self.comparison_run_id,
            label="comparison_run_id",
        )
        if baseline_run_id == comparison_run_id:
            raise ValueError("baseline_run_id 與 comparison_run_id 必須不同")
        if type(self.parameter_difference) is not ComparisonParameterDifference:
            raise TypeError("parameter_difference 必須是 exact ComparisonParameterDifference")
        site_ids = _snapshot_site_ids(self.site_ids)
        age_edges = _readonly_edges(
            self.age_bin_edges_seconds,
            label="age_bin_edges_seconds",
        )
        if type(self.kde_primary_bandwidth_m) is not float:
            raise TypeError("kde_primary_bandwidth_m 必須是原生 float")
        if not math.isfinite(self.kde_primary_bandwidth_m) or self.kde_primary_bandwidth_m <= 0.0:
            raise ValueError("kde_primary_bandwidth_m 必須是有限正值")
        hdr_levels = tuple(self.hdr_levels)
        if (
            not hdr_levels
            or any(type(level) is not float or not 0.0 < level < 1.0 for level in hdr_levels)
            or any(left >= right for left, right in zip(hdr_levels, hdr_levels[1:], strict=False))
        ):
            raise ValueError("hdr_levels 必須是嚴格遞增的有限 float tuple")

        outcome: dict[str, Mapping[str, ComparisonRatioDifference]] = {}
        for site_id, ratios in self.outcome_ratio_differences_by_site.items():
            if site_id not in site_ids or not isinstance(ratios, Mapping):
                raise ValueError("outcome ratio mapping 的 site key 或 value 不合法")
            inner: dict[str, ComparisonRatioDifference] = {}
            for outcome_id, difference in ratios.items():
                _require_text(outcome_id, label="outcome status key")
                if type(difference) is not ComparisonRatioDifference:
                    raise TypeError("outcome difference 必須是 exact ComparisonRatioDifference")
                inner[outcome_id] = difference
            outcome[site_id] = MappingProxyType(inner)
        if set(outcome) != set(site_ids):
            raise ValueError("outcome ratio mapping 必須完整覆蓋 site_ids")

        connectivity_difference = _readonly_array(
            self.connectivity_visit_fraction_difference,
            dtype=np.dtype(np.float64),
            ndim=2,
            label="connectivity_visit_fraction_difference",
        )
        expected_matrix_shape = (len(site_ids), len(site_ids))
        if connectivity_difference.shape != expected_matrix_shape:
            raise ValueError("connectivity difference shape 必須是 source×target site axis")
        connectivity_status = _readonly_array(
            self.connectivity_visit_fraction_status,
            dtype=np.dtype(object),
            ndim=2,
            label="connectivity_visit_fraction_status",
        )
        if connectivity_status.shape != expected_matrix_shape:
            raise ValueError("connectivity status shape 必須是 source×target site axis")
        for index in np.ndindex(connectivity_status.shape):
            value = connectivity_status[index]
            if index[0] == index[1]:
                if value is not None:
                    raise ValueError("connectivity 對角線 status 必須是 None")
            elif (
                not isinstance(value, tuple)
                or len(value) != 2
                or any(type(status) is not EstimateStatus for status in value)
            ):
                raise TypeError("connectivity status 必須是兩側 EstimateStatus tuple")
        connectivity_denominators = _snapshot_denominator_pairs(
            self.connectivity_valid_member_denominator_by_source_site,
            site_ids=site_ids,
            label="connectivity_valid_member_denominator_by_source_site",
        )

        source_ratio = _snapshot_source_ratio_mapping(
            self.source_receptor_conditional_ratio_differences_by_key,
            label="source_receptor_conditional_ratio_differences_by_key",
        )
        source_age = _snapshot_source_scalar_mapping(
            self.source_receptor_travel_age_median_differences_by_key,
            label="source_receptor_travel_age_median_differences_by_key",
        )
        if set(source_ratio) != set(source_age):
            raise ValueError("source-receptor ratio／travel-age key set 必須一致")
        if any(key.study_site_id not in site_ids for key in source_ratio):
            raise ValueError("source-receptor key 超出 site axis")

        pathway_difference = _snapshot_array_mapping(
            self.pathway_visit_fraction_difference_by_site,
            site_ids=site_ids,
            label="pathway_visit_fraction_difference_by_site",
        )
        residence_difference = _snapshot_array_mapping(
            self.pathway_mean_residence_seconds_difference_by_site,
            site_ids=site_ids,
            label="pathway_mean_residence_seconds_difference_by_site",
        )
        pathway_status = _snapshot_status_array_mapping(
            self.pathway_visit_fraction_status_by_site,
            site_ids=site_ids,
            label="pathway_visit_fraction_status_by_site",
        )
        if any(
            pathway_difference[site_id].shape != residence_difference[site_id].shape
            or pathway_difference[site_id].shape != pathway_status[site_id].shape
            for site_id in site_ids
        ):
            raise ValueError("pathway difference／status shape 必須一致")
        pathway_denominators = _snapshot_denominator_pairs(
            self.pathway_valid_member_denominator_by_site,
            site_ids=site_ids,
            label="pathway_valid_member_denominator_by_site",
        )
        pathway_x_edges = _snapshot_edges_mapping(
            self.pathway_x_edges_m_by_site,
            site_ids=site_ids,
            label="pathway_x_edges_m_by_site",
        )
        pathway_y_edges = _snapshot_edges_mapping(
            self.pathway_y_edges_m_by_site,
            site_ids=site_ids,
            label="pathway_y_edges_m_by_site",
        )

        hdr_values = _snapshot_hdr_value_mapping(
            self.kde_primary_hdr_jaccard_by_site,
            site_ids=site_ids,
            hdr_levels=hdr_levels,
        )
        hdr_status = _snapshot_hdr_status_mapping(
            self.kde_primary_hdr_status_by_site,
            site_ids=site_ids,
            hdr_levels=hdr_levels,
        )
        kde_counts = _snapshot_denominator_pairs(
            self.kde_primary_raw_point_count_by_site,
            site_ids=site_ids,
            label="kde_primary_raw_point_count_by_site",
        )
        kde_x_edges = _snapshot_edges_mapping(
            self.kde_x_edges_m_by_site,
            site_ids=site_ids,
            label="kde_x_edges_m_by_site",
        )
        kde_y_edges = _snapshot_edges_mapping(
            self.kde_y_edges_m_by_site,
            site_ids=site_ids,
            label="kde_y_edges_m_by_site",
        )
        material_bed = _snapshot_material_ratio_mapping(
            self.material_bed_contact_ratio_differences_by_site_material,
            label="material_bed_contact_ratio_differences_by_site_material",
        )
        material_deposited = _snapshot_material_ratio_mapping(
            self.material_deposited_ratio_differences_by_site_material,
            label="material_deposited_ratio_differences_by_site_material",
        )
        if set(material_bed) != set(material_deposited):
            raise ValueError("material bed-contact／deposited key set 必須一致")
        if any(key[0] not in site_ids for key in material_bed):
            raise ValueError("material key 超出 site axis")

        object.__setattr__(self, "baseline_run_id", baseline_run_id)
        object.__setattr__(self, "comparison_run_id", comparison_run_id)
        object.__setattr__(self, "site_ids", site_ids)
        object.__setattr__(self, "age_bin_edges_seconds", age_edges)
        object.__setattr__(self, "hdr_levels", hdr_levels)
        object.__setattr__(self, "outcome_ratio_differences_by_site", MappingProxyType(outcome))
        object.__setattr__(self, "connectivity_visit_fraction_difference", connectivity_difference)
        object.__setattr__(self, "connectivity_visit_fraction_status", connectivity_status)
        object.__setattr__(
            self,
            "connectivity_valid_member_denominator_by_source_site",
            connectivity_denominators,
        )
        object.__setattr__(
            self,
            "source_receptor_conditional_ratio_differences_by_key",
            source_ratio,
        )
        object.__setattr__(
            self,
            "source_receptor_travel_age_median_differences_by_key",
            source_age,
        )
        object.__setattr__(self, "pathway_visit_fraction_difference_by_site", pathway_difference)
        object.__setattr__(
            self,
            "pathway_mean_residence_seconds_difference_by_site",
            residence_difference,
        )
        object.__setattr__(self, "pathway_valid_member_denominator_by_site", pathway_denominators)
        object.__setattr__(self, "pathway_visit_fraction_status_by_site", pathway_status)
        object.__setattr__(self, "pathway_x_edges_m_by_site", pathway_x_edges)
        object.__setattr__(self, "pathway_y_edges_m_by_site", pathway_y_edges)
        object.__setattr__(self, "kde_primary_hdr_jaccard_by_site", hdr_values)
        object.__setattr__(self, "kde_primary_hdr_status_by_site", hdr_status)
        object.__setattr__(self, "kde_primary_raw_point_count_by_site", kde_counts)
        object.__setattr__(self, "kde_x_edges_m_by_site", kde_x_edges)
        object.__setattr__(self, "kde_y_edges_m_by_site", kde_y_edges)
        object.__setattr__(
            self,
            "material_bed_contact_ratio_differences_by_site_material",
            material_bed,
        )
        object.__setattr__(
            self,
            "material_deposited_ratio_differences_by_site_material",
            material_deposited,
        )

    @property
    def outcome_ratio_difference_by_site(self) -> Mapping[str, Mapping[str, ComparisonRatioDifference]]:
        """outcome_ratio_differences_by_site 的單數欄位別名。"""

        return self.outcome_ratio_differences_by_site

    @property
    def connectivity_visit_fraction_differences(self) -> np.ndarray:
        """回傳 connectivity visit-fraction 差值唯讀矩陣。"""

        return self.connectivity_visit_fraction_difference

    @property
    def source_receptor_conditional_ratio_difference_by_key(
        self,
    ) -> Mapping[SourceReceptorAggregateKey, ComparisonRatioDifference]:
        """source-receptor conditional ratio mapping 的單數別名。"""

        return self.source_receptor_conditional_ratio_differences_by_key

    @property
    def source_receptor_travel_age_median_difference_by_key(
        self,
    ) -> Mapping[SourceReceptorAggregateKey, ComparisonScalarDifference]:
        """source-receptor travel-age median mapping 的單數別名。"""

        return self.source_receptor_travel_age_median_differences_by_key

    @property
    def pathway_visit_fraction_differences_by_site(self) -> Mapping[str, np.ndarray]:
        """pathway_visit_fraction_difference_by_site 的複數別名。"""

        return self.pathway_visit_fraction_difference_by_site

    @property
    def pathway_mean_residence_difference_by_site(self) -> Mapping[str, np.ndarray]:
        """pathway mean residence seconds 差值的簡短別名。"""

        return self.pathway_mean_residence_seconds_difference_by_site

    @property
    def hdr_jaccard_overlap_by_site(self) -> Mapping[str, Mapping[float, float | None]]:
        """kde_primary_hdr_jaccard_by_site 的語意別名。"""

        return self.kde_primary_hdr_jaccard_by_site

    @property
    def material_ratio_differences_by_site_material(
        self,
    ) -> Mapping[tuple[str, str], Mapping[str, ComparisonRatioDifference]]:
        """將兩種 material ratio 差值組合成唯讀的 bed_contact/deposited mapping。"""

        return MappingProxyType(
            {
                key: MappingProxyType(
                    {
                        "bed_contact": self.material_bed_contact_ratio_differences_by_site_material[key],
                        "deposited": self.material_deposited_ratio_differences_by_site_material[key],
                    }
                )
                for key in self.material_bed_contact_ratio_differences_by_site_material
            }
        )


def _snapshot_denominator_pairs(
    value: Mapping[str, tuple[int, int]],
    *,
    site_ids: tuple[str, ...],
    label: str,
) -> Mapping[str, tuple[int, int]]:
    """封存 site 到兩側 raw denominator/count 的 tuple mapping。"""

    if not isinstance(value, Mapping) or set(value) != set(site_ids):
        raise ValueError(f"{label} 必須完整覆蓋 site axis")
    copied: dict[str, tuple[int, int]] = {}
    for site_id in site_ids:
        pair = value[site_id]
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(type(item) is not int for item in pair)
            or any(item < 0 for item in pair)
        ):
            raise TypeError(f"{label}[{site_id!r}] 必須是兩側非負原生 int tuple")
        copied[site_id] = pair
    return MappingProxyType(copied)


def _snapshot_array_mapping(
    value: Mapping[str, np.ndarray],
    *,
    site_ids: tuple[str, ...],
    label: str,
) -> Mapping[str, np.ndarray]:
    """複製 site 到 float array mapping 並鎖定每個陣列。"""

    if not isinstance(value, Mapping) or set(value) != set(site_ids):
        raise ValueError(f"{label} 必須完整覆蓋 site axis")
    copied = {
        site_id: _readonly_array(
            value[site_id],
            dtype=np.dtype(np.float64),
            ndim=2,
            label=f"{label}[{site_id}]",
        )
        for site_id in site_ids
    }
    return MappingProxyType(copied)


def _snapshot_status_array_mapping(
    value: Mapping[str, np.ndarray],
    *,
    site_ids: tuple[str, ...],
    label: str,
) -> Mapping[str, np.ndarray]:
    """複製 pathway status pair array mapping並驗證每格兩側狀態。"""

    if not isinstance(value, Mapping) or set(value) != set(site_ids):
        raise ValueError(f"{label} 必須完整覆蓋 site axis")
    copied: dict[str, np.ndarray] = {}
    for site_id in site_ids:
        array = _readonly_array(
            value[site_id],
            dtype=np.dtype(object),
            ndim=2,
            label=f"{label}[{site_id}]",
        )
        for status_pair in array.flat:
            if (
                not isinstance(status_pair, tuple)
                or len(status_pair) != 2
                or any(type(status) is not EstimateStatus for status in status_pair)
            ):
                raise TypeError(f"{label}[{site_id!r}] 每格必須是兩側 EstimateStatus tuple")
        copied[site_id] = array
    return MappingProxyType(copied)


def _snapshot_edges_mapping(
    value: Mapping[str, np.ndarray],
    *,
    site_ids: tuple[str, ...],
    label: str,
) -> Mapping[str, np.ndarray]:
    """複製各站公尺制邊界 mapping，保留 axis 單位與順序。"""

    if not isinstance(value, Mapping) or set(value) != set(site_ids):
        raise ValueError(f"{label} 必須完整覆蓋 site axis")
    copied = {
        site_id: _readonly_edges(value[site_id], label=f"{label}[{site_id}]")
        for site_id in site_ids
    }
    return MappingProxyType(copied)


def _snapshot_source_ratio_mapping(
    value: Mapping[SourceReceptorAggregateKey, ComparisonRatioDifference],
    *,
    label: str,
) -> Mapping[SourceReceptorAggregateKey, ComparisonRatioDifference]:
    """封存 source-receptor ratio 差值 mapping 與 exact key。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必須是 mapping")
    copied: dict[SourceReceptorAggregateKey, ComparisonRatioDifference] = {}
    for key, difference in value.items():
        if type(key) is not SourceReceptorAggregateKey:
            raise TypeError(f"{label} key 必須是 exact SourceReceptorAggregateKey")
        if type(difference) is not ComparisonRatioDifference:
            raise TypeError(f"{label} value 必須是 exact ComparisonRatioDifference")
        copied[key] = difference
    return MappingProxyType(copied)


def _snapshot_source_scalar_mapping(
    value: Mapping[SourceReceptorAggregateKey, ComparisonScalarDifference],
    *,
    label: str,
) -> Mapping[SourceReceptorAggregateKey, ComparisonScalarDifference]:
    """封存 source-receptor scalar 差值 mapping 與 exact key。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必須是 mapping")
    copied: dict[SourceReceptorAggregateKey, ComparisonScalarDifference] = {}
    for key, difference in value.items():
        if type(key) is not SourceReceptorAggregateKey:
            raise TypeError(f"{label} key 必須是 exact SourceReceptorAggregateKey")
        if type(difference) is not ComparisonScalarDifference:
            raise TypeError(f"{label} value 必須是 exact ComparisonScalarDifference")
        copied[key] = difference
    return MappingProxyType(copied)


def _snapshot_hdr_value_mapping(
    value: Mapping[str, Mapping[float, float | None]],
    *,
    site_ids: tuple[str, ...],
    hdr_levels: tuple[float, ...],
) -> Mapping[str, Mapping[float, float | None]]:
    """封存各站各 HDR level 的 Jaccard scalar mapping。"""

    if not isinstance(value, Mapping) or set(value) != set(site_ids):
        raise ValueError("HDR Jaccard mapping 必須完整覆蓋 site axis")
    copied: dict[str, Mapping[float, float | None]] = {}
    for site_id in site_ids:
        levels = value[site_id]
        if not isinstance(levels, Mapping) or tuple(levels) != hdr_levels:
            raise ValueError(f"site {site_id!r} 的 HDR level axis 不相容")
        inner: dict[float, float | None] = {}
        for level in hdr_levels:
            number = levels[level]
            if number is not None and (type(number) is not float or not math.isfinite(number)):
                raise TypeError("HDR Jaccard 必須是有限原生 float 或 None")
            if number is not None and not 0.0 <= number <= 1.0:
                raise ValueError("HDR Jaccard 必須位於 [0, 1]")
            inner[level] = number
        copied[site_id] = MappingProxyType(inner)
    return MappingProxyType(copied)


def _snapshot_hdr_status_mapping(
    value: Mapping[str, Mapping[float, ComparisonHDRStatus]],
    *,
    site_ids: tuple[str, ...],
    hdr_levels: tuple[float, ...],
) -> Mapping[str, Mapping[float, ComparisonHDRStatus]]:
    """封存各站各 HDR level 的兩側 availability status。"""

    if not isinstance(value, Mapping) or set(value) != set(site_ids):
        raise ValueError("HDR status mapping 必須完整覆蓋 site axis")
    copied: dict[str, Mapping[float, ComparisonHDRStatus]] = {}
    for site_id in site_ids:
        levels = value[site_id]
        if not isinstance(levels, Mapping) or tuple(levels) != hdr_levels:
            raise ValueError(f"site {site_id!r} 的 HDR status level axis 不相容")
        inner: dict[float, ComparisonHDRStatus] = {}
        for level in hdr_levels:
            status = levels[level]
            if type(status) is not ComparisonHDRStatus:
                raise TypeError("HDR status 必須是 exact ComparisonHDRStatus")
            inner[level] = status
        copied[site_id] = MappingProxyType(inner)
    return MappingProxyType(copied)


def _snapshot_material_ratio_mapping(
    value: Mapping[tuple[str, str], ComparisonRatioDifference],
    *,
    label: str,
) -> Mapping[tuple[str, str], ComparisonRatioDifference]:
    """封存 site×material ratio 差值 mapping。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必須是 mapping")
    copied: dict[tuple[str, str], ComparisonRatioDifference] = {}
    for key, difference in value.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(type(item) is not str or not item for item in key)
        ):
            raise TypeError(f"{label} key 必須是 (site_id, material_id) tuple")
        if type(difference) is not ComparisonRatioDifference:
            raise TypeError(f"{label} value 必須是 exact ComparisonRatioDifference")
        copied[key] = difference
    return MappingProxyType(copied)


def build_report_comparison_statistics(
    baseline: ReportStatistics,
    comparison: ReportStatistics,
    parameter_difference: ComparisonParameterDifference,
) -> ReportComparisonStatistics:
    """建立 F11/T05 比較統計，所有差值方向固定為比較組減基準組。

    Args:
        baseline: 基準組統計，須帶已驗證的來源聚合資料（AggregateReleasePayload）
            及報告規格（ReportSpec）；不接受無來源身分的純產品組合。
        comparison: 比較組統計，須有相同型別的已驗證來源聚合資料與報告規格。
        parameter_difference: 呼叫端明示的單一物理參數名稱、兩側數值與單位。

    Returns:
        不可變的 ReportComparisonStatistics。比例、旅行年齡、路徑格網及高密度區域
        （HDR）差值保留兩側原始分母、狀態或可追溯的座標軸。材質來源核對尚未完成，
        目前不提供材質差值，其保留欄位由本入口回傳空對照表，不代表全部 F11/T05 差值可用。

    Raises:
        TypeError／ValueError: 輸入型別、來源身分、座標軸語意、報告政策或差值契約
            不相容時。任一不可估計比例不會被轉成數值零。
    """

    (
        baseline,
        comparison,
        baseline_payload,
        comparison_payload,
        baseline_spec,
        _comparison_spec,
    ) = _validate_input_and_compatibility(
        baseline,
        comparison,
        parameter_difference,
    )
    site_ids = baseline.outcome_statistics.site_ids

    outcome_differences: dict[str, Mapping[str, ComparisonRatioDifference]] = {}
    for site_id in site_ids:
        baseline_ratios = baseline.outcome.outcome_ratios_by_site[site_id]
        comparison_ratios = comparison.outcome.outcome_ratios_by_site[site_id]
        outcome_differences[site_id] = MappingProxyType(
            {
                outcome_id: _comparison_ratio(
                    baseline_ratios[outcome_id],
                    comparison_ratios[outcome_id],
                )
                for outcome_id in baseline_ratios
            }
        )

    baseline_connectivity = baseline.connectivity
    comparison_connectivity = comparison.connectivity
    connectivity_difference = _difference_array(
        baseline_connectivity.visit_fraction,
        comparison_connectivity.visit_fraction,
        unavailable=baseline_connectivity.diagonal_not_applicable_mask
        | np.isnan(baseline_connectivity.visit_fraction)
        | np.isnan(comparison_connectivity.visit_fraction),
    )
    connectivity_status_baseline = baseline_connectivity.visit_fraction_status
    connectivity_status_comparison = comparison_connectivity.visit_fraction_status
    connectivity_status = _status_pair_array(
        connectivity_status_baseline,
        connectivity_status_comparison,
        diagonal_none=True,
    )
    connectivity_denominators = {
        site_id: (
            baseline_connectivity.valid_member_denominator_by_source_site[site_id],
            comparison_connectivity.valid_member_denominator_by_source_site[site_id],
        )
        for site_id in site_ids
    }

    baseline_source = baseline.source_receptor
    comparison_source = comparison.source_receptor
    source_ratio_differences: dict[SourceReceptorAggregateKey, ComparisonRatioDifference] = {}
    source_age_differences: dict[SourceReceptorAggregateKey, ComparisonScalarDifference] = {}
    for key in baseline_source.by_key:
        baseline_record = baseline_source.by_key[key]
        comparison_record = comparison_source.by_key[key]
        source_ratio_differences[key] = _comparison_ratio(
            baseline_record.conditional_ratio,
            comparison_record.conditional_ratio,
        )
        if 0.5 not in baseline_record.travel_age.quantile_values_seconds:
            raise ValueError("source-receptor travel-age quantiles 必須包含 0.5 中位數")
        source_age_differences[key] = ComparisonScalarDifference(
            baseline_value=baseline_record.travel_age.quantile_values_seconds[0.5],
            baseline_status=baseline_record.travel_age.status,
            comparison_value=comparison_record.travel_age.quantile_values_seconds[0.5],
            comparison_status=comparison_record.travel_age.status,
            difference=(
                None
                if (
                    baseline_record.travel_age.quantile_values_seconds[0.5] is None
                    or comparison_record.travel_age.quantile_values_seconds[0.5] is None
                )
                else float(
                    comparison_record.travel_age.quantile_values_seconds[0.5]
                    - baseline_record.travel_age.quantile_values_seconds[0.5]
                )
            ),
        )

    pathway_visit_differences: dict[str, np.ndarray] = {}
    pathway_residence_differences: dict[str, np.ndarray] = {}
    pathway_statuses: dict[str, np.ndarray] = {}
    pathway_denominators: dict[str, tuple[int, int]] = {}
    pathway_x_edges: dict[str, np.ndarray] = {}
    pathway_y_edges: dict[str, np.ndarray] = {}
    for site_id in site_ids:
        baseline_pathway = baseline.pathway_by_site[site_id]
        comparison_pathway = comparison.pathway_by_site[site_id]
        unavailable = (
            np.full(baseline_pathway.visit_fraction.shape, False, dtype=bool)
            if (
                baseline_pathway.valid_member_denominator > 0
                and comparison_pathway.valid_member_denominator > 0
            )
            else np.full(baseline_pathway.visit_fraction.shape, True, dtype=bool)
        )
        pathway_visit_differences[site_id] = _difference_array(
            baseline_pathway.visit_fraction,
            comparison_pathway.visit_fraction,
            unavailable=unavailable,
        )
        baseline_mean_residence = _pathway_mean_residence(baseline_pathway)
        comparison_mean_residence = _pathway_mean_residence(comparison_pathway)
        pathway_residence_differences[site_id] = _difference_array(
            baseline_mean_residence,
            comparison_mean_residence,
            unavailable=unavailable,
        )
        pathway_statuses[site_id] = _pathway_status_pairs(
            baseline_pathway,
            comparison_pathway,
        )
        pathway_denominators[site_id] = (
            baseline_pathway.valid_member_denominator,
            comparison_pathway.valid_member_denominator,
        )
        pathway_x_edges[site_id] = np.array(
            baseline_pathway.x_edges_m,
            dtype=np.float64,
            copy=True,
        )
        pathway_y_edges[site_id] = np.array(
            baseline_pathway.y_edges_m,
            dtype=np.float64,
            copy=True,
        )

    hdr_values: dict[str, Mapping[float, float | None]] = {}
    hdr_statuses: dict[str, Mapping[float, ComparisonHDRStatus]] = {}
    kde_counts: dict[str, tuple[int, int]] = {}
    kde_x_edges: dict[str, np.ndarray] = {}
    kde_y_edges: dict[str, np.ndarray] = {}
    hdr_levels = tuple(float(level) for level in baseline_payload.aggregate_spec.hdr_levels)
    primary_bandwidth = float(baseline_spec.primary_kde_bandwidth_m)
    for site_id in site_ids:
        baseline_product = baseline.kde_by_site[site_id]
        comparison_product = comparison.kde_by_site[site_id]
        baseline_layer = baseline_product.layers[primary_bandwidth]
        comparison_layer = comparison_product.layers[primary_bandwidth]
        values, status = _hdr_overlap(
            baseline_layer,
            comparison_layer,
            levels=hdr_levels,
        )
        hdr_values[site_id] = values
        hdr_statuses[site_id] = MappingProxyType(
            {level: status for level in hdr_levels}
        )
        kde_counts[site_id] = (
            baseline_product.raw_point_count,
            comparison_product.raw_point_count,
        )
        kde_x_edges[site_id] = np.array(
            baseline_product.x_edges_m,
            dtype=np.float64,
            copy=True,
        )
        kde_y_edges[site_id] = np.array(
            baseline_product.y_edges_m,
            dtype=np.float64,
            copy=True,
        )

    material_bed: dict[tuple[str, str], ComparisonRatioDifference] = {}
    material_deposited: dict[tuple[str, str], ComparisonRatioDifference] = {}
    baseline_material = baseline.material_statistics
    comparison_material = comparison.material_statistics
    if baseline_material is not None and comparison_material is not None:
        for key in baseline_material.statistics:
            baseline_record = baseline_material.statistics[key]
            comparison_record = comparison_material.statistics[key]
            baseline_bed = _material_ratio(
                baseline_record.first_bed_contact_member_count,
                baseline_record.valid_member_denominator,
            )
            comparison_bed = _material_ratio(
                comparison_record.first_bed_contact_member_count,
                comparison_record.valid_member_denominator,
            )
            baseline_deposited = _material_ratio(
                baseline_record.deposited_member_count,
                baseline_record.valid_member_denominator,
            )
            comparison_deposited = _material_ratio(
                comparison_record.deposited_member_count,
                comparison_record.valid_member_denominator,
            )
            material_bed[key] = _comparison_ratio(baseline_bed, comparison_bed)
            material_deposited[key] = _comparison_ratio(
                baseline_deposited,
                comparison_deposited,
            )

    return ReportComparisonStatistics(
        baseline_run_id=baseline_payload.run_id,
        comparison_run_id=comparison_payload.run_id,
        parameter_difference=parameter_difference,
        site_ids=site_ids,
        age_bin_edges_seconds=np.array(
            baseline_payload.aggregate_spec.age_bin_edges_seconds,
            dtype=np.float64,
            copy=True,
        ),
        kde_primary_bandwidth_m=primary_bandwidth,
        hdr_levels=hdr_levels,
        outcome_ratio_differences_by_site=outcome_differences,
        connectivity_visit_fraction_difference=connectivity_difference,
        connectivity_visit_fraction_status=connectivity_status,
        connectivity_valid_member_denominator_by_source_site=connectivity_denominators,
        source_receptor_conditional_ratio_differences_by_key=source_ratio_differences,
        source_receptor_travel_age_median_differences_by_key=source_age_differences,
        pathway_visit_fraction_difference_by_site=pathway_visit_differences,
        pathway_mean_residence_seconds_difference_by_site=pathway_residence_differences,
        pathway_valid_member_denominator_by_site=pathway_denominators,
        pathway_visit_fraction_status_by_site=pathway_statuses,
        pathway_x_edges_m_by_site=pathway_x_edges,
        pathway_y_edges_m_by_site=pathway_y_edges,
        kde_primary_hdr_jaccard_by_site=hdr_values,
        kde_primary_hdr_status_by_site=hdr_statuses,
        kde_primary_raw_point_count_by_site=kde_counts,
        kde_x_edges_m_by_site=kde_x_edges,
        kde_y_edges_m_by_site=kde_y_edges,
        material_bed_contact_ratio_differences_by_site_material=material_bed,
        material_deposited_ratio_differences_by_site_material=material_deposited,
    )
