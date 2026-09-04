"""提供繪圖與表格產製端（renderer）使用的統一報告資料介面（facade）。

本模組組合型別及內容已驗證的統計產品（typed products），或從已驗證聚合資料
（AggregateReleasePayload，以下稱來源資料）依報告規格（ReportSpec）建立統計。
它不讀寫檔案或重開軌跡分片，也不接受繪圖端臨時指定分位數、平滑帶寬或分母。
需要衍生產品時，會先核對報告規格與聚合規格的來源關聯，再沿用已登錄的政策。

核心產品包含停止結果 ``OutcomeStatistics``、方向性跨站 ``ConnectivityStatistics``
及來源段—受體 ``SourceReceptorStatistics``。提供來源時，核心與路徑格網產品會和
同一來源直接衍生的完整資料逐欄核對；核密度估計（KDE）在本類別內依來源單次產生，不接受
帶來源身分的外部密度覆寫。材料統計缺少可由聚合資料核對的逐粒子來源，因此只供
沒有執行批次（run）身分的純產品入口使用。各層對照表會複製後封存，供繪圖端
讀取既有原始計數（raw count）、比例與狀態，不在圖表函式內另訂分母或補零。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType

import numpy as np

from .aggregate_release_payload import AggregateReleasePayload
from .report_kde_statistics import KDESensitivityProduct, build_kde_sensitivity_product
from .report_material_statistics import MaterialStatisticsProduct
from .report_matrix_statistics import (
    ConnectivityStatistics,
    OutcomeStatistics,
    build_connectivity_statistics,
    build_outcome_statistics,
)
from .report_pathway_statistics import PathwayGridStatistics, build_pathway_grid_statistics
from .report_source_receptor_statistics import (
    SourceReceptorStatistics,
    build_source_receptor_statistics,
)
from .report_spec import ReportSpec, validate_report_spec_against_aggregate_spec

__all__ = [
    "ReportStatistics",
    "ReportStatisticsFacade",
    "ReportStatisticsProduct",
    "ReportStatisticsResult",
    "build_report_products",
    "build_report_statistics",
]


def _snapshot_product_mapping(
    value: object | None,
    *,
    product_type: type,
    label: str,
    expected_site_ids: tuple[str, ...] | None = None,
) -> Mapping[str, object]:
    """複製 site 到 typed product mapping，並檢查 key set 與 exact product type。

    renderer 需要穩定的 site join；因此此處不接受子類別、duck object 或缺列，亦不
    以空 mapping 靜默代表 caller 忘了提供完整產品。mapping 只做 defensive snapshot，
    不會改變產品內部的 immutable 陣列。
    """

    if value is None:
        copied: dict[str, object] = {}
    else:
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} 必須是 site 到 typed product 的 mapping")
        copied = {}
        for site_id, product in value.items():
            if type(site_id) is not str or not site_id or site_id != site_id.strip():
                raise ValueError(f"{label} 的 site key 必須是非空原生 str")
            if type(product) is not product_type:
                raise TypeError(f"{label}[{site_id!r}] 必須是 exact {product_type.__name__}")
            if site_id in copied:
                raise ValueError(f"{label} 不可有重複 site key")
            copied[site_id] = product
    if expected_site_ids is not None and set(copied) != set(expected_site_ids):
        raise ValueError(f"{label} 的 site key 必須精確等於 facade core site set")
    return MappingProxyType(copied)


def _require_core_products(
    outcome: object | None,
    connectivity: object | None,
    source_receptor: object | None,
) -> tuple[OutcomeStatistics, ConnectivityStatistics, SourceReceptorStatistics]:
    """驗證 facade 的核心三項產品必須是 exact immutable 類別。"""

    if type(outcome) is not OutcomeStatistics:
        raise TypeError("outcome_statistics 必須是 exact OutcomeStatistics")
    if type(connectivity) is not ConnectivityStatistics:
        raise TypeError("connectivity_statistics 必須是 exact ConnectivityStatistics")
    if type(source_receptor) is not SourceReceptorStatistics:
        raise TypeError(
            "source_receptor_statistics 必須是 exact SourceReceptorStatistics"
        )
    return outcome, connectivity, source_receptor


def _require_same_product(actual: object, expected: object, *, label: str) -> None:
    """逐欄核對報告產品，不以同形狀、同總數或呼叫端提供的雜湊代替來源證據。

    待驗產品 actual 與基準產品 expected 都在記憶體中；後者須由目前來源聚合資料
    （payload）的既有統計建立函式（builder）產生。本函式比較資料類別、對照表
    （mapping）、序列及陣列，不讀檔或重算核密度估計。浮點陣列僅允許相同位置的
    非數值標記（NaN）相等，以保留零分母的不可用語意；其餘值、陣列資料型別
    （dtype）及形狀均須精確相等。不符時拋出 ValueError，label 指出第一個不符欄位；
    通過時不回傳資料、不修改輸入，也不為原始計數、分母或座標軸補值。
    """

    if type(actual) is not type(expected):
        raise ValueError(f"{label} 與 payload 衍生產品型別不一致")
    if isinstance(expected, np.ndarray):
        if actual.dtype != expected.dtype or not np.array_equal(
            actual, expected, equal_nan=expected.dtype.kind in "fc"
        ):
            raise ValueError(f"{label} 與 payload 衍生陣列不一致")
    elif is_dataclass(expected):
        for item in fields(expected):
            _require_same_product(
                getattr(actual, item.name), getattr(expected, item.name),
                label=f"{label}.{item.name}",
            )
    elif isinstance(expected, Mapping):
        if actual.keys() != expected.keys():
            raise ValueError(f"{label} 與 payload 衍生產品 key 不一致")
        for key, value in expected.items():
            _require_same_product(actual[key], value, label=f"{label}[{key!r}]")
    elif isinstance(expected, tuple):
        if len(actual) != len(expected):
            raise ValueError(f"{label} 與 payload 衍生產品長度不一致")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _require_same_product(left, right, label=f"{label}[{index}]")
    elif actual != expected:
        raise ValueError(f"{label} 與 payload 衍生產品數值不一致")


def _validate_payload_products(
    payload: AggregateReleasePayload,
    outcome: OutcomeStatistics,
    connectivity: ConnectivityStatistics,
    source_receptor: SourceReceptorStatistics,
    pathway: Mapping[str, PathwayGridStatistics],
    report_spec: ReportSpec | None,
) -> None:
    """由已驗證的來源聚合資料重建可查證產品，拒絕混入其他執行批次的計數。

    核對包含停止／失敗格網、跨站方向矩陣、受體與來源段識別碼、全部原始計數與分母、
    旅行年齡直方圖及其秒軸。路徑產品另核對公尺格線、訪格數、停留秒數及由來源
    直方圖重建的分位數；不改變既有有效成員分母。若有 ReportSpec，政策亦須經獨立
    的驗證關卡（gate）核對；未提供規格時只沿用產品的顯式統計政策，不宣稱規格已
    綁定。輸入不符時由逐欄比較拋出 ValueError；通過時不回傳或修改任何產品。
    """

    expected = (
        build_outcome_statistics(payload, minimum_count=outcome.minimum_count),
        build_connectivity_statistics(
            payload, minimum_count=connectivity.minimum_count,
            a_zone_site_ids=connectivity.a_zone_site_ids,
        ),
        build_source_receptor_statistics(
            payload, report_spec=report_spec, quantiles=source_receptor.quantiles,
            minimum_count=source_receptor.minimum_count,
        ),
    )
    for name, actual, reference in zip(
        ("outcome_statistics", "connectivity_statistics", "source_receptor_statistics"),
        (outcome, connectivity, source_receptor), expected, strict=True,
    ):
        _require_same_product(actual, reference, label=name)
    for site_id, product in pathway.items():
        reference = build_pathway_grid_statistics(
            payload.pathway_by_site[site_id],
            valid_member_denominator=payload.event_aggregate.valid_member_denominator_by_site[site_id],
            quantiles=tuple(product.first_passage_quantiles),
            low_sample_min_member_count=product.low_sample_min_member_count,
        )
        _require_same_product(product, reference, label=f"pathway_statistics_by_site[{site_id!r}]")


def _validate_report_policy_products(
    outcome: OutcomeStatistics,
    connectivity: ConnectivityStatistics,
    source_receptor: SourceReceptorStatistics,
    *,
    pathway: Mapping[str, PathwayGridStatistics],
    report_spec: ReportSpec,
) -> None:
    """核對呼叫端已有產品是否服從來源所綁定的固定報告統計政策。

    本介面可接受先前建立的統計產品；同時提供 ``AggregateReleasePayload`` 與
    ``ReportSpec`` 時，除了站點集合，還必須確認低樣本門檻及旅行年齡／路徑分位數
    沒有在介面外被替換。不符時拋出 ValueError，通過時不修改輸入。本函式只核對政策；
    完整原始計數及衍生值另由來源核對函式檢查。KDE 只接受依來源與規格現場產生，
    不在此處接受或替外部密度產品宣告來源。
    """

    minimum_count = report_spec.low_sample_min_member_count
    if (
        outcome.minimum_count != minimum_count
        or connectivity.minimum_count != minimum_count
        or source_receptor.minimum_count != minimum_count
    ):
        raise ValueError(
            "核心 report product 的 minimum_count 必須精確等於 report_spec.low_sample_min_member_count"
        )
    if source_receptor.quantiles != report_spec.travel_age_quantiles:
        raise ValueError(
            "SourceReceptorStatistics quantiles 必須精確等於 report_spec.travel_age_quantiles"
        )

    expected_pathway_quantiles = report_spec.pathway_first_passage_quantiles
    for site_id, product in pathway.items():
        if product.low_sample_min_member_count != minimum_count:
            raise ValueError(
                f"pathway_statistics_by_site[{site_id!r}] 的 minimum_count 與 report_spec 不一致"
            )
        if tuple(product.first_passage_quantiles) != expected_pathway_quantiles:
            raise ValueError(
                f"pathway_statistics_by_site[{site_id!r}] 的 quantiles 與 report_spec 不一致"
            )


@dataclass(frozen=True, slots=True)
class ReportStatistics:
    """封存供繪圖與表格產製端直接讀取的統一報告統計集合。

    核心欄位是不可變的停止、跨站及來源段—受體產品；選用的路徑格網
    ``pathway_statistics_by_site`` 在提供來源聚合資料時，須與來源逐欄一致。
    此時 ``kde_statistics_by_site=None`` 要求依同一 ReportSpec 單次建立核密度估計，
    空對照表表示省略；非空外部密度及材料產品一律拒絕，因本入口無法充分查證其
    衍生來源。沒有來源聚合資料的純產品入口仍可組合這些產品，但 ``run_id`` 必為 None。

    ``products`` 是固定名稱索引，只保存已建立的核心及實際提供的選用產品，不將
    ``None`` 或不可用狀態轉成零值。比例、原始計數、秒／公尺軸與失敗狀態仍由各
    統計產品保存。本類別使用既有建立函式核對內容，不另訂科學分母或估計方法。
    """

    outcome_statistics: OutcomeStatistics
    connectivity_statistics: ConnectivityStatistics
    source_receptor_statistics: SourceReceptorStatistics
    pathway_statistics_by_site: Mapping[str, PathwayGridStatistics] = field(default_factory=dict)
    kde_statistics_by_site: Mapping[str, KDESensitivityProduct] | None = field(default_factory=dict)
    material_statistics: MaterialStatisticsProduct | None = None
    aggregate_payload: AggregateReleasePayload | None = None
    report_spec: ReportSpec | None = None
    _products: Mapping[str, object] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """驗證完整來源、政策與站點範圍後建立唯讀索引；錯配資料不取得執行批次身分。"""

        outcome, connectivity, source_receptor = _require_core_products(
            self.outcome_statistics,
            self.connectivity_statistics,
            self.source_receptor_statistics,
        )
        payload = self.aggregate_payload
        if payload is not None and type(payload) is not AggregateReleasePayload:
            raise TypeError("aggregate_payload 必須是 exact AggregateReleasePayload 或 None")
        report_spec = self.report_spec
        if report_spec is not None and type(report_spec) is not ReportSpec:
            raise TypeError("report_spec 必須是 exact ReportSpec 或 None")
        if report_spec is not None and payload is None:
            raise ValueError("report_spec 必須與 AggregateReleasePayload 一起提供")
        if payload is not None and report_spec is not None:
            validate_report_spec_against_aggregate_spec(report_spec, payload.aggregate_spec)

        core_site_ids = tuple(outcome.site_ids)
        if connectivity.site_ids != core_site_ids:
            raise ValueError("OutcomeStatistics 與 ConnectivityStatistics 的 site axis 必須一致")
        if set(source_receptor.site_ids) != set(core_site_ids):
            raise ValueError(
                "SourceReceptorStatistics 的 site set 必須與停止／連通產品一致"
            )
        pathway = _snapshot_product_mapping(
            self.pathway_statistics_by_site,
            product_type=PathwayGridStatistics,
            label="pathway_statistics_by_site",
            expected_site_ids=(
                core_site_ids
                if self.pathway_statistics_by_site
                else None
            ),
        )
        kde = _snapshot_product_mapping(
            self.kde_statistics_by_site,
            product_type=KDESensitivityProduct,
            label="kde_statistics_by_site",
            expected_site_ids=(
                core_site_ids
                if self.kde_statistics_by_site
                else None
            ),
        )
        material = self.material_statistics
        if material is not None and type(material) is not MaterialStatisticsProduct:
            raise TypeError("material_statistics 必須是 exact MaterialStatisticsProduct 或 None")
        if payload is not None:
            if material is not None:
                raise ValueError("payload-bound material_statistics 無法驗證來源；請使用純 products 入口")
            if kde:
                raise ValueError("payload-bound KDE 不接受外部覆寫；請使用 None 由來源建立或純 products 入口")
        if report_spec is not None and payload is not None:
            _validate_report_policy_products(
                outcome,
                connectivity,
                source_receptor,
                pathway=pathway,
                report_spec=report_spec,
            )
        if payload is not None:
            _validate_payload_products(
                payload, outcome, connectivity, source_receptor, pathway, report_spec,
            )
            if self.kde_statistics_by_site is None:
                if report_spec is None:
                    raise ValueError("由 payload 建立 KDE 必須提供 report_spec")
                # 核心／路徑來源通過後才計算核密度，每站只建立一次；None 是要求計算，
                # 不是呼叫端提供的驗證旗標。外部密度即使原始總計數相同也不能注入。
                kde = MappingProxyType({
                    site_id: build_kde_sensitivity_product(
                        payload.event_aggregate.site_grid_counts[site_id].local_first_exit_count,
                        site_id=site_id,
                        aggregate_spec=payload.aggregate_spec,
                        report_spec=report_spec,
                    )
                    for site_id in core_site_ids
                })

        products: dict[str, object] = {
            "outcome": outcome,
            "connectivity": connectivity,
            "source_receptor": source_receptor,
        }
        if pathway:
            products["pathway_by_site"] = pathway
        if kde:
            products["kde_by_site"] = kde
        if material is not None:
            products["material"] = material

        object.__setattr__(self, "outcome_statistics", outcome)
        object.__setattr__(self, "connectivity_statistics", connectivity)
        object.__setattr__(self, "source_receptor_statistics", source_receptor)
        object.__setattr__(self, "pathway_statistics_by_site", pathway)
        object.__setattr__(self, "kde_statistics_by_site", kde)
        object.__setattr__(self, "material_statistics", material)
        object.__setattr__(self, "report_spec", report_spec)
        object.__setattr__(self, "_products", MappingProxyType(products))

    @property
    def outcome(self) -> OutcomeStatistics:
        """停止結果產品的簡短 renderer-facing 別名。"""

        return self.outcome_statistics

    @property
    def connectivity(self) -> ConnectivityStatistics:
        """方向性跨站連通產品的簡短 renderer-facing 別名。"""

        return self.connectivity_statistics

    @property
    def source_receptor(self) -> SourceReceptorStatistics:
        """來源段—受體產品的簡短 renderer-facing 別名。"""

        return self.source_receptor_statistics

    @property
    def pathway_by_site(self) -> Mapping[str, PathwayGridStatistics]:
        """``pathway_statistics_by_site`` 的簡短別名。"""

        return self.pathway_statistics_by_site

    @property
    def kde_by_site(self) -> Mapping[str, KDESensitivityProduct]:
        """``kde_statistics_by_site`` 的簡短別名。"""

        return self.kde_statistics_by_site

    @property
    def products(self) -> Mapping[str, object]:
        """回傳固定核心名稱與 optional products 的唯讀 mapping。"""

        return self._products

    @property
    def run_id(self) -> str | None:
        """回傳已核對來源的執行批次識別碼；沒有來源的純產品入口回傳 None。"""

        return None if self.aggregate_payload is None else self.aggregate_payload.run_id


# 讓呼叫端可用 facade 語意名稱引用同一個 exact immutable class。
ReportStatisticsFacade = ReportStatistics


def _resolve_first_inputs(
    first: object | None,
    *,
    payload: AggregateReleasePayload | None,
    outcome_statistics: OutcomeStatistics | None,
) -> tuple[AggregateReleasePayload | None, object | None, object | None]:
    """解析 payload／outcome positional 與 keyword 來源，拒絕歧義重複輸入。"""

    if payload is not None:
        if type(payload) is not AggregateReleasePayload:
            raise TypeError("payload 必須是 exact AggregateReleasePayload")
        if first is not None or outcome_statistics is not None:
            raise ValueError("payload 不可與 positional／keyword outcome data 同時提供")
        return payload, None, None
    if type(first) is AggregateReleasePayload:
        if outcome_statistics is not None:
            raise ValueError("AggregateReleasePayload 不可與 outcome_statistics 同時提供")
        return first, None, None
    if first is not None and outcome_statistics is not None:
        raise ValueError("positional outcome 與 outcome_statistics 不可同時提供")
    return None, first, outcome_statistics


def _build_payload_products(
    payload: AggregateReleasePayload,
    *,
    report_spec: ReportSpec | None,
    outcome_statistics: object | None,
    connectivity_statistics: object | None,
    source_receptor_statistics: object | None,
    pathway_statistics_by_site: Mapping[str, PathwayGridStatistics] | None,
    kde_statistics_by_site: Mapping[str, KDESensitivityProduct] | None,
    material_statistics: MaterialStatisticsProduct | None,
) -> ReportStatistics:
    """從已驗證來源及報告規格組合產品，交由建構子核對完整來源。

    缺少的核心與路徑統計由既有方法產生；既有產品則保留給來源核對。核密度估計
    延至建構子通過來源檢查後單次建立，避免為驗證來源重算昂貴平滑。回傳具來源身分
    的 ReportStatistics；缺規格、來源不符或含不可驗證的外部覆寫時拒絕建立。
    """

    if report_spec is None and any(
        product is None
        for product in (
            outcome_statistics,
            connectivity_statistics,
            source_receptor_statistics,
        )
    ):
        raise ValueError(
            "由 AggregateReleasePayload 建立缺少的 report product 時必須提供 exact report_spec"
        )
    if report_spec is not None:
        if type(report_spec) is not ReportSpec:
            raise TypeError("report_spec 必須是 exact ReportSpec")
        validate_report_spec_against_aggregate_spec(report_spec, payload.aggregate_spec)
        minimum_count = report_spec.low_sample_min_member_count
    else:
        minimum_count = 1

    event = payload.event_aggregate
    built_outcome = (
        build_outcome_statistics(payload, minimum_count=minimum_count)
        if outcome_statistics is None
        else outcome_statistics
    )
    built_connectivity = (
        build_connectivity_statistics(payload, minimum_count=minimum_count)
        if connectivity_statistics is None
        else connectivity_statistics
    )
    built_source_receptor = (
        build_source_receptor_statistics(payload, report_spec=report_spec)
        if source_receptor_statistics is None
        else source_receptor_statistics
    )

    core_outcome, core_connectivity, core_source = _require_core_products(
        built_outcome,
        built_connectivity,
        built_source_receptor,
    )
    site_ids = tuple(core_outcome.site_ids)

    if pathway_statistics_by_site is None:
        if report_spec is None:
            pathway = {}
        else:
            pathway = {
                site_id: build_pathway_grid_statistics(
                    payload.pathway_by_site[site_id],
                    valid_member_denominator=event.valid_member_denominator_by_site[site_id],
                    quantiles=report_spec.pathway_first_passage_quantiles,
                    low_sample_min_member_count=report_spec.low_sample_min_member_count,
                )
                for site_id in site_ids
            }
    else:
        pathway = pathway_statistics_by_site

    # 將自動核密度估計交由建構子在完整來源核對後執行，避免建立一次後又為
    # 驗證來源重算昂貴平滑。空對照表仍是呼叫端明示省略密度產品的合法用法。
    kde = {} if kde_statistics_by_site is None and report_spec is None else kde_statistics_by_site

    return ReportStatistics(
        outcome_statistics=core_outcome,
        connectivity_statistics=core_connectivity,
        source_receptor_statistics=core_source,
        pathway_statistics_by_site=pathway,
        kde_statistics_by_site=kde,
        material_statistics=material_statistics,
        aggregate_payload=payload,
        report_spec=report_spec,
    )


def build_report_statistics(
    aggregate_payload: AggregateReleasePayload | OutcomeStatistics | None = None,
    connectivity_statistics: ConnectivityStatistics | None = None,
    source_receptor_statistics: SourceReceptorStatistics | None = None,
    *,
    payload: AggregateReleasePayload | None = None,
    report_spec: ReportSpec | None = None,
    outcome_statistics: OutcomeStatistics | None = None,
    pathway_statistics_by_site: Mapping[str, PathwayGridStatistics] | None = None,
    pathway_by_site: Mapping[str, PathwayGridStatistics] | None = None,
    kde_statistics_by_site: Mapping[str, KDESensitivityProduct] | None = None,
    kde_by_site: Mapping[str, KDESensitivityProduct] | None = None,
    material_statistics: MaterialStatisticsProduct | None = None,
) -> ReportStatistics:
    """建立報告統計的統一讀取介面，或組合已有的三項核心統計產品。

    Args:
        aggregate_payload: 來源聚合資料，須為 AggregateReleasePayload 本身的型別。
            函式依 report_spec 建立缺少的停止、跨站、來源段—受體、路徑及核密度產品；
            也可傳 OutcomeStatistics 作為無來源身分之純產品入口的第一個位置參數。
        connectivity_statistics: 既有跨站產品，須為 ConnectivityStatistics 本身的型別。
        source_receptor_statistics: 既有來源段—受體產品，須為 SourceReceptorStatistics。
        payload: aggregate_payload 的關鍵字別名，不可和其他來源或停止產品參數混用。
        report_spec: 報告規格，負責綁定聚合規格、分位數、低樣本門檻及核密度政策；
            只組合已建產品時可省略，但不再宣稱特定報告規格已綁定。
        outcome_statistics: 純產品入口以關鍵字指定停止產品的形式。
        pathway_statistics_by_site: 提供來源時逐欄核對路徑產品，純產品入口保留原物件。
        kde_statistics_by_site: 提供來源時只允許省略（由來源單次產生）或空對照表
            （不產生核密度）；非空外部覆寫只供無執行批次身分的純產品入口使用。
        pathway_by_site／kde_by_site: 上述兩個對照表的簡短別名，不可與長名稱同時提供。
        material_statistics: 材料產品，須為 MaterialStatisticsProduct；只供純產品入口
            使用，因聚合資料未保存足以驗證逐粒子材料統計的來源。

    Returns:
        不可變的 ReportStatistics；products 提供固定核心名稱及實際存在的選用產品。
        原始計數、分母、狀態與秒／公尺軸均保留在子產品，不將缺值或不可用比例補零。

    Raises:
        TypeError／ValueError: 來源的原始計數、分母、軸、衍生產品或規格政策不一致，
            帶來源入口混入不可驗證的核密度／材料覆寫，或型別、參數別名不合法時。
    """

    if pathway_statistics_by_site is not None and pathway_by_site is not None:
        raise ValueError("pathway_statistics_by_site 與 pathway_by_site 不可同時提供")
    if kde_statistics_by_site is not None and kde_by_site is not None:
        raise ValueError("kde_statistics_by_site 與 kde_by_site 不可同時提供")
    resolved_pathway = (
        pathway_statistics_by_site
        if pathway_statistics_by_site is not None
        else pathway_by_site
    )
    resolved_kde = kde_statistics_by_site if kde_statistics_by_site is not None else kde_by_site

    resolved_payload, positional_outcome, keyword_outcome = _resolve_first_inputs(
        aggregate_payload,
        payload=payload,
        outcome_statistics=outcome_statistics,
    )
    resolved_outcome = positional_outcome if positional_outcome is not None else keyword_outcome
    if resolved_payload is not None:
        return _build_payload_products(
            resolved_payload,
            report_spec=report_spec,
            outcome_statistics=resolved_outcome,
            connectivity_statistics=connectivity_statistics,
            source_receptor_statistics=source_receptor_statistics,
            pathway_statistics_by_site=resolved_pathway,
            kde_statistics_by_site=resolved_kde,
            material_statistics=material_statistics,
        )

    if report_spec is not None:
        raise ValueError("純 products facade 沒有 AggregateReleasePayload 可供 report_spec binding")
    if resolved_outcome is None:
        raise TypeError("純 products facade 必須提供 outcome_statistics")
    outcome, connectivity, source_receptor = _require_core_products(
        resolved_outcome,
        connectivity_statistics,
        source_receptor_statistics,
    )
    pathway = {} if resolved_pathway is None else resolved_pathway
    kde = {} if resolved_kde is None else resolved_kde
    return ReportStatistics(
        outcome_statistics=outcome,
        connectivity_statistics=connectivity,
        source_receptor_statistics=source_receptor,
        pathway_statistics_by_site=pathway,
        kde_statistics_by_site=kde,
        material_statistics=material_statistics,
    )


# 統一 facade 的常見產品／builder 名稱別名；它們都指向同一份 immutable 實作，避免
# renderer 或上層 pipeline 因名稱差異複製產品或另訂一套資料契約。
ReportStatisticsProduct = ReportStatistics
ReportStatisticsResult = ReportStatistics
build_report_products = build_report_statistics
