"""彙整分析規格的嚴格資料模型與 JSON 載入器。

本模組描述的是已驗收資料產品進行空間彙整、核密度估計與條件式來源
足跡分析時所需的固定設定。所有距離、座標與帶寬均以公尺（m）表示；
`x`、`y` 是由每站明示的固定本地區域方位等距投影（azimuthal equidistant,
AEQD）產生的公尺制運算座標，而非把經緯度直接當成平面距離。travel age
軸以秒（s）表示，且由
`age_bin_edges_seconds` 固定 pathway 與事件直方圖的共同分箱。載入器只
接受 UTF-8 編碼的 JSON 普通檔案，並在讀取原始位元組後計算來源雜湊，避免
格式化差異掩蓋輸入檔的實際變更。

這裡保存的結果只能描述「條件式來源足跡」或「相對來源權重」。在尚未
建立先驗、似然與觀測驗證前，這些欄位不得被解釋為絕對來源機率或因果
歸因；本模組只負責規格的結構、數值限制與可重現雜湊，不宣稱任何科學
推論本身已經完成。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn
from uuid import uuid4

from shapely.geometry.base import BaseGeometry

from .boundaries import BoundaryGeometry
from .manifests import BoundaryGeometryBundle

__all__ = [
    "AGGREGATE_SPEC_SCHEMA_VERSION",
    "AggregateSpec",
    "SiteBoundarySegments",
    "SiteGridSpec",
    "SiteMetricCRSSpec",
    "load_aggregate_spec",
    "validate_aggregate_spec_against_boundaries",
    "write_aggregate_spec_from_boundaries",
]


# 公開版本常數是輸入 JSON 必須完全相等的 schema 版本；版本不相容時不
# 允許載入器以寬鬆方式猜測欄位，避免不同研究批次使用不同語意的設定。
AGGREGATE_SPEC_SCHEMA_VERSION = "1.0.0"

# JSON 檔案的根層只允許這些設定欄位。兩個 SHA-256 欄位是載入後由原始
# 位元組與正規化內容推導的 provenance metadata，因此不屬於輸入根層。
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "grid_cell_size_m",
        "site_grids",
        "site_metric_crs",
        "boundary_bin_size_m",
        "boundary_segment_lengths_m",
        "site_boundary_segment_ids",
        "kde_bandwidths_m",
        "hdr_levels",
        "age_bin_edges_seconds",
        "bootstrap_replicates",
        "bootstrap_confidence_level",
        "bootstrap_seed",
        "denominator_policy",
    }
)

# 每個站點的網格範圍必須恰好由四個邊界欄位描述；額外欄位可能讓同一
# 份設定在不同程式中產生不同的網格，因此不予忽略。
_SITE_GRID_KEYS = frozenset(
    {"x_min_m", "x_max_m", "y_min_m", "y_max_m"}
)

# 每站的公尺制計算座標必須完整保存投影方法、中心與軸序；不接受只寫
# 座標參考系統識別碼（例如 EPSG code）或省略中心的簡寫，因為相同經緯度在
# 不同 local projection 下會
# 對應到不同的 x/y 數值，無法重建每一條 flow 的 metric geometry。
_SITE_METRIC_CRS_KEYS = frozenset(
    {
        "projection_method",
        "center_lon_deg",
        "center_lat_deg",
        "linear_unit",
        "axis_order",
    }
)

# 每個站點的邊界分類只允許本地邊界與外圍邊界兩組。欄位採 exact-key
# policy，避免拼字錯誤或尚未定義語意的分類被下游流程靜默忽略。
_SITE_BOUNDARY_SEGMENT_KEYS = frozenset({"local", "outer"})

# 高密度區域（high-density region, HDR）層級是固定的展示與彙整契約，
# 不接受任意自訂層級，以確保不同 run 的區域權重具有可比性。
_HDR_LEVELS = (0.5, 0.75, 0.9)

# 此政策名稱同時是資料缺口與數值失敗的分母處理契約；它不是可自由延伸
# 的描述文字，因為下游統計結果必須能由名稱唯一決定。
_DENOMINATOR_POLICY = "exclude_data_gap_and_numerical_failure_v1"

# 網格寬高與網格尺寸的商必須接近正整數。相對與絕對容差都固定為規格
# 要求的 1e-9，避免呼叫端使用不同容差造成同一設定可載入性不一致。
_GRID_ALIGNMENT_REL_TOL = 1e-9
_GRID_ALIGNMENT_ABS_TOL = 1e-9

# seed 是無號 128 位元整數；上限寫成常數可讓邊界條件清楚且避免散落
# 位元運算在驗證流程中。
_MAX_BOOTSTRAP_SEED = 2**128 - 1

# run、站點與邊界段名稱共用同一個 ASCII slug 契約，確保它們既能作為
# 穩定識別碼，也不會被誤當成含有目錄分隔符的檔案路徑。
_SLUG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class _AggregateSpecError(ValueError):
    """內部驗證例外；對外仍統一呈現為不含檔案路徑的 ValueError。"""


def _fail(message: str) -> NoReturn:
    """以不攜帶輸入檔路徑的訊息中止規格驗證。"""

    raise _AggregateSpecError(message)


def _require_finite_number(value: object, *, label: str) -> int | float:
    """驗證只允許原生 int/float 且排除浮點非有限值的數字。

    Python 的 bool 是 int 的子類別，但在資料契約中布林值不能冒充數值，
    所以此處刻意檢查精確型別。任意大小的原生整數在數學上仍為有限值；
    浮點數則必須通過 `math.isfinite`，以攔截 NaN、正負無限大及 JSON
    極大指數轉出的非有限結果。
    """

    value_type = type(value)
    if value_type is int:
        return value
    if value_type is float and math.isfinite(value):
        return value
    _fail(f"{label} 必須是有限的 int 或 float。")


def _require_positive_number(value: object, *, label: str) -> int | float:
    """驗證有限數字且大於零，供所有長度與帶寬欄位共用。"""

    number = _require_finite_number(value, label=label)
    if number <= 0:
        _fail(f"{label} 必須大於零。")
    return number


def _require_slug(value: object, *, label: str) -> str:
    """驗證不含路徑語意的 ASCII slug 識別碼。

    識別碼必須以英文字母或阿拉伯數字起首，後續只能使用英文字母、
    阿拉伯數字、句點、底線或連字號，長度介於 1 到 128 個字元；明確
    排除 `.`、`..` 與 slash，避免識別碼在檔案或報表路由中形成路徑穿越
    語意。此限制也適用於 JSON mapping 的鍵。
    """

    if type(value) is not str:
        _fail(f"{label} 必須是 slug 字串。")
    if value in {".", ".."} or not _SLUG_PATTERN.fullmatch(value):
        _fail(f"{label} 不符合 slug 格式。")
    return value


def _copy_mapping(value: object, *, label: str) -> dict[Any, Any]:
    """將 mapping 複製成新的普通 dict，隔離呼叫端後續的可變狀態。"""

    if not isinstance(value, Mapping):
        _fail(f"{label} 必須是 object mapping。")
    try:
        return dict(value)
    except Exception:
        # 自訂 Mapping 可能在迭代時拋出任意例外；對外不暴露實作細節。
        _fail(f"{label} 無法複製。")


def _copy_tuple(value: object, *, label: str) -> tuple[Any, ...]:
    """把陣列型輸入固定為 tuple，並拒絕會產生模糊語意的字串容器。"""

    if isinstance(value, (str, bytes, bytearray)):
        _fail(f"{label} 必須是陣列。")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except Exception:
        _fail(f"{label} 必須是陣列。")


def _require_sha256(value: object, *, label: str) -> str:
    """驗證由 hashlib 產生的 64 碼小寫 SHA-256 十六進位字串。"""

    if type(value) is not str or not _SHA256_PATTERN.fullmatch(value):
        _fail(f"{label} 必須是小寫 SHA-256 十六進位字串。")
    return value


def _require_grid_alignment(
    grid: SiteGridSpec,
    grid_cell_size_m: int | float,
) -> None:
    """驗證站點範圍可由整數個公尺制網格單元完整表示。

    寬度與高度分別由 `x_max_m - x_min_m` 及 `y_max_m - y_min_m` 定義，
    不進行任何座標平移或四捨五入。一般浮點輸入使用固定的相對與絕對
    容差。整數與浮點數都依相同的比例與近似判定處理；若極大值無法換算
    成有限比例則採封閉式失敗，不接受無法可靠驗證的網格。這個檢查只
    保證幾何網格一致，不代表資料點皆位於海域或任何物理邊界內。
    """

    for lower, upper, axis_label in (
        (grid.x_min_m, grid.x_max_m, "x"),
        (grid.y_min_m, grid.y_max_m, "y"),
    ):
        width = upper - lower
        if type(width) is float and not math.isfinite(width):
            _fail(f"站點 {axis_label} 軸寬度必須是有限值。")
        try:
            ratio = width / grid_cell_size_m
        except (OverflowError, ZeroDivisionError, TypeError):
            _fail(f"站點 {axis_label} 軸範圍無法換算為網格單元數。")
        if type(ratio) is not float or not math.isfinite(ratio):
            _fail(f"站點 {axis_label} 軸網格單元數必須是有限值。")

        nearest_integer = round(ratio)
        if nearest_integer <= 0 or not math.isclose(
            ratio,
            nearest_integer,
            rel_tol=_GRID_ALIGNMENT_REL_TOL,
            abs_tol=_GRID_ALIGNMENT_ABS_TOL,
        ):
            _fail(f"站點 {axis_label} 軸範圍不是接近正整數個網格單元。")


@dataclass(frozen=True, slots=True)
class SiteGridSpec:
    """單一站點的公尺制矩形網格範圍。

    `x_min_m`、`x_max_m`、`y_min_m` 與 `y_max_m` 都是有限的原生 int 或
    float，單位為公尺；每一軸都要求最大值嚴格大於最小值。站點範圍是否
    能被整數個 `grid_cell_size_m` 覆蓋，會在外層 `AggregateSpec` 驗證，
    因為該條件需要同時知道整份彙整規格的網格尺寸。
    """

    x_min_m: int | float
    x_max_m: int | float
    y_min_m: int | float
    y_max_m: int | float

    def __post_init__(self) -> None:
        """固定座標邊界的型別、有限性與嚴格矩形條件。"""

        _require_finite_number(self.x_min_m, label="x_min_m")
        _require_finite_number(self.x_max_m, label="x_max_m")
        _require_finite_number(self.y_min_m, label="y_min_m")
        _require_finite_number(self.y_max_m, label="y_max_m")
        if self.x_max_m <= self.x_min_m:
            _fail("x_max_m 必須大於 x_min_m。")
        if self.y_max_m <= self.y_min_m:
            _fail("y_max_m 必須大於 y_min_m。")


@dataclass(frozen=True, slots=True)
class SiteBoundarySegments:
    """單一站點引用的本地與外圍邊界段識別碼。

    `local_segment_ids` 表示與站點局部空間範圍直接相關的邊界段，
    `outer_segment_ids` 表示外圍比較或來源足跡彙整所使用的邊界段。兩組
    都必須是非空、安全 slug tuple，且各組內不得重複；同一識別碼可同時
    出現在 local 與 outer，因為分類角色可能隨分析用途重疊。建構時會
    防禦性複製為 tuple，避免呼叫端修改原始 list 後改變條件式來源足跡
    的邊界分組。識別碼是否存在於全域邊界長度表，由 `AggregateSpec`
    進一步驗證。
    """

    local_segment_ids: tuple[str, ...]
    outer_segment_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """固定兩組識別碼並驗證非空、slug 安全性與組內唯一性。"""

        local_segment_ids = _copy_tuple(
            self.local_segment_ids,
            label="local_segment_ids",
        )
        outer_segment_ids = _copy_tuple(
            self.outer_segment_ids,
            label="outer_segment_ids",
        )

        for segment_ids, label in (
            (local_segment_ids, "local_segment_ids"),
            (outer_segment_ids, "outer_segment_ids"),
        ):
            if not segment_ids:
                _fail(f"{label} 不得為空。")
            for segment_id in segment_ids:
                _require_slug(segment_id, label=label)
            if len(set(segment_ids)) != len(segment_ids):
                _fail(f"{label} 不得包含重複識別碼。")

        object.__setattr__(self, "local_segment_ids", local_segment_ids)
        object.__setattr__(self, "outer_segment_ids", outer_segment_ids)

    @property
    def local(self) -> tuple[str, ...]:
        """回傳本地邊界段的唯讀 tuple，供 JSON `local` 語意直接取用。"""

        return self.local_segment_ids

    @property
    def outer(self) -> tuple[str, ...]:
        """回傳外圍邊界段的唯讀 tuple，供 JSON `outer` 語意直接取用。"""

        return self.outer_segment_ids


@dataclass(frozen=True, slots=True)
class SiteMetricCRSSpec:
    """單一站點固定的公尺制區域方位等距投影（AEQD）描述。

    這組欄位對應 ``geometry.DomainProjection`` 的 per-flow fixed local
    projection：``center_lon_deg`` 與 ``center_lat_deg`` 是投影中心的經緯度，
    單位分別為度；投影後的 x 軸必須代表向東的公尺距離，y 軸必須代表向北的
    公尺距離。每個 site 都必須明示一份設定，即使 A 區兩站刻意共用同一個
    投影中心也不能由下游程式猜測或從其他站複製。

    ``projection_method``、``linear_unit`` 與 ``axis_order`` 是固定字串，
    不是可任意延伸的標籤。此資料模型只保存可重建 geometry 的必要契約，
    不保證輸入邊界本身無自交、點位都在海域內，也不把經緯度當作平面距離
    直接運算；所有 metric distance 必須使用這裡明示的投影結果。
    """

    projection_method: str
    center_lon_deg: int | float
    center_lat_deg: int | float
    linear_unit: str
    axis_order: str

    def __post_init__(self) -> None:
        """驗證投影方法、中心經緯度範圍、長度單位與 x/y 軸序契約。"""

        if (
            type(self.projection_method) is not str
            or self.projection_method != "azimuthal_equidistant_wgs84"
        ):
            _fail(
                "projection_method 必須精確為 "
                "azimuthal_equidistant_wgs84。"
            )

        center_lon_deg = _require_finite_number(
            self.center_lon_deg,
            label="center_lon_deg",
        )
        if not -180 <= center_lon_deg <= 180:
            _fail("center_lon_deg 必須介於 -180 與 180 度之間（含邊界）。")

        center_lat_deg = _require_finite_number(
            self.center_lat_deg,
            label="center_lat_deg",
        )
        if not -90 < center_lat_deg < 90:
            _fail("center_lat_deg 必須嚴格介於 -90 與 90 度之間。")

        if type(self.linear_unit) is not str or self.linear_unit != "m":
            _fail("linear_unit 必須精確為 m。")
        if (
            type(self.axis_order) is not str
            or self.axis_order != "x_east_y_north"
        ):
            _fail("axis_order 必須精確為 x_east_y_north。")


@dataclass(frozen=True, slots=True)
class AggregateSpec:
    """完整的彙整、核密度估計（kernel density estimation, KDE）、
    高密度區域（high-density region, HDR）與自助法（bootstrap）設定及
    其 provenance 雜湊。

    所有長度、邊界與核密度估計帶寬均以公尺（m）表示；`site_grids` 是
    站點 slug 到矩形範圍的 mapping，`site_metric_crs` 是站點 slug 到
    固定 local AEQD 公尺制投影的 mapping，`boundary_segment_lengths_m`
    是邊界段 slug 到其公尺長度的 mapping，`site_boundary_segment_ids` 則
    保存每站本地與外圍邊界段的引用。四個 mapping 會在建構後複製並包裝成
    `MappingProxyType`，帶寬、HDR 層級、年齡邊界與站點邊界引用則固定為
    tuple，避免呼叫端在規格建立後悄悄改變統計分母、空間離散化、投影基準
    或邊界分類。

    站點邊界 mapping 的 site key 必須與 `site_grids` 完全一致；每個引用
    都必須存在於 `boundary_segment_lengths_m`，而長度表中的每個邊界段
    至少要被任一站點的 local 或 outer 組引用一次。同一邊界段可以跨站
    共用，也可以在同站的 local 與 outer 同時出現，這只表示彙整角色重疊，
    不代表重複計算或絕對來源機率。

    `age_bin_edges_seconds` 是 pathway 與事件 travel-age histogram 共用的
    秒數邊界；至少要有兩點、從精確的 0 s 開始並嚴格遞增。第 i 個箱採用
    `[edge_i, edge_{i+1})`，但最後一箱採用
    `[edge_{n-2}, edge_{n-1}]`，因此恰好等於最後邊界的 travel age 不會
    被排除。這只是分箱索引的重建契約，不會替資料缺口、域外或數值失敗
    狀態補值。

    `hdr_levels` 固定為 0.5、0.75、0.9；bootstrap replicate 數必須是
    正整數，信賴水準介於 0 與 1 之間，seed 是 0 到 2^128-1 的整數。
    `denominator_policy` 固定為排除資料缺口與數值失敗的版本化政策。這些
    設定只支援「條件式來源足跡」或「相對來源權重」的可重現彙整，不把
    結果提升為未經先驗、似然與觀測驗證的絕對來源機率或因果歸因。

    `source_sha256` 是輸入 JSON 原始 bytes 的 SHA-256；`canonical_sha256`
    是移除這兩個雜湊欄位後，對 `to_dict()` 產生固定分隔符與排序鍵的
    UTF-8 JSON 所計算的 SHA-256。輸入 JSON 本身不允許攜帶這兩個欄位。
    """

    schema_version: str
    run_id: str
    grid_cell_size_m: int | float
    site_grids: Mapping[str, SiteGridSpec]
    site_metric_crs: Mapping[str, SiteMetricCRSSpec]
    boundary_bin_size_m: int | float
    boundary_segment_lengths_m: Mapping[str, int | float]
    site_boundary_segment_ids: Mapping[str, SiteBoundarySegments]
    kde_bandwidths_m: tuple[int | float, ...]
    hdr_levels: tuple[int | float, ...]
    age_bin_edges_seconds: tuple[int | float, ...]
    bootstrap_replicates: int
    bootstrap_confidence_level: int | float
    bootstrap_seed: int
    denominator_policy: str
    source_sha256: str
    canonical_sha256: str

    def __post_init__(self) -> None:
        """驗證完整資料契約並建立不可變的 mapping 與序列快照。"""

        if type(self.schema_version) is not str:
            _fail("schema_version 必須是字串。")
        if self.schema_version != AGGREGATE_SPEC_SCHEMA_VERSION:
            _fail("schema_version 不受支援。")
        _require_slug(self.run_id, label="run_id")

        grid_cell_size_m = _require_positive_number(
            self.grid_cell_size_m,
            label="grid_cell_size_m",
        )

        # 先複製再驗證與包裝；如此即使呼叫端保留原始 dict，也不能在
        # 建構期間或建構後透過該 dict 改寫規格內容。
        site_grids = _copy_mapping(self.site_grids, label="site_grids")
        if not site_grids:
            _fail("site_grids 不得為空。")
        for site_name, grid in site_grids.items():
            _require_slug(site_name, label="site_grids key")
            if not isinstance(grid, SiteGridSpec):
                _fail("site_grids value 必須是 SiteGridSpec。")
            _require_grid_alignment(grid, grid_cell_size_m)
        object.__setattr__(self, "site_grids", MappingProxyType(site_grids))

        # 投影中心是每站重建 x/y metric geometry 的必要輸入；先複製 mapping
        # 再檢查 site set 與每個 immutable value，避免呼叫端保留的 dict 在
        # 建構後改變某站的投影基準，進而使同一批軌跡無法重現。
        site_metric_crs = _copy_mapping(
            self.site_metric_crs,
            label="site_metric_crs",
        )
        if set(site_metric_crs) != set(site_grids):
            _fail("site_metric_crs keys 必須與 site_grids 完全一致。")
        for site_name, metric_crs in site_metric_crs.items():
            _require_slug(site_name, label="site_metric_crs key")
            if not isinstance(metric_crs, SiteMetricCRSSpec):
                _fail("site_metric_crs value 必須是 SiteMetricCRSSpec。")
        object.__setattr__(
            self,
            "site_metric_crs",
            MappingProxyType(site_metric_crs),
        )

        _require_positive_number(
            self.boundary_bin_size_m,
            label="boundary_bin_size_m",
        )
        boundary_segments = _copy_mapping(
            self.boundary_segment_lengths_m,
            label="boundary_segment_lengths_m",
        )
        if not boundary_segments:
            _fail("boundary_segment_lengths_m 不得為空。")
        for segment_name, segment_length in boundary_segments.items():
            _require_slug(segment_name, label="boundary segment key")
            _require_positive_number(
                segment_length,
                label="boundary segment length",
            )
        object.__setattr__(
            self,
            "boundary_segment_lengths_m",
            MappingProxyType(boundary_segments),
        )

        # 站點集合、引用存在性與全域覆蓋率必須同時成立，才能避免某站缺少
        # 邊界分類或長度表含有永遠不會進入條件式來源足跡的孤立邊界段。
        site_boundary_segments = _copy_mapping(
            self.site_boundary_segment_ids,
            label="site_boundary_segment_ids",
        )
        if set(site_boundary_segments) != set(site_grids):
            _fail("site_boundary_segment_ids keys 必須與 site_grids 完全一致。")

        referenced_segment_ids: set[str] = set()
        for site_name, segment_groups in site_boundary_segments.items():
            _require_slug(site_name, label="site_boundary_segment_ids key")
            if not isinstance(segment_groups, SiteBoundarySegments):
                _fail(
                    "site_boundary_segment_ids value 必須是 "
                    "SiteBoundarySegments。"
                )
            for segment_id in (
                *segment_groups.local_segment_ids,
                *segment_groups.outer_segment_ids,
            ):
                if segment_id not in boundary_segments:
                    _fail("站點引用的 boundary segment 不存在於長度表。")
                referenced_segment_ids.add(segment_id)

        if referenced_segment_ids != set(boundary_segments):
            _fail("每個 boundary segment 都必須至少被一個站點分類引用。")
        object.__setattr__(
            self,
            "site_boundary_segment_ids",
            MappingProxyType(site_boundary_segments),
        )

        kde_bandwidths_m = _copy_tuple(
            self.kde_bandwidths_m,
            label="kde_bandwidths_m",
        )
        if len(kde_bandwidths_m) != 3:
            _fail("kde_bandwidths_m 必須恰有三個值。")
        for bandwidth in kde_bandwidths_m:
            _require_positive_number(bandwidth, label="kde bandwidth")
        if not (
            kde_bandwidths_m[0]
            < kde_bandwidths_m[1]
            < kde_bandwidths_m[2]
        ):
            _fail("kde_bandwidths_m 必須嚴格遞增。")
        object.__setattr__(self, "kde_bandwidths_m", kde_bandwidths_m)

        hdr_levels = _copy_tuple(self.hdr_levels, label="hdr_levels")
        if any(
            type(level) not in {int, float}
            or (type(level) is float and not math.isfinite(level))
            for level in hdr_levels
        ):
            _fail("hdr_levels 必須只含有限的 int 或 float。")
        if hdr_levels != _HDR_LEVELS:
            _fail("hdr_levels 必須精確為 [0.5, 0.75, 0.9]。")
        object.__setattr__(self, "hdr_levels", hdr_levels)

        # age edges 是跨 pathway／事件產品共用的時間軸。先逐點驗證原生
        # JSON number 與有限性，再轉成 float；如此下游不會因 int/float
        # 混用而得到不同 dtype，而最後一個 edge 的閉區間語意可固定重建。
        age_bin_edges_seconds = _copy_tuple(
            self.age_bin_edges_seconds,
            label="age_bin_edges_seconds",
        )
        if len(age_bin_edges_seconds) < 2:
            _fail("age_bin_edges_seconds 必須至少包含兩個邊界。")
        normalized_age_edges: list[float] = []
        for edge in age_bin_edges_seconds:
            finite_edge = _require_finite_number(
                edge,
                label="age_bin_edges_seconds edge",
            )
            try:
                normalized_edge = float(finite_edge)
            except (OverflowError, ValueError):
                _fail("age_bin_edges_seconds 必須可正規化為有限 float。")
            if not math.isfinite(normalized_edge):
                _fail("age_bin_edges_seconds 必須可正規化為有限 float。")
            normalized_age_edges.append(normalized_edge)

        if normalized_age_edges[0] != 0.0:
            _fail("age_bin_edges_seconds 的第一個邊界必須精確為 0 秒。")
        if not all(
            left < right
            for left, right in zip(
                normalized_age_edges,
                normalized_age_edges[1:],
                strict=False,
            )
        ):
            _fail("age_bin_edges_seconds 必須嚴格遞增。")
        object.__setattr__(
            self,
            "age_bin_edges_seconds",
            tuple(normalized_age_edges),
        )

        if type(self.bootstrap_replicates) is not int:
            _fail("bootstrap_replicates 必須是正整數。")
        if self.bootstrap_replicates <= 0:
            _fail("bootstrap_replicates 必須是正整數。")
        _require_finite_number(
            self.bootstrap_confidence_level,
            label="bootstrap_confidence_level",
        )
        if not (
            0 < self.bootstrap_confidence_level < 1
        ):
            _fail("bootstrap_confidence_level 必須嚴格介於 0 與 1 之間。")
        if type(self.bootstrap_seed) is not int:
            _fail("bootstrap_seed 必須是 0 到 2^128-1 的整數。")
        if not 0 <= self.bootstrap_seed <= _MAX_BOOTSTRAP_SEED:
            _fail("bootstrap_seed 超出 128 位元無號整數範圍。")

        if type(self.denominator_policy) is not str:
            _fail("denominator_policy 必須是字串。")
        if self.denominator_policy != _DENOMINATOR_POLICY:
            _fail("denominator_policy 不受支援。")

        _require_sha256(self.source_sha256, label="source_sha256")
        _require_sha256(self.canonical_sha256, label="canonical_sha256")

    def to_dict(self) -> dict[str, Any]:
        """回傳可 JSON 序列化的深層資料快照。

        mapping 會轉回新的普通 dict，tuple 會轉回新的 list，確保呼叫端
        修改回傳值不會影響 frozen 物件內部。回傳值包含兩個 provenance
        雜湊欄位；計算 `canonical_sha256` 時必須將 `source_sha256` 與
        `canonical_sha256` 一併排除，因為它們是由同一份設定衍生的結果。
        輸入 JSON 的根層白名單仍只包含設定欄位，不可直接把這個包含雜湊
        metadata 的完整快照當作新的輸入檔。
        """

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "grid_cell_size_m": self.grid_cell_size_m,
            "site_grids": {
                site_name: {
                    "x_min_m": grid.x_min_m,
                    "x_max_m": grid.x_max_m,
                    "y_min_m": grid.y_min_m,
                    "y_max_m": grid.y_max_m,
                }
                for site_name, grid in self.site_grids.items()
            },
            "site_metric_crs": {
                site_name: {
                    "projection_method": metric_crs.projection_method,
                    "center_lon_deg": metric_crs.center_lon_deg,
                    "center_lat_deg": metric_crs.center_lat_deg,
                    "linear_unit": metric_crs.linear_unit,
                    "axis_order": metric_crs.axis_order,
                }
                for site_name, metric_crs in self.site_metric_crs.items()
            },
            "boundary_bin_size_m": self.boundary_bin_size_m,
            "boundary_segment_lengths_m": dict(
                self.boundary_segment_lengths_m
            ),
            "site_boundary_segment_ids": {
                site_name: {
                    "local": list(segment_groups.local_segment_ids),
                    "outer": list(segment_groups.outer_segment_ids),
                }
                for site_name, segment_groups in (
                    self.site_boundary_segment_ids.items()
                )
            },
            "kde_bandwidths_m": list(self.kde_bandwidths_m),
            "hdr_levels": list(self.hdr_levels),
            "age_bin_edges_seconds": list(self.age_bin_edges_seconds),
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_confidence_level": self.bootstrap_confidence_level,
            "bootstrap_seed": self.bootstrap_seed,
            "denominator_policy": self.denominator_policy,
            "source_sha256": self.source_sha256,
            "canonical_sha256": self.canonical_sha256,
        }


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """作為 JSON object_pairs_hook，拒絕每一層 object 的重複鍵。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON object 不得包含 duplicate keys。")
        result[key] = value
    return result


def _reject_json_constant(_token: str) -> NoReturn:
    """拒絕 Python JSON 解碼器額外接受的 NaN/Infinity 常數。"""

    _fail("JSON 不允許 NaN 或 Infinity。")


def _reject_non_finite_values(value: object) -> None:
    """遞迴攔截極大指數解碼後形成的非有限 float。

    `parse_constant` 只能攔截文字形式的 NaN/Infinity；例如 `1e999` 可能
    先被標準解碼器轉成 `inf`，所以仍需在 object 與 array 內逐層檢查。
    """

    if type(value) is float and not math.isfinite(value):
        _fail("JSON 數值必須是有限值。")
    if isinstance(value, dict):
        for nested_value in value.values():
            _reject_non_finite_values(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            _reject_non_finite_values(nested_value)


def _read_regular_file_bytes(path: object) -> bytes:
    """以非 symlink 普通檔案語意讀取原始 bytes。

    先以 `lstat` 檢查路徑本身，再用 `O_NOFOLLOW` 開啟並以 `fstat` 確認
    開啟的檔案仍是 regular file；如此來源雜湊不會悄悄落到符號連結或
    目錄上。所有作業系統例外都轉成不含 path 的 ValueError，避免錯誤
    訊息洩漏使用者輸入的檔案位置。
    """

    file_descriptor: int | None = None
    try:
        filesystem_path = os.fspath(path)
        if not isinstance(filesystem_path, (str, bytes)):
            _fail("path 必須是檔案系統路徑。")

        link_status = os.lstat(filesystem_path)
        if stat.S_ISLNK(link_status.st_mode):
            _fail("輸入檔不得是 symlink。")
        if not stat.S_ISREG(link_status.st_mode):
            _fail("輸入檔必須是普通檔案。")

        open_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(filesystem_path, open_flags)
        opened_status = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_status.st_mode):
            _fail("開啟的輸入物件必須是普通檔案。")

        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except _AggregateSpecError:
        raise
    except Exception:
        _fail("無法讀取輸入檔。")
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)


def _canonical_sha256(spec: AggregateSpec) -> str:
    """依固定 JSON 序列化規則計算設定本體的 SHA-256。"""

    payload = spec.to_dict()
    # 雜湊欄位是 payload 的衍生 metadata，必須一併移除才能避免自我參照
    # 或因原始檔案格式變化而改變 canonical hash。
    del payload["source_sha256"]
    del payload["canonical_sha256"]
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _load_document(
    document: object,
    *,
    source_sha256: str,
) -> AggregateSpec:
    """驗證已解碼 JSON 並建立帶有兩個 provenance 雜湊的規格物件。"""

    if not isinstance(document, dict):
        _fail("JSON root 必須是 object。")
    if set(document) != _ROOT_KEYS:
        _fail("JSON root keys 不符合固定 schema。")

    raw_site_grids = document["site_grids"]
    if type(raw_site_grids) is not dict or not raw_site_grids:
        _fail("site_grids 必須是非空 object。")
    site_grids: dict[str, SiteGridSpec] = {}
    for site_name, raw_grid in raw_site_grids.items():
        _require_slug(site_name, label="site_grids key")
        if type(raw_grid) is not dict or set(raw_grid) != _SITE_GRID_KEYS:
            _fail("每個 site grid 必須精確包含四個邊界欄位。")
        site_grids[site_name] = SiteGridSpec(
            x_min_m=raw_grid["x_min_m"],
            x_max_m=raw_grid["x_max_m"],
            y_min_m=raw_grid["y_min_m"],
            y_max_m=raw_grid["y_max_m"],
        )

    raw_site_metric_crs = document["site_metric_crs"]
    if type(raw_site_metric_crs) is not dict:
        _fail("site_metric_crs 必須是 JSON object。")
    site_metric_crs: dict[str, SiteMetricCRSSpec] = {}
    for site_name, raw_metric_crs in raw_site_metric_crs.items():
        _require_slug(site_name, label="site_metric_crs key")
        if (
            type(raw_metric_crs) is not dict
            or set(raw_metric_crs) != _SITE_METRIC_CRS_KEYS
        ):
            _fail(
                "每個 site_metric_crs value 必須精確包含五個投影欄位。"
            )
        site_metric_crs[site_name] = SiteMetricCRSSpec(
            projection_method=raw_metric_crs["projection_method"],
            center_lon_deg=raw_metric_crs["center_lon_deg"],
            center_lat_deg=raw_metric_crs["center_lat_deg"],
            linear_unit=raw_metric_crs["linear_unit"],
            axis_order=raw_metric_crs["axis_order"],
        )

    raw_segments = document["boundary_segment_lengths_m"]
    if type(raw_segments) is not dict or not raw_segments:
        _fail("boundary_segment_lengths_m 必須是非空 object。")
    boundary_segments: dict[str, int | float] = {}
    for segment_name, segment_length in raw_segments.items():
        _require_slug(segment_name, label="boundary segment key")
        boundary_segments[segment_name] = _require_positive_number(
            segment_length,
            label="boundary segment length",
        )

    raw_site_boundary_segments = document["site_boundary_segment_ids"]
    if type(raw_site_boundary_segments) is not dict:
        _fail("site_boundary_segment_ids 必須是 JSON object。")
    site_boundary_segments: dict[str, SiteBoundarySegments] = {}
    for site_name, raw_segment_groups in raw_site_boundary_segments.items():
        _require_slug(site_name, label="site_boundary_segment_ids key")
        if (
            type(raw_segment_groups) is not dict
            or set(raw_segment_groups) != _SITE_BOUNDARY_SEGMENT_KEYS
        ):
            _fail("每個站點邊界分類必須精確包含 local 與 outer。")
        raw_local_segment_ids = raw_segment_groups["local"]
        raw_outer_segment_ids = raw_segment_groups["outer"]
        if type(raw_local_segment_ids) is not list:
            _fail("站點 local 邊界段必須是 JSON array。")
        if type(raw_outer_segment_ids) is not list:
            _fail("站點 outer 邊界段必須是 JSON array。")
        site_boundary_segments[site_name] = SiteBoundarySegments(
            local_segment_ids=tuple(raw_local_segment_ids),
            outer_segment_ids=tuple(raw_outer_segment_ids),
        )

    raw_bandwidths = document["kde_bandwidths_m"]
    if type(raw_bandwidths) is not list:
        _fail("kde_bandwidths_m 必須是 JSON array。")
    raw_hdr_levels = document["hdr_levels"]
    if type(raw_hdr_levels) is not list:
        _fail("hdr_levels 必須是 JSON array。")
    raw_age_bin_edges_seconds = document["age_bin_edges_seconds"]
    if type(raw_age_bin_edges_seconds) is not list:
        _fail("age_bin_edges_seconds 必須是 JSON array。")

    # 先以全零 canonical hash 建立完整物件，使所有結構驗證都集中在
    # AggregateSpec.__post_init__；接著只替換真正計算出的 canonical hash。
    provisional_spec = AggregateSpec(
        schema_version=document["schema_version"],
        run_id=document["run_id"],
        grid_cell_size_m=document["grid_cell_size_m"],
        site_grids=site_grids,
        site_metric_crs=site_metric_crs,
        boundary_bin_size_m=document["boundary_bin_size_m"],
        boundary_segment_lengths_m=boundary_segments,
        site_boundary_segment_ids=site_boundary_segments,
        kde_bandwidths_m=tuple(raw_bandwidths),
        hdr_levels=tuple(raw_hdr_levels),
        age_bin_edges_seconds=tuple(raw_age_bin_edges_seconds),
        bootstrap_replicates=document["bootstrap_replicates"],
        bootstrap_confidence_level=document["bootstrap_confidence_level"],
        bootstrap_seed=document["bootstrap_seed"],
        denominator_policy=document["denominator_policy"],
        source_sha256=source_sha256,
        canonical_sha256="0" * 64,
    )
    return AggregateSpec(
        schema_version=provisional_spec.schema_version,
        run_id=provisional_spec.run_id,
        grid_cell_size_m=provisional_spec.grid_cell_size_m,
        site_grids=provisional_spec.site_grids,
        site_metric_crs=provisional_spec.site_metric_crs,
        boundary_bin_size_m=provisional_spec.boundary_bin_size_m,
        boundary_segment_lengths_m=provisional_spec.boundary_segment_lengths_m,
        site_boundary_segment_ids=provisional_spec.site_boundary_segment_ids,
        kde_bandwidths_m=provisional_spec.kde_bandwidths_m,
        hdr_levels=provisional_spec.hdr_levels,
        age_bin_edges_seconds=provisional_spec.age_bin_edges_seconds,
        bootstrap_replicates=provisional_spec.bootstrap_replicates,
        bootstrap_confidence_level=provisional_spec.bootstrap_confidence_level,
        bootstrap_seed=provisional_spec.bootstrap_seed,
        denominator_policy=provisional_spec.denominator_policy,
        source_sha256=provisional_spec.source_sha256,
        canonical_sha256=_canonical_sha256(provisional_spec),
    )


def load_aggregate_spec(path: str | Path) -> AggregateSpec:
    """從嚴格 UTF-8 JSON 普通檔案載入並驗證 `AggregateSpec`。

    讀取流程先保留原始 bytes 計算 `source_sha256`，再以拒絕 duplicate
    keys、NaN/Infinity、非有限極大指數、未知欄位與錯誤資料型別的方式
    解碼。根層與站點欄位採 exact-key policy；所有距離與帶寬維持公尺制，
    投影中心以度表示，travel age 以秒表示，網格範圍必須是整數個 cell
    size 的近似值。成功回傳的四個 mapping、站點邊界分組與三個數值序列
    欄位都是防禦性不可變快照；age edges 另固定為 float tuple。

    任何檔案、編碼、JSON 或資料契約錯誤都以 `ValueError` 回報，錯誤訊息
    不包含傳入的 path。規格描述的來源結果仍僅是條件式來源足跡或相對
    來源權重，不能在此資料模型之外被宣稱為絕對來源機率或因果歸因。
    """

    try:
        raw_bytes = _read_regular_file_bytes(path)
        source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        try:
            text = raw_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            _fail("輸入檔必須是嚴格 UTF-8。")

        try:
            document = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except _AggregateSpecError:
            raise
        except Exception:
            _fail("輸入檔不是有效的嚴格 JSON。")

        _reject_non_finite_values(document)
        return _load_document(document, source_sha256=source_sha256)
    except _AggregateSpecError as error:
        # 轉成公開的 ValueError，並避免把內部例外類別暴露成 API 契約。
        raise ValueError(str(error)) from None
    except Exception:
        # 包含檔案系統、雜湊、型別與序列化邊界例外；訊息刻意不帶 path。
        raise ValueError("aggregate spec 載入或驗證失敗。") from None


def _derive_boundary_contract(
    boundaries: BoundaryGeometryBundle | Mapping[str, BoundaryGeometry],
    *,
    site_metric_centers_deg: Mapping[str, tuple[int | float, int | float]],
    grid_cell_size_m: int | float,
) -> tuple[
    dict[str, SiteGridSpec],
    dict[str, SiteMetricCRSSpec],
    dict[str, float],
    dict[str, SiteBoundarySegments],
]:
    """只由記憶體中的邊界與明示中心推導四份新的 boundary contract mapping。

    這是 writer 與公開 validator 共用的唯一幾何推導入口。它不讀取或寫入檔案，
    不建立 ``AggregateSpec``，也不計算 ``source_sha256``／``canonical_sha256``；
    因此 validator 可以在 pipeline 已經載入規格後，重新驗證 caller 傳入的
    ``AggregateSpec`` 是否確實來自同一份已驗證 ``BoundaryGeometryBundle``。

    ``flow_domain`` 的 bounds 先在已投影的公尺座標中以 ``floor``／``ceil`` 向外
    對齊 ``grid_cell_size_m``，所以不會因浮點或負座標把有效流場範圍向內裁掉。每站
    的投影只使用 ``site_metric_centers_deg`` 明示的 WGS84 經緯度（單位是度）建立
    固定區域方位等距投影（azimuthal equidistant, AEQD）描述；不會從
    ``DomainProjection`` 的私有狀態猜中心。open-water 邊界只接受有效且非空的
    ``LineString``／``MultiLineString``，長度在目前公尺制 geometry 中以正的有限
    公尺數保存；coastline 不會被放進 segment table。``local_equals_flow`` 時，
    flow segment 同時擔任 local 與 outer role；其他站點則分別登錄 local 與 flow
    segment。相同 segment ID 若跨站或跨 role 出現，弧長必須 exact 相同。

    回傳的四個普通 ``dict`` 都是本次推導新建立的物件，供 caller 後續建立規格或
    做逐項 exact 比對。輸入 mapping、geometry 與中心不會被修改。此處測試資料只
    驗證工程 binding；通過 synthetic 測試不代表任何正式 OCM／NWW 科學成果。
    """

    # Bundle 需額外驗證 projection site set；但實際投影中心仍必須由 caller 明示，
    # 因為 projections 物件的內部狀態不是這個資料契約允許的隱含輸入。一般 mapping
    # 保留 writer 既有允許行為，不要求它附帶 projections metadata。
    if isinstance(boundaries, BoundaryGeometryBundle):
        geometry_mapping = dict(boundaries.by_study_site)
        if set(boundaries.projections) != set(geometry_mapping):
            raise ValueError("BoundaryGeometryBundle 的 projections site set 必須與 geometries 一致。")
    elif isinstance(boundaries, Mapping):
        try:
            geometry_mapping = dict(boundaries)
        except Exception as error:
            raise ValueError("boundaries mapping 無法複製。") from error
    else:
        raise ValueError("boundaries 必須是 BoundaryGeometryBundle 或 site-to-geometry mapping。")

    if not geometry_mapping:
        raise ValueError("boundaries 不得為空。")
    for site_id in geometry_mapping:
        _require_slug(site_id, label="boundaries site key")
    site_ids = tuple(sorted(geometry_mapping))

    try:
        center_mapping = dict(site_metric_centers_deg)
    except Exception as error:
        raise ValueError("site_metric_centers_deg 必須是 mapping。") from error
    for site_id in center_mapping:
        _require_slug(site_id, label="site_metric_centers_deg key")
    if set(center_mapping) != set(site_ids):
        raise ValueError("site_metric_centers_deg keys 必須與 boundaries site set 完全一致。")

    try:
        cell_size = float(_require_positive_number(grid_cell_size_m, label="grid_cell_size_m"))
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("grid_cell_size_m 必須是可表示的正有限數值。") from error
    if not math.isfinite(cell_size):
        raise ValueError("grid_cell_size_m 必須是可表示的正有限數值。")

    def line_length(line: object, *, label: str) -> float:
        """驗證開放邊界是有效二維線，並回傳其公尺長度。"""

        if not isinstance(line, BaseGeometry):
            raise ValueError(f"{label} 必須是 Shapely geometry。")
        if line.is_empty or not line.is_valid or line.geom_type not in {
            "LineString",
            "MultiLineString",
        }:
            raise ValueError(f"{label} 必須是有效且非空的 LineString 或 MultiLineString。")
        try:
            length = float(line.length)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"{label} 長度無法轉成有限公尺數值。") from error
        if not math.isfinite(length) or length <= 0:
            raise ValueError(f"{label} 長度必須是正的有限公尺數值。")
        return length

    def aligned_bounds(flow_domain: object, *, site_id: str) -> SiteGridSpec:
        """將 flow polygon bounds 逐軸向外對齊到固定公尺格網。"""

        if not isinstance(flow_domain, BaseGeometry):
            raise ValueError(f"站點 {site_id} 的 flow_domain 必須是 Shapely geometry。")
        if flow_domain.is_empty or not flow_domain.is_valid or flow_domain.geom_type != "Polygon":
            raise ValueError(f"站點 {site_id} 的 flow_domain 必須是有效且非空的 Polygon。")
        bounds = flow_domain.bounds
        if len(bounds) != 4 or not all(math.isfinite(float(value)) for value in bounds):
            raise ValueError(f"站點 {site_id} 的 flow_domain bounds 必須是有限四元組。")
        x_min, y_min, x_max, y_max = (float(value) for value in bounds)
        if not (x_min < x_max and y_min < y_max):
            raise ValueError(f"站點 {site_id} 的 flow_domain bounds 必須形成正矩形。")

        def outward(value: float, *, lower: bool) -> float:
            """以 floor/ceil 保證對應下界／上界不會向內裁切。"""

            try:
                index = math.floor(value / cell_size) if lower else math.ceil(value / cell_size)
                aligned = float(index * cell_size)
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(f"站點 {site_id} 的 flow bounds 無法對齊 grid。") from error
            if not math.isfinite(aligned):
                raise ValueError(f"站點 {site_id} 的對齊 grid bounds 必須是有限值。")
            return aligned

        return SiteGridSpec(
            x_min_m=outward(x_min, lower=True),
            x_max_m=outward(x_max, lower=False),
            y_min_m=outward(y_min, lower=True),
            y_max_m=outward(y_max, lower=False),
        )

    site_grids: dict[str, SiteGridSpec] = {}
    site_metric_crs: dict[str, SiteMetricCRSSpec] = {}
    site_boundary_segments: dict[str, SiteBoundarySegments] = {}
    segment_lengths: dict[str, float] = {}

    def register_segment(segment_id: str, length: float, *, site_id: str) -> None:
        """登錄全域 segment ID，拒絕同 ID 對應不同的公尺長度。"""

        _require_slug(segment_id, label="boundary segment ID")
        previous = segment_lengths.get(segment_id)
        if previous is not None and previous != length:
            raise ValueError(
                f"boundary segment {segment_id} 在不同 site/role 的長度不一致。"
            )
        segment_lengths[segment_id] = length

    for site_id in site_ids:
        geometry = geometry_mapping[site_id]
        if not isinstance(geometry, BoundaryGeometry):
            raise ValueError("boundaries value 必須是 BoundaryGeometry。")
        if type(geometry.local_equals_flow) is not bool:
            raise ValueError(f"站點 {site_id} 的 local_equals_flow 必須是 bool。")
        site_grids[site_id] = aligned_bounds(geometry.flow_domain, site_id=site_id)

        center = center_mapping[site_id]
        if isinstance(center, (str, bytes, bytearray)):
            raise ValueError(f"站點 {site_id} 的 metric center 必須是 lon/lat 二元素陣列。")
        try:
            center_values = tuple(center)
        except Exception as error:
            raise ValueError(f"站點 {site_id} 的 metric center 必須是 lon/lat 二元素陣列。") from error
        if len(center_values) != 2:
            raise ValueError(f"站點 {site_id} 的 metric center 必須是 lon/lat 二元素陣列。")
        site_metric_crs[site_id] = SiteMetricCRSSpec(
            projection_method="azimuthal_equidistant_wgs84",
            center_lon_deg=center_values[0],
            center_lat_deg=center_values[1],
            linear_unit="m",
            axis_order="x_east_y_north",
        )

        flow_segment_id = geometry.flow_boundary_segment_id
        flow_length = line_length(
            geometry.flow_open_boundary,
            label=f"站點 {site_id} 的 flow_open_boundary",
        )
        register_segment(flow_segment_id, flow_length, site_id=site_id)

        if geometry.local_equals_flow:
            local_segment_ids = (flow_segment_id,)
        else:
            local_segment_id = geometry.own_local_boundary_segment_id
            local_length = line_length(
                geometry.own_local_open_boundary,
                label=f"站點 {site_id} 的 own_local_open_boundary",
            )
            register_segment(local_segment_id, local_length, site_id=site_id)
            local_segment_ids = (local_segment_id,)
        site_boundary_segments[site_id] = SiteBoundarySegments(
            local_segment_ids=local_segment_ids,
            outer_segment_ids=(flow_segment_id,),
        )

    return (
        site_grids,
        site_metric_crs,
        segment_lengths,
        site_boundary_segments,
    )


def validate_aggregate_spec_against_boundaries(
    spec: AggregateSpec,
    boundaries: BoundaryGeometryBundle | Mapping[str, BoundaryGeometry],
    *,
    site_metric_centers_deg: Mapping[str, tuple[int | float, int | float]],
) -> None:
    """驗證已載入 ``AggregateSpec`` 是否由指定邊界與明示投影中心導出。

    ``spec`` 必須是 exact ``AggregateSpec`` 實例，不接受 subclass 或只模仿欄位的
    duck type。函式以 ``spec.grid_cell_size_m`` 重新執行與 writer 完全相同的純記憶體
    推導：flow domain 的公尺 bounds 向外對齊 grid、每站明示的經緯度中心建立 AEQD
    metric CRS、open-water LineString／MultiLineString 的正有限公尺長度，以及
    local／outer segment role。四份 mapping（site grids、site metric CRS、全域
    segment length、每站 segment IDs）都以 site key 與 immutable value 逐項 exact
    比對；任一站點、bounds、中心、投影方法／單位／軸序、segment ID、弧長或 role
    差異都回報 ``ValueError``。

    這個 validator 不驗證 ``source_sha256`` 或 ``canonical_sha256`` 的 bytes 真實性，
    因為那是 ``load_aggregate_spec`` 讀檔與 writer 發布流程的責任；也不從 geometry
    猜測 ``run_id``，因為幾何本身沒有 run identity。pipeline 必須另行核對
    ``spec.run_id == plan.run_id``。函式只讀輸入、回傳 ``None``，不寫檔、不修改
    ``spec``、boundaries 或 centers。測試使用的資料是 synthetic 工程 binding，並非
    OCM／NWW 科學成果。
    """

    if type(spec) is not AggregateSpec:
        raise ValueError("spec 必須是 exact AggregateSpec 實例。")

    (
        derived_site_grids,
        derived_site_metric_crs,
        derived_segment_lengths,
        derived_site_boundary_segments,
    ) = _derive_boundary_contract(
        boundaries,
        site_metric_centers_deg=site_metric_centers_deg,
        grid_cell_size_m=spec.grid_cell_size_m,
    )

    if set(spec.site_grids) != set(derived_site_grids):
        raise ValueError("spec.site_grids site set 與 boundaries 推導結果不一致。")
    for site_id in sorted(derived_site_grids):
        if spec.site_grids[site_id] != derived_site_grids[site_id]:
            raise ValueError(f"spec.site_grids[{site_id}] 與 boundaries 推導結果不一致。")

    if set(spec.site_metric_crs) != set(derived_site_metric_crs):
        raise ValueError("spec.site_metric_crs site set 與 boundaries 推導結果不一致。")
    for site_id in sorted(derived_site_metric_crs):
        if spec.site_metric_crs[site_id] != derived_site_metric_crs[site_id]:
            raise ValueError(f"spec.site_metric_crs[{site_id}] 與 boundaries 推導結果不一致。")

    if set(spec.boundary_segment_lengths_m) != set(derived_segment_lengths):
        raise ValueError("spec.boundary_segment_lengths_m keys 與 boundaries 推導結果不一致。")
    for segment_id in sorted(derived_segment_lengths):
        if spec.boundary_segment_lengths_m[segment_id] != derived_segment_lengths[segment_id]:
            raise ValueError(
                f"spec.boundary_segment_lengths_m[{segment_id}] 與 boundaries 推導結果不一致。"
            )

    if set(spec.site_boundary_segment_ids) != set(derived_site_boundary_segments):
        raise ValueError("spec.site_boundary_segment_ids site set 與 boundaries 推導結果不一致。")
    for site_id in sorted(derived_site_boundary_segments):
        if spec.site_boundary_segment_ids[site_id] != derived_site_boundary_segments[site_id]:
            raise ValueError(
                f"spec.site_boundary_segment_ids[{site_id}] 與 boundaries 推導結果不一致。"
            )


def write_aggregate_spec_from_boundaries(
    target: str | Path,
    boundaries: BoundaryGeometryBundle | Mapping[str, BoundaryGeometry],
    *,
    run_id: str,
    site_metric_centers_deg: Mapping[str, tuple[int | float, int | float]],
    grid_cell_size_m: int | float,
    boundary_bin_size_m: int | float,
    kde_bandwidths_m: tuple[int | float, ...],
    age_bin_edges_seconds: tuple[int | float, ...],
    bootstrap_replicates: int,
    bootstrap_confidence_level: int | float,
    bootstrap_seed: int,
) -> Path:
    """由已驗證的邊界幾何建立並原子發布彙整規格 JSON。

    Args:
        target: 要發布的 JSON 普通檔案路徑。既有檔案、目錄與符號連結都
            不可被覆寫；暫存檔會建立在同一個 parent，讓最後的
            ``os.replace`` 保持同一檔案系統內的原子交換語意。
        boundaries: 已由上游 geometry manifest 驗證的
            :class:`BoundaryGeometryBundle`，或等價的 site-to-
            :class:`BoundaryGeometry` mapping。每個站點的 flow domain
            bounds 以公尺表示；開放邊界線段的弧長也以公尺表示。Bundle
            另外必須有與 geometry 完全相同的 projection site set，但
            投影中心仍由 ``site_metric_centers_deg`` 明示提供，不從
            ``DomainProjection`` 的私有狀態猜測。
        run_id: 此彙整規格所屬 run 的安全識別碼。
        site_metric_centers_deg: 每站固定 AEQD 投影中心
            ``(center_lon_deg, center_lat_deg)``，單位為度。中心是重建
            metric geometry 的必要資料，因此不能由站點之間隱含繼承。
        grid_cell_size_m: 站點彙整格網的公尺邊長。flow domain 的四個
            bounds 會分別以此尺寸向外取整，確保不裁掉有效流場範圍。
        boundary_bin_size_m: 一維邊界弧長分箱尺寸，單位為公尺。
        kde_bandwidths_m: 三個嚴格遞增的 KDE 帶寬，單位為公尺。
        age_bin_edges_seconds: pathway 與事件共用的 travel-age 秒數
            邊界；第一點必須為 0，最後一箱包含最後邊界。
        bootstrap_replicates: bootstrap 重抽樣次數。
        bootstrap_confidence_level: bootstrap 信賴水準。
        bootstrap_seed: bootstrap 使用的 128 位元無號整數 seed。

    Returns:
        成功以原子方式發布後的 ``Path``。發布前會先用
        :func:`load_aggregate_spec` 重新讀取同一份 partial JSON，因此
        回傳的檔案已通過完整 schema、數值、site set 與 canonical hash
        驗證。

    Raises:
        ValueError: 幾何、站點、投影中心、線段長度或 AggregateSpec
            資料契約不合法，或檔案無法安全建立／驗證。
        FileExistsError: ``target`` 已存在或本身是符號連結；函式不會
            覆寫既有成果。

    Notes:
        ``local_equals_flow`` 代表同一個開放 flow line 同時擔任 local
        與 outer 語意，所以 local 與 outer 都引用 flow segment ID；
        其他站點則使用自己的 local line/ID。相同 segment ID 不論在哪一
        個站點或角色出現，都必須得到完全相同的公尺長度，避免同一邊界
        在下游以不同弧長格線重建。此函式只發布設定，不讀取 raw
        NetCDF，也不把 coastline contact 假造為 open-water segment。
    """

    (
        site_grids,
        site_metric_crs,
        segment_lengths,
        site_boundary_segments,
    ) = _derive_boundary_contract(
        boundaries,
        site_metric_centers_deg=site_metric_centers_deg,
        grid_cell_size_m=grid_cell_size_m,
    )

    # 先建構 provisional spec，集中觸發 AggregateSpec 的完整 schema 驗證；
    # JSON 輸入不攜帶兩個衍生 hash，source hash 會由 partial 原始 bytes
    # 計算，而 canonical hash 會由 loader 依固定序列化語意重建。
    provisional_spec = AggregateSpec(
        schema_version=AGGREGATE_SPEC_SCHEMA_VERSION,
        run_id=run_id,
        grid_cell_size_m=grid_cell_size_m,
        site_grids=site_grids,
        site_metric_crs=site_metric_crs,
        boundary_bin_size_m=boundary_bin_size_m,
        boundary_segment_lengths_m=segment_lengths,
        site_boundary_segment_ids=site_boundary_segments,
        kde_bandwidths_m=tuple(kde_bandwidths_m),
        hdr_levels=_HDR_LEVELS,
        age_bin_edges_seconds=tuple(age_bin_edges_seconds),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_seed=bootstrap_seed,
        denominator_policy=_DENOMINATOR_POLICY,
        source_sha256="0" * 64,
        canonical_sha256="0" * 64,
    )
    payload = provisional_spec.to_dict()
    del payload["source_sha256"]
    del payload["canonical_sha256"]
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    target_path = Path(target)
    if target_path.exists() or target_path.is_symlink():
        raise FileExistsError(f"不可覆寫既有 aggregate spec：{target_path.name}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = target_path.parent / f".{target_path.name}.partial-{uuid4().hex}"
    try:
        # 使用 xb 避免極小機率的 partial 名稱碰撞造成既有暫存內容被覆寫；
        # fsync 讓 load/replace 前的 bytes 已交給檔案系統，最後只交換同父目錄
        # 的完整檔案，不發布半寫入 JSON。
        with partial_path.open("xb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        load_aggregate_spec(partial_path)
        if target_path.exists() or target_path.is_symlink():
            raise FileExistsError(f"不可覆寫既有 aggregate spec：{target_path.name}")
        os.replace(partial_path, target_path)
    except Exception:
        with suppress(OSError):
            partial_path.unlink(missing_ok=True)
        raise
    return target_path
