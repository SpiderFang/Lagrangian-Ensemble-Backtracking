"""建立停止結果與方向性跨站連通的 renderer-facing 統計產品。

本模組只處理已通過 aggregate release 驗證的計數與分母，不讀取檔案、不重新開啟
trajectory shard，也不從圖面或缺列推測資料。停止結果沿用事件聚合保存的
``ParticleStatus`` 非 ``ACTIVE`` 狀態；``DATA_GAP`` 與 ``NUMERICAL_FAILURE`` 只
保留在 outcome／failure exposure 診斷，``PRE_WINDOW_DEPOSITION`` 則只保留 outcome；
三者都不進入有效成員分母，且 pre-window 不會被誤列為資料或數值失敗。

跨站矩陣的方向固定為 ``source_study_site_id → target_study_site_id``，矩陣軸不是
空間 ``(y_cell, x_cell)`` 軸，因此另以 site ID tuple 保存 axis label。矩陣對角線
在 aggregate 拓撲中沒有合法的 cross-site key；其 raw matrix 位置使用結構性零值，
並以 ``diagonal_not_applicable_mask`` 明確標記不適用，避免把不適用誤讀成零連通。
非對角線的 visit fraction 使用 source site 有效成員分母；cross-site event share
則使用同一 source row 的所有跨站 event raw count 作分母。兩者都保存 raw count、
分母與 ``EstimateStatus``，不能互相替代。

所有陣列在 immutable product 邊界重新複製並設為唯讀。比例透過既有
``CountRatio``／``build_count_ratio`` 建立，故零分母會是 ``None`` 加明確
``zero_denominator``，而不是零值。這些輸出描述指定條件下的條件式來源足跡或相對
來源權重，不是絕對來源機率，也不構成因果歸因。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np

from .aggregate_release_payload import AggregateReleasePayload
from .event_aggregation import CrossSiteAggregateKey
from .models import ParticleStatus
from .report_ratio_statistics import CountRatio, EstimateStatus, build_count_ratio

__all__ = [
    "ConnectivityStatistics",
    "ConnectivityStatisticsProduct",
    "OutcomeStatistics",
    "OutcomeStatisticsProduct",
    "build_connectivity_statistics",
    "build_outcome_statistics",
]


_INT64_MAX: Final[int] = int(np.iinfo(np.int64).max)
_EXPECTED_OUTCOME_KEYS: Final[frozenset[str]] = frozenset(
    status.value for status in ParticleStatus if status is not ParticleStatus.ACTIVE
)
_FAILURE_KEYS: Final[tuple[str, str]] = (
    ParticleStatus.DATA_GAP.value,
    ParticleStatus.NUMERICAL_FAILURE.value,
)


def _require_native_nonnegative_int(value: object, *, label: str) -> int:
    """驗證報告計數是可安全保存的非負原生 ``int``。

    aggregate release 的部分跨 shard 計數在更底層可使用 Python 任意精度整數，但
    ``CountRatio`` 與 renderer-facing NumPy raw matrix 的交換契約固定為 signed
    ``int64``。因此在報告邊界先檢查上限，再進行任何 NumPy 轉換，避免大數靜默
    截斷或繞回成負值。
    """

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


def _snapshot_site_ids(value: object, *, label: str = "site_ids") -> tuple[str, ...]:
    """複製 site axis 的識別碼並保留 caller 宣告的矩陣軸順序。

    site 順序會直接決定 ``raw_matrix[i, j]`` 的 source／target 解讀，因此不能在
    builder 中依目前計數重新排序或補站。payload 路徑會以 AggregateSpec 的固定
    site mapping 順序提供此 tuple；純 mapping 路徑則要求 caller 明示順序，並拒絕
    重複、空白或含首尾空白的識別碼。
    """

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} 必須是非字串、非 mapping 的 site sequence")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} 必須是可 materialize 的 site sequence") from error
    if not items:
        raise ValueError(f"{label} 不可為空")
    normalized: list[str] = []
    for index, item in enumerate(items):
        if type(item) is not str:
            raise TypeError(f"{label}[{index}] 必須是原生 str")
        if not item or item != item.strip():
            raise ValueError(f"{label}[{index}] 必須是非空且沒有首尾空白的文字")
        if item in normalized:
            raise ValueError(f"{label} 不可包含重複 site ID")
        normalized.append(item)
    return tuple(normalized)


def _snapshot_count_mapping(
    value: object,
    *,
    key_type: type,
    label: str,
) -> Mapping[object, int]:
    """複製 typed key 到 raw count 的 mapping，隔離 caller 後續修改。

    ``CrossSiteAggregateKey`` 與其他 immutable key 是資料契約的一部分；不接受
    ``("source", "target")`` 等相似 tuple，避免 renderer 取得不同 key 形式後
    產生不一致的方向 join。每個計數先經 Python int64 上限檢查，再封存為
    ``MappingProxyType``。
    """

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必須是 mapping")
    copied: dict[object, int] = {}
    try:
        items = tuple(value.items())
    except Exception as error:
        raise ValueError(f"{label} 無法穩定讀取") from error
    for key, raw_count in items:
        if type(key) is not key_type:
            raise TypeError(f"{label} 的 key 必須是 exact {key_type.__name__}")
        if key in copied:
            raise ValueError(f"{label} 不可有重複 key")
        copied[key] = _require_native_nonnegative_int(
            raw_count,
            label=f"{label}[{key!r}]",
        )
    return MappingProxyType(copied)


def _snapshot_site_count_mapping(
    value: object,
    *,
    site_ids: tuple[str, ...],
    label: str,
) -> Mapping[str, int]:
    """複製每站 scalar count mapping，並要求 key set 完整對齊 site axis。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必須是 mapping")
    copied: dict[str, int] = {}
    for key, raw_count in value.items():
        if type(key) is not str:
            raise TypeError(f"{label} 的 key 必須是原生 str")
        if not key or key != key.strip():
            raise ValueError(f"{label} 的 key 必須是非空且沒有首尾空白的文字")
        if key in copied:
            raise ValueError(f"{label} 不可有重複 site ID")
        copied[key] = _require_native_nonnegative_int(
            raw_count,
            label=f"{label}[{key!r}]",
        )
    if set(copied) != set(site_ids):
        raise ValueError(f"{label} 的 site key 必須精確等於 site_ids")
    # mapping 的迭代順序也固定依 site axis 重建；這避免同一份資料因 caller
    # 使用不同 insertion order 而讓 renderer 的表格列順序漂移。
    return MappingProxyType({site_id: copied[site_id] for site_id in site_ids})


def _snapshot_outcome_mapping(
    value: object,
    *,
    site_ids: tuple[str, ...],
    label: str,
) -> Mapping[str, Mapping[str, int]]:
    """封存每站非 ACTIVE 停止狀態的完整 raw count 拓撲。

    每個狀態即使計數為零也必須存在；缺列與零事件的科學語意不同，不能在報告層
    以 ``dict.get(..., 0)`` 靜默補齊。這裡的完整狀態集合與既有 ``ParticleStatus``
    enum 同步，並由後續守恆檢查確認總數等於該站 total denominator。
    """

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必須是 site 到 outcome mapping")
    copied_by_site: dict[str, Mapping[str, int]] = {}
    for site_id, raw_outcomes in value.items():
        if type(site_id) is not str or not site_id or site_id != site_id.strip():
            raise ValueError(f"{label} 的 site key 必須是非空原生 str")
        if not isinstance(raw_outcomes, Mapping):
            raise TypeError(f"{label}[{site_id!r}] 必須是 outcome mapping")
        copied: dict[str, int] = {}
        for outcome, raw_count in raw_outcomes.items():
            if type(outcome) is not str or not outcome or outcome != outcome.strip():
                raise ValueError(f"{label}[{site_id!r}] 的 outcome key 不合法")
            if outcome in copied:
                raise ValueError(f"{label}[{site_id!r}] 不可有重複 outcome")
            copied[outcome] = _require_native_nonnegative_int(
                raw_count,
                label=f"{label}[{site_id!r}][{outcome!r}]",
            )
        if set(copied) != _EXPECTED_OUTCOME_KEYS:
            raise ValueError(
                f"{label}[{site_id!r}] 必須完整包含 ParticleStatus 的所有非 ACTIVE 狀態"
            )
        copied_by_site[site_id] = MappingProxyType(copied)
    if set(copied_by_site) != set(site_ids):
        raise ValueError(f"{label} 的 site key 必須精確等於 site_ids")
    # 外層 mapping 依 canonical site axis 重建，與 raw matrix 的 row/column join
    # 一致；內層 outcome 則仍保留 ParticleStatus 的完整 key 拓撲。
    return MappingProxyType({site_id: copied_by_site[site_id] for site_id in site_ids})


def _readonly_array(value: object, *, dtype: np.dtype, ndim: int, label: str) -> np.ndarray:
    """建立指定 dtype、shape 維度與唯讀旗標的獨立 NumPy snapshot。"""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 必須是 {ndim} 維 NumPy 陣列") from error
    if raw.ndim != ndim:
        raise ValueError(f"{label} 必須是 {ndim} 維陣列")
    if dtype == np.dtype(np.int64):
        if raw.dtype.kind not in "iu":
            raise TypeError(f"{label} 必須是整數 dtype")
        for element in raw.flat:
            _require_native_nonnegative_int(int(element), label=label)
    elif dtype == np.dtype(np.float64):
        if raw.dtype.kind not in "iuf":
            raise TypeError(f"{label} 必須是數值 dtype")
    elif dtype == np.dtype(np.bool_):
        if raw.dtype.kind != "b":
            raise TypeError(f"{label} 必須是 bool dtype")
    copied = np.array(raw, dtype=dtype, copy=True)
    if dtype == np.dtype(np.float64) and np.any(np.isinf(copied)):
        raise ValueError(f"{label} 不可包含無限值")
    copied.setflags(write=False)
    return copied


def _snapshot_failure_grid_mapping(
    value: object | None,
    *,
    site_ids: tuple[str, ...],
    label: str,
) -> Mapping[str, np.ndarray]:
    """封存 F10 failure-grid 診斷，並保留其 `(y_cell, x_cell)` 軸語意。

    failure grid 是資料缺口或數值失敗成員的空間診斷，不是有效來源 numerator。
    因此只有 caller 明示提供的 grid 才會出現在產品；省略時返回空 mapping，絕不
    依 outcome count 猜測格點或建立全零陣列。若提供 grid，所有站點都必須有一個
    非負 int64 二維陣列，且後續會核對其總和等於對應 failure raw count。
    """

    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必須是 site 到 failure grid 的 mapping")
    copied: dict[str, np.ndarray] = {}
    for site_id, array in value.items():
        if type(site_id) is not str or not site_id or site_id != site_id.strip():
            raise ValueError(f"{label} 的 site key 必須是非空原生 str")
        if site_id in copied:
            raise ValueError(f"{label} 不可有重複 site key")
        copied[site_id] = _readonly_array(
            array,
            dtype=np.dtype(np.int64),
            ndim=2,
            label=f"{label}[{site_id!r}]",
        )
    if set(copied) != set(site_ids):
        raise ValueError(f"{label} 的 site key 必須精確等於 site_ids")
    return MappingProxyType(copied)


def _snapshot_status_grid(
    value: object | None,
    *,
    expected: np.ndarray,
    label: str,
) -> np.ndarray:
    """封存並核對逐格 ``EstimateStatus``，對角線以 ``None`` 表示不適用。

    status 是缺值與低樣本的語意來源，不能由 renderer 從比例的 NaN 或零值重新猜測。
    對角線不是估計失敗，而是拓撲上沒有 cross-site 定義，所以以獨立 mask 加
    ``None`` 保存，不冒充 ``zero_denominator``。
    """

    if value is None:
        copied = np.array(expected, dtype=object, copy=True)
    else:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} 必須是 status grid") from error
        if raw.shape != expected.shape:
            raise ValueError(f"{label} shape 必須等於 {expected.shape}")
        copied = np.array(raw, dtype=object, copy=True)
    if not np.array_equal(copied, expected):
        raise ValueError(f"{label} 必須精確符合 raw count、分母與不適用遮罩")
    copied.setflags(write=False)
    return copied


def _ratio_status(
    numerator: int,
    denominator: int,
    *,
    minimum_count: int,
) -> EstimateStatus:
    """以既有 CountRatio 的同一規則取得逐格狀態。"""

    return build_count_ratio(
        numerator,
        denominator,
        minimum_count=minimum_count,
    ).status


@dataclass(frozen=True, slots=True)
class OutcomeStatistics:
    """封存每站停止狀態 raw count、total 分母與失敗曝光。

    ``outcome_count_by_site`` 的內層 key 是 ``ParticleStatus`` 的非 ``ACTIVE``
    字串值；每個站點完整保存所有狀態，即使 raw count 為零。每個狀態的
    ``CountRatio`` 都以 ``total_member_denominator_by_site`` 為分母，故表格可以
    同列輸出 raw numerator、raw denominator、ratio 與明確狀態。

    ``data_gap_failure_grid_by_site`` 與 ``numerical_failure_grid_by_site`` 若有提供，
    會保留事件聚合的 ``(y_cell, x_cell)`` 空間診斷；它們的總和必須分別等於
    DATA_GAP／NUMERICAL_FAILURE raw count，不能當成有效來源格網。
    ``valid_member_denominator_by_site`` 必須等於
    ``total - DATA_GAP - NUMERICAL_FAILURE - PRE_WINDOW_DEPOSITION``。這個 cross-field
    gate 確保三種被排除成員都不進後續條件式來源母體；前兩者保留 failure exposure，
    pre-window 只保留 outcome，不算資料或數值失敗。失敗曝光仍使用 total 分母，與
    有效來源比例的分母語意明確分離。這些結果只能描述條件式來源足跡／相對來源權重，
    不是絕對來源機率或因果歸因。
    """

    site_ids: tuple[str, ...]
    outcome_count_by_site: Mapping[str, Mapping[str, int]]
    total_member_denominator_by_site: Mapping[str, int]
    valid_member_denominator_by_site: Mapping[str, int] | None = None
    data_gap_failure_grid_by_site: Mapping[str, np.ndarray] | None = None
    numerical_failure_grid_by_site: Mapping[str, np.ndarray] | None = None
    minimum_count: int = 1
    _outcome_ratios_by_site: Mapping[str, Mapping[str, CountRatio]] = field(
        init=False,
        repr=False,
    )
    _failure_exposure_by_site: Mapping[str, Mapping[str, CountRatio]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """驗證停止計數守恆、失敗排除規則與比例狀態，然後建立唯讀 mapping。"""

        site_ids = _snapshot_site_ids(self.site_ids)
        outcomes = _snapshot_outcome_mapping(
            self.outcome_count_by_site,
            site_ids=site_ids,
            label="outcome_count_by_site",
        )
        totals = _snapshot_site_count_mapping(
            self.total_member_denominator_by_site,
            site_ids=site_ids,
            label="total_member_denominator_by_site",
        )
        if self.valid_member_denominator_by_site is None:
            valid_input: Mapping[str, int] = {
                site_id: totals[site_id]
                - outcomes[site_id][ParticleStatus.DATA_GAP.value]
                - outcomes[site_id][ParticleStatus.NUMERICAL_FAILURE.value]
                - outcomes[site_id][ParticleStatus.PRE_WINDOW_DEPOSITION.value]
                for site_id in site_ids
            }
        else:
            valid_input = self.valid_member_denominator_by_site
        valid = _snapshot_site_count_mapping(
            valid_input,
            site_ids=site_ids,
            label="valid_member_denominator_by_site",
        )
        if (self.data_gap_failure_grid_by_site is None) != (
            self.numerical_failure_grid_by_site is None
        ):
            raise ValueError(
                "data_gap_failure_grid_by_site 與 numerical_failure_grid_by_site 必須同時提供"
            )
        data_gap_grid = _snapshot_failure_grid_mapping(
            self.data_gap_failure_grid_by_site,
            site_ids=site_ids,
            label="data_gap_failure_grid_by_site",
        )
        numerical_grid = _snapshot_failure_grid_mapping(
            self.numerical_failure_grid_by_site,
            site_ids=site_ids,
            label="numerical_failure_grid_by_site",
        )
        minimum_count = _require_positive_native_int(
            self.minimum_count,
            label="minimum_count",
        )

        ratios_by_site: dict[str, Mapping[str, CountRatio]] = {}
        failure_by_site: dict[str, Mapping[str, CountRatio]] = {}
        for site_id in site_ids:
            site_outcomes = outcomes[site_id]
            total = totals[site_id]
            if sum(site_outcomes.values()) != total:
                raise ValueError(
                    f"site {site_id!r} 的 outcome count 合計必須等於 total denominator"
                )
            gap = site_outcomes[ParticleStatus.DATA_GAP.value]
            numerical = site_outcomes[ParticleStatus.NUMERICAL_FAILURE.value]
            pre_window = site_outcomes[ParticleStatus.PRE_WINDOW_DEPOSITION.value]
            expected_valid = total - gap - numerical - pre_window
            if valid[site_id] != expected_valid:
                raise ValueError(
                    f"site {site_id!r} 的 valid denominator 必須排除 DATA_GAP、"
                    "NUMERICAL_FAILURE 與 PRE_WINDOW_DEPOSITION"
                )
            if data_gap_grid and int(sum(int(element) for element in data_gap_grid[site_id].flat)) != gap:
                raise ValueError(
                    f"site {site_id!r} 的 data_gap failure grid 合計必須等於 DATA_GAP raw count"
                )
            numerical_grid_total = (
                int(sum(int(element) for element in numerical_grid[site_id].flat))
                if numerical_grid
                else None
            )
            if numerical_grid_total is not None and numerical_grid_total != numerical:
                raise ValueError(
                    f"site {site_id!r} 的 numerical failure grid 合計必須等於 NUMERICAL_FAILURE raw count"
                )
            ratios_by_site[site_id] = MappingProxyType(
                {
                    outcome: build_count_ratio(
                        count,
                        total,
                        minimum_count=minimum_count,
                    )
                    for outcome, count in site_outcomes.items()
                }
            )
            failure_by_site[site_id] = MappingProxyType(
                {
                    failure: ratios_by_site[site_id][failure]
                    for failure in _FAILURE_KEYS
                }
            )

        object.__setattr__(self, "site_ids", site_ids)
        object.__setattr__(self, "outcome_count_by_site", outcomes)
        object.__setattr__(self, "total_member_denominator_by_site", totals)
        object.__setattr__(self, "valid_member_denominator_by_site", valid)
        object.__setattr__(self, "data_gap_failure_grid_by_site", data_gap_grid)
        object.__setattr__(self, "numerical_failure_grid_by_site", numerical_grid)
        object.__setattr__(self, "minimum_count", minimum_count)
        object.__setattr__(self, "_outcome_ratios_by_site", MappingProxyType(ratios_by_site))
        object.__setattr__(self, "_failure_exposure_by_site", MappingProxyType(failure_by_site))

    @property
    def outcome_ratios_by_site(self) -> Mapping[str, Mapping[str, CountRatio]]:
        """回傳每站每個停止狀態的 immutable ``CountRatio`` mapping。"""

        return self._outcome_ratios_by_site

    @property
    def outcome_ratio_by_site(self) -> Mapping[str, Mapping[str, CountRatio]]:
        """``outcome_ratios_by_site`` 的單數欄位別名。"""

        return self._outcome_ratios_by_site

    @property
    def ratios_by_site(self) -> Mapping[str, Mapping[str, CountRatio]]:
        """供 renderer／table adapter 使用的比例 mapping 別名。"""

        return self._outcome_ratios_by_site

    @property
    def failure_exposure_by_site(self) -> Mapping[str, Mapping[str, CountRatio]]:
        """回傳 DATA_GAP／NUMERICAL_FAILURE 的 total-denominator exposure。"""

        return self._failure_exposure_by_site

    @property
    def data_gap_exposure_by_site(self) -> Mapping[str, CountRatio]:
        """回傳每站資料缺口曝光比例及其 raw count／total 分母。"""

        return MappingProxyType(
            {
                site_id: self._failure_exposure_by_site[site_id][ParticleStatus.DATA_GAP.value]
                for site_id in self.site_ids
            }
        )

    @property
    def numerical_failure_exposure_by_site(self) -> Mapping[str, CountRatio]:
        """回傳每站數值失敗曝光比例及其 raw count／total 分母。"""

        return MappingProxyType(
            {
                site_id: self._failure_exposure_by_site[site_id][
                    ParticleStatus.NUMERICAL_FAILURE.value
                ]
                for site_id in self.site_ids
            }
        )

    @property
    def outcome_counts_by_site(self) -> Mapping[str, Mapping[str, int]]:
        """``outcome_count_by_site`` 的複數欄位別名。"""

        return self.outcome_count_by_site

    @property
    def total_denominator_by_site(self) -> Mapping[str, int]:
        """total member denominator 的簡短欄位別名。"""

        return self.total_member_denominator_by_site

    @property
    def total_member_count_by_site(self) -> Mapping[str, int]:
        """事件 aggregate 欄位名稱的 total member count 別名。"""

        return self.total_member_denominator_by_site

    @property
    def valid_member_count_by_site(self) -> Mapping[str, int]:
        """有效成員數的簡短欄位別名；資料缺口、數值失敗與 pre-window 已排除。"""

        return self.valid_member_denominator_by_site

    @property
    def ratio_by_site(self) -> Mapping[str, Mapping[str, CountRatio]]:
        """``outcome_ratios_by_site`` 的單數欄位別名。"""

        return self._outcome_ratios_by_site

    @property
    def data_gap_ratio_by_site(self) -> Mapping[str, CountRatio]:
        """資料缺口 exposure ratio 的欄位別名，仍保留 raw／total 分母。"""

        return self.data_gap_exposure_by_site

    @property
    def numerical_failure_ratio_by_site(self) -> Mapping[str, CountRatio]:
        """數值失敗 exposure ratio 的欄位別名，仍保留 raw／total 分母。"""

        return self.numerical_failure_exposure_by_site

    @property
    def data_gap_grid_by_site(self) -> Mapping[str, np.ndarray]:
        """F10 資料缺口 failure grid 的欄位別名；未提供時為空 mapping。"""

        return self.data_gap_failure_grid_by_site

    @property
    def numerical_grid_by_site(self) -> Mapping[str, np.ndarray]:
        """F10 數值失敗 grid 的簡短欄位別名。"""

        return self.numerical_failure_grid_by_site


# 與既有報告產品的命名慣例相容；alias 指向同一個 exact frozen class，不複製資料
# 或建立第二套欄位契約。這些名稱只改善 renderer／table adapter 的語意可讀性。
OutcomeStatisticsProduct = OutcomeStatistics


@dataclass(frozen=True, slots=True)
class ConnectivityStatistics:
    """封存方向性跨站連通矩陣與兩種互不相同的比例。

    ``site_ids`` 同時是 source 與 target axis；``raw_matrix[i, j]`` 代表由
    ``site_ids[i]`` 來源站點進入 ``site_ids[j]`` 目標站點的 unique-member raw
    count。矩陣 axis 不是空間 cell，因此不使用 ``(y_cell, x_cell)`` 命名；任何
    downstream spatial product 仍須保持自己的 ``(y_cell, x_cell[, age_bin])`` 軸。

    ``row_valid_member_denominator[i]`` 是來源站點有效成員數，決定
    ``visit_fraction[i, j]``；``row_event_denominator[i]`` 是該 row 所有非對角
    cross-site event raw count 合計，決定 ``cross_site_event_share[i, j]``。前者
    衡量有效成員中有多少跨到某目標，後者衡量跨站事件中有多少落在某目標；兩者
    不得混稱。對角線以 ``diagonal_not_applicable_mask`` 標示，比例填 NaN 只是
    payload sentinel，真正不適用語意來自 mask。
    """

    site_ids: tuple[str, ...]
    raw_matrix: np.ndarray
    valid_member_denominator_by_source_site: Mapping[str, int]
    visit_fraction: np.ndarray
    cross_site_event_share: np.ndarray
    diagonal_not_applicable_mask: np.ndarray
    row_event_denominator: np.ndarray
    minimum_count: int = 1
    visit_fraction_status: np.ndarray | None = None
    cross_site_event_share_status: np.ndarray | None = None
    a_zone_site_ids: tuple[str, str] | None = None
    _row_valid_member_denominator: np.ndarray = field(init=False, repr=False)
    _a_zone_indices: tuple[int, int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """驗證方向矩陣、列分母、兩種比例與不適用 mask 的 cross-field invariants。"""

        site_ids = _snapshot_site_ids(self.site_ids)
        site_count = len(site_ids)
        denominators = _snapshot_site_count_mapping(
            self.valid_member_denominator_by_source_site,
            site_ids=site_ids,
            label="valid_member_denominator_by_source_site",
        )
        raw_matrix = _readonly_array(
            self.raw_matrix,
            dtype=np.dtype(np.int64),
            ndim=2,
            label="raw_matrix",
        )
        expected_shape = (site_count, site_count)
        if raw_matrix.shape != expected_shape:
            raise ValueError(f"raw_matrix shape 必須是 {expected_shape}，軸為 source×target")
        if np.any(np.diag(raw_matrix) != 0):
            raise ValueError("raw_matrix 對角線只能是結構性零值，實際語意由 mask 表示")

        diagonal_mask = _readonly_array(
            self.diagonal_not_applicable_mask,
            dtype=np.dtype(np.bool_),
            ndim=2,
            label="diagonal_not_applicable_mask",
        )
        if diagonal_mask.shape != expected_shape or not np.array_equal(
            diagonal_mask,
            np.eye(site_count, dtype=bool),
        ):
            raise ValueError("diagonal_not_applicable_mask 必須精確標記矩陣對角線")

        minimum_count = _require_positive_native_int(
            self.minimum_count,
            label="minimum_count",
        )
        if self.a_zone_site_ids is None:
            a_zone_site_ids = None
            a_zone_indices = None
        else:
            a_zone_site_ids = _snapshot_site_ids(
                self.a_zone_site_ids,
                label="a_zone_site_ids",
            )
            if len(a_zone_site_ids) != 2:
                raise ValueError("a_zone_site_ids 必須精確包含兩個不同站點")
            a_zone_indices = (
                site_ids.index(a_zone_site_ids[0]),
                site_ids.index(a_zone_site_ids[1]),
            )
        row_denominator = np.array(
            [denominators[site_id] for site_id in site_ids],
            dtype=np.int64,
            copy=True,
        )
        row_denominator.setflags(write=False)

        event_denominator_values: list[int] = []
        for row_index, source_id in enumerate(site_ids):
            valid_denominator = denominators[source_id]
            for target_index in range(site_count):
                if row_index == target_index:
                    continue
                count = int(raw_matrix[row_index, target_index])
                if count > valid_denominator:
                    raise ValueError(
                        f"raw_matrix[{source_id!r}, {site_ids[target_index]!r}] 不可大於 "
                        "source site valid denominator"
                    )
            event_total = sum(
                int(raw_matrix[row_index, target_index])
                for target_index in range(site_count)
                if target_index != row_index
            )
            if event_total > _INT64_MAX:
                raise ValueError("row_event_denominator 不得超過 signed int64 上限")
            event_denominator_values.append(event_total)
        expected_event_denominator = np.array(event_denominator_values, dtype=np.int64)
        expected_event_denominator.setflags(write=False)
        supplied_event_denominator = _readonly_array(
            self.row_event_denominator,
            dtype=np.dtype(np.int64),
            ndim=1,
            label="row_event_denominator",
        )
        if supplied_event_denominator.shape != (site_count,) or not np.array_equal(
            supplied_event_denominator,
            expected_event_denominator,
        ):
            raise ValueError("row_event_denominator 必須等於每列非對角 raw count 合計")

        expected_visit = np.full(expected_shape, np.nan, dtype=np.float64)
        expected_share = np.full(expected_shape, np.nan, dtype=np.float64)
        expected_visit_status = np.empty(expected_shape, dtype=object)
        expected_share_status = np.empty(expected_shape, dtype=object)
        for row_index, source_id in enumerate(site_ids):
            valid_denominator = denominators[source_id]
            event_denominator = event_denominator_values[row_index]
            for target_index in range(site_count):
                if row_index == target_index:
                    expected_visit_status[row_index, target_index] = None
                    expected_share_status[row_index, target_index] = None
                    continue
                count = int(raw_matrix[row_index, target_index])
                if valid_denominator > 0:
                    expected_visit[row_index, target_index] = count / valid_denominator
                if event_denominator > 0:
                    expected_share[row_index, target_index] = count / event_denominator
                expected_visit_status[row_index, target_index] = _ratio_status(
                    count,
                    valid_denominator,
                    minimum_count=minimum_count,
                )
                expected_share_status[row_index, target_index] = _ratio_status(
                    count,
                    event_denominator,
                    minimum_count=minimum_count,
                )

        visit = _readonly_array(
            self.visit_fraction,
            dtype=np.dtype(np.float64),
            ndim=2,
            label="visit_fraction",
        )
        share = _readonly_array(
            self.cross_site_event_share,
            dtype=np.dtype(np.float64),
            ndim=2,
            label="cross_site_event_share",
        )
        if visit.shape != expected_shape or not np.array_equal(
            visit,
            expected_visit,
            equal_nan=True,
        ):
            raise ValueError("visit_fraction 必須精確等於 raw matrix 除以列有效分母")
        if share.shape != expected_shape or not np.array_equal(
            share,
            expected_share,
            equal_nan=True,
        ):
            raise ValueError("cross_site_event_share 必須精確等於 raw matrix 除以列內 event 分母")

        visit_status = _snapshot_status_grid(
            self.visit_fraction_status,
            expected=expected_visit_status,
            label="visit_fraction_status",
        )
        share_status = _snapshot_status_grid(
            self.cross_site_event_share_status,
            expected=expected_share_status,
            label="cross_site_event_share_status",
        )

        object.__setattr__(self, "site_ids", site_ids)
        object.__setattr__(self, "raw_matrix", raw_matrix)
        object.__setattr__(self, "valid_member_denominator_by_source_site", denominators)
        object.__setattr__(self, "visit_fraction", visit)
        object.__setattr__(self, "cross_site_event_share", share)
        object.__setattr__(self, "diagonal_not_applicable_mask", diagonal_mask)
        object.__setattr__(self, "row_event_denominator", expected_event_denominator)
        object.__setattr__(self, "minimum_count", minimum_count)
        object.__setattr__(self, "visit_fraction_status", visit_status)
        object.__setattr__(self, "cross_site_event_share_status", share_status)
        object.__setattr__(self, "a_zone_site_ids", a_zone_site_ids)
        object.__setattr__(self, "_row_valid_member_denominator", row_denominator)
        object.__setattr__(self, "_a_zone_indices", a_zone_indices)

    @property
    def source_site_ids(self) -> tuple[str, ...]:
        """回傳 source axis 的 site ID tuple。"""

        return self.site_ids

    @property
    def target_site_ids(self) -> tuple[str, ...]:
        """回傳 target axis 的 site ID tuple。"""

        return self.site_ids

    @property
    def matrix(self) -> np.ndarray:
        """raw source×target matrix 的簡短別名。"""

        return self.raw_matrix

    @property
    def raw_counts(self) -> np.ndarray:
        """raw cross-site unique-member count matrix 的複數欄位別名。"""

        return self.raw_matrix

    @property
    def row_valid_member_denominator(self) -> np.ndarray:
        """回傳依 source axis 排列的有效成員分母唯讀向量。"""

        return self._row_valid_member_denominator

    @property
    def valid_member_denominator_by_site(self) -> Mapping[str, int]:
        """``valid_member_denominator_by_source_site`` 的簡短別名。"""

        return self.valid_member_denominator_by_source_site

    @property
    def raw_count_matrix(self) -> np.ndarray:
        """``raw_matrix`` 的完整欄位名稱別名。"""

        return self.raw_matrix

    @property
    def valid_member_denominator_by_source(self) -> Mapping[str, int]:
        """source row 有效 member 分母的欄位名稱別名。"""

        return self.valid_member_denominator_by_source_site

    @property
    def source_row_denominator(self) -> np.ndarray:
        """``row_valid_member_denominator`` 的語意別名。"""

        return self._row_valid_member_denominator

    @property
    def row_event_denominator_by_source(self) -> np.ndarray:
        """依 source axis 排列的 cross-site event share 分母。"""

        return self.row_event_denominator

    @property
    def visit_fractions(self) -> np.ndarray:
        """逐目標 visit fraction 的複數欄位別名。"""

        return self.visit_fraction

    @property
    def event_share(self) -> np.ndarray:
        """cross-site event 內部 share 的簡短欄位別名。"""

        return self.cross_site_event_share

    @property
    def cross_site_event_internal_share(self) -> np.ndarray:
        """cross-site event 內部 share 的完整語意別名。"""

        return self.cross_site_event_share

    @property
    def diagonal_mask(self) -> np.ndarray:
        """對角線不適用遮罩的簡短別名。"""

        return self.diagonal_not_applicable_mask

    @property
    def visit_fraction_matrix(self) -> np.ndarray:
        """``visit_fraction`` 的矩陣欄位別名。"""

        return self.visit_fraction

    @property
    def event_share_matrix(self) -> np.ndarray:
        """``cross_site_event_share`` 的矩陣欄位別名。"""

        return self.cross_site_event_share

    def _a_zone_slice(self, value: np.ndarray, *, label: str) -> np.ndarray:
        """複製已登錄 A 區兩站的 2×2 子矩陣，不對未登錄站點猜測。"""

        if self._a_zone_indices is None:
            raise ValueError(f"{label} 只有在明示 a_zone_site_ids 時才可讀取")
        indices = self._a_zone_indices
        copied = np.array(value[np.ix_(indices, indices)], copy=True)
        copied.setflags(write=False)
        return copied

    @property
    def a_zone_2x2_raw_matrix(self) -> np.ndarray:
        """回傳明示 A 區兩站的方向性 2×2 raw matrix。"""

        return self._a_zone_slice(self.raw_matrix, label="a_zone_2x2_raw_matrix")

    @property
    def a_zone_raw_matrix(self) -> np.ndarray:
        """``a_zone_2x2_raw_matrix`` 的簡短別名。"""

        return self.a_zone_2x2_raw_matrix

    @property
    def a_zone_2x2_visit_fraction(self) -> np.ndarray:
        """回傳明示 A 區兩站的 2×2 visit fraction。"""

        return self._a_zone_slice(self.visit_fraction, label="a_zone_2x2_visit_fraction")

    @property
    def a_zone_2x2_event_share(self) -> np.ndarray:
        """回傳明示 A 區兩站的 2×2 cross-site event share。"""

        return self._a_zone_slice(self.cross_site_event_share, label="a_zone_2x2_event_share")

    @property
    def a_zone_2x2_diagonal_not_applicable_mask(self) -> np.ndarray:
        """回傳 A 區 2×2 對角線不適用遮罩。"""

        return self._a_zone_slice(
            self.diagonal_not_applicable_mask,
            label="a_zone_2x2_diagonal_not_applicable_mask",
        )

    @property
    def a_zone_row_valid_member_denominator(self) -> np.ndarray:
        """回傳 A 區 source row 對應的有效 member 分母。"""

        if self._a_zone_indices is None:
            raise ValueError("a_zone_row_valid_member_denominator 需要明示 a_zone_site_ids")
        copied = np.array(self._row_valid_member_denominator[list(self._a_zone_indices)], copy=True)
        copied.setflags(write=False)
        return copied


# 與 ``MaterialStatisticsProduct`` 等既有產品的命名慣例相容；不建立第二個資料型別。
ConnectivityStatisticsProduct = ConnectivityStatistics


def _resolve_payload_or_mapping(
    first: object | None,
    *,
    aggregate_payload: AggregateReleasePayload | None,
    payload: AggregateReleasePayload | None,
    label: str,
) -> tuple[AggregateReleasePayload | None, object | None]:
    """統一 positional／keyword payload 入口，避免同時指定兩份來源。"""

    candidates = [candidate for candidate in (aggregate_payload, payload) if candidate is not None]
    if len(candidates) > 1:
        raise ValueError("aggregate_payload 與 payload 不可同時提供")
    resolved_payload = candidates[0] if candidates else None
    if resolved_payload is not None:
        if type(resolved_payload) is not AggregateReleasePayload:
            raise TypeError(f"{label} 必須是 exact AggregateReleasePayload")
        if first is not None:
            raise ValueError("payload 已由 keyword 提供時不可再指定 positional data")
        return resolved_payload, None
    if type(first) is AggregateReleasePayload:
        return first, None
    return None, first


def build_outcome_statistics(
    outcome_count_by_site: AggregateReleasePayload | Mapping[str, Mapping[str, int]] | None = None,
    total_member_denominator_by_site: Mapping[str, int] | None = None,
    *,
    valid_member_denominator_by_site: Mapping[str, int] | None = None,
    total_member_count_by_site: Mapping[str, int] | None = None,
    aggregate_payload: AggregateReleasePayload | None = None,
    payload: AggregateReleasePayload | None = None,
    minimum_count: int = 1,
) -> OutcomeStatistics:
    """由已驗證 aggregate counts 建立停止結果與 failure exposure 產品。

    Args:
        outcome_count_by_site: 可直接傳 exact ``AggregateReleasePayload``；此時函式
            會使用其 event aggregate 的 outcome、total 與 valid 分母。若傳 mapping，
            其內層必須完整列出所有非 ``ACTIVE`` ``ParticleStatus`` 值及 raw count。
        total_member_denominator_by_site: 純 mapping 入口的每站 total member 分母。
            ``total_member_count_by_site`` 是相同欄位的明示別名，兩者不可同時提供。
        valid_member_denominator_by_site: 純 mapping 入口的有效成員分母；若省略，
            依 DATA_GAP、NUMERICAL_FAILURE 與 PRE_WINDOW_DEPOSITION raw count 建立預期值。
            payload 入口一定重用 aggregate 已保存的 valid denominator，並重新核對三類排除。
        minimum_count: ``CountRatio`` 使用的正整數 raw numerator 低樣本門檻。

    Returns:
        frozen ``OutcomeStatistics``；各狀態 ratio 都保留 raw numerator／denominator，
        零 total 分母為 ``None`` 加 ``zero_denominator``，不以零替代不可用值。

    Raises:
        TypeError／ValueError: payload、mapping、停止守恆、分母或門檻不符合契約時。
    """

    resolved_payload, raw_outcomes = _resolve_payload_or_mapping(
        outcome_count_by_site,
        aggregate_payload=aggregate_payload,
        payload=payload,
        label="outcome_count_by_site",
    )
    if total_member_count_by_site is not None and total_member_denominator_by_site is not None:
        raise ValueError("total_member_count_by_site 與 total_member_denominator_by_site 不可同時提供")
    if resolved_payload is not None:
        if any(
            value is not None
            for value in (
                total_member_denominator_by_site,
                total_member_count_by_site,
                valid_member_denominator_by_site,
            )
        ):
            raise ValueError("payload 入口不可混用手動 outcome／denominator mapping")
        event = resolved_payload.event_aggregate
        # 報告層所有 site axis 都以 AggregateSpec 的登錄順序為 canonical；event
        # mapping 雖已由 payload 驗證 site set，仍不應讓 caller 的 mapping insertion
        # order 使 OutcomeStatistics 與 ConnectivityStatistics 產生不同軸順序。
        site_ids = tuple(resolved_payload.aggregate_spec.site_grids)
        return OutcomeStatistics(
            site_ids=site_ids,
            outcome_count_by_site=event.outcome_count_by_site,
            total_member_denominator_by_site=event.total_member_count_by_site,
            valid_member_denominator_by_site=event.valid_member_denominator_by_site,
            data_gap_failure_grid_by_site={
                site_id: event.site_grid_counts[site_id].data_gap_failure_count
                for site_id in site_ids
            },
            numerical_failure_grid_by_site={
                site_id: event.site_grid_counts[site_id].numerical_failure_count
                for site_id in site_ids
            },
            minimum_count=minimum_count,
        )

    if raw_outcomes is None:
        raise TypeError("必須提供 AggregateReleasePayload 或 outcome_count_by_site mapping")
    if not isinstance(raw_outcomes, Mapping):
        raise TypeError("outcome_count_by_site 必須是 mapping")
    totals = total_member_count_by_site or total_member_denominator_by_site
    if totals is None:
        raise TypeError("純 mapping 入口必須提供 total member denominator")
    site_ids = tuple(raw_outcomes)
    return OutcomeStatistics(
        site_ids=site_ids,
        outcome_count_by_site=raw_outcomes,
        total_member_denominator_by_site=totals,
        valid_member_denominator_by_site=valid_member_denominator_by_site,
        minimum_count=minimum_count,
    )


def build_connectivity_statistics(
    cross_site_unique_member_count: AggregateReleasePayload
    | Mapping[CrossSiteAggregateKey, int]
    | None = None,
    valid_member_denominator_by_source_site: Mapping[str, int] | None = None,
    *,
    site_ids: Sequence[str] | None = None,
    valid_member_denominator_by_site: Mapping[str, int] | None = None,
    cross_site_counts: Mapping[CrossSiteAggregateKey, int] | None = None,
    aggregate_payload: AggregateReleasePayload | None = None,
    payload: AggregateReleasePayload | None = None,
    minimum_count: int = 1,
    a_zone_site_ids: Sequence[str] | None = None,
) -> ConnectivityStatistics:
    """由有向 cross-site raw count 建立連通矩陣產品。

    Args:
        cross_site_unique_member_count: 可直接傳 exact ``AggregateReleasePayload``，
            或傳 ``CrossSiteAggregateKey`` 到 raw unique-member count 的完整 mapping。
            mapping 入口不得省略零事件的 ordered distinct pair；缺列與零值不是同一
            語意。``raw_matrix[i, j]`` 的方向永遠是 source ``site_ids[i]`` 到
            target ``site_ids[j]``。
        valid_member_denominator_by_source_site: 純 mapping 入口的 source row 有效
            成員分母；``valid_member_denominator_by_site`` 是同一欄位的明示別名。
        site_ids: 純 mapping 入口的完整矩陣 axis，順序會被保存。payload 入口使用
            AggregateSpec 的 site mapping，不接受 caller 另造 axis。
        minimum_count: 兩種比例共用的 raw numerator 低樣本門檻。
        a_zone_site_ids: optional、明示的 A 區兩站 site ID 順序。提供後產品會額外
            暴露方向性 2×2 raw／比例子矩陣；省略時不會猜測哪兩站屬於 A 區。

    Returns:
        frozen ``ConnectivityStatistics``，包含 raw matrix、列分母、逐目標 visit
        fraction、cross-site event 內部 share、兩組 status 與 diagonal mask。

    Raises:
        TypeError／ValueError: key 拓撲、分母、raw count、方向、守恆或 shape 不合法時。
    """

    resolved_payload, raw_counts = _resolve_payload_or_mapping(
        cross_site_unique_member_count,
        aggregate_payload=aggregate_payload,
        payload=payload,
        label="cross_site_unique_member_count",
    )
    if valid_member_denominator_by_site is not None and valid_member_denominator_by_source_site is not None:
        raise ValueError(
            "valid_member_denominator_by_site 與 valid_member_denominator_by_source_site 不可同時提供"
        )
    if cross_site_counts is not None and raw_counts is not None:
        raise ValueError("cross_site_counts 與 positional cross-site mapping 不可同時提供")
    if resolved_payload is not None:
        if any(
            value is not None
            for value in (
                valid_member_denominator_by_source_site,
                valid_member_denominator_by_site,
                site_ids,
                cross_site_counts,
            )
        ):
            raise ValueError("payload 入口不可混用手動 site／denominator／count mapping")
        event = resolved_payload.event_aggregate
        resolved_site_ids = tuple(resolved_payload.aggregate_spec.site_grids)
        resolved_counts = event.cross_site_unique_member_count
        resolved_denominators = event.valid_member_denominator_by_site
    else:
        resolved_counts = cross_site_counts if cross_site_counts is not None else raw_counts
        if resolved_counts is None:
            raise TypeError("必須提供 AggregateReleasePayload 或 cross-site count mapping")
        resolved_denominators = (
            valid_member_denominator_by_site
            if valid_member_denominator_by_site is not None
            else valid_member_denominator_by_source_site
        )
        if resolved_denominators is None:
            raise TypeError("純 mapping 入口必須提供 source row valid denominator")
        if site_ids is None:
            if not isinstance(resolved_denominators, Mapping):
                raise TypeError("site_ids 省略時 denominator 必須是 mapping")
            site_ids = tuple(resolved_denominators)
        resolved_site_ids = _snapshot_site_ids(site_ids)

    normalized_site_ids = _snapshot_site_ids(resolved_site_ids)
    counts = _snapshot_count_mapping(
        resolved_counts,
        key_type=CrossSiteAggregateKey,
        label="cross_site_unique_member_count",
    )
    denominators = _snapshot_site_count_mapping(
        resolved_denominators,
        site_ids=normalized_site_ids,
        label="valid_member_denominator_by_source_site",
    )
    expected_keys = {
        CrossSiteAggregateKey(source, target)
        for source in normalized_site_ids
        for target in normalized_site_ids
        if source != target
    }
    if set(counts) != expected_keys:
        raise ValueError(
            "cross_site_unique_member_count 的 key set 必須完整覆蓋所有 ordered distinct site pairs"
        )

    site_index = {site_id: index for index, site_id in enumerate(normalized_site_ids)}
    matrix = np.zeros((len(normalized_site_ids), len(normalized_site_ids)), dtype=np.int64)
    for key, count in counts.items():
        source_index = site_index[key.source_study_site_id]
        target_index = site_index[key.target_study_site_id]
        matrix[source_index, target_index] = count
    matrix.setflags(write=False)

    row_event_denominator_values: list[int] = []
    for row_index in range(len(normalized_site_ids)):
        event_total = sum(
            int(matrix[row_index, target_index])
            for target_index in range(len(normalized_site_ids))
            if target_index != row_index
        )
        if event_total > _INT64_MAX:
            raise ValueError("row_event_denominator 不得超過 signed int64 上限")
        row_event_denominator_values.append(event_total)
    row_event_denominator = np.array(row_event_denominator_values, dtype=np.int64)
    visit = np.full(matrix.shape, np.nan, dtype=np.float64)
    share = np.full(matrix.shape, np.nan, dtype=np.float64)
    diagonal_mask = np.eye(len(normalized_site_ids), dtype=bool)
    visit_status = np.empty(matrix.shape, dtype=object)
    share_status = np.empty(matrix.shape, dtype=object)
    for row_index, source_id in enumerate(normalized_site_ids):
        valid_denominator = denominators[source_id]
        event_denominator = int(row_event_denominator[row_index])
        for target_index in range(len(normalized_site_ids)):
            if row_index == target_index:
                visit_status[row_index, target_index] = None
                share_status[row_index, target_index] = None
                continue
            count = int(matrix[row_index, target_index])
            if valid_denominator > 0:
                visit[row_index, target_index] = count / valid_denominator
            if event_denominator > 0:
                share[row_index, target_index] = count / event_denominator
            visit_status[row_index, target_index] = _ratio_status(
                count,
                valid_denominator,
                minimum_count=_require_positive_native_int(minimum_count, label="minimum_count"),
            )
            share_status[row_index, target_index] = _ratio_status(
                count,
                event_denominator,
                minimum_count=minimum_count,
            )

    return ConnectivityStatistics(
        site_ids=normalized_site_ids,
        raw_matrix=matrix,
        valid_member_denominator_by_source_site=denominators,
        visit_fraction=visit,
        cross_site_event_share=share,
        diagonal_not_applicable_mask=diagonal_mask,
        row_event_denominator=row_event_denominator,
        minimum_count=minimum_count,
        visit_fraction_status=visit_status,
        cross_site_event_share_status=share_status,
        a_zone_site_ids=a_zone_site_ids,
    )
