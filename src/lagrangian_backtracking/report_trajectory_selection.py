"""F03 報告代表軌跡的固定容量、八層分層雜湊選樣器。

本模組只為 F03 代表軌跡圖保存可重建的小型軌跡子集，不把入選頻率、雜湊順位或圖面
案例解讀成絕對來源機率、相對來源權重或因果歸因。候選先沿用報告有效成員分母政策：
``DATA_GAP`` 與 ``NUMERICAL_FAILURE`` 只累計排除數，尚未終止的 ``ACTIVE`` 結果立即
失敗；有效成員若不是 ``DJF/MAM/JJA/SON × spring_proxy/neap_proxy`` 核心八層，則視為
event arrival 排除，不能混入八層配額。

每站的代表軌跡總數 ``K`` 由 exact ``ReportSpec`` 決定，八層各保留 ``K/8`` 筆最小
SHA-256 priority digest。selector 只保存各層目前入選的固定容量排序 list、八層 eligible
計數及兩種排除計數，不保存未入選 ``ParticleResult``，因此軌跡記憶體上限是
``O(site_count × K)``，不隨已串流候選總數成長。成功 finalize 後，所有 sequence 都轉成
tuple，所有內外 mapping 都做 defensive copy 並包成 ``MappingProxyType``，供 renderer
只讀使用。為維持這個上限，selector 只對目前 retained bucket 執行重複 identity 防線；
正式 caller 必須先由後續 validated trajectory stream 保證全 run identity 唯一，不能要求
selector 另存隨候選數量成長的全量 seen set。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from .engine import ParticleResult
from .report_spec import ReportSpec
from .report_trajectory_identity import (
    CORE_SEASONS,
    CORE_TIDE_CLASSES,
    REPRESENTATIVE_SELECTION_POLICY,
    RepresentativeTrajectory,
    is_valid_report_member,
    representative_priority_digest,
)

__all__ = [
    "RepresentativeSelection",
    "RepresentativeTrajectorySelector",
]


StratumKey = tuple[str, str]
_CORE_STRATA: Final[tuple[StratumKey, ...]] = tuple(
    (season, tide_class)
    for season in CORE_SEASONS
    for tide_class in CORE_TIDE_CLASSES
)
_CORE_STRATA_SET: Final[frozenset[StratumKey]] = frozenset(_CORE_STRATA)
_EVENT_TIDE_CLASS: Final[str] = "event"
_MAX_UINT128: Final[int] = 2**128 - 1


def _require_text(value: object, *, label: str) -> str:
    """要求 identity／站點欄位是原生、非空且沒有首尾空白的 ``str``。

    selector 的文字會直接參與站點路由、report strata key 或 canonical digest；不做 trim
    與隱式轉型，才能避免同一視覺名稱在不同 shard 被分成兩個實際 identity。
    """

    if type(value) is not str:
        raise TypeError(f"{label} 必須是原生 str")
    if not value or value != value.strip():
        raise ValueError(f"{label} 必須是非空且沒有首尾空白的文字")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    """要求計數是非負原生 Python ``int``，拒絕 bool 與 NumPy scalar。"""

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 int，且不可是 bool 或 NumPy scalar")
    if value < 0:
        raise ValueError(f"{label} 必須是非負整數")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    """要求容量是大於零的原生 Python ``int``。"""

    number = _require_nonnegative_int(value, label=label)
    if number == 0:
        raise ValueError(f"{label} 必須大於 0")
    return number


def _require_selection_policy(value: object) -> str:
    """要求快照與 selector 使用目前唯一受支援的原生 policy 文字。"""

    if type(value) is not str:
        raise TypeError("selection_policy 必須是原生 str")
    if value != REPRESENTATIVE_SELECTION_POLICY:
        raise ValueError("selection_policy 不受支援")
    return value


def _require_selection_seed(value: object) -> int:
    """要求 selection seed 是跨語言摘要契約允許的原生 128 位元無號整數。"""

    if type(value) is not int:
        raise TypeError("selection_seed 必須是原生 int，且不可是 bool 或 NumPy scalar")
    if not 0 <= value <= _MAX_UINT128:
        raise ValueError("selection_seed 超出 128 位元無號整數範圍")
    return value


def _snapshot_site_ids(value: object) -> tuple[str, ...]:
    """複製、驗證並排序站點 ID，拒絕空集合、字串容器與重複值。

    輸入順序不是科學契約；canonical site order 以字典序固定，使不同 shard caller 即使
    以不同順序提供相同站點，也會產生一致的 finalize mapping 順序。
    """

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("site_ids 必須是非字串、非 mapping 的 iterable")
    try:
        raw_site_ids = tuple(iter(value))  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("site_ids 必須是 iterable") from error
    if not raw_site_ids:
        raise ValueError("site_ids 不可為空")
    site_ids = tuple(
        _require_text(site_id, label=f"site_ids[{index}]")
        for index, site_id in enumerate(raw_site_ids)
    )
    if len(set(site_ids)) != len(site_ids):
        raise ValueError("site_ids 不可重複")
    return tuple(sorted(site_ids))


def _mapping_items(value: object, *, label: str) -> tuple[tuple[object, object], ...]:
    """複製 mapping items，避免後續驗證持有 caller 的可變 view。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必須是 mapping")
    try:
        return tuple(value.items())
    except Exception as error:
        raise ValueError(f"{label} 無法讀取 mapping") from error


def _require_stratum_key(value: object, *, label: str) -> StratumKey:
    """要求 mapping key 是 exact 二元素 tuple，且精確屬於核心八層。"""

    if type(value) is not tuple:
        raise TypeError(f"{label} 必須是 exact tuple")
    if len(value) != 2:
        raise ValueError(f"{label} 必須是 (season, tide_class) 二元素 tuple")
    season = _require_text(value[0], label=f"{label}.season")
    tide_class = _require_text(value[1], label=f"{label}.tide_class")
    key = (season, tide_class)
    if key not in _CORE_STRATA_SET:
        raise ValueError(f"{label} 不屬於核心 season/tide 八層")
    return key


def _trajectory_identity(value: RepresentativeTrajectory) -> tuple[str | int, ...]:
    """提取 digest 必須唯一綁定的十欄完整粒子與 report strata identity。"""

    return (
        value.particle_id,
        value.scenario_id,
        value.member_id,
        value.study_site_id,
        value.analysis_region_id,
        value.receptor_id,
        value.material_id,
        value.arrival_time_id,
        value.season,
        value.tide_class,
    )


def _snapshot_trajectory_tuple(
    value: object,
    *,
    site_id: str,
    stratum: StratumKey,
    capacity: int,
    selection_policy: str,
    selection_seed: int,
) -> tuple[RepresentativeTrajectory, ...]:
    """複製單層軌跡，並依快照 provenance 重算摘要及驗證嚴格遞增。"""

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("selected stratum 必須是 RepresentativeTrajectory iterable")
    try:
        records = tuple(iter(value))  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("selected stratum 必須是 RepresentativeTrajectory iterable") from error
    if len(records) != capacity:
        raise ValueError("selected 每個 stratum 必須精確等於 capacity_per_stratum")

    previous_digest: str | None = None
    for index, record in enumerate(records):
        if type(record) is not RepresentativeTrajectory:
            raise TypeError(f"selected stratum[{index}] 必須是 exact RepresentativeTrajectory")
        if record.study_site_id != site_id:
            raise ValueError("selected trajectory 的 study_site_id 與 outer site key 不一致")
        if (record.season, record.tide_class) != stratum:
            raise ValueError("selected trajectory 的 season/tide 與 stratum key 不一致")
        # policy 已在快照入口精確驗證；現行 digest helper 以該版本常數建立 canonical
        # payload。這裡仍接收 policy，明示 record 驗證是由快照自身 provenance 驅動，
        # 而不是信任 selector 曾經算過的值。
        if selection_policy != REPRESENTATIVE_SELECTION_POLICY:
            raise ValueError("selected trajectory 的 selection policy 不受支援")
        expected_digest = representative_priority_digest(
            selection_seed,
            record.particle_id,
            record.scenario_id,
            record.member_id,
            record.study_site_id,
            record.analysis_region_id,
            record.receptor_id,
            record.material_id,
            record.arrival_time_id,
            record.season,
            record.tide_class,
        )
        if record.priority_digest != expected_digest:
            raise ValueError("selected trajectory digest 與 selection provenance／identity 不一致")
        if previous_digest is not None and record.priority_digest <= previous_digest:
            raise ValueError("selected trajectory digest 必須嚴格遞增且不可重複")
        previous_digest = record.priority_digest
    return records


def _snapshot_selected(
    value: object,
    *,
    site_ids: tuple[str, ...],
    capacity: int,
    selection_policy: str,
    selection_seed: int,
) -> Mapping[str, Mapping[StratumKey, tuple[RepresentativeTrajectory, ...]]]:
    """建立雙層唯讀快照，並以快照 policy/seed 驗證每筆 trajectory digest。"""

    outer_items = _mapping_items(value, label="selected")
    outer: dict[str, object] = {}
    for raw_site_id, inner_value in outer_items:
        site_id = _require_text(raw_site_id, label="selected site key")
        if site_id in outer:
            raise ValueError("selected site key 不可重複")
        outer[site_id] = inner_value
    if set(outer) != set(site_ids):
        raise ValueError("selected site key 必須精確等於 site_ids")

    copied_outer: dict[str, Mapping[StratumKey, tuple[RepresentativeTrajectory, ...]]] = {}
    for site_id in site_ids:
        inner_items = _mapping_items(outer[site_id], label=f"selected[{site_id}]")
        copied_inner: dict[StratumKey, tuple[RepresentativeTrajectory, ...]] = {}
        for raw_key, raw_records in inner_items:
            key = _require_stratum_key(raw_key, label=f"selected[{site_id}] stratum key")
            if key in copied_inner:
                raise ValueError("selected stratum key 不可重複")
            copied_inner[key] = _snapshot_trajectory_tuple(
                raw_records,
                site_id=site_id,
                stratum=key,
                capacity=capacity,
                selection_policy=selection_policy,
                selection_seed=selection_seed,
            )
        if set(copied_inner) != _CORE_STRATA_SET:
            raise ValueError("selected 每站必須精確包含核心八層")
        ordered_inner = {key: copied_inner[key] for key in _CORE_STRATA}
        copied_outer[site_id] = MappingProxyType(ordered_inner)
    return MappingProxyType(copied_outer)


def _snapshot_eligible_counts(
    value: object,
    *,
    site_ids: tuple[str, ...],
    capacity: int,
) -> Mapping[str, Mapping[StratumKey, int]]:
    """建立每站八層 eligible 計數的雙層唯讀 defensive snapshot。"""

    outer_items = _mapping_items(value, label="eligible_count_by_site_stratum")
    outer: dict[str, object] = {}
    for raw_site_id, inner_value in outer_items:
        site_id = _require_text(raw_site_id, label="eligible count site key")
        if site_id in outer:
            raise ValueError("eligible count site key 不可重複")
        outer[site_id] = inner_value
    if set(outer) != set(site_ids):
        raise ValueError("eligible count site key 必須精確等於 site_ids")

    copied_outer: dict[str, Mapping[StratumKey, int]] = {}
    for site_id in site_ids:
        inner_items = _mapping_items(outer[site_id], label=f"eligible[{site_id}]")
        copied_inner: dict[StratumKey, int] = {}
        for raw_key, raw_count in inner_items:
            key = _require_stratum_key(raw_key, label=f"eligible[{site_id}] stratum key")
            if key in copied_inner:
                raise ValueError("eligible stratum key 不可重複")
            count = _require_nonnegative_int(raw_count, label="eligible count")
            if count < capacity:
                raise ValueError("成功 selection 的每層 eligible count 不可小於容量")
            copied_inner[key] = count
        if set(copied_inner) != _CORE_STRATA_SET:
            raise ValueError("eligible count 每站必須精確包含核心八層")
        ordered_inner = {key: copied_inner[key] for key in _CORE_STRATA}
        copied_outer[site_id] = MappingProxyType(ordered_inner)
    return MappingProxyType(copied_outer)


def _snapshot_site_counts(
    value: object,
    *,
    site_ids: tuple[str, ...],
    label: str,
) -> Mapping[str, int]:
    """建立站點排除計數的唯讀 defensive mapping，要求 exact key closure。"""

    items = _mapping_items(value, label=label)
    copied: dict[str, int] = {}
    for raw_site_id, raw_count in items:
        site_id = _require_text(raw_site_id, label=f"{label} site key")
        if site_id in copied:
            raise ValueError(f"{label} site key 不可重複")
        copied[site_id] = _require_nonnegative_int(raw_count, label=f"{label}[{site_id}]")
    if set(copied) != set(site_ids):
        raise ValueError(f"{label} site key 必須精確等於 site_ids")
    return MappingProxyType({site_id: copied[site_id] for site_id in site_ids})


@dataclass(frozen=True, slots=True)
class RepresentativeSelection:
    """成功完成的 F03 代表軌跡不可變快照。

    ``selection_policy`` 與 ``selection_seed`` 是 release 可自行稽核的選樣 provenance；
    建構快照時會用它們與每筆 trajectory 的十欄 identity 重算 SHA-256，不信任 selector
    內部歷史或 caller 提供的摘要。``selected`` 與 ``eligible_count_by_site_stratum`` 精確包含每站的八個核心
    ``(season, tide_class)`` tuple key。每層 selected tuple 長度必須等於
    ``capacity_per_stratum``，並依完整 priority digest 嚴格遞增；eligible count 則保存
    所有通過狀態、identity 與至少兩筆 observation 閘門的核心候選數，不因 top-K 淘汰而
    減少。兩種 site-level 排除計數分別記錄 event arrival 與資料缺口／數值失敗成員，
    不會被混入有效分母。

    所有外層與內層 mapping 都會複製後包成 ``MappingProxyType``，trajectory sequence
    複製成 tuple；因此 caller 修改原 dict/list 或嘗試經由結果改值，都不會污染已完成
    selection。這份資料只供代表圖與稽核，不是機率估計產品。
    """

    site_ids: tuple[str, ...]
    capacity_per_stratum: int
    selection_policy: str
    selection_seed: int
    selected: Mapping[str, Mapping[StratumKey, tuple[RepresentativeTrajectory, ...]]]
    eligible_count_by_site_stratum: Mapping[str, Mapping[StratumKey, int]]
    excluded_event_arrival_count_by_site: Mapping[str, int]
    excluded_invalid_member_count_by_site: Mapping[str, int]

    def __post_init__(self) -> None:
        """驗證完整八層閉包、計數與排序，並封存全部 nested containers。"""

        site_ids = _snapshot_site_ids(self.site_ids)
        capacity = _require_positive_int(
            self.capacity_per_stratum,
            label="capacity_per_stratum",
        )
        selection_policy = _require_selection_policy(self.selection_policy)
        selection_seed = _require_selection_seed(self.selection_seed)
        selected = _snapshot_selected(
            self.selected,
            site_ids=site_ids,
            capacity=capacity,
            selection_policy=selection_policy,
            selection_seed=selection_seed,
        )
        eligible = _snapshot_eligible_counts(
            self.eligible_count_by_site_stratum,
            site_ids=site_ids,
            capacity=capacity,
        )
        excluded_event = _snapshot_site_counts(
            self.excluded_event_arrival_count_by_site,
            site_ids=site_ids,
            label="excluded_event_arrival_count_by_site",
        )
        excluded_invalid = _snapshot_site_counts(
            self.excluded_invalid_member_count_by_site,
            site_ids=site_ids,
            label="excluded_invalid_member_count_by_site",
        )

        object.__setattr__(self, "site_ids", site_ids)
        object.__setattr__(self, "capacity_per_stratum", capacity)
        object.__setattr__(self, "selection_policy", selection_policy)
        object.__setattr__(self, "selection_seed", selection_seed)
        object.__setattr__(self, "selected", selected)
        object.__setattr__(self, "eligible_count_by_site_stratum", eligible)
        object.__setattr__(self, "excluded_event_arrival_count_by_site", excluded_event)
        object.__setattr__(self, "excluded_invalid_member_count_by_site", excluded_invalid)


class RepresentativeTrajectorySelector:
    """逐筆串流保留每站八層最小 SHA-256 priority 的固定容量 selector。

    Args:
        report_spec: 必須是 exact ``ReportSpec``，policy 必須精確等於
            ``REPRESENTATIVE_SELECTION_POLICY``；每站總容量 ``K`` 至少 8 且可被 8
            整除，故每個 season/tide stratum 容量固定為 ``K/8``。
        site_ids: 非空、原生、唯一的站點 ID iterable。constructor 會 defensive-copy 並
            排序，輸入順序不影響結果。

    每次 ``add`` 只在 candidate 進入目前 top-K 時保存 immutable
    ``RepresentativeTrajectory``；未入選 ``ParticleResult`` 不會被保留。除了固定八層
    list 外，只維護整數計數，因此 RAM 為 ``O(site_count × K)``。成功 ``finalize`` 後
    selector 封閉，重複 finalize 回傳同一個 immutable snapshot，任何後續 add 都拒絕。
    全 run identity 唯一性由正式 validated trajectory stream 保證；本 selector 只掃描
    retained bucket 拒絕重複或 digest 衝突，作為不破壞固定 RAM 上限的額外防線。
    """

    def __init__(self, report_spec: ReportSpec, site_ids: object) -> None:
        """驗證 exact spec、policy、容量與站點後建立空的固定容量八層 reservoir。"""

        if type(report_spec) is not ReportSpec:
            raise TypeError("report_spec 必須是 exact ReportSpec")
        selection_policy = _require_selection_policy(
            report_spec.representative_selection_policy
        )
        total_capacity = report_spec.representative_trajectory_count_per_site
        if type(total_capacity) is not int:
            raise TypeError("representative_trajectory_count_per_site 必須是原生 int")
        if total_capacity < 8 or total_capacity % 8 != 0:
            raise ValueError("representative_trajectory_count_per_site 必須至少 8 且可被 8 整除")
        selection_seed = _require_selection_seed(report_spec.representative_selection_seed)

        normalized_site_ids = _snapshot_site_ids(site_ids)
        self._site_ids = normalized_site_ids
        self._site_id_set = frozenset(normalized_site_ids)
        self._capacity_per_stratum = total_capacity // len(_CORE_STRATA)
        self._selection_policy = selection_policy
        self._selection_seed = selection_seed
        self._selected: dict[str, dict[StratumKey, list[RepresentativeTrajectory]]] = {
            site_id: {key: [] for key in _CORE_STRATA}
            for site_id in normalized_site_ids
        }
        self._eligible_count: dict[str, dict[StratumKey, int]] = {
            site_id: {key: 0 for key in _CORE_STRATA}
            for site_id in normalized_site_ids
        }
        self._excluded_event_count = {site_id: 0 for site_id in normalized_site_ids}
        self._excluded_invalid_count = {site_id: 0 for site_id in normalized_site_ids}
        self._finalized_selection: RepresentativeSelection | None = None

    @property
    def site_ids(self) -> tuple[str, ...]:
        """回傳已排序且不可變的站點 ID。"""

        return self._site_ids

    @property
    def capacity_per_stratum(self) -> int:
        """回傳每站每個核心 season/tide stratum 的固定保留容量。"""

        return self._capacity_per_stratum

    @property
    def retained_count(self) -> int:
        """回傳目前實際保存的軌跡數，用於確認 RAM 不隨候選總數成長。"""

        return sum(
            len(bucket)
            for site_buckets in self._selected.values()
            for bucket in site_buckets.values()
        )

    def _registered_site_id(self, result: ParticleResult) -> str:
        """由已通過 exact result gate 的 final state 取得並驗證已登錄站點。"""

        site_id = _require_text(
            result.final_state.study_site_id,
            label="result.final_state.study_site_id",
        )
        if site_id not in self._site_id_set:
            raise ValueError("ParticleResult 的 study_site_id 未在 selector 登錄")
        return site_id

    def _check_retained_digest_conflict(
        self,
        bucket: list[RepresentativeTrajectory],
        candidate: RepresentativeTrajectory,
    ) -> None:
        """對 retained bucket 拒絕完整重複、identity 衝突及 SHA-256 碰撞。

        selector 不保存未入選候選，否則記憶體會隨 N 成長；因此 exact duplicate gate 的
        可稽核範圍是目前 retained top-K。相同 digest 若 identity 不同，無論是否由測試
        注入或實際碰撞，都立即 fail closed；相同 identity 的重複或 observation 衝突也
        不會重複計入 eligible counter。
        """

        for retained in bucket:
            if retained.priority_digest != candidate.priority_digest:
                continue
            if _trajectory_identity(retained) != _trajectory_identity(candidate):
                raise ValueError("retained priority digest 對應到不同完整 identity")
            if retained == candidate:
                raise ValueError("同一代表軌跡不可重複 add")
            raise ValueError("相同代表軌跡 identity 的 observation payload 不一致")

    def _retain_candidate(
        self,
        bucket: list[RepresentativeTrajectory],
        candidate: RepresentativeTrajectory,
    ) -> None:
        """依 digest 升冪插入候選，並在超過容量時移除目前最大 digest。"""

        insertion_index = 0
        while (
            insertion_index < len(bucket)
            and bucket[insertion_index].priority_digest < candidate.priority_digest
        ):
            insertion_index += 1

        if len(bucket) < self._capacity_per_stratum:
            bucket.insert(insertion_index, candidate)
            return
        if insertion_index < self._capacity_per_stratum:
            bucket.insert(insertion_index, candidate)
            bucket.pop()

    def add(
        self,
        result: ParticleResult,
        *,
        material_id: str,
        arrival_time_id: str,
        season: str,
        tide_class: str,
    ) -> None:
        """串流處理一筆粒子結果，更新排除計數或單一核心層 top-K。

        ``is_valid_report_member`` 先執行 exact ``ParticleResult`` 與終止狀態政策：
        ``DATA_GAP``／``NUMERICAL_FAILURE`` 只增加 invalid counter；``ACTIVE`` 直接拋錯。
        有效結果的四個 report strata identity 必須是原生非空文字。event arrival 只接受
        核心 season 搭配精確 ``event`` tide，並只增加 event counter；核心 tide 也必須搭配
        核心 season 才能選樣。任何其他 season/tide 標籤都視為分類契約錯誤而 fail closed，
        不會因拼字錯誤污染 event 排除數。核心候選會建立包含完整十欄 identity 的 digest
        與 defensive ``RepresentativeTrajectory``，少於兩筆或非 exact observation 會在
        任何 eligible 計數變更前失敗。

        每層只保存目前最小的固定數量 digest；大於 cutoff 的結果完成驗證與 eligible
        計數後立即釋放，不保存原始 ``ParticleResult``。成功 finalize 後不可再呼叫。
        """

        if self._finalized_selection is not None:
            raise RuntimeError("selector finalize 後不可再 add")

        valid_member = is_valid_report_member(result)
        site_id = self._registered_site_id(result)
        if not valid_member:
            self._excluded_invalid_count[site_id] += 1
            return

        normalized_material = _require_text(material_id, label="material_id")
        normalized_arrival = _require_text(arrival_time_id, label="arrival_time_id")
        normalized_season = _require_text(season, label="season")
        normalized_tide = _require_text(tide_class, label="tide_class")
        if normalized_tide == _EVENT_TIDE_CLASS:
            if normalized_season not in CORE_SEASONS:
                raise ValueError("event arrival 的 season 必須精確屬於核心四季")
            self._excluded_event_count[site_id] += 1
            return
        if normalized_tide not in CORE_TIDE_CLASSES:
            raise ValueError("tide_class 必須是核心 tide 或精確的 event")
        if normalized_season not in CORE_SEASONS:
            raise ValueError("核心 tide 的 season 必須精確屬於核心四季")

        state = result.final_state
        digest = representative_priority_digest(
            self._selection_seed,
            state.particle_id,
            state.scenario_id,
            state.member_id,
            state.study_site_id,
            state.analysis_region_id,
            state.receptor_id,
            normalized_material,
            normalized_arrival,
            normalized_season,
            normalized_tide,
        )
        candidate = RepresentativeTrajectory(
            particle_id=state.particle_id,
            scenario_id=state.scenario_id,
            member_id=state.member_id,
            study_site_id=state.study_site_id,
            analysis_region_id=state.analysis_region_id,
            receptor_id=state.receptor_id,
            material_id=normalized_material,
            arrival_time_id=normalized_arrival,
            season=normalized_season,
            tide_class=normalized_tide,
            priority_digest=digest,
            observations=result.observations,
        )

        key = (normalized_season, normalized_tide)
        bucket = self._selected[site_id][key]
        self._check_retained_digest_conflict(bucket, candidate)
        self._eligible_count[site_id][key] += 1
        self._retain_candidate(bucket, candidate)

    def finalize(self) -> RepresentativeSelection:
        """驗證每站八層配額完整後回傳同一份 immutable selection snapshot。

        每層 eligible count 必須至少等於容量，retained list 必須精確等於容量；任一站點
        或 strata 不足即 fail closed，不建立部分結果，也不封閉 selector，caller 可補入
        缺少的合法候選後重試。首次成功後保存 snapshot，後續 finalize 直接回傳同一物件。
        """

        if self._finalized_selection is not None:
            return self._finalized_selection

        for site_id in self._site_ids:
            for key in _CORE_STRATA:
                eligible_count = self._eligible_count[site_id][key]
                retained_count = len(self._selected[site_id][key])
                if eligible_count < self._capacity_per_stratum:
                    raise ValueError(
                        f"代表軌跡 strata 候選不足：site={site_id}, stratum={key}"
                    )
                if retained_count != self._capacity_per_stratum:
                    raise ValueError(
                        f"代表軌跡 retained 容量不符：site={site_id}, stratum={key}"
                    )

        selection = RepresentativeSelection(
            site_ids=self._site_ids,
            capacity_per_stratum=self._capacity_per_stratum,
            selection_policy=self._selection_policy,
            selection_seed=self._selection_seed,
            selected=self._selected,
            eligible_count_by_site_stratum=self._eligible_count,
            excluded_event_arrival_count_by_site=self._excluded_event_count,
            excluded_invalid_member_count_by_site=self._excluded_invalid_count,
        )
        self._finalized_selection = selection
        return selection
