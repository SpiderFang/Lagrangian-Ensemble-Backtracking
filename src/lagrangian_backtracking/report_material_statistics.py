"""建立沉底海廢材質的 member-level 報告統計。

本模組是報告層的純計算邊界：它不讀取檔案、不修改 ``ParticleResult``，也不把
iOcean 清除統計或主管的定性關注轉成速度、數量權重或來源先驗。呼叫端提供已通過
資料契約的 scenario strata 與一條只遍歷一次的 ``ParticleResult`` iterable；模組依
``study_site_id × material_id`` 建立固定拓撲，排除 ``DATA_GAP``／
``NUMERICAL_FAILURE`` 後，統計每個有效系集成員是否曾發生海床接觸，以及是否為沉積
終止結果。

事件資料的物理語意如下：``BED_CONTACT`` 是可重複出現的海床接觸診斷，
``DEPOSITED`` 是終止沉積事件。第一項統計只保留每個 member 的布林結果，故同一成員
有多個接觸事件仍只計一次；基線 ``deposit_on_first_contact_and_stop`` 不需要提供
repeated-contact 欄位，也不把 repeated contact 當成核心分母或分子。沉積統計以
``ParticleStatus.DEPOSITED`` 或 ``DEPOSITED`` event 證明，並同樣按 member 去重。

輸出是 frozen dataclass 加上唯讀 mapping；比例在有效分母為零時使用 ``None``，而非
零值。這些數值只能解讀為指定情境下的條件式來源足跡／相對來源權重，不能解讀為絕對
來源機率或因果歸因。所有位置單位、公尺垂向方向與 Observation 環境脈絡沿用既有
``engine.py`` 資料契約；本最小產品目前不另行重算 depth-age histogram。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from .aggregate_release_records import ScenarioStratum
from .engine import Observation, ParticleResult
from .models import BoundaryEvent, EventType, ParticleState, ParticleStatus
from .scenarios import Scenario

__all__ = [
    "FISHING_GEAR_MATERIAL_ID",
    "MaterialStatistics",
    "MaterialStatisticsAccumulator",
    "MaterialStatisticsProduct",
    "MaterialStatisticsResult",
    "MaterialStatisticsSummary",
    "build_material_statistics",
]


# 這是目前十種非上浮基線中主管特別關注的漁業用具代理；它只用於報告排序／篩選
# 的穩定識別，不會改變情境清單、沉降速度或情境數量。
FISHING_GEAR_MATERIAL_ID: Final[str] = "oca_fishinggear_open_mesh_bundle"

_INVALID_MEMBER_STATUSES: Final[frozenset[ParticleStatus]] = frozenset(
    {
        ParticleStatus.DATA_GAP,
        ParticleStatus.NUMERICAL_FAILURE,
    }
)
_BED_EVENT_TYPES: Final[frozenset[EventType]] = frozenset(
    {EventType.BED_CONTACT, EventType.DEPOSITED}
)
_MAX_INT64: Final[int] = 2**63 - 1


def _require_text(value: object, *, label: str) -> str:
    """驗證站點、材質與 scenario identity 的非空原生字串。

    這些文字會成為報告產品的 join key；不自動 trim 或轉型，可避免不同 shard 把
    ``"site-a"`` 與 ``" site-a"`` 悄悄分成不同物理群組。
    """

    if type(value) is not str:
        raise TypeError(f"{label} 必須是原生 str")
    if not value or value != value.strip():
        raise ValueError(f"{label} 必須是非空且沒有首尾空白的文字")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    """驗證可進入報告計數的非負原生 ``int``，拒絕 bool 與 NumPy scalar。"""

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 int，且不可是 bool 或 NumPy scalar")
    if value < 0:
        raise ValueError(f"{label} 不可為負數")
    if value > _MAX_INT64:
        raise ValueError(f"{label} 不可超過 signed int64 上限")
    return value


def _require_native_int(value: object, *, label: str) -> int:
    """驗證 UTC 奈秒欄位是原生整數，但保留 Unix epoch 前的合法負值。"""

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 int，且不可是 bool 或 NumPy scalar")
    return value


def _require_finite_number(value: object, *, label: str) -> float:
    """驗證事件座標或比例是有限數值，避免 malformed event 進入統計。"""

    if type(value) not in (int, float):
        raise TypeError(f"{label} 必須是原生 int 或 float")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} 必須是有限值")
    return normalized


def _snapshot_ids(value: object, *, label: str) -> tuple[str, ...]:
    """複製並驗證站點／材質 ID iterable，保留字典序以固定輸出排列。"""

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} 必須是非字串、非 mapping iterable")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} 必須是 iterable") from error
    if not items:
        raise ValueError(f"{label} 不可為空")
    normalized = tuple(_require_text(item, label=f"{label}[{index}]") for index, item in enumerate(items))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} 不可重複")
    return tuple(sorted(normalized))


def _normalise_scenario_mapping(
    scenarios_by_id: object | None,
    scenario_strata: object | None,
) -> dict[str, Scenario | ScenarioStratum]:
    """把 Scenario 或 ScenarioStratum 來源整理成唯一 scenario ID mapping。

    正式 aggregate release 會提供 ``ScenarioStratum``，reference engine 測試與早期
    呼叫端則通常只持有 ``Scenario``。兩者只要保存本模組需要的 identity 欄位即可共用
    同一 reducer；若同時提供兩份來源則立即失敗，避免 caller 不知道哪一份是分組真相。
    """

    if scenarios_by_id is not None and scenario_strata is not None:
        raise ValueError("scenarios_by_id 與 scenario_strata 不可同時提供")
    if scenarios_by_id is None and scenario_strata is None:
        raise ValueError("必須提供 scenarios_by_id 或 scenario_strata")

    if scenarios_by_id is not None:
        if not isinstance(scenarios_by_id, Mapping):
            raise TypeError("scenarios_by_id 必須是 mapping")
        try:
            items = tuple(scenarios_by_id.items())
        except Exception as error:
            raise ValueError("scenarios_by_id 無法穩定讀取") from error
    else:
        if isinstance(scenario_strata, (str, bytes, bytearray, Mapping)):
            if isinstance(scenario_strata, Mapping):
                try:
                    items = tuple(scenario_strata.items())
                except Exception as error:
                    raise ValueError("scenario_strata mapping 無法穩定讀取") from error
            else:
                raise TypeError("scenario_strata 必須是 ScenarioStratum iterable 或 mapping")
        else:
            try:
                strata = tuple(scenario_strata)  # type: ignore[arg-type]
            except (TypeError, ValueError) as error:
                raise TypeError("scenario_strata 必須是 ScenarioStratum iterable") from error
            items = tuple((getattr(stratum, "scenario_id", None), stratum) for stratum in strata)

    if not items:
        raise ValueError("scenario strata 不可為空")

    normalized: dict[str, Scenario | ScenarioStratum] = {}
    for mapping_key, scenario in items:
        key = _require_text(mapping_key, label="scenario mapping key")
        if type(scenario) not in (Scenario, ScenarioStratum):
            raise TypeError("scenario strata value 必須是 exact Scenario 或 ScenarioStratum")
        scenario_id = _require_text(scenario.scenario_id, label="scenario_id")
        if key != scenario_id:
            raise ValueError("scenario mapping key 必須與 scenario_id 完全一致")
        if scenario_id in normalized:
            raise ValueError("scenario_id 不可重複")
        _require_text(scenario.study_site_id, label=f"{scenario_id}.study_site_id")
        _require_text(scenario.material_id, label=f"{scenario_id}.material_id")
        _require_text(scenario.analysis_region_id, label=f"{scenario_id}.analysis_region_id")
        _require_text(scenario.receptor_id, label=f"{scenario_id}.receptor_id")
        _require_native_int(scenario.arrival_time_utc_ns, label=f"{scenario_id}.arrival_time_utc_ns")
        normalized[scenario_id] = scenario
    return normalized


def _resolve_group_ids(
    scenarios: Mapping[str, Scenario | ScenarioStratum],
    *,
    site_ids: object | None,
    material_ids: object | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """由 scenario strata 與 optional declared topology 產生完整 site×material 集合。

    若 caller 顯式提供 site/material 清單，未出現在結果中的組合仍會保留 zero
    denominator row，讓「無事件」和「沒有這個分類」不被混為一談。已提供的 scenario
    若不在聲明拓撲內則拒絕，而不是將資料靜默丟棄。
    """

    derived_sites = tuple(sorted({scenario.study_site_id for scenario in scenarios.values()}))
    derived_materials = tuple(sorted({scenario.material_id for scenario in scenarios.values()}))
    resolved_sites = _snapshot_ids(site_ids, label="site_ids") if site_ids is not None else derived_sites
    resolved_materials = (
        _snapshot_ids(material_ids, label="material_ids")
        if material_ids is not None
        else derived_materials
    )
    declared_pairs = {
        (site_id, material_id)
        for site_id in resolved_sites
        for material_id in resolved_materials
    }
    if any(
        (scenario.study_site_id, scenario.material_id) not in declared_pairs
        for scenario in scenarios.values()
    ):
        raise ValueError("scenario strata 超出宣告的 site_ids × material_ids 拓撲")
    return resolved_sites, resolved_materials


@dataclass(frozen=True, slots=True)
class MaterialStatistics:
    """一個 ``study_site_id × material_id`` 的 immutable member-level 統計列。

    ``valid_member_denominator`` 是依既有報告政策排除資料缺口／數值失敗後的有效成員
    數。``first_bed_contact_member_count`` 是曾出現至少一筆 ``BED_CONTACT`` 或
    ``DEPOSITED`` 的不同 member 數；``deposited_member_count`` 則是終止狀態或事件
    證明已沉積的不同 member 數。兩個 fraction 都以有效分母計算，零分母時為 ``None``。
    repeated contact 只影響「是否曾接觸」的布林判定，不是此基線產品的必要欄位。
    """

    study_site_id: str
    material_id: str
    valid_member_denominator: int
    first_bed_contact_member_count: int
    first_bed_contact_fraction: float | None
    deposited_member_count: int
    deposited_fraction: float | None

    def __post_init__(self) -> None:
        """驗證計數與比例的一致性，並封存原生 immutable scalar。"""

        site_id = _require_text(self.study_site_id, label="study_site_id")
        material_id = _require_text(self.material_id, label="material_id")
        denominator = _require_nonnegative_int(
            self.valid_member_denominator,
            label="valid_member_denominator",
        )
        first_count = _require_nonnegative_int(
            self.first_bed_contact_member_count,
            label="first_bed_contact_member_count",
        )
        deposited_count = _require_nonnegative_int(
            self.deposited_member_count,
            label="deposited_member_count",
        )
        if first_count > denominator or deposited_count > denominator:
            raise ValueError("海床接觸／沉積 member 計數不可大於有效分母")
        if deposited_count > first_count:
            raise ValueError("沉積 member 計數必須是海床接觸 member 的子集合")

        expected_first = None if denominator == 0 else float(first_count / denominator)
        expected_deposited = None if denominator == 0 else float(deposited_count / denominator)
        for value, expected, label in (
            (self.first_bed_contact_fraction, expected_first, "first_bed_contact_fraction"),
            (self.deposited_fraction, expected_deposited, "deposited_fraction"),
        ):
            if expected is None:
                if value is not None:
                    raise ValueError(f"{label} 在零分母時必須是 None")
            else:
                if type(value) is not float:
                    raise TypeError(f"{label} 必須是原生 float")
                if not math.isfinite(value) or value != expected:
                    raise ValueError(f"{label} 必須精確等於 member 計數除以有效分母")

        object.__setattr__(self, "study_site_id", site_id)
        object.__setattr__(self, "material_id", material_id)
        object.__setattr__(self, "valid_member_denominator", denominator)
        object.__setattr__(self, "first_bed_contact_member_count", first_count)
        object.__setattr__(self, "deposited_member_count", deposited_count)

    @property
    def valid_member_count(self) -> int:
        """有效成員分母的常用別名。"""

        return self.valid_member_denominator

    @property
    def first_bed_contact_count(self) -> int:
        """首次海床接觸 member 計數的簡短欄位別名。"""

        return self.first_bed_contact_member_count

    @property
    def first_bed_contact_ratio(self) -> float | None:
        """首次海床接觸比例的語意別名。"""

        return self.first_bed_contact_fraction

    @property
    def deposited_count(self) -> int:
        """沉積 member 計數的簡短欄位別名。"""

        return self.deposited_member_count

    @property
    def deposited_ratio(self) -> float | None:
        """沉積比例的語意別名。"""

        return self.deposited_fraction


@dataclass(frozen=True, slots=True)
class MaterialStatisticsProduct:
    """保存完整 site×material 拓撲的 immutable 材質統計產品。

    ``records`` 依 site ID、material ID 排序且每個 pair 僅一列；``site_ids`` 或
    ``material_ids`` 若由 caller 聲明，產品會要求完整笛卡兒積，故無事件組合仍保留
    zero-denominator row。mapping key 是 ``(study_site_id, material_id)`` tuple，供
    renderer 以固定 join key 讀取；所有 mapping 與 tuple 都不保留 caller alias。
    """

    records: tuple[MaterialStatistics, ...]
    site_ids: tuple[str, ...] = ()
    material_ids: tuple[str, ...] = ()
    _statistics_by_site_material: Mapping[tuple[str, str], MaterialStatistics] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """驗證 pair closure、排序與防禦性 mapping snapshot。"""

        if isinstance(self.records, (str, bytes, bytearray, Mapping)):
            raise TypeError("records 必須是 MaterialStatistics iterable")
        try:
            records = tuple(self.records)
        except (TypeError, ValueError) as error:
            raise TypeError("records 必須是 MaterialStatistics iterable") from error
        if not records:
            raise ValueError("records 不可為空")
        if any(type(record) is not MaterialStatistics for record in records):
            raise TypeError("records 每一列必須是 exact MaterialStatistics")
        by_key: dict[tuple[str, str], MaterialStatistics] = {}
        for record in records:
            key = (record.study_site_id, record.material_id)
            if key in by_key:
                raise ValueError("records 的 site×material key 不可重複")
            by_key[key] = record

        normalized_sites = (
            _snapshot_ids(self.site_ids, label="site_ids")
            if self.site_ids
            else tuple(sorted({record.study_site_id for record in records}))
        )
        normalized_materials = (
            _snapshot_ids(self.material_ids, label="material_ids")
            if self.material_ids
            else tuple(sorted({record.material_id for record in records}))
        )
        expected_keys = {
            (site_id, material_id)
            for site_id in normalized_sites
            for material_id in normalized_materials
        }
        if set(by_key) != expected_keys:
            raise ValueError("records 必須完整覆蓋 site_ids × material_ids")
        ordered_records = tuple(by_key[key] for key in sorted(by_key))
        ordered_mapping = {key: by_key[key] for key in sorted(by_key)}
        object.__setattr__(self, "records", ordered_records)
        object.__setattr__(self, "site_ids", normalized_sites)
        object.__setattr__(self, "material_ids", normalized_materials)
        object.__setattr__(
            self,
            "_statistics_by_site_material",
            MappingProxyType(ordered_mapping),
        )

    @property
    def statistics_by_site_material(self) -> Mapping[tuple[str, str], MaterialStatistics]:
        """回傳以 ``(study_site_id, material_id)`` 為 key 的唯讀統計 mapping。"""

        return self._statistics_by_site_material

    @property
    def by_site_material(self) -> Mapping[tuple[str, str], MaterialStatistics]:
        """``statistics_by_site_material`` 的簡短別名。"""

        return self._statistics_by_site_material

    @property
    def stats_by_site_material(self) -> Mapping[tuple[str, str], MaterialStatistics]:
        """供表格／renderer 使用的 mapping 別名。"""

        return self._statistics_by_site_material

    @property
    def statistics(self) -> Mapping[tuple[str, str], MaterialStatistics]:
        """一般化的統計 mapping 別名。"""

        return self._statistics_by_site_material

    def __getitem__(self, key: tuple[str, str]) -> MaterialStatistics:
        """以 site×material tuple 直接讀取統計列。"""

        return self._statistics_by_site_material[key]

    def __iter__(self):
        """依固定排序迭代 site×material keys。"""

        return iter(self._statistics_by_site_material)

    def __len__(self) -> int:
        """回傳 site×material 統計列數。"""

        return len(self._statistics_by_site_material)


# 這些 alias 讓不同報告入口可用同一份產品而不複製資料或另訂 schema 名稱。
MaterialStatisticsResult = MaterialStatisticsProduct
MaterialStatisticsSummary = MaterialStatisticsProduct


@dataclass(slots=True)
class _MutableCounts:
    """reducer 內部的三項 Python 計數，不對外暴露可變狀態。"""

    valid_member_denominator: int = 0
    first_bed_contact_member_count: int = 0
    deposited_member_count: int = 0


class MaterialStatisticsAccumulator:
    """以一次 iterable 遍歷累加 site×material 材質統計的固定拓撲 reducer。

    constructor 只保存 scenario strata 的 identity 與預先宣告的 site×material keys；
    ``add`` 每次驗證一個 ``ParticleResult``，完成 member-level 事件判定後立即釋放該
    result，不保存軌跡、observation 或 event 歷史。``_seen_member_keys`` 只保存已處理
    的 logical ``(scenario_id, member_id)`` identity，確保重複輸入不會讓分母膨脹；正式
    shard iterator 應本來就保證每個 member 只出現一次，重複 identity 會 fail-closed。
    ``BED_CONTACT``／``DEPOSITED`` event 可缺席，因為沒有事件的 member 只是未接觸海床；
    這正是 baseline 不把 repeated contact 當成必要結果的設計。
    """

    _OPEN = "open"
    _FINALIZED = "finalized"
    _FAILED = "failed"

    def __init__(
        self,
        *,
        scenarios_by_id: Mapping[str, Scenario | ScenarioStratum] | None = None,
        scenario_strata: Iterable[ScenarioStratum] | Mapping[str, ScenarioStratum] | None = None,
        site_ids: Iterable[str] | None = None,
        material_ids: Iterable[str] | None = None,
    ) -> None:
        """建立完整 zero topology，並封存 scenario 分層索引。

        ``scenarios_by_id`` 適合 reference engine；``scenario_strata`` 適合正式 aggregate
        release。兩者互斥且至少提供一種，因為 ParticleResult 本身沒有 material_id，
        必須由已驗證 scenario strata 連回材質與站點。
        """

        scenarios = _normalise_scenario_mapping(scenarios_by_id, scenario_strata)
        resolved_sites, resolved_materials = _resolve_group_ids(
            scenarios,
            site_ids=site_ids,
            material_ids=material_ids,
        )
        self._scenarios = MappingProxyType(dict(scenarios))
        self._site_ids = resolved_sites
        self._material_ids = resolved_materials
        self._counts = {
            (site_id, material_id): _MutableCounts()
            for site_id in resolved_sites
            for material_id in resolved_materials
        }
        self._seen_member_keys: set[tuple[str, int]] = set()
        self._state = self._OPEN
        self._result: MaterialStatisticsProduct | None = None

    @property
    def site_ids(self) -> tuple[str, ...]:
        """回傳 reducer 已登錄的排序站點 IDs。"""

        return self._site_ids

    @property
    def material_ids(self) -> tuple[str, ...]:
        """回傳 reducer 已登錄的排序材質 IDs。"""

        return self._material_ids

    @property
    def member_count(self) -> int:
        """回傳已接受的 logical member 數，包含有效與失敗 member。"""

        return len(self._seen_member_keys)

    def _ensure_open(self) -> None:
        """拒絕 finalize／失敗後繼續累加，避免 caller 使用部分產品。"""

        if self._state != self._OPEN:
            raise ValueError("MaterialStatisticsAccumulator 已關閉")

    def _mark_failed(self) -> None:
        """將任何 add 例外轉成不可恢復狀態。"""

        self._state = self._FAILED

    def _validate_result(
        self,
        result: object,
    ) -> tuple[
        Scenario | ScenarioStratum,
        ParticleState,
        ParticleStatus,
        tuple[BoundaryEvent, ...],
    ]:
        """驗證一筆結果的 scenario/member/event identity，並回傳固定 tuple。

        只做材質統計所需的欄位 gate：observation 必須是既有 ``Observation``，但不在
        此處重算路徑或環境深度；事件位置只驗證有限性，真正的海床幾何判定已由引擎
        產生 ``BED_CONTACT``／``DEPOSITED`` 類型完成。失敗成員仍完整走完 identity
        驗證，之後才依既有 denominator policy 排除計數。
        """

        if type(result) is not ParticleResult:
            raise TypeError("result 必須是 exact ParticleResult")
        final_state = result.final_state
        if type(final_state) is not ParticleState:
            raise TypeError("result.final_state 必須是 exact ParticleState")
        scenario_id = _require_text(final_state.scenario_id, label="final_state.scenario_id")
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError("ParticleResult 的 scenario_id 不存在於 scenario strata")
        particle_id = _require_text(final_state.particle_id, label="final_state.particle_id")
        study_site_id = _require_text(final_state.study_site_id, label="final_state.study_site_id")
        analysis_region_id = _require_text(
            final_state.analysis_region_id,
            label="final_state.analysis_region_id",
        )
        receptor_id = _require_text(final_state.receptor_id, label="final_state.receptor_id")
        if (
            study_site_id != scenario.study_site_id
            or analysis_region_id != scenario.analysis_region_id
            or receptor_id != scenario.receptor_id
        ):
            raise ValueError("ParticleResult 的 scenario/site/region/receptor identity 不一致")
        member_id = _require_nonnegative_int(final_state.member_id, label="final_state.member_id")
        status = final_state.status
        if type(status) is not ParticleStatus:
            raise TypeError("final_state.status 必須是 exact ParticleStatus")
        if status is ParticleStatus.ACTIVE:
            raise ValueError("ACTIVE ParticleResult 尚未終止，不可進入報告")
        final_time_utc_ns = _require_native_int(
            final_state.time_utc_ns,
            label="final_state.time_utc_ns",
        )
        if final_time_utc_ns > scenario.arrival_time_utc_ns:
            raise ValueError("final_state.time_utc_ns 不可晚於 scenario arrival time")
        final_age_seconds = _require_finite_number(
            final_state.age_seconds,
            label="final_state.age_seconds",
        )
        if final_age_seconds < 0.0:
            raise ValueError("final_state.age_seconds 不可為負值")
        _require_finite_number(final_state.x_m, label="final_state.x_m")
        _require_finite_number(final_state.y_m, label="final_state.y_m")
        _require_finite_number(final_state.z_m, label="final_state.z_m")

        try:
            observations = tuple(result.observations)
        except (TypeError, ValueError) as error:
            raise TypeError("result.observations 必須是 Observation iterable") from error
        for index, observation in enumerate(observations):
            if type(observation) is not Observation:
                raise TypeError(f"observations[{index}] 必須是 exact Observation")
            if observation.particle_id != particle_id:
                raise ValueError("Observation.particle_id 必須與 final_state.particle_id 一致")
            _require_native_int(
                observation.time_utc_ns,
                label=f"observations[{index}].time_utc_ns",
            )
            observation_age_seconds = _require_finite_number(
                observation.age_seconds,
                label=f"observations[{index}].age_seconds",
            )
            if observation_age_seconds < 0.0:
                raise ValueError(f"observations[{index}].age_seconds 不可為負值")
            _require_finite_number(observation.x_m, label=f"observations[{index}].x_m")
            _require_finite_number(observation.y_m, label=f"observations[{index}].y_m")
            _require_finite_number(observation.z_m, label=f"observations[{index}].z_m")
            if type(observation.status) is not ParticleStatus:
                raise TypeError(f"observations[{index}].status 必須是 exact ParticleStatus")

        try:
            events = tuple(result.events)
        except (TypeError, ValueError) as error:
            raise TypeError("result.events 必須是 BoundaryEvent iterable") from error
        for index, event in enumerate(events):
            if type(event) is not BoundaryEvent:
                raise TypeError(f"events[{index}] 必須是 exact BoundaryEvent")
            if (
                event.particle_id != particle_id
                or event.scenario_id != final_state.scenario_id
                or event.member_id != member_id
                or event.study_site_id != scenario.study_site_id
                or event.analysis_region_id != scenario.analysis_region_id
                or event.receptor_id != scenario.receptor_id
            ):
                raise ValueError(f"events[{index}] identity 與 final_state／scenario 不一致")
            if type(event.event_type) is not EventType:
                raise TypeError(f"events[{index}].event_type 必須是 EventType")
            event_time_utc_ns = _require_native_int(
                event.time_utc_ns,
                label=f"events[{index}].time_utc_ns",
            )
            if event_time_utc_ns < final_time_utc_ns or event_time_utc_ns > scenario.arrival_time_utc_ns:
                raise ValueError(f"events[{index}].time_utc_ns 必須落在 arrival 與 final time 之間")
            _require_finite_number(event.x_m, label=f"events[{index}].x_m")
            _require_finite_number(event.y_m, label=f"events[{index}].y_m")
            _require_finite_number(event.z_m, label=f"events[{index}].z_m")
            fraction = _require_finite_number(event.fraction, label=f"events[{index}].fraction")
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"events[{index}].fraction 必須落在 [0, 1]")
        return scenario, final_state, status, events

    def add(self, result: ParticleResult) -> None:
        """驗證並立即吸收一個 member 的結果，遇到錯誤後永久關閉 reducer。

        分母與兩個 numerator 都是 member-level 計數：同一 member 的多個 bed events
        只產生一個布林判定。``DATA_GAP``／``NUMERICAL_FAILURE`` 只留下在外部輸入中
        已存在的 identity，不進入任何有效材質列；其餘終止狀態才增加有效分母。
        """

        self._ensure_open()
        try:
            scenario, final_state, status, events = self._validate_result(result)
            member_key = (scenario.scenario_id, final_state.member_id)
            if member_key in self._seen_member_keys:
                raise ValueError("同一 scenario_id × member_id 不可重複輸入")
            self._seen_member_keys.add(member_key)
            if status in _INVALID_MEMBER_STATUSES:
                return

            group_key = (scenario.study_site_id, scenario.material_id)
            counts = self._counts[group_key]
            counts.valid_member_denominator += 1
            # terminal DEPOSITED 本身就是一次已證明的海床接觸；即使某個 legacy／最小
            # synthetic writer 沒有另寫 BED_CONTACT，首次接觸分子仍不能被低估。
            bed_event_present = status is ParticleStatus.DEPOSITED or any(
                event.event_type in _BED_EVENT_TYPES for event in events
            )
            deposited_present = status is ParticleStatus.DEPOSITED or any(
                event.event_type is EventType.DEPOSITED for event in events
            )
            if bed_event_present:
                counts.first_bed_contact_member_count += 1
            if deposited_present:
                counts.deposited_member_count += 1
        except Exception:
            self._mark_failed()
            raise

    def add_result(self, result: ParticleResult) -> None:
        """``add`` 的語意別名，方便與其他 report reducer 入口一致。"""

        self.add(result)

    def add_many(self, results: Iterable[ParticleResult]) -> None:
        """一次遍歷吸收 iterable，不先轉成 list 或 tuple 保存所有結果。"""

        self._ensure_open()
        if isinstance(results, (str, bytes, bytearray, Mapping)):
            raise TypeError("results 必須是非字串、非 mapping iterable")
        try:
            iterator = iter(results)
        except TypeError as error:
            self._mark_failed()
            raise TypeError("results 必須是 iterable") from error
        try:
            for result in iterator:
                self.add(result)
        except Exception:
            # add 已將自身錯誤轉成 failed；iterator 取值錯誤也不能讓 caller 繼續使用
            # 可能只吸收半段的 reducer。
            self._mark_failed()
            raise

    def finalize(self) -> MaterialStatisticsProduct:
        """封存並回傳完整 site×material 產品；空輸入保留 zero-denominator rows。"""

        self._ensure_open()
        records: list[MaterialStatistics] = []
        for site_id in self._site_ids:
            for material_id in self._material_ids:
                counts = self._counts[(site_id, material_id)]
                denominator = counts.valid_member_denominator
                first_fraction = (
                    None
                    if denominator == 0
                    else float(counts.first_bed_contact_member_count / denominator)
                )
                deposited_fraction = (
                    None
                    if denominator == 0
                    else float(counts.deposited_member_count / denominator)
                )
                records.append(
                    MaterialStatistics(
                        study_site_id=site_id,
                        material_id=material_id,
                        valid_member_denominator=denominator,
                        first_bed_contact_member_count=counts.first_bed_contact_member_count,
                        first_bed_contact_fraction=first_fraction,
                        deposited_member_count=counts.deposited_member_count,
                        deposited_fraction=deposited_fraction,
                    )
                )
        self._result = MaterialStatisticsProduct(
            records=tuple(records),
            site_ids=self._site_ids,
            material_ids=self._material_ids,
        )
        self._state = self._FINALIZED
        return self._result


def build_material_statistics(
    results: Iterable[ParticleResult],
    *,
    scenarios_by_id: Mapping[str, Scenario | ScenarioStratum] | None = None,
    scenario_strata: Iterable[ScenarioStratum] | Mapping[str, ScenarioStratum] | None = None,
    site_ids: Iterable[str] | None = None,
    material_ids: Iterable[str] | None = None,
) -> MaterialStatisticsProduct:
    """以一次遍歷建立 site×material 材質統計產品。

    Args:
        results: ``ParticleResult`` 的一次性 iterable；不會被 materialize 成全案 list。
        scenarios_by_id: reference engine 使用的 scenario ID mapping。
        scenario_strata: 正式 aggregate release 使用的 ``ScenarioStratum`` iterable 或
            mapping；與 ``scenarios_by_id`` 二選一。
        site_ids: 可選的完整站點拓撲；未出現結果的站點仍輸出零分母列。
        material_ids: 可選的完整材質拓撲；應傳入十種基線 material ID 以保留零事件類別。

    Returns:
        ``MaterialStatisticsProduct``。每個 row 都保存有效 member denominator、首次
        海床接觸 member count/fraction 與 deposited member count/fraction。

    Raises:
        TypeError／ValueError: scenario strata、結果 identity、事件欄位或 topology 不符
            契約時；任何失敗都不回傳部分產品。
    """

    accumulator = MaterialStatisticsAccumulator(
        scenarios_by_id=scenarios_by_id,
        scenario_strata=scenario_strata,
        site_ids=site_ids,
        material_ids=material_ids,
    )
    accumulator.add_many(results)
    return accumulator.finalize()
