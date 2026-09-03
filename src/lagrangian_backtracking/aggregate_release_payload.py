"""彙整 release payload 的不可變封存容器與跨產品資料契約。

本模組只負責把已由下游資料型別驗證過的彙整規格、trajectory shard 綁定、
scenario 分層、事件聚合與路徑聚合封存成一個可核對的 release reference；不讀取
檔案、不重算 aggregate、不修補計數，也不執行任何新的科學推論。所有距離與網格
邊界沿用公尺（m），所有 travel age 邊界沿用秒（s）；事件與路徑的水平陣列軸
固定為 ``(y_cell, x_cell)``。

封存結果描述的是指定 run、受體與輸入條件下的「條件式來源足跡」或「相對來源
權重」。在尚未建立先驗、似然與觀測驗證前，任何計數、分母或路徑權重都不是
絕對來源機率，也不構成因果歸因。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np

from .aggregate_release_records import AggregateShardBinding, ScenarioStratum
from .aggregate_spec import AggregateSpec
from .event_aggregation import (
    BoundaryAggregateKey,
    CrossSiteAggregateKey,
    EventAggregateChunk,
    SiteEventGridCounts,
    SourceReceptorAggregateKey,
    initialize_event_aggregate,
)
from .models import ParticleStatus
from .streaming_aggregation import StreamingPathwayAggregate

__all__ = ["AGGREGATE_RELEASE_SCHEMA_VERSION", "AggregateReleasePayload"]


# release payload 的版本是精確相等的資料契約；未知版本不能靠猜測欄位語意繼續執行。
AGGREGATE_RELEASE_SCHEMA_VERSION = "1.0.0"

# run 與 experiment case 識別碼會出現在路徑、manifest 與跨產品 join key，因此只接受
# 不含路徑分隔符與控制字元的 ASCII 安全 slug，且長度上限與 AggregateSpec 一致。
_SAFE_SLUG_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)

# 所有 provenance digest 都必須是完整的小寫 SHA-256；不接受大寫、前綴或任意短摘要。
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_RUN_KINDS: Final[frozenset[str]] = frozenset(
    {"synthetic", "pilot", "formal"}
)

# EventAggregateChunk 的六個事件網格欄位是固定資料契約。集中列舉欄位可避免只驗證
# 部分陣列，並明確保留每一站的 (y_cell, x_cell) 軸語意。
_EVENT_GRID_FIELDS: Final[tuple[str, ...]] = (
    "local_first_exit_count",
    "outer_first_exit_count",
    "bed_first_contact_count",
    "bed_repeated_contact_count",
    "data_gap_failure_count",
    "numerical_failure_count",
)


def _require_exact_text(value: object, *, label: str) -> str:
    """驗證資料契約需要的原生非空文字，不以 trim 或隱式轉型修補輸入。

    這些欄位會成為 schema、模式、run 或 provenance 的穩定識別值；接受字串子類別、
    空字串或含首尾空白的值，可能讓序列化後的 join key 與來源 manifest 產生歧義，
    因此型別與內容都採 fail-closed 政策。
    """

    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} 必須是非空且沒有首尾空白的原生 str")
    return value


def _require_safe_slug(value: object, *, label: str) -> str:
    """驗證可安全用作識別碼的 ASCII slug。

    slug 不含 slash、反斜線、空白或 ``.``／``..`` 路徑語意，長度限制為 1 至 128
    個字元。函式只驗證並回傳原值，不會自動小寫、刪除字元或改寫 caller 的識別碼。
    """

    text = _require_exact_text(value, label=label)
    if _SAFE_SLUG_RE.fullmatch(text) is None:
        raise ValueError(f"{label} 必須是安全 ASCII slug")
    return text


def _require_sha256(value: object, *, label: str) -> str:
    """驗證 provenance 欄位是 64 碼小寫 SHA-256 十六進位字串。"""

    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} 必須是 64 碼小寫 SHA-256")
    return value


def _require_positive_native_int(value: object, *, label: str) -> int:
    """驗證成員倍率是排除 bool 與 NumPy scalar 的正原生 Python int。

    ``members_per_scenario`` 會同時決定 shard 粒子數、站點粒子分母與整份 payload
    的總粒子數；若讓 ``bool``、NumPy 整數或可轉型物件通過，跨檔案序列化後可能得到
    不同型別或不同 overflow 語意，所以此邊界不做任何隱式轉換。
    """

    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} 必須是大於 0 的原生 Python int")
    return value


def _copy_nonempty_tuple(value: object, *, label: str) -> tuple[object, ...]:
    """防禦性複製非空序列，拒絕字串容器與無法穩定 materialize 的輸入。

    payload 只保存 tuple 快照，不保存 caller 的 list 或 generator；元素本身則由呼叫端
    的 immutable record constructor 負責封存。函式不排序、不去重，也不以任何方式補齊
    scenario 或 shard，因為原順序是 release manifest 的資料語意。
    """

    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} 必須是非空元素序列，不可為字串容器")
    try:
        copied = tuple(value)  # type: ignore[arg-type]
    except Exception as error:
        raise ValueError(f"{label} 必須是可 materialize 的序列") from error
    if not copied:
        raise ValueError(f"{label} 不得為空")
    return copied


def _copy_nonempty_mapping(value: object, *, label: str) -> dict[object, object]:
    """防禦性複製非空 mapping，隔離 caller 對 pathway site index 的後續修改。

    mapping 的 key/value 仍在上層逐項驗證；這裡只建立獨立普通 dict，避免直接以
    ``MappingProxyType`` 包裝 caller 的可變 dict 而留下外部修改通道。
    """

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 必須是 mapping")
    try:
        copied = dict(value)
    except Exception as error:
        raise ValueError(f"{label} 無法防禦性複製") from error
    if not copied:
        raise ValueError(f"{label} 不得為空")
    return copied


@dataclass(frozen=True, slots=True)
class AggregateReleasePayload:
    """封存一個 release run 的規格、分層與兩類 aggregate reference。

    ``aggregate_spec`` 提供每站公尺制矩形、cell 尺寸與秒數 age 軸；
    ``event_aggregate`` 的六類事件陣列與 ``pathway_by_site`` 的路徑陣列均採
    ``(y_cell, x_cell)``。``shard_bindings`` 的 scenario 半開區間使用原始 tuple 順序，
    ``scenario_strata`` 則是一列一個 scenario×receptor×arrival 分層資料。所有粒子
    計數都以 ``members_per_scenario`` 對應到完整分層列，不把缺值、乾點、資料缺口或
    數值失敗以零值替代；底層 EventAggregateChunk 與 StreamingPathwayAggregate 的
    constructor invariants 視為既有資料契約，本容器只做跨產品一致性驗證。

    本類別只封存已驗證的 reference，不重算、clip、平移或修補任何輸入 aggregate。
    產物可支持條件式來源足跡或相對來源權重的發布；未建立先驗、似然與觀測驗證前，
    不得將其解讀為絕對來源機率或因果歸因。
    """

    schema_version: str
    run_id: str
    run_kind: str
    experiment_case_id: str
    members_per_scenario: int
    config_hash: str
    checkpoint_input_binding_hash: str
    source_run_plan_sha256: str
    source_run_progress_sha256: str
    source_normalized_config_sha256: str
    source_input_inventory_sha256: str
    aggregate_spec: AggregateSpec
    shard_bindings: tuple[AggregateShardBinding, ...]
    scenario_strata: tuple[ScenarioStratum, ...]
    event_aggregate: EventAggregateChunk
    pathway_by_site: Mapping[str, StreamingPathwayAggregate]

    def __post_init__(self) -> None:
        """先驗證基本型別與 provenance，再封存容器並執行跨產品不變量檢查。

        這個方法刻意把型別錯誤視為資料契約錯誤而立即拒絕；不在驗證失敗時嘗試
        從別的欄位推導替代值。後半段檢查只讀取既有 aggregate 的 metadata、shape、
        edges 與計數，不會改動陣列內容，也不會重新計算事件或路徑統計。
        """

        if type(self.schema_version) is not str:
            raise ValueError("schema_version 必須是原生 str")
        if self.schema_version != AGGREGATE_RELEASE_SCHEMA_VERSION:
            raise ValueError("schema_version 不受支援")

        run_id = _require_safe_slug(self.run_id, label="run_id")
        run_kind = _require_exact_text(self.run_kind, label="run_kind")
        if run_kind not in _ALLOWED_RUN_KINDS:
            raise ValueError("run_kind 只允許 synthetic、pilot 或 formal")
        experiment_case_id = _require_safe_slug(
            self.experiment_case_id,
            label="experiment_case_id",
        )
        members_per_scenario = _require_positive_native_int(
            self.members_per_scenario,
            label="members_per_scenario",
        )

        # 六個 digest 是 run plan、進度、設定與輸入 inventory 的 provenance 鎖點；任何
        # 一個欄位型別或大小寫不符，都不能讓下游把不同來源誤認成同一份 release。
        digest_fields = (
            "config_hash",
            "checkpoint_input_binding_hash",
            "source_run_plan_sha256",
            "source_run_progress_sha256",
            "source_normalized_config_sha256",
            "source_input_inventory_sha256",
        )
        digests = {
            field_name: _require_sha256(
                getattr(self, field_name),
                label=field_name,
            )
            for field_name in digest_fields
        }

        if type(self.aggregate_spec) is not AggregateSpec:
            raise ValueError("aggregate_spec 必須是 exact AggregateSpec 實例")
        if type(self.event_aggregate) is not EventAggregateChunk:
            raise ValueError("event_aggregate 必須是 exact EventAggregateChunk 實例")

        raw_shards = _copy_nonempty_tuple(
            self.shard_bindings,
            label="shard_bindings",
        )
        raw_strata = _copy_nonempty_tuple(
            self.scenario_strata,
            label="scenario_strata",
        )
        raw_pathways = _copy_nonempty_mapping(
            self.pathway_by_site,
            label="pathway_by_site",
        )
        for index, shard in enumerate(raw_shards):
            if type(shard) is not AggregateShardBinding:
                raise ValueError(
                    f"shard_bindings[{index}] 必須是 exact AggregateShardBinding 實例"
                )
        for index, stratum in enumerate(raw_strata):
            if type(stratum) is not ScenarioStratum:
                raise ValueError(
                    f"scenario_strata[{index}] 必須是 exact ScenarioStratum 實例"
                )
        for site_id, pathway in raw_pathways.items():
            if type(site_id) is not str or not site_id:
                raise ValueError("pathway_by_site 的 key 必須是非空原生 str")
            if type(pathway) is not StreamingPathwayAggregate:
                raise ValueError(
                    "pathway_by_site 的 value 必須是 exact StreamingPathwayAggregate 實例"
                )

        # 只有通過基本契約後才寫入 defensive copy；此處保留 aggregate、record 與 pathway
        # value 的原始 reference，因為它們各自的 constructor 已完成底層封存與唯讀保證。
        object.__setattr__(self, "schema_version", self.schema_version)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "run_kind", run_kind)
        object.__setattr__(self, "experiment_case_id", experiment_case_id)
        object.__setattr__(self, "members_per_scenario", members_per_scenario)
        for field_name, digest in digests.items():
            object.__setattr__(self, field_name, digest)
        object.__setattr__(self, "shard_bindings", raw_shards)
        object.__setattr__(self, "scenario_strata", raw_strata)
        object.__setattr__(self, "pathway_by_site", MappingProxyType(raw_pathways))

        self._validate_cross_product_invariants()

    def _validate_cross_product_invariants(self) -> None:
        """驗證 spec、shard、strata、event 與 pathway 間的完整跨產品契約。

        此流程只比較識別碼、半開 scenario range、Python 計數、NumPy shape 與既有邊界
        陣列的 exact equality。計算中的座標仍是公尺制 x/y，age 軸仍是秒；任何事件或
        路徑陣列的第一、二軸不符合 ``(y_cell, x_cell)`` 都會 fail closed，而不透過
        transpose、clip、padding 或重新聚合來掩蓋來源產品錯誤。
        """

        spec = self.aggregate_spec
        event = self.event_aggregate
        spec_sites = _mapping_keys_as_set(spec.site_grids, label="aggregate_spec.site_grids")

        if spec.run_id != self.run_id:
            raise ValueError("aggregate_spec.run_id 必須精確等於 payload.run_id")

        # scenario_strata 是跨事件分母與 pathway 站點拓撲的唯一列舉來源；先逐列驗證
        # scenario_id 唯一，再建立站點與 (site, receptor) 集合，避免重複列被靜默合併。
        scenario_ids: set[str] = set()
        scenario_sites: set[str] = set()
        scenario_receptor_keys: set[tuple[str, str]] = set()
        strata_count_by_site: dict[str, int] = {}
        for index, stratum in enumerate(self.scenario_strata):
            scenario_id = stratum.scenario_id
            if scenario_id in scenario_ids:
                raise ValueError(f"scenario_strata[{index}] 的 scenario_id 不得重複")
            scenario_ids.add(scenario_id)

            site_id = stratum.study_site_id
            if site_id not in spec_sites:
                raise ValueError(
                    f"scenario_strata[{index}] 的 study_site_id 不存在於 aggregate_spec"
                )
            scenario_sites.add(site_id)
            receptor_key = (site_id, stratum.receptor_id)
            scenario_receptor_keys.add(receptor_key)
            strata_count_by_site[site_id] = strata_count_by_site.get(site_id, 0) + 1

            # formal release 必須每一列都有完整的 receptor×arrival 動態初始條件；None
            # 代表未知、乾點、資料缺口或未提供，不得在此層以零值替代。
            if self.run_kind == "formal" and stratum.has_dynamic_initial_condition is not True:
                raise ValueError(
                    f"formal scenario_strata[{index}] 必須具備動態初始條件"
                )

        if scenario_sites != spec_sites:
            raise ValueError(
                "scenario_strata 的 study_site_id site set 必須與 aggregate_spec.site_grids 完全相等"
            )

        _validate_shard_bindings(
            self.shard_bindings,
            scenario_count=len(self.scenario_strata),
            members_per_scenario=self.members_per_scenario,
        )

        # 四個事件站點 mapping 與 pathway index 必須共同描述同一組站點；不能因某個
        # 產品缺列而讓後續報表把缺資料誤當作零事件或零粒子。
        event_site_mappings = {
            "site_grid_counts": event.site_grid_counts,
            "outcome_count_by_site": event.outcome_count_by_site,
            "valid_member_denominator_by_site": event.valid_member_denominator_by_site,
            "total_member_count_by_site": event.total_member_count_by_site,
        }
        for label, mapping in event_site_mappings.items():
            actual_sites = _mapping_keys_as_set(mapping, label=f"event_aggregate.{label}")
            if actual_sites != spec_sites:
                raise ValueError(
                    f"event_aggregate.{label} 的 site set 必須與 aggregate_spec 完全相等"
                )

        pathway_sites = _mapping_keys_as_set(
            self.pathway_by_site,
            label="pathway_by_site",
        )
        if pathway_sites != spec_sites:
            raise ValueError(
                "pathway_by_site 的 site set 必須與 aggregate_spec 完全相等"
            )

        _validate_member_counts(
            event=event,
            pathway_by_site=self.pathway_by_site,
            strata_count_by_site=strata_count_by_site,
            scenario_count=len(self.scenario_strata),
            shard_particle_count=sum(
                shard.particle_count for shard in self.shard_bindings
            ),
            members_per_scenario=self.members_per_scenario,
        )

        if not np.array_equal(
            event.age_bin_edges_seconds,
            spec.age_bin_edges_seconds,
        ):
            raise ValueError(
                "event_aggregate.age_bin_edges_seconds 必須以 np.array_equal 精確等於 aggregate_spec"
            )

        _validate_exact_event_topology(
            spec=spec,
            event=event,
            scenario_receptor_keys=scenario_receptor_keys,
        )

        # 事件的六類網格都只接受既有 SiteEventGridCounts 的 shape reference；此處逐站
        # 核對 (y_cell, x_cell)，不讀取或重算任何計數值，讓資料缺口與數值失敗仍保留
        # 在各自欄位而不被本容器混成單一零值狀態。
        for site_id, grid_counts in event.site_grid_counts.items():
            if type(grid_counts) is not SiteEventGridCounts:
                raise ValueError(
                    f"event_aggregate.site_grid_counts[{site_id!r}] 必須是 exact SiteEventGridCounts"
                )
            expected_shape = _grid_shape_from_spec(spec, site_id)
            for field_name in _EVENT_GRID_FIELDS:
                field_value = getattr(grid_counts, field_name)
                if field_value.shape != expected_shape:
                    raise ValueError(
                        f"event_aggregate.site_grid_counts[{site_id!r}].{field_name} "
                        f"shape 必須是 (y_cell, x_cell)={expected_shape}"
                    )

        # pathway 邊界是以 AggregateSpec 的公尺制 min、cell size 與 max 重建的 canonical
        # 幾何；最後一點明確固定為 max，避免浮點累乘的尾端誤差被當成另一個網格。age
        # 邊界則必須與事件共用同一組秒數軸，禁止以近似值或不同 bin 定義替代。
        for site_id, pathway in self.pathway_by_site.items():
            grid = spec.site_grids[site_id]
            x_cells, y_cells = _grid_cell_counts_from_spec(spec, site_id)
            expected_x_edges = _rebuild_metric_edges(
                grid.x_min_m,
                grid.x_max_m,
                spec.grid_cell_size_m,
                x_cells,
            )
            expected_y_edges = _rebuild_metric_edges(
                grid.y_min_m,
                grid.y_max_m,
                spec.grid_cell_size_m,
                y_cells,
            )
            if not np.array_equal(pathway.x_edges_m, expected_x_edges):
                raise ValueError(
                    f"pathway_by_site[{site_id!r}].x_edges_m 不符合 spec 公尺制 x 格線"
                )
            if not np.array_equal(pathway.y_edges_m, expected_y_edges):
                raise ValueError(
                    f"pathway_by_site[{site_id!r}].y_edges_m 不符合 spec 公尺制 y 格線"
                )
            if not np.array_equal(
                pathway.age_bin_edges_seconds,
                spec.age_bin_edges_seconds,
            ):
                raise ValueError(
                    f"pathway_by_site[{site_id!r}].age_bin_edges_seconds 必須精確等於 spec age 軸"
                )

        # 受體分母 key 必須逐一對應 strata 的 (study_site_id, receptor_id)，而非只對
        # 站點做總數比較；缺少任何一個 key 都會使該受體的有效成員分母不可追溯。
        actual_receptor_keys = _receptor_key_set(
            event.valid_member_denominator_by_receptor
        )
        if actual_receptor_keys != scenario_receptor_keys:
            raise ValueError(
                "event.valid_member_denominator_by_receptor 的 key set 必須精確等於 "
                "strata 的 (site, receptor) 集合"
            )


def _validate_exact_event_topology(
    *,
    spec: AggregateSpec,
    event: EventAggregateChunk,
    scenario_receptor_keys: set[tuple[str, str]],
) -> None:
    """驗證事件產品保留完整的零事件拓撲與 canonical 邊界格線。

    這裡的零拓撲是由規格與 scenario strata 的資料契約預先決定的列集合，並不
    是「沒有資料」或「觀測到零事件」：即使某邊界、受體組合、跨站方向或終止
    狀態在本次 run 的計數為零，對應的 key 仍必須存在，才能讓下游辨識零值與
    缺列的差異。函式只讀取既有 mapping 與陣列，canonical 邊界則透過公開的
    ``initialize_event_aggregate`` 依 AggregateSpec 建立，不重算任何科學計數。
    """

    # 由 immutable AggregateSpec 的每站 local／outer segment 引用明確建立預期
    # 邊界列。這個集合是資料拓撲，不是依現有事件計數反推的觀測結果；因此同一
    # segment 在不同站點或不同角色出現時，仍須保留各自獨立的 BoundaryAggregateKey。
    expected_boundary_keys: set[BoundaryAggregateKey] = set()
    for site_id, segment_groups in spec.site_boundary_segment_ids.items():
        for boundary_kind, segment_ids in (
            ("local", segment_groups.local_segment_ids),
            ("outer", segment_groups.outer_segment_ids),
        ):
            for segment_id in segment_ids:
                expected_boundary_keys.add(
                    BoundaryAggregateKey(
                        study_site_id=site_id,
                        boundary_kind=boundary_kind,
                        boundary_segment_id=segment_id,
                    )
                )

    for field_name in (
        "boundary_bin_edges_m",
        "boundary_arclength_raw_count",
        "boundary_travel_age_histogram",
    ):
        actual_keys = _mapping_keys_as_set(
            getattr(event, field_name),
            label=f"event_aggregate.{field_name}",
        )
        if actual_keys != expected_boundary_keys:
            raise ValueError(
                f"event_aggregate.{field_name} 的 key set 必須精確等於 "
                "aggregate_spec.site_boundary_segment_ids 建立的 boundary 拓撲"
            )

    # initialize_event_aggregate 是事件模組公開的 canonical 邏輯，會依規格的
    # segment 長度與 boundary bin 尺寸建立從 0 m 到精確終點的 float64 邊界。逐 key
    # 比對完整陣列，不能只比長度或用近似值，避免最後一個短 bin、尾端浮點值或
    # 中間分箱被替換後仍被當成相容產品；這一步不讀取或改寫任何事件計數。
    canonical_event = initialize_event_aggregate(
        spec,
        age_bin_edges_seconds=event.age_bin_edges_seconds,
    )
    for boundary_key in expected_boundary_keys:
        if not np.array_equal(
            event.boundary_bin_edges_m[boundary_key],
            canonical_event.boundary_bin_edges_m[boundary_key],
        ):
            raise ValueError(
                "event_aggregate.boundary_bin_edges_m 的邊界格線必須以 "
                f"np.array_equal 精確等於 canonical spec 格線：{boundary_key!r}"
            )

    # 每個 scenario strata 的 (site, receptor) 都必須與該站所有預期 boundary
    # key 形成完整笛卡兒積。source-receptor 的 raw count 與 travel-age histogram
    # 即使全為零也要留下列，否則缺列會和零事件混淆並破壞 release 的固定資料表
    # 拓撲。這裡只建立 key set，不重算任何 raw count 或 histogram 數值。
    expected_source_receptor_keys: set[SourceReceptorAggregateKey] = set()
    for site_id, receptor_id in scenario_receptor_keys:
        for boundary_key in expected_boundary_keys:
            if boundary_key.study_site_id != site_id:
                continue
            expected_source_receptor_keys.add(
                SourceReceptorAggregateKey(
                    study_site_id=site_id,
                    receptor_id=receptor_id,
                    boundary_kind=boundary_key.boundary_kind,
                    boundary_segment_id=boundary_key.boundary_segment_id,
                )
            )

    for field_name in (
        "source_receptor_raw_count",
        "source_receptor_travel_age_histogram",
    ):
        actual_keys = _mapping_keys_as_set(
            getattr(event, field_name),
            label=f"event_aggregate.{field_name}",
        )
        if actual_keys != expected_source_receptor_keys:
            raise ValueError(
                f"event_aggregate.{field_name} 的 key set 必須精確等於 "
                "scenario strata 與 site boundary 拓撲的完整 cross product"
            )

    # 跨站統計是有方向的 source→target 關係；每一對不同站點都要有一列，
    # 單站時則沒有任何合法方向，所以預期集合必須精確為空。這個檢查維持
    # 零計數方向與缺列方向的可區分性，不會從計數值推導或補建統計結果。
    site_ids = _mapping_keys_as_set(spec.site_grids, label="aggregate_spec.site_grids")
    expected_cross_site_keys = {
        CrossSiteAggregateKey(
            source_study_site_id=source_site_id,
            target_study_site_id=target_site_id,
        )
        for source_site_id in site_ids
        for target_site_id in site_ids
        if source_site_id != target_site_id
    }
    actual_cross_site_keys = _mapping_keys_as_set(
        event.cross_site_unique_member_count,
        label="event_aggregate.cross_site_unique_member_count",
    )
    if actual_cross_site_keys != expected_cross_site_keys:
        raise ValueError(
            "event_aggregate.cross_site_unique_member_count 的 key set 必須精確等於 "
            "所有 ordered distinct site pairs"
        )

    # outcome mapping 的內層 key 由 ParticleStatus 的正式非 ACTIVE value 完整
    # 定義。每站即使某個終止原因計數為零也不可省略，因為 outcome 欄位的固定
    # 拓撲是狀態契約，不是只列出本批實際出現的事件摘要。
    expected_outcome_keys = {
        status.value
        for status in ParticleStatus
        if status != ParticleStatus.ACTIVE
    }
    for site_id in site_ids:
        actual_outcome_keys = _mapping_keys_as_set(
            event.outcome_count_by_site[site_id],
            label=f"event_aggregate.outcome_count_by_site[{site_id!r}]",
        )
        if actual_outcome_keys != expected_outcome_keys:
            raise ValueError(
                f"event_aggregate.outcome_count_by_site[{site_id!r}] 的 key set 必須精確等於 "
                "ParticleStatus 中除 ACTIVE 外的所有 value"
            )


def _mapping_keys_as_set(value: Mapping[object, object], *, label: str) -> set[object]:
    """讀取 mapping key set 並把異常轉為一致的 fail-closed ValueError。

    跨產品 site set 是拓撲契約，不需要排序或重建；這裡只建立短暫的 Python set，
    不修改輸入 mapping。若遭遇被竄改的 mapping 或不可雜湊 key，應立即拒絕 release，
    不能把缺失站點當作空產品繼續發布。
    """

    try:
        return set(value)
    except Exception as error:
        raise ValueError(f"{label} 無法建立安全的 key set") from error


def _grid_cell_counts_from_spec(
    spec: AggregateSpec,
    site_id: str,
) -> tuple[int, int]:
    """由站點公尺制矩形與公尺 cell 尺寸取得 ``(x_cell, y_cell)`` 數量。

    AggregateSpec 已在底層驗證每一軸可由整數個 cell 覆蓋；本函式只用相同的 round
    語意取得整數數量，不平移座標、不裁切範圍，也不以 pathway 或 event 的 shape
    反推規格。回傳順序刻意是 x、y，呼叫端再組成產品要求的 ``(y_cell, x_cell)``。
    """

    try:
        grid = spec.site_grids[site_id]
        x_cells = int(
            round((grid.x_max_m - grid.x_min_m) / spec.grid_cell_size_m)
        )
        y_cells = int(
            round((grid.y_max_m - grid.y_min_m) / spec.grid_cell_size_m)
        )
    except (KeyError, OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"站點 {site_id!r} 無法由 spec 建立整數網格數") from error
    if x_cells <= 0 or y_cells <= 0:
        raise ValueError(f"站點 {site_id!r} 的 x/y 網格數必須為正整數")
    return x_cells, y_cells


def _grid_shape_from_spec(spec: AggregateSpec, site_id: str) -> tuple[int, int]:
    """取得事件與 pathway 共用的 ``(y_cell, x_cell)`` 陣列 shape。

    ``x`` 與 ``y`` 是每站固定 local projection 下的公尺制運算軸；這個 shape 只描述
    矩形被 cell 離散化後的索引數量，不代表海域遮罩、陸地、乾點或任何有效物理事件。
    """

    x_cells, y_cells = _grid_cell_counts_from_spec(spec, site_id)
    return y_cells, x_cells


def _rebuild_metric_edges(
    minimum_m: int | float,
    maximum_m: int | float,
    cell_size_m: int | float,
    cell_count: int,
) -> np.ndarray:
    """以公尺制 min、cell size 與 cell_count 重建格線並固定最後 max。

    前 ``cell_count`` 個間距由 ``minimum_m + arange * cell_size_m`` 產生，最後一點
    強制寫成 ``maximum_m``，只為重建規格定義的格線 reference。此 helper 不會拿
    pathway 內容來校準或修補 spec，也不會對輸入 aggregate 陣列做任何 in-place 修改。
    """

    try:
        edges = np.asarray(minimum_m, dtype=np.float64) + (
            np.arange(cell_count + 1, dtype=np.float64)
            * np.asarray(cell_size_m, dtype=np.float64)
        )
        edges[-1] = np.asarray(maximum_m, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("公尺制 pathway 格線無法由 spec 安全重建") from error
    if edges.ndim != 1 or edges.size != cell_count + 1:
        raise ValueError("公尺制 pathway 格線 shape 不符合 spec cell 數")
    return edges


def _validate_shard_bindings(
    shard_bindings: tuple[AggregateShardBinding, ...],
    *,
    scenario_count: int,
    members_per_scenario: int,
) -> None:
    """驗證 shard 原順序的 scenario 半開區間與粒子數守恆。

    每個 shard 的 ``[scenario_start_index, scenario_stop_index)`` 都是 immutable
    scenario 順序的半開區間，不是檔案 row 或粒子索引。游標必須從 0 開始逐 shard
    無縫前進到 strata 長度；每列 scenario 對應固定 ``members_per_scenario`` 個成員，
    因而 shard 粒子數與整份粒子總數都能以 Python int 精確核對，不接受 overlap、gap、
    重複 shard_id 或以零值替代缺失範圍。
    """

    cursor = 0
    total_particle_count = 0
    shard_ids: set[str] = set()
    for index, shard in enumerate(shard_bindings):
        shard_id = shard.shard_id
        if shard_id in shard_ids:
            raise ValueError(f"shard_bindings[{index}] 的 shard_id 不得重複")
        shard_ids.add(shard_id)

        start = shard.scenario_start_index
        stop = shard.scenario_stop_index
        if type(start) is not int or type(stop) is not int:
            raise ValueError(
                f"shard_bindings[{index}] 的 scenario range 必須是原生 Python int"
            )
        if start != cursor:
            raise ValueError(
                f"shard_bindings[{index}] 的 scenario range 必須從前一 shard 連續銜接"
            )
        if stop <= start:
            raise ValueError(f"shard_bindings[{index}] 的 scenario range 必須非空")

        expected_particles = (stop - start) * members_per_scenario
        if type(shard.particle_count) is not int or shard.particle_count != expected_particles:
            raise ValueError(
                f"shard_bindings[{index}].particle_count 必須等於 scenario span×members_per_scenario"
            )
        total_particle_count += shard.particle_count
        cursor = stop

    if cursor != scenario_count:
        raise ValueError(
            "shard_bindings 的 scenario ranges 必須從 0 連續覆蓋到 len(scenario_strata)"
        )
    expected_total = scenario_count * members_per_scenario
    if total_particle_count != expected_total:
        raise ValueError("所有 shard 的 particle_count 總和必須等於 strata×M")


def _validate_member_counts(
    *,
    event: EventAggregateChunk,
    pathway_by_site: Mapping[str, StreamingPathwayAggregate],
    strata_count_by_site: Mapping[str, int],
    scenario_count: int,
    shard_particle_count: int,
    members_per_scenario: int,
) -> None:
    """精確核對站點、event、pathway、shard 與整份 payload 的粒子分母。

    對每一站，strata 列數乘以 ``M`` 就是該站應有的成員粒子數；這個數必須同時等於
    event 的 ``total_member_count_by_site`` 與 pathway 的 ``input_particle_count``。
    所有加總以 Python int 進行，避免 NumPy 固定寬度 scalar 造成繞回。此處只核對既有
    計數 reference，不會重算事件、路徑、有效分母或任何條件式來源權重。
    """

    expected_total = scenario_count * members_per_scenario
    if shard_particle_count != expected_total:
        raise ValueError("所有 shard particle 總數必須等於 strata×M")
    if type(event.input_particle_count) is not int or event.input_particle_count != expected_total:
        raise ValueError("event.input_particle_count 必須精確等於 strata×M")

    site_total = 0
    pathway_total = 0
    for site_id, stratum_count in strata_count_by_site.items():
        expected_site_count = stratum_count * members_per_scenario
        event_count = event.total_member_count_by_site[site_id]
        pathway_count = pathway_by_site[site_id].input_particle_count
        if type(event_count) is not int or event_count != expected_site_count:
            raise ValueError(
                f"event.total_member_count_by_site[{site_id!r}] 必須等於該站 strata 列數×M"
            )
        if type(pathway_count) is not int or pathway_count != expected_site_count:
            raise ValueError(
                f"pathway_by_site[{site_id!r}].input_particle_count 必須等於該站 strata 列數×M"
            )
        site_total += event_count
        pathway_total += pathway_count

    if site_total != expected_total:
        raise ValueError("所有站點 event 粒子總數必須等於 strata×M")
    if pathway_total != expected_total:
        raise ValueError("所有站點 pathway 粒子總數必須等於 strata×M")
    if site_total != event.input_particle_count or pathway_total != event.input_particle_count:
        raise ValueError("event、pathway 與所有站點粒子總數必須完全一致")


def _receptor_key_set(
    value: Mapping[object, object],
) -> set[tuple[str, str]]:
    """把受體分母 mapping 的公開 key 投影為 ``(site_id, receptor_id)`` 集合。

    EventAggregateChunk 已負責受體 key 的底層型別與非負分母驗證；本層只做跨產品
    join key 核對，確保每一個 scenario strata 受體都有明確有效分母。若 key 被竄改到
    缺欄位或欄位不可形成穩定 tuple，則以 ValueError 中止，不猜測或補建分母。
    """

    result: set[tuple[str, str]] = set()
    for key in value:
        try:
            site_id = key.study_site_id
            receptor_id = key.receptor_id
        except AttributeError as error:
            raise ValueError(
                "valid_member_denominator_by_receptor 的 key 必須具有 site/receptor 欄位"
            ) from error
        if type(site_id) is not str or type(receptor_id) is not str:
            raise ValueError(
                "valid_member_denominator_by_receptor 的 key 欄位必須是原生 str"
            )
        result.add((site_id, receptor_id))
    return result
