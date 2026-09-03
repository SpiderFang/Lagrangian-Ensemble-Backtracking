"""YAML 設定載入、跨欄位科學契約與正式發布閘門。

example config 同時保存已定案設計與尚待 SERVER／pilot 衍生的 ``null`` 欄位。開發模式
允許這些欄位存在，以便合成測試與 pilot 前進；``formal_release=True`` 接受研究團隊核定的
全部可得 2024–2025 資料契約，但仍拒絕缺少重建驗證、expanded A、M/時步/回溯期等
manifest 的設定。這個分層避免把上游 ``trial_ready`` 名稱誤作外部補件阻擋，也不會讓
尚未驗證的缺口重建被直接送入正式兩年批次。
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# 這組名稱取自海洋保育署 iOcean 海洋廢棄物管理頁於 2026-08-27 顯示的查詢類別。
# 常數只用來驗證臺灣情境的分類追溯是否完整；該網站的清除重量與件數不含單體物性，
# 因此不得用這組分類或統計量反推密度、阻力或終端沉降速度。
EXPECTED_OCA_CATEGORIES_ZH = frozenset(
    {
        "竹木",
        "保麗龍",
        "廢漁網漁具",
        "其他/不可回收",
        "鐵罐",
        "鋁罐",
        "寶特瓶",
        "玻璃瓶",
        "廢紙",
        "其他/可回收",
    }
)


def _validate_non_rising_material_contract(settling: Any, *, expected_count: int) -> None:
    """驗證設定中的十類材質／形狀代理均為嚴格負值且可追溯。

    ``settling`` 來自 YAML 的 ``physics.settling``。因該區塊同時保存研究說明與未來可
    擴充欄位，目前仍以 mapping 讀取；本函式補上不可放寬的科學契約：z 軸向上為正、
    零速與上浮一律拒絕、十個 iOcean 類別一對一、材質／形狀／適用條件與證據欄位不可
    缺漏。檢查通過只代表設計一致，不代表暫定速度已由現地樣本校準。
    """

    if not isinstance(settling, dict):
        raise ValueError("physics.settling 必須是 mapping")
    if settling.get("z_positive_up_sign_convention") is not True:
        raise ValueError("沉降速度必須採 z positive-up 符號慣例")
    if settling.get("require_strictly_negative_velocity") is not True:
        raise ValueError("非上浮基線必須要求 settling_velocity_mps 嚴格小於 0")
    if settling.get("positive_or_zero_velocity_policy") != "reject_config":
        raise ValueError("零速或正值物性必須在設定驗證階段拒絕")

    records = settling.get("material_classes")
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"physics.settling.material_classes 必須恰有 {expected_count} 筆")
    material_ids: list[str] = []
    categories: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"material_classes[{index}] 必須是 mapping")
        required_fields = (
            "material_id",
            "oca_category_zh",
            "material_family_zh",
            "representative_shape_zh",
            "applicability_condition_zh",
            "calibration_status",
            "evidence_grade",
        )
        if any(
            not isinstance(record.get(field), str) or not record[field].strip() for field in required_fields
        ):
            raise ValueError(f"material_classes[{index}] 缺少材質、形狀、適用條件或證據欄位")
        velocity = record.get("settling_velocity_mps")
        if isinstance(velocity, bool) or not isinstance(velocity, (int, float)) or velocity >= 0.0:
            raise ValueError(f"{record['material_id']} 的 settling_velocity_mps 必須嚴格小於 0")
        if record.get("behavior_class") != "sinking":
            raise ValueError(f"{record['material_id']} 的 behavior_class 必須是 sinking")
        material_ids.append(record["material_id"])
        categories.append(record["oca_category_zh"])
    if len(set(material_ids)) != len(material_ids):
        raise ValueError("material_classes 的 material_id 必須唯一")
    if set(categories) != EXPECTED_OCA_CATEGORIES_ZH:
        raise ValueError("material_classes 必須一對一涵蓋 iOcean 十個海廢類別")


class StrictModel(BaseModel):
    """允許文件型額外欄位、但禁止未知型別被任意轉成物件的共同基底。"""

    model_config = ConfigDict(extra="allow", frozen=True)


class InputContract(StrictModel):
    """上游 root、全部可得資料決策、時間正規化與重建證據。"""

    ocm_native_root_env: str
    ocm_surface_root_env: str
    nww_analysis_root_env: str
    years: list[int]
    ocm_contract: dict[str, Any]
    nww_contract: dict[str, Any]
    available_data_contract: dict[str, Any]
    time_axis_contract: dict[str, Any]
    ocm_gap_reconstruction_manifest: str | None = None
    ocm_gap_safe_arrival_manifest: str | None = None
    nww_full_hourly_analysis_manifest: str | None = None


class StudyAreaConfig(StrictModel):
    """forcing 層與站點層不可混用的全域計數契約。"""

    expected_flow_domain_count: int = 4
    expected_analysis_region_count: int = 4
    expected_study_site_count: int = 5
    primary_analysis_unit: str
    allow_overlapping_local_domains: bool
    shared_forcing_domain_preserves_hydrodynamic_connectivity: bool
    primary_local_boundary_scope: str
    other_site_local_domain_crossing_policy: str


class DomainConfig(StrictModel):
    """單一 forcing domain 的 bbox、投影與發布角色。"""

    analysis_region_id: str
    analysis_region_name_zh: str
    flow_domain_id: str
    center_lonlat: tuple[float, float]
    bbox_lon_lat: tuple[float, float, float, float]
    metric_crs_policy: str
    current_domain_role: str | None = None
    formal_release_flow_domain_id: str | None = None
    formal_release_domain_status: str | None = None

    def resolved_flow_domain_id(self, *, formal: bool = False) -> str:
        """回傳目前執行模式應使用的 flow-domain 識別碼。

        pilot 與開發模式固定使用現行 ``flow_domain_id``；formal 模式若已核定
        ``formal_release_flow_domain_id``，則改用該正式上游產品 ID。集中在 domain 設定
        提供此 resolver，可避免 receptor 初始條件、geometry、preflight 與未來 runtime
        各自複製 A 區 expanded-domain 判斷。此方法只解析識別碼，不讀取 OCM 或驗證檔案。
        """

        if formal and self.formal_release_flow_domain_id:
            return self.formal_release_flow_domain_id
        return self.flow_domain_id

    @model_validator(mode="after")
    def validate_bbox(self) -> DomainConfig:
        """拒絕倒置 bbox 或位於 bbox 外的投影中心。"""

        lon_min, lon_max, lat_min, lat_max = self.bbox_lon_lat
        lon, lat = self.center_lonlat
        if not (lon_min < lon_max and lat_min < lat_max):
            raise ValueError(f"{self.flow_domain_id} bbox 必須嚴格遞增")
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            raise ValueError(f"{self.flow_domain_id} center 必須位於 bbox 內")
        return self


class StudySiteConfig(StrictModel):
    """獨立研究站點與其 local-domain 生成政策。"""

    study_site_id: str
    study_site_name_zh: str
    analysis_region_id: str
    flow_domain_id: str
    formal_release_flow_domain_id: str | None = None
    anchor_lonlat: tuple[float, float] | None = None
    receptor_core_radius_m: float | None = None
    local_domain_baseline_radius_m: float | None = None
    local_domain_sensitivity_radii_m: list[float] = Field(default_factory=list)
    local_domain_policy: str | None = None

    def resolved_flow_domain_id(self, *, formal: bool = False) -> str:
        """回傳 study site 在目前執行模式應使用的 flow-domain ID。

        `flow_domain_id` 保留 pilot／開發來源；正式 release 可另外保存與所屬 region
        一致的 expanded 或正式來源 ID。這個欄位不改變站點的獨立情境身分，只讓 site-level
        runtime、geometry 與 input inventory 能明確指向同一套 accepted forcing。
        """

        if formal and self.formal_release_flow_domain_id:
            return self.formal_release_flow_domain_id
        return self.flow_domain_id


class ScenarioConfig(StrictModel):
    """五站完整交叉與 member/seed 的不可變計數契約。"""

    expected_receptor_count_per_site: int
    expected_receptor_count: int
    expected_arrival_time_count_per_site: int
    expected_material_count: int
    scenario_count_per_site: int
    scenario_count_region_A: int
    scenario_count: int
    receptor_manifest: str | None = None
    arrival_time_manifest: str | None = None
    material_manifest: str | None = None
    receptor_arrival_initial_condition_manifest: str | None = None
    members_per_scenario: int | None = None
    master_seed: int | None = None
    seed_policy: str | None = None


class ExecutionConfig(StrictModel):
    """CPU backend、分片、checkpoint cadence 與 forcing cache 的工程欄位。

    ``checkpoint_interval_sweeps`` 的單位是完整批次 sweep，不是輸出觀測點；兩者在
    adaptive time-step 下不等價。``active_chunk_size`` 控制一次散射／回寫的粒子數，
    ``max_resident_forcing_months`` 控制單一 process 的月份 cache 上限。這些欄位只描述
    執行策略，不改變 Scenario、粒子 seed 或物理方程。
    """

    reference_backend: str
    production_backend: str
    shard_scenario_count: int | None = None
    checkpoint_interval_sweeps: int | None = None
    active_chunk_size: int | None = None
    max_resident_forcing_months: int | None = 2
    fail_if_dirty_git: bool
    atomic_publish: bool
    input_change_policy: str

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_checkpoint_field(cls, value: Any) -> Any:
        """拒絕舊的 output-step 欄位，避免 YAML extra=allow 造成靜默誤讀。

        專案仍允許文件型額外設定欄位，但 ``checkpoint_interval_output_steps`` 曾被誤用
        為 checkpoint cadence；若讓它落入 ``model_extra``，呼叫端可能以為設定已生效而
        實際採用另一個預設值。因此只對這個已淘汰欄位建立局部 before gate，不改變全域
        extra 欄位相容性。
        """

        if isinstance(value, dict) and "checkpoint_interval_output_steps" in value:
            raise ValueError(
                "execution.checkpoint_interval_output_steps 已淘汰，請改用 checkpoint_interval_sweeps"
            )
        return value


class BoundaryConfig(StrictModel):
    """local、foreign-local 與 outer boundary 的事件政策。"""

    local_domain_first_exit: str
    other_site_local_domain_enter: str
    other_site_local_domain_exit: str
    other_site_local_domain_changes_study_site: bool
    flow_domain_open_boundary: str
    max_backtrack_days: float | None = None
    maximum_step_count: int | None = None


class IntegrationConfig(StrictModel):
    """signed-time RK4、隨機 split 與 adaptive step 尚待核定的數值欄位。"""

    deterministic_method: str
    time_direction: str
    stochastic_method: str
    stochastic_variance_uses_absolute_dt: bool
    output_interval_seconds: float | None = None
    dt_min_seconds: float | None = None
    dt_max_seconds: float | None = None


class ProjectConfig(StrictModel):
    """可由 CLI 驗證的完整專案設定。

    跨欄位驗證會鎖定 4 domains、5 sites、每站 10,000、A 區 20,000、全案 50,000，
    並確認貢寮與龜山島共用 A 區 forcing 而不是共用情境。這些是已裁決需求，不能
    透過修改單一 YAML 數值靜默縮減。
    """

    schema_version: str
    config_status: str
    design_version: str
    project_id: str
    time_standard: str
    inputs: InputContract
    study_area: StudyAreaConfig
    domains: list[DomainConfig]
    study_sites: list[StudySiteConfig]
    integration: IntegrationConfig
    boundaries: BoundaryConfig
    scenarios: ScenarioConfig
    execution: ExecutionConfig
    forcing: dict[str, Any]
    physics: dict[str, Any]
    geometry: dict[str, Any]
    outputs: dict[str, Any]

    @model_validator(mode="after")
    def validate_scientific_contract(self) -> ProjectConfig:
        """驗證五站完整交叉、唯一 ID 與 A 區共用 forcing 契約。"""

        if self.time_standard != "UTC":
            raise ValueError("time_standard 必須固定為 UTC")
        if len(self.domains) != self.study_area.expected_flow_domain_count:
            raise ValueError("flow domain 數量與 study_area 契約不符")
        if len(self.study_sites) != self.study_area.expected_study_site_count:
            raise ValueError("study site 數量與 study_area 契約不符")
        domain_ids = [item.flow_domain_id for item in self.domains]
        region_ids = [item.analysis_region_id for item in self.domains]
        site_ids = [item.study_site_id for item in self.study_sites]
        if len(set(domain_ids)) != len(domain_ids) or len(set(region_ids)) != len(region_ids):
            raise ValueError("flow_domain_id 與 analysis_region_id 必須唯一")
        if len(set(site_ids)) != len(site_ids):
            raise ValueError("study_site_id 必須唯一")
        domain_by_region = {item.analysis_region_id: item.flow_domain_id for item in self.domains}
        formal_domain_by_region = {
            item.analysis_region_id: item.formal_release_flow_domain_id for item in self.domains
        }
        for site in self.study_sites:
            if domain_by_region.get(site.analysis_region_id) != site.flow_domain_id:
                raise ValueError(f"{site.study_site_id} 的 region 與 flow domain 對應不一致")
            formal_domain = formal_domain_by_region.get(site.analysis_region_id)
            if (
                site.formal_release_flow_domain_id is not None
                and site.formal_release_flow_domain_id != formal_domain
            ):
                raise ValueError(f"{site.study_site_id} 的 formal flow domain 與 region 設定不一致")
        northeast = {site.study_site_id: site for site in self.study_sites if site.analysis_region_id == "A"}
        if set(northeast) != {"gongliao", "guishan"}:
            raise ValueError("A 區必須恰含獨立的 gongliao 與 guishan 站點")
        if len({site.flow_domain_id for site in northeast.values()}) != 1:
            raise ValueError("貢寮與龜山島必須共用同一 A 區 forcing domain")
        counts = self.scenarios
        if (
            counts.expected_material_count != 10
            or counts.expected_receptor_count_per_site != 20
            or counts.expected_arrival_time_count_per_site != 50
            or counts.scenario_count_per_site != 10_000
            or counts.scenario_count_region_A != 20_000
            or counts.scenario_count != 50_000
            or counts.expected_receptor_count != 100
        ):
            raise ValueError("情境契約必須維持每站 10×20×50、A 區 20,000、全案 50,000")
        _validate_non_rising_material_contract(
            self.physics.get("settling"),
            expected_count=counts.expected_material_count,
        )
        if self.boundaries.other_site_local_domain_changes_study_site:
            raise ValueError("foreign-local crossing 不得改變 study_site_id")
        return self

    def normalized_payload(self) -> dict[str, Any]:
        """回傳排序前可 JSON 序列化內容，供 hash、manifest 與差異比較。"""

        return self.model_dump(mode="json", exclude_none=False)

    def config_hash(self) -> str:
        """以 canonical JSON 計算 SHA-256；與 YAML 排版及 key 順序無關。"""

        encoded = json.dumps(
            self.normalized_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def assert_formal_release_ready(self) -> None:
        """拒絕尚含 pilot/derived-pending 欄位的正式批次設定。

        這裡只驗證設定本身；上游月份的 schema、可用 coverage、canonical UTC 與物理
        reconstruction skill 仍由 preflight／reconstruction manifests 驗證。兩層 gate 都
        通過前，CLI 不得啟動正式 run。
        """

        blockers: list[str] = []
        if self.config_status != "approved":
            blockers.append("config_status 必須是 approved")
        available_contract = self.inputs.available_data_contract
        if available_contract.get("status") != "decided":
            blockers.append("全部可得 2024–2025 資料契約尚未凍結")
        time_contract = self.inputs.time_axis_contract
        if time_contract.get("canonicalization_policy") != "sort_and_deduplicate_prefer_last":
            blockers.append("正式時間軸必須使用 sort_and_deduplicate_prefer_last")
        if not (self.inputs.ocm_gap_reconstruction_manifest or self.inputs.ocm_gap_safe_arrival_manifest):
            blockers.append("OCM approved reconstruction 或 gap-safe arrival/horizon manifest 尚未產出")
        if not self.inputs.nww_full_hourly_analysis_manifest:
            blockers.append("NWW 完整逐時 analysis manifest 尚未產出")
        if self.scenarios.members_per_scenario is None or self.scenarios.members_per_scenario < 1:
            blockers.append("正式 members_per_scenario 尚未由收斂測試核定")
        if self.scenarios.master_seed is None or self.scenarios.master_seed < 0:
            blockers.append("正式 master_seed 尚未核定")
        for field_name in (
            "receptor_manifest",
            "arrival_time_manifest",
            "material_manifest",
            "receptor_arrival_initial_condition_manifest",
        ):
            if not getattr(self.scenarios, field_name):
                blockers.append(f"scenarios.{field_name} 尚未產出")
        for field_name in (
            "domain_manifest",
            "local_domain_manifest",
            "open_boundary_manifest",
            "receptor_manifest",
        ):
            if not self.geometry.get(field_name):
                blockers.append(f"geometry.{field_name} 尚未產出")
        settling = self.physics.get("settling", {})
        if not isinstance(settling, dict) or not settling.get("material_manifest"):
            blockers.append("physics.settling.material_manifest 尚未產出")
        horizontal_diffusion = self.physics.get("horizontal_diffusion", {})
        vertical_diffusion = self.physics.get("vertical_diffusion", {})
        if not isinstance(horizontal_diffusion, dict) or horizontal_diffusion.get("constant_kh_m2ps") is None:
            blockers.append("正式 constant_kh_m2ps 尚未由 pilot 核定")
        if not isinstance(vertical_diffusion, dict) or vertical_diffusion.get("constant_kz_m2ps") is None:
            blockers.append("正式 constant_kz_m2ps 尚未由 pilot 核定")
        ocm_forcing = self.forcing.get("ocm", {})
        if not isinstance(ocm_forcing, dict) or ocm_forcing.get("wetdry_semantics_decision_status") not in {
            "confirmed",
            "approved",
        }:
            blockers.append("OCM wetdry 語意尚未確認")
        nww_forcing = self.forcing.get("nww3", {})
        if not isinstance(nww_forcing, dict) or nww_forcing.get("convention_evidence_status") not in {
            "confirmed",
            "approved",
        }:
            blockers.append("NWW 波向慣例證據尚未升為 confirmed/approved")
        if self.integration.dt_min_seconds is None or self.integration.dt_max_seconds is None:
            blockers.append("正式 dt_min/dt_max 尚未核定")
        if self.integration.output_interval_seconds is None:
            blockers.append("正式 output_interval_seconds 尚未核定")
        if self.boundaries.max_backtrack_days is None or self.boundaries.maximum_step_count is None:
            blockers.append("正式 max_backtrack_days/maximum_step_count 尚未核定")
        if self.execution.shard_scenario_count is None or self.execution.shard_scenario_count < 1:
            blockers.append("正式 shard_scenario_count 尚未核定")
        if self.execution.checkpoint_interval_sweeps is None or self.execution.checkpoint_interval_sweeps < 1:
            blockers.append("正式 checkpoint_interval_sweeps 尚未核定")
        region_a = next(item for item in self.domains if item.analysis_region_id == "A")
        if not region_a.formal_release_flow_domain_id:
            blockers.append("A 區 expanded formal_release_flow_domain_id 尚未產出")
        if region_a.formal_release_flow_domain_id == region_a.flow_domain_id:
            blockers.append("A 區正式 domain 不得沿用僅供 pilot 的現行 v3 ID")
        for site in self.study_sites:
            if (
                site.analysis_region_id == "A"
                and site.formal_release_flow_domain_id is not None
                and site.formal_release_flow_domain_id != region_a.formal_release_flow_domain_id
            ):
                blockers.append(f"{site.study_site_id} formal flow domain 未與 A 區 source 一致")
        if blockers:
            raise ValueError("正式發布設定未通過：" + "；".join(blockers))


def resolve_flow_domain_id(config: ProjectConfig, analysis_region_id: str, *, formal: bool = False) -> str:
    """依 analysis region 解析 pilot 或 formal runtime 應使用的 flow-domain ID。

    設定中的 ``flow_domain_id`` 是開發／pilot 的現行產品識別碼；formal 且有核定值時，
    ``formal_release_flow_domain_id`` 才是正式 OCM/NWW runtime 應讀取的識別碼。這個公開
    resolver 只做 config lookup，不讀取 OCM、不產生 manifest，也不修改設定；後續 geometry、
    dynamic initial-condition loader 與 runtime 應共用它，避免不同模組對 A 區 expanded
    domain 做出不一致判斷。
    """

    if not isinstance(analysis_region_id, str) or not analysis_region_id.strip():
        raise ValueError("analysis_region_id 不可為空白")
    matches = [domain for domain in config.domains if domain.analysis_region_id == analysis_region_id]
    if len(matches) != 1:
        raise ValueError(f"config 必須恰有一個 analysis_region_id：{analysis_region_id}")
    return matches[0].resolved_flow_domain_id(formal=formal)


def load_config(path: str | Path, *, formal_release: bool = False) -> ProjectConfig:
    """由 UTF-8 YAML 載入並驗證專案設定。

    YAML 根節點必須是 mapping；``yaml.safe_load`` 禁止任意 Python object tag。若要求
    ``formal_release``，會在結構驗證後再套用衍生閘門，讓正式 CLI fail closed。
    """

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"設定根節點必須是 mapping：{config_path}")
    config = ProjectConfig.model_validate(payload)
    if formal_release:
        config.assert_formal_release_ready()
    return config
