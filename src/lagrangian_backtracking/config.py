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
    anchor_lonlat: tuple[float, float] | None = None
    receptor_core_radius_m: float | None = None
    local_domain_baseline_radius_m: float | None = None
    local_domain_sensitivity_radii_m: list[float] = Field(default_factory=list)
    local_domain_policy: str | None = None


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
    members_per_scenario: int | None = None
    master_seed: int | None = None


class ExecutionConfig(StrictModel):
    """reference/production backend、分片與 checkpoint 的正式工程欄位。"""

    reference_backend: str
    production_backend: str
    shard_scenario_count: int | None = None
    checkpoint_interval_output_steps: int | None = None
    fail_if_dirty_git: bool
    atomic_publish: bool
    input_change_policy: str


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
        for site in self.study_sites:
            if domain_by_region.get(site.analysis_region_id) != site.flow_domain_id:
                raise ValueError(f"{site.study_site_id} 的 region 與 flow domain 對應不一致")
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
        if not (
            self.inputs.ocm_gap_reconstruction_manifest
            or self.inputs.ocm_gap_safe_arrival_manifest
        ):
            blockers.append(
                "OCM approved reconstruction 或 gap-safe arrival/horizon manifest 尚未產出"
            )
        if not self.inputs.nww_full_hourly_analysis_manifest:
            blockers.append("NWW 完整逐時 analysis manifest 尚未產出")
        if self.scenarios.members_per_scenario is None or self.scenarios.members_per_scenario < 1:
            blockers.append("正式 members_per_scenario 尚未由收斂測試核定")
        if self.scenarios.master_seed is None or self.scenarios.master_seed < 0:
            blockers.append("正式 master_seed 尚未核定")
        for field_name in ("receptor_manifest", "arrival_time_manifest", "material_manifest"):
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
        if (
            self.execution.checkpoint_interval_output_steps is None
            or self.execution.checkpoint_interval_output_steps < 1
        ):
            blockers.append("正式 checkpoint_interval_output_steps 尚未核定")
        region_a = next(item for item in self.domains if item.analysis_region_id == "A")
        if not region_a.formal_release_flow_domain_id:
            blockers.append("A 區 expanded formal_release_flow_domain_id 尚未產出")
        if region_a.formal_release_flow_domain_id == region_a.flow_domain_id:
            blockers.append("A 區正式 domain 不得沿用僅供 pilot 的現行 v3 ID")
        if blockers:
            raise ValueError("正式發布設定未通過：" + "；".join(blockers))


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
