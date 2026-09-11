"""YAML 設定載入、跨欄位科學契約與正式發布閘門。

example config 同時保存已定案設計與尚待 SERVER／pilot 衍生的 ``null`` 欄位。開發模式
允許這些欄位存在，以便合成測試與 pilot 前進；``formal_release=True`` 接受研究團隊核定的
全部可得 2024–2025 資料契約，並依明示的 ``formal_domain_policy`` 套用來源範圍 gate。
舊 ``expanded_domain_v1`` 維持 A 區 expanded source 的既有檢查；新的
``v3_local20km_20260909_v1`` 在三套 forcing 共同有效格網與 20 km 邊界證據尚未由
validator／producer 實際驗證前，必定 fail closed。這個分層避免把上游 ``trial_ready``
名稱誤作外部補件阻擋，也不會讓尚未驗證的缺口重建被直接送入正式兩年批次。
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

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

# 設計版本是整個設定／manifest／checkpoint 身分的一部分，不能讓不同模組自行拼接
# 版本字串。CURRENT_DESIGN_VERSION 代表本期 A 區 v3/20 km 與沉降基線；v2 只作為
# 舊設定的相容邊界，並不表示目前範例仍採用 v2。凡是正式或 pilot 設定若使用 v3
# design，A 區都必須明示對應的 versioned formal_domain_policy；反向也同樣成立。
CURRENT_DESIGN_VERSION = "design_baseline_v3_non_rising_a_v3_local20_20260909"
LEGACY_DESIGN_VERSION_V2 = "design_baseline_v2_non_rising_oca_proxy"
# 提供較短的舊版本別名給既有外部工具／測試；兩個名稱代表完全相同的 v2 legacy
# 字串，實際驗證仍集中在 _validate_design_domain_binding。
LEGACY_DESIGN_VERSION = LEGACY_DESIGN_VERSION_V2

# 這些識別碼是設定資料契約的一部分。``expanded_domain_v1`` 是未明示新政策的
# 舊設定所採用的相容語意；它保留既有 A 區南擴 candidate／formal source 流程。
# ``v3_local20km_20260909_v1`` 則只描述本期已決定的 A 區研究範圍：兩個站點共用
# ``northeast_taiwan_common_cache_v3``，local 半徑為 20 km。後者尚未代表三套 forcing
# 已完成共同有效網格與邊界驗證，因此正式發布 gate 仍會無條件阻擋。
FORMAL_DOMAIN_POLICY_EXPANDED_V1 = "expanded_domain_v1"
FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1 = "v3_local20km_20260909_v1"
FormalDomainPolicy = Literal[
    "expanded_domain_v1",
    "v3_local20km_20260909_v1",
]

NORTHEAST_V3_FLOW_DOMAIN_ID = "northeast_taiwan_common_cache_v3"
NORTHEAST_V3_BBOX_LON_LAT = (121.306315, 122.793685, 24.600844, 25.499156)
NORTHEAST_V3_COMMON_FORCING_PRODUCTS = frozenset(
    {"ocm_native", "ocm_surface", "nww3_analysis"}
)
NORTHEAST_V3_LOCAL_SITE_IDS = frozenset({"gongliao", "guishan"})

# 舊正式輸入採七日缺口安全基線；此值只供相容性查詢，不把所有舊 runtime 或
# 一日工程試跑改成七日。新欄位未明示時，既有建置／執行驗證仍各自維持原有規則。
DEFAULT_BACKTRACK_SUPPORT_DAYS = 7


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
    """上游 root、資料決策、時間正規化、重建證據與回溯支援窗。

    ``backtrack_support_days`` 是要求輸入建置與驗證的完整整日支援窗，填值本身不代表
    已有資料證據。單次執行的回溯長度仍由 ``boundaries.max_backtrack_days`` 指定，且
    不得超過明示的支援上限。欄位採正整數，拒絕布林值、字串、浮點數、零與負數；
    ``None`` 只適合尚未定案的準備性設定，不能建成新版共同母體。未出現在舊 YAML 時，
    保留舊流程的驗證規則，並在設定雜湊中省略由 Pydantic 補出的預設欄位。
    """

    ocm_native_root_env: str
    ocm_surface_root_env: str
    nww_analysis_root_env: str
    years: list[int]
    ocm_contract: dict[str, Any]
    nww_contract: dict[str, Any]
    available_data_contract: dict[str, Any]
    time_axis_contract: dict[str, Any]
    backtrack_support_days: StrictInt | None = None
    ocm_gap_reconstruction_manifest: str | None = None
    ocm_gap_safe_arrival_manifest: str | None = None
    nww_full_hourly_analysis_manifest: str | None = None

    @model_validator(mode="after")
    def validate_backtrack_support_days(self) -> InputContract:
        """確認明示的母體支援窗是正整日；未明示欄位保留舊設定語意。"""

        value = self.backtrack_support_days
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("inputs.backtrack_support_days 必須是正整數日")
        return self


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
    """單一 forcing domain 的 bbox、投影與發布角色。

    ``formal_domain_policy`` 是 versioned research scope，而不是檔案系統中產品是否
    已驗收的旗標。缺少此欄位時固定回到 ``expanded_domain_v1``，讓既有設定維持原本
    的 normalized payload／hash 語意；新設定必須明示 ``v3_local20km_20260909_v1``，
    才會套用 A 區 v3 與 20 km local 的跨欄位契約。
    """

    analysis_region_id: str
    analysis_region_name_zh: str
    flow_domain_id: str
    center_lonlat: tuple[float, float]
    bbox_lon_lat: tuple[float, float, float, float]
    metric_crs_policy: str
    formal_domain_policy: FormalDomainPolicy = FORMAL_DOMAIN_POLICY_EXPANDED_V1
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
    """獨立研究站點與其 local-domain、受體核心生成政策。

    ``anchor_lonlat`` 與 ``receptor_core_radius_m`` 是同一個受體候選核心的兩個
    不可分欄位：前者是經緯度資料交換座標，後者是以站點 flow-domain 的既有 AEQD
    公尺投影計算的半徑。這個 schema-level gate 先於 inputs-build 執行，避免設定只
    填一半、或以無限／非正半徑進入幾何 intersection 後才得到難以定位的錯誤。沒有
    明示核心的站點仍保留既有 local／flow candidate 行為；這裡不會把 local domain
    自動縮成核心圓，也不驗證核心是否落在實際 mesh 或 ocean polygon 內。
    """

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
    # C 區研究者明示的候選子區以 GeoJSON mapping 保存，讓紅框數位化座標、2+3
    # 配額、選點 policy 與來源影像 provenance 進入 normalized config hash。這些欄位
    # 只限制 receptor 候選，不代表 OCM/NWW forcing 支援；實際 input derivation 仍須
    # 與 approved flow/local geometry 交集，並通過 static ocean、persistent wet/dry、
    # NWW 四角及垂向 zcor gate。未設定時保留其他站點的原有 local/flow selector。
    receptor_candidate_regions: list[dict[str, Any]] | None = None
    receptor_candidate_selection: dict[str, Any] | None = None
    receptor_candidate_regions_provenance: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_receptor_core_pair(self) -> StudySiteConfig:
        """驗證受體核心 anchor／半徑成對，且半徑是有限正的公尺值。

        Pydantic 先把 YAML 數值轉成欄位型別；此 validator 再檢查跨欄位條件與有限性。
        ``anchor_lonlat`` 的有限性也一併驗證，因為 ``NaN`` 經緯度會讓後續 AEQD 投影
        與 Shapely intersection 失去可重建性。這個檢查只處理設定內可判定的 schema
        契約；核心與 local/static ocean polygon 的實際交集仍由 input derivation 在
        讀到 domain geometry 後 fail closed。
        """

        has_anchor = self.anchor_lonlat is not None
        has_radius = self.receptor_core_radius_m is not None
        if has_anchor != has_radius:
            raise ValueError(
                f"{self.study_site_id} 的 receptor core 必須同時明示 anchor_lonlat 與 "
                "receptor_core_radius_m"
            )
        if not has_anchor:
            return self
        assert self.anchor_lonlat is not None
        assert self.receptor_core_radius_m is not None
        if not all(math.isfinite(float(value)) for value in self.anchor_lonlat):
            raise ValueError(f"{self.study_site_id} 的 anchor_lonlat 必須是有限數值")
        radius = float(self.receptor_core_radius_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"{self.study_site_id} 的 receptor_core_radius_m 必須是有限正數")
        return self

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

    @model_validator(mode="before")
    @classmethod
    def reject_new_horizon_coercion(cls, value: Any) -> Any:
        """在新支援窗契約啟用時，先拒絕 Pydantic 可能悄悄轉型的布林／字串。

        舊 YAML 沒有 ``backtrack_support_days`` 時完全跳過這個 before gate，以免
        改變既有設定的載入行為。新契約的 requested horizon 若寫成 ``true``、字串
        或其他非數值，不能等 Pydantic 轉成 ``1.0`` 後再判定，否則會把錯誤設定誤認
        成一日執行窗。
        """

        if not isinstance(value, dict):
            return value
        inputs = value.get("inputs")
        if not isinstance(inputs, dict) or "backtrack_support_days" not in inputs:
            return value
        requested = (value.get("boundaries") or {}).get("max_backtrack_days")
        if requested is not None and (
            isinstance(requested, bool) or not isinstance(requested, (int, float))
        ):
            raise ValueError("boundaries.max_backtrack_days 必須是有限正數")
        return value

    @model_validator(mode="after")
    def validate_scientific_contract(self) -> ProjectConfig:
        """驗證五站完整交叉、回溯支援窗、唯一 ID 與 A 區共用 forcing 契約。"""

        self._validate_backtrack_support_contract()

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
        self.assert_research_domain_policy()
        return self

    @property
    def effective_backtrack_support_days(self) -> int | None:
        """回傳輸入母體可供執行設定使用的支援日數。

        舊 YAML 未寫 ``inputs.backtrack_support_days`` 時，回傳 7 日相容基線，供
        legacy artifact 的 release 摘要保持可讀；若 YAML 明示 ``null``，則保留
        「尚未具備支援證據」的狀態。這個屬性只讀設定，不宣告實際產品已通過
        validator；實際支援仍須由 input artifact 的逐到達時刻紀錄證明，也不會把
        7 日預設倒灌到舊 runtime 的研究期檢查。
        """

        if "backtrack_support_days" not in self.inputs.model_fields_set:
            return DEFAULT_BACKTRACK_SUPPORT_DAYS
        return self.inputs.backtrack_support_days

    def _validate_backtrack_support_contract(self) -> None:
        """驗證研究期與輸入母體支援窗的關係，避免執行設定超出實際檢查範圍。

        ``max_backtrack_days`` 保留浮點型別以支援既有 pilot 的小於一日視窗；新支援窗
        契約啟用且已指定執行日數時，必須是有限正數，並且不得超過明示的整日支援窗。
        若支援欄位明示 ``null``，代表母體尚未定案，只有同樣未指定研究期的準備性
        config 可以載入。這裡是設定層的必要邊界，實際資料是否真的覆蓋仍由
        ``input_derivation``、release binding 與 runtime inventory 各自驗證。
        """

        # 舊 YAML 沒有這個欄位時，所有新增支援窗檢查都必須停用；舊 selector、pilot
        # 與 runtime 的既有 horizon 規則仍由各自模組維持，不能因 Pydantic 補出 None
        # 而改變舊 run 的拒絕時機或可接受範圍。
        if "backtrack_support_days" not in self.inputs.model_fields_set:
            return
        support_days = self.inputs.backtrack_support_days
        requested = self.boundaries.max_backtrack_days
        if requested is None:
            return
        if isinstance(requested, bool) or not isinstance(requested, (int, float)):
            raise ValueError("boundaries.max_backtrack_days 必須是有限正數")
        requested_float = float(requested)
        if not math.isfinite(requested_float) or requested_float <= 0.0:
            raise ValueError("boundaries.max_backtrack_days 必須是有限正數")
        if support_days is None:
            raise ValueError(
                "已指定 boundaries.max_backtrack_days，但 inputs.backtrack_support_days 尚未定案"
            )
        if requested_float > float(support_days):
            raise ValueError(
                "boundaries.max_backtrack_days 不得超過 inputs.backtrack_support_days"
            )

    def assert_research_domain_policy(self) -> None:
        """驗證研究範圍 policy 與 domain／site source binding 的完整契約。

        ``expanded_domain_v1`` 維持既有行為：formal A 區可使用設定明示的 expanded
        source，並由 ``assert_formal_release_ready`` 繼續執行原本的 formal gate。
        ``v3_local20km_20260909_v1`` 只允許 A 區精確使用 ``northeast_taiwan_common_cache_v3``
        與原始 bbox，且貢寮、龜山島必須各自保留 20,000 m local、12,500 m receptor core、
        空的本期 local sensitivity 與三套 forcing 的 2 格 margin 宣告。這裡只檢查設定
        內可判定的研究設計與來源綁定；三套產品的實際共同有效格網／20 km 邊界證據仍由
        後續 validator／producer 產出，不能用 ``approved``、任意 boolean 或非空路徑繞過。

        此方法會在 ``ProjectConfig`` schema 驗證及 input derivation 開始時呼叫，讓未知
        policy、A 區錯誤 ID、半徑、核心或 margin 在任何 source I/O 前被拒絕。缺少
        ``formal_domain_policy`` 的舊模型已由欄位預設為 ``expanded_domain_v1``，因此
        不會把舊設定誤套用本期 v3 設計。
        """

        # 版本與 scope 必須雙向綁定。DomainConfig 的 default 只服務 v2 舊 YAML 的
        # hash／載入相容性；若本期 v3 design 省略 policy，這裡不得把 default 當成
        # v3，而要在任何 forcing I/O 前直接拒絕。反向若 policy 已寫成 v3，design
        # 也必須 exact 使用目前唯一版本，避免以任意新舊字串借用本期研究範圍。
        region_a_candidates = [
            domain for domain in self.domains if domain.analysis_region_id == "A"
        ]
        if len(region_a_candidates) != 1:
            raise ValueError("研究範圍版本綁定要求恰有一個 A 區 domain")
        region_a_policy = region_a_candidates[0].formal_domain_policy
        region_a_policy_explicit = "formal_domain_policy" in region_a_candidates[0].model_fields_set
        if self.design_version == CURRENT_DESIGN_VERSION:
            if (
                not region_a_policy_explicit
                or region_a_policy != FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1
            ):
                raise ValueError(
                    f"{CURRENT_DESIGN_VERSION} 必須明示 A 區 formal_domain_policy="
                    f"{FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1}"
                )
        elif region_a_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1:
            raise ValueError(
                f"A 區 {FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1} 只能搭配 "
                f"{CURRENT_DESIGN_VERSION}"
            )
        elif self.design_version != LEGACY_DESIGN_VERSION_V2:
            raise ValueError(
                "design_version 必須是目前 CURRENT_DESIGN_VERSION 或已登錄的 v2 legacy："
                f"{CURRENT_DESIGN_VERSION}、{LEGACY_DESIGN_VERSION_V2}"
            )

        allowed_policies = {
            FORMAL_DOMAIN_POLICY_EXPANDED_V1,
            FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1,
        }
        invalid = [
            f"{domain.analysis_region_id}={domain.formal_domain_policy!r}"
            for domain in self.domains
            if domain.formal_domain_policy not in allowed_policies
        ]
        if invalid:
            raise ValueError("未知 formal_domain_policy：" + ", ".join(invalid))

        v3_domains = [
            domain
            for domain in self.domains
            if domain.formal_domain_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1
        ]
        if not v3_domains:
            return
        if len(v3_domains) != 1 or v3_domains[0].analysis_region_id != "A":
            raise ValueError("v3_local20km_20260909_v1 只能且必須套用 A 區")

        region_a = v3_domains[0]
        if region_a.flow_domain_id != NORTHEAST_V3_FLOW_DOMAIN_ID:
            raise ValueError("v3_local20km_20260909_v1 的 A 區 flow_domain_id 必須 exact v3")
        if region_a.formal_release_flow_domain_id != NORTHEAST_V3_FLOW_DOMAIN_ID:
            raise ValueError("v3_local20km_20260909_v1 的 A 區 formal flow-domain 必須 exact v3")
        if tuple(float(value) for value in region_a.bbox_lon_lat) != NORTHEAST_V3_BBOX_LON_LAT:
            raise ValueError("v3_local20km_20260909_v1 的 A 區 bbox 必須維持 northeast v3 bbox")
        if region_a.formal_release_domain_status != "pending_common_support":
            raise ValueError(
                "v3_local20km_20260909_v1 的 A 區 formal_release_domain_status 必須是 "
                "pending_common_support"
            )

        # 新 policy 不接受任何 expanded candidate 欄位；即使同一根目錄仍有 v4，也不
        # 能由 resolver 轉讀。這些欄位只在 legacy fixture／舊 expanded config 中存在。
        expanded_fields = {
            "expanded_domain_candidate_id",
            "expanded_bbox_lon_lat",
            "expanded_south_boundary_at_or_south_of_deg",
            "radius_25000_formal_requires_expanded_domain",
            "radius_35000_formal_requires_expanded_domain",
        }
        active_expanded = sorted(
            field for field in expanded_fields if field in (region_a.model_extra or {})
        )
        if active_expanded:
            raise ValueError(
                "v3_local20km_20260909_v1 不得含 expanded candidate／mandatory 欄位："
                + ", ".join(active_expanded)
            )

        domain_extra = region_a.model_extra or {}
        domain_margin = domain_extra.get("minimum_common_forcing_margin_grid_cells")
        if (
            isinstance(domain_margin, bool)
            or not isinstance(domain_margin, (int, float))
            or not math.isfinite(float(domain_margin))
            or not math.isclose(float(domain_margin), 2.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                "v3_local20km_20260909_v1 的 A 區 minimum_common_forcing_margin_grid_cells "
                "必須明示為 2"
            )
        forcing_names = domain_extra.get("margin_required_for_forcings")
        if (
            not isinstance(forcing_names, (list, tuple))
            or not all(isinstance(name, str) for name in forcing_names)
            or set(forcing_names) != NORTHEAST_V3_COMMON_FORCING_PRODUCTS
            or len(forcing_names) != len(NORTHEAST_V3_COMMON_FORCING_PRODUCTS)
        ):
            raise ValueError(
                "v3_local20km_20260909_v1 必須明示 ocm_native、ocm_surface、nww3_analysis "
                "三套 forcing margin"
            )

        sites_a = [site for site in self.study_sites if site.analysis_region_id == "A"]
        if {site.study_site_id for site in sites_a} != NORTHEAST_V3_LOCAL_SITE_IDS:
            raise ValueError("v3_local20km_20260909_v1 的 A 區必須恰含 gongliao 與 guishan")
        for site in sites_a:
            if site.flow_domain_id != NORTHEAST_V3_FLOW_DOMAIN_ID:
                raise ValueError(f"{site.study_site_id} 的 v3 policy flow-domain 必須 exact v3")
            if site.formal_release_flow_domain_id != NORTHEAST_V3_FLOW_DOMAIN_ID:
                raise ValueError(f"{site.study_site_id} 的 v3 policy formal flow-domain 必須 exact v3")
            if "radius_35000_requires_expanded_flow_domain" in (site.model_extra or {}):
                raise ValueError(
                    f"{site.study_site_id} 的 v3 policy 不得含 expanded radius mandatory 欄位"
                )
            if site.receptor_core_radius_m is None or not math.isclose(
                float(site.receptor_core_radius_m), 12_500.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{site.study_site_id} 的 v3 policy receptor core 必須是 12500 m")
            if site.local_domain_baseline_radius_m is None or not math.isclose(
                float(site.local_domain_baseline_radius_m), 20_000.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{site.study_site_id} 的 v3 policy local radius 必須是 20000 m")
            if site.local_domain_sensitivity_radii_m:
                raise ValueError(
                    f"{site.study_site_id} 的 v3 policy 本期 local sensitivity 必須是空列表"
                )
            site_extra = site.model_extra or {}
            site_margin = site_extra.get("minimum_flow_domain_margin_local_grid_scales")
            if (
                isinstance(site_margin, bool)
                or not isinstance(site_margin, (int, float))
                or not math.isfinite(float(site_margin))
                or not math.isclose(float(site_margin), 2.0, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError(
                    f"{site.study_site_id} 的 v3 policy local flow-domain margin 必須明示為 2"
                )

        exclusions = (self.boundaries.model_extra or {}).get("sensitivity_case_region_exclusions")
        expanded_exclusion = exclusions.get("expanded_domain") if isinstance(exclusions, dict) else None
        if not isinstance(expanded_exclusion, list) or expanded_exclusion != ["A"]:
            raise ValueError(
                "v3_local20km_20260909_v1 必須明示 expanded_domain 對 A 區的本期 scope exclusion"
            )

    def normalized_payload(self) -> dict[str, Any]:
        """回傳排序前可 JSON 序列化內容，供 hash、manifest 與差異比較。

        新增 policy 欄位採用明示即入 hash 的相容策略。對由舊 YAML 載入、未曾提供
        ``formal_domain_policy`` 的 nested ``DomainConfig``，只在 canonical payload 移除
        Pydantic 的預設值，保留舊 run／checkpoint 的 config hash；只要來源 YAML 明示
        ``expanded_domain_v1`` 或本期 v3 policy，欄位便會留在 payload，形成有意義的
        version boundary。
        """

        payload = self.model_dump(mode="json", exclude_none=False)
        inputs_payload = payload.get("inputs")
        if (
            isinstance(inputs_payload, dict)
            and "backtrack_support_days" in inputs_payload
            and "backtrack_support_days" not in self.inputs.model_fields_set
        ):
            # 未明示的新欄位只是 Pydantic 為了型別完整性補上的 None。刪除它可讓
            # 舊 YAML 保留既有 canonical hash；若 YAML 明示 null，欄位仍會留下來，
            # 讓「尚未具備支援證據」的意圖可被追溯。
            del inputs_payload["backtrack_support_days"]
        domain_payloads = payload.get("domains")
        if isinstance(domain_payloads, list):
            for domain, domain_payload in zip(self.domains, domain_payloads, strict=False):
                if (
                    isinstance(domain_payload, dict)
                    and "formal_domain_policy" in domain_payload
                    and "formal_domain_policy" not in domain.model_fields_set
                ):
                    del domain_payload["formal_domain_policy"]
        # C 區候選設定是本期新增的 optional schema。舊 v2 YAML 沒有這三個 key 時，
        # Pydantic 仍會以 default 建立欄位；若直接將 default 寫入 canonical payload，
        # 會讓既有 run/checkpoint hash 無科學變更地漂移。因此只有 YAML 曾明示欄位時才
        # 讓候選資料進入 hash；明示 null 仍保留，讓人工刪除／停用意圖可被追溯。
        site_payloads = payload.get("study_sites")
        if isinstance(site_payloads, list):
            candidate_fields = (
                "receptor_candidate_regions",
                "receptor_candidate_selection",
                "receptor_candidate_regions_provenance",
            )
            for site, site_payload in zip(self.study_sites, site_payloads, strict=False):
                if not isinstance(site_payload, dict):
                    continue
                for field_name in candidate_fields:
                    if field_name in site_payload and field_name not in site.model_fields_set:
                        del site_payload[field_name]
        return payload

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
        通過前，CLI 不得啟動正式 run。對 ``v3_local20km_20260909_v1``，即使其他
        formal 欄位全部填滿，也必須保留 ``v3/20km共同forcing邊界支援尚待實際驗證``
        blocker；本輪沒有 validator／producer 可提供三套 forcing 的共同有效格網證據，
        因此不可用 approved status、boolean 或非空 manifest path 偽造通過。舊的
        ``expanded_domain_v1`` 則維持原本 expanded A formal ID gate。
        """

        self.assert_research_domain_policy()
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
        if region_a.formal_domain_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1:
            # 此 blocker 刻意不讀取任何「已批准」欄位；實際共同有效格網與 20 km
            # 邊界證據尚未由 validator／producer 建立前，任何完整設定都不能啟動 formal。
            blockers.append("v3/20km共同forcing邊界支援尚待實際驗證")
        else:
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
