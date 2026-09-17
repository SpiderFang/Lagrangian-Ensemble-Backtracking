"""YAML 設定載入、跨欄位科學契約與正式發布閘門。

example config 同時保存已定案設計與尚待 SERVER／pilot 衍生的 ``null`` 欄位。開發模式
允許這些欄位存在，以便合成測試與 pilot 前進；``formal_release=True`` 接受研究團隊核定的
全部可得 2024–2025 資料契約，並依明示的 ``formal_domain_policy`` 套用來源範圍 gate。
舊 ``expanded_domain_v1`` 維持 A 區 expanded source 的既有檢查；新的
``v3_local20km_20260909_v1`` 固定使用 A 區既有 v3 forcing 與兩站 20 km local domain；
空間支援不宣稱預先量測共同網格 margin，而由版本化的 runtime policy 要求每個速度
取樣階段遇到無效 forcing 時 fail closed。這個政策同時要求資料缺口及外層開放邊界
採停止語意；正式輸入仍須獨立通過 accepted products、時間窗、manifest 與 hash 閘門。
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator
from shapely.geometry import Polygon

from .accelerated import PHYSICS_KERNEL_BACKEND_NUMPY_V1
from .bed_residence import (
    BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
    BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
    BED_RESIDENCE_SAMPLING_POLICY_ID,
)
from .gap_policy import (
    APPROVED_OCM_HYBRID_RECONSTRUCTION_POLICY_ID,
    EXCLUDE_DATA_GAP_NUMERICAL_FAILURE_AND_PRE_WINDOW_DEPOSITION_DENOMINATOR_POLICY_ID,
    OBSERVED_GAP_CENSORED_STOP_AT_FIRST_GAP_POLICY_ID,
    REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID,
)

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
# 版本字串。新的 CURRENT_DESIGN_VERSION 同時綁定「2025 observation、2024–2025
# forcing」與「遇第一個已知缺口即截尾」母體政策；前一版 observation-2025 保留為
# 唯讀 legacy，較早的 v3/20 km 與 v2 也只能載入其既有 artifact。所有 v3-family
# 都必須繼續使用同一個 A 區 v3/local20 空間政策，不能藉版本切換回 expanded domain。
LEGACY_DESIGN_VERSION_V3_LOCAL20_20260909 = (
    "design_baseline_v3_non_rising_a_v3_local20_20260909"
)
CURRENT_DESIGN_VERSION = (
    "design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916"
)
# 舊版已選 observation-2025、但尚未把已知 OCM 缺口正式納入母體政策；它只能讀取
# 舊 artifact，不得被當成新版「遇第一個缺口截尾」的正式設計。保留 exact 字串是為
# 了讓既有 manifest／config 可唯讀稽核，不讓新版 loader 以模糊前綴接受它。
LEGACY_DESIGN_VERSION_OBSERVATION_2025 = (
    "design_baseline_v3_non_rising_a_v3_local20_observation_2025_20260916"
)
LEGACY_DESIGN_VERSION_V2 = "design_baseline_v2_non_rising_oca_proxy"
# 提供較短的舊版本別名給既有外部工具／測試；兩個名稱代表完全相同的 v2 legacy
# 字串，實際驗證仍集中在 _validate_design_domain_binding。
LEGACY_DESIGN_VERSION = LEGACY_DESIGN_VERSION_V2

# 這些識別碼是設定資料契約的一部分。``expanded_domain_v1`` 是未明示新政策的
# 舊設定所採用的相容語意；它保留既有 A 區南擴 candidate／formal source 流程。
# ``v3_local20km_20260909_v1`` 描述本期已決定的 A 區研究範圍：兩站共用
# ``northeast_taiwan_common_cache_v3``，local 半徑為 20 km；空間有效性由逐 stage
# runtime gate 處理，不宣稱已量測三套 forcing 的共同網格 margin。
FORMAL_DOMAIN_POLICY_EXPANDED_V1 = "expanded_domain_v1"
FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1 = "v3_local20km_20260909_v1"
RUNTIME_SPATIAL_SUPPORT_POLICY_V3_FAIL_CLOSED_NO_EXPANSION_V1 = (
    "runtime_stage_fail_closed_no_expansion_v1"
)
FORMAL_RELEASE_DOMAIN_STATUS_V3_FAIL_CLOSED_NO_EXPANSION = (
    "no_expansion_runtime_stage_fail_closed"
)
FormalDomainPolicy = Literal[
    "expanded_domain_v1",
    "v3_local20km_20260909_v1",
]
RuntimeSpatialSupportPolicy = Literal[
    "runtime_stage_fail_closed_no_expansion_v1",
]

# OCM 內層插值的版本化選項。這兩個識別碼只描述 OCM 垂向、水平及時間插值所使用的
# 實作，不會改寫 ``production_backend`` 的整體粒子引擎語意；完整 Python RK4、NWW3、
# Stokes、邊界、品質檢查與 checkpoint 仍由既有 reference orchestration 負責。
OCM_INTERPOLATION_BACKEND_NUMPY_V1 = "numpy_v1"
OCM_INTERPOLATION_BACKEND_NUMBA_V1 = "numba_ocm_v1"
OcmInterpolationBackend = Literal[
    "numpy_v1",
    "numba_ocm_v1",
]

# 完整 CPU 數值 primitive 的版本化選項與 OCM 插值開關分開保存；前者控制 Stokes、
# 步長候選、RK4 純量組合與擴散位移，後者仍只控制 OCM 網格內插。
PhysicsKernelBackend = Literal["numpy_v1", "numba_cpu_v1"]

NORTHEAST_V3_FLOW_DOMAIN_ID = "northeast_taiwan_common_cache_v3"
NORTHEAST_V3_BBOX_LON_LAT = (121.306315, 122.793685, 24.600844, 25.499156)
NORTHEAST_V3_LOCAL_SITE_IDS = frozenset({"gongliao", "guishan"})

# 四個 forcing domain 的空間範圍由相鄰的 OCM-SVD-Analysis 正式契約提供。這裡再
# 以不可變常數保存一份輸入閘門，原因是 LBT 只能重用同一套 OCM/NWW 網格，不能因
# 研究站名稱變更而自行平移或縮放 bbox。座標順序固定為
# ``(經度最小值、經度最大值、緯度最小值、緯度最大值)``，所有物理計算仍會在各域
# 的公尺制投影座標進行。
FORMAL_FLOW_DOMAIN_BBOXES_LON_LAT = {
    "A": NORTHEAST_V3_BBOX_LON_LAT,
    "B": (119.70812, 121.19188, 24.300844, 25.199156),
    "C": (120.16671, 121.62, 21.550844, 22.449156),
    "D": (119.19912, 120.70088, 25.750844, 26.649156),
}
FORMAL_FLOW_DOMAIN_IDS_BY_REGION = {
    "A": "northeast_taiwan_common_cache_v3",
    "B": "hsinchu_cache_v3",
    "C": "houwan_nmmba_cache_v3",
    "D": "lienchiang_common_cache_v3",
}

# 現行正式母體以南灣為 C 區研究站；``houwan`` 只會出現在明確標示的歷史／唯讀
# artifact。將 site ID 集中定義，可讓 config、scenario coverage 與後續 validator
# 使用相同的五站集合，避免 C 區只改中文名稱卻遺留舊站點識別碼。
CURRENT_FORMAL_STUDY_SITE_IDS = frozenset(
    {"gongliao", "guishan", "hsinchu", "nanwan", "lienchiang"}
)
LEGACY_FORMAL_STUDY_SITE_IDS = frozenset(
    {"gongliao", "guishan", "hsinchu", "houwan", "lienchiang"}
)

# 正式圖面核對後固定的研究站 anchor。這些數值是 WGS84 資料交換座標；受體核心的
# 公尺距離與 mesh 定位仍由 input derivation 依 flow-domain projection 驗證。A 區
# 貢寮 anchor 特別保留核對圖的高精度值，不沿用舊版約略位置。
FORMAL_STUDY_SITE_ANCHORS_LON_LAT = {
    "gongliao": (121.9223889, 25.0964444),
    "guishan": (121.951606, 24.843127),
    "hsinchu": (120.45, 24.75),
    "nanwan": (120.763161, 21.946577),
    "lienchiang": (119.95, 26.2),
}

# 新竹 24 小時工程試跑已核定的五個水平受體順序與來源 manifest 摘要。正式母體
# 必須逐點在同一份 OCM 原生 mesh 尋找 persistent-wet face，不得重新以 maximin
# 抽出另一組候選；來源 manifest 不放進 Git，但其 SHA-256 會和座標順序一併寫入
# canonical config 與 receptor provenance。
HSINCHU_FIXED_HORIZONTAL_RECEPTOR_COORDINATES_LON_LAT = (
    (120.32880147298177, 24.730445861816406),
    (120.39448547363281, 24.85012690226237),
    (120.45531717936198, 24.75184504191081),
    (120.47039794921875, 24.640532811482746),
    (120.57318623860677, 24.753894170125324),
)
HSINCHU_FIXED_HORIZONTAL_RECEPTOR_SOURCE_SHA256 = (
    "f625c3cbaf339220994e14926c5a2bd65974ec40706e0d81c0fa3f7404128055"
)
HSINCHU_FIXED_HORIZONTAL_RECEPTOR_POLICY_ID = (
    "hsinchu_24h_pilot_fixed_horizontal_manifest_coordinates_v1"
)

# 龜山島受體選點採「soft priority」而非第二個研究域：這個 WGS84 polygon 只在已由
# anchor 及 12.5 km core、local/flow geometry 與 persistent-wet 篩出的候選中提供優先
# 順序。若走廊內有效 face 不足五個，input-build 必須退回同一個 12.5 km core，不能
# 擴大 local/domain、改寫 forcing bbox 或用走廊取代後續 NWW/OCM 垂向支援 gate。
GUISHAN_SOFT_PRIORITY_POLYGON_LON_LAT = (
    (121.78, 24.79),
    (121.91, 24.76),
    (121.94, 24.80),
    (121.94, 24.90),
    (121.82, 24.92),
    (121.76, 24.87),
)
GUISHAN_SOFT_PRIORITY_POLICY_ID = "guishan_soft_priority_corridor_core_fallback_v1"

# 現行正式母體的受體選樣契約。水平與垂向分開命名，讓 manifest 能辨識「從哪一個
# 候選 face 池抽樣」與「在單一 face 的完整水柱內抽哪四個 normalized fraction」；兩者
# 都由同一個 master seed 經 SHA-256 派生，但每個站點／face 使用獨立 stream。這些
# token 是資料契約而非實作細節，任何固定座標、maximin 或 10/40/70/near-bed target
# 都不能冒充目前正式母體的 random policy。
HORIZONTAL_RECEPTOR_SELECTION_POLICY_ID = (
    "seeded_uniform_random_without_replacement_v1"
)
VERTICAL_RECEPTOR_SELECTION_POLICY_ID = "seeded_uniform_random_open_interval_v1"
RECEPTOR_SELECTION_SEED_POLICY_ID = "sha256_v1_pcg64dxsm"
RANDOM_VERTICAL_RECEPTOR_ID_PREFIX = "random_vertical_draw"

# 舊正式輸入採七日缺口安全基線；此值只供相容性查詢，不把所有舊 runtime 或
# 一日工程試跑改成七日。新欄位未明示時，既有建置／執行驗證仍各自維持原有規則。
DEFAULT_BACKTRACK_SUPPORT_DAYS = 7

# 新版 arrival policy 的固定識別碼。policy 不只是文件標籤，而是 selector 的分層
# 數量、observation 年份與 metadata 欄位的資料契約；寫入設定／manifest 後不可由
# 呼叫端自行改用另一套選時算法。
ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1 = "observation_year_stratified_48_plus_2_v1"
ARRIVAL_SELECTION_POLICY_LEGACY_TWO_YEAR_V1 = "two_years_stratified_48_plus_2_v1"


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
    # 已核准 OCM hybrid reconstruction patch 的 root 環境變數名稱；路徑只在 runtime
    # CLI／worker 啟動時解析，絕不寫入 immutable run plan。舊 YAML 未明示時保持 None。
    ocm_reconstruction_root_env: str | None = None
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
    才會套用 A 區 v3 與 20 km local 的跨欄位契約。A 區另以
    ``runtime_spatial_support_policy`` 明示逐 stage 空間支援檢查版本；其值不代表任何
    產品 margin 已量測或通過，只指定粒子 runtime 遇到無效 forcing 時採停止語意。
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
    runtime_spatial_support_policy: RuntimeSpatialSupportPolicy | None = None

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
    # 某些站點的水平受體已由研究者透過上一輪工程試跑核定。若有明示座標，正式
    # input-build 必須依此順序逐點映射到同一份原生 OCM mesh；這些欄位不是把受體
    # 當成 forcing domain，也不繞過 persistent-wet、50 個 arrival 與垂向支援 gate。
    horizontal_receptor_coordinates: list[tuple[float, float]] | None = None
    horizontal_receptor_source_manifest_sha256: str | None = None
    horizontal_receptor_selection_policy: str | None = None
    horizontal_receptor_coordinate_tolerance_m: float | None = None
    # 龜山島走廊是受體候選的 soft priority polygon，不是新的 local/flow domain。它
    # 只會在 core∩local 的 persistent-wet pool 中優先選點；不足五點時回到同一 core
    # pool，且兩條路徑都必須通過 NWW exact-hour 與 OCM 垂向支援檢查。
    receptor_priority_polygon: list[tuple[float, float]] | None = None
    receptor_priority_selection: dict[str, Any] | None = None
    # 舊後灣 pilot 曾以 GeoJSON 紅框保存 2+3 候選、選點 policy 與影像 provenance。
    # 欄位保留給歷史唯讀 loader，現行南灣正式設定不可再填入；未設定時沿用一般
    # core/local selector，並由 input derivation 逐站執行 static ocean、persistent
    # wet/dry、NWW 四角與垂向 zcor gate。
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

    @model_validator(mode="after")
    def validate_fixed_horizontal_receptors(self) -> StudySiteConfig:
        """驗證歷史固定水平受體的座標、順序摘要與公尺制匹配公差。

        固定受體座標來自已核定的外部工程 manifest，並不代表這些點一定能在另一個
        OCM mesh 上使用。因此 schema 只先檢查座標數量、WGS84 範圍、來源 SHA-256
        格式與公尺制匹配公差；實際是否為同一 source face、是否 persistent-wet 以及
        是否通過所有 arrival 的 OCM/NWW 支援，仍由 input derivation 在讀取 accepted
        products 後逐點驗證。這些欄位目前只供明確標示的歷史／唯讀 pilot；current
        formal 會在 ``ProjectConfig.validate_scientific_contract`` 另外拒絕它們。未
        提供固定座標的舊設定維持原本 deterministic maximin 相容行為。
        """

        coordinates = self.horizontal_receptor_coordinates
        if coordinates is None:
            if any(
                value is not None
                for value in (
                    self.horizontal_receptor_source_manifest_sha256,
                    self.horizontal_receptor_selection_policy,
                    self.horizontal_receptor_coordinate_tolerance_m,
                )
            ):
                raise ValueError(
                    f"{self.study_site_id} 固定水平受體欄位必須和 horizontal_receptor_coordinates 一起明示"
                )
            return self
        if self.receptor_priority_polygon is not None or self.receptor_priority_selection is not None:
            # 固定點與 priority 都是「如何選出五個水平 face」的互斥契約；若同時
            # 存在，selector 無法判定應以宣告座標還是走廊偏好為準，容易把 B 區
            # 已核定位置悄悄換成另一組。因此在 schema 層直接拒絕，而非讓 builder
            # 依欄位順序產生難以追溯的結果。
            raise ValueError(
                f"{self.study_site_id} 固定水平受體不得同時設定 receptor priority polygon"
            )
        if len(coordinates) != 5:
            raise ValueError(
                f"{self.study_site_id}.horizontal_receptor_coordinates 必須恰有 5 個水平位置"
            )
        for index, coordinate in enumerate(coordinates):
            if len(coordinate) != 2 or not all(math.isfinite(float(value)) for value in coordinate):
                raise ValueError(
                    f"{self.study_site_id}.horizontal_receptor_coordinates[{index}] 必須是有限 lon/lat"
                )
            lon, lat = (float(value) for value in coordinate)
            if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
                raise ValueError(
                    f"{self.study_site_id}.horizontal_receptor_coordinates[{index}] 超出 WGS84 bounds"
                )
        source_hash = self.horizontal_receptor_source_manifest_sha256
        if source_hash is not None:
            if not isinstance(source_hash, str) or len(source_hash) != 64:
                raise ValueError(
                    f"{self.study_site_id}.horizontal_receptor_source_manifest_sha256 必須是 64 位 SHA-256"
                )
            try:
                int(source_hash, 16)
            except ValueError as exc:
                raise ValueError(
                    f"{self.study_site_id}.horizontal_receptor_source_manifest_sha256 必須是十六進位字串"
                ) from exc
            if source_hash.lower() != source_hash:
                raise ValueError(
                    f"{self.study_site_id}.horizontal_receptor_source_manifest_sha256 必須使用小寫"
                )
        tolerance = self.horizontal_receptor_coordinate_tolerance_m
        if tolerance is not None and (
            not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0
        ):
            raise ValueError(
                f"{self.study_site_id}.horizontal_receptor_coordinate_tolerance_m 必須是有限正數"
            )
        return self

    @model_validator(mode="after")
    def validate_receptor_priority_polygon(self) -> StudySiteConfig:
        """驗證 soft-priority 候選走廊的 WGS84 幾何與 policy 綁定。

        priority polygon 只描述在既有 12.5 km receptor core 內的選點偏好，不是 forcing
        domain、local boundary 或資料支援範圍。schema 層先拒絕退化、自交與超出 WGS84
        的座標；實際與 core/local polygon 的交集、persistent-wet face 數量與 forcing
        支援仍由 input derivation 逐站計算。沒有 polygon 時不能單獨留下 selection
        policy，避免設定看似啟用優先走廊卻沒有幾何可追溯。
        """

        polygon = self.receptor_priority_polygon
        policy = self.receptor_priority_selection
        if polygon is None:
            if policy is not None:
                raise ValueError(
                    f"{self.study_site_id} receptor_priority_selection 必須和 polygon 一起明示"
                )
            return self
        if len(polygon) < 3:
            raise ValueError(
                f"{self.study_site_id}.receptor_priority_polygon 至少需要三個頂點"
            )
        values: list[tuple[float, float]] = []
        for index, coordinate in enumerate(polygon):
            if len(coordinate) != 2:
                raise ValueError(
                    f"{self.study_site_id}.receptor_priority_polygon[{index}] 必須是 lon/lat"
                )
            lon, lat = (float(value) for value in coordinate)
            if not math.isfinite(lon) or not math.isfinite(lat):
                raise ValueError(
                    f"{self.study_site_id}.receptor_priority_polygon[{index}] 必須是有限數值"
                )
            if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
                raise ValueError(
                    f"{self.study_site_id}.receptor_priority_polygon[{index}] 超出 WGS84 bounds"
                )
            values.append((lon, lat))
        polygon_geometry = Polygon(values)
        if polygon_geometry.is_empty or not polygon_geometry.is_valid or polygon_geometry.area <= 0.0:
            raise ValueError(
                f"{self.study_site_id}.receptor_priority_polygon 必須是有效、非退化 Polygon"
            )
        if policy is not None:
            if policy.get("policy_id") != GUISHAN_SOFT_PRIORITY_POLICY_ID:
                raise ValueError("receptor_priority_selection.policy_id 未登錄")
            if policy.get("fallback") != "same_core_pool":
                raise ValueError("receptor_priority_selection.fallback 必須是 same_core_pool")
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


class BedResidenceTimeConfig(StrictModel):
    """五站共用沉底年齡抽樣與回溯模式切換的嚴格契約。

    此區塊只有在 YAML 明示時才啟用；啟用後所有欄位都必須明確提供，且固定為本期已定
    案的 90 日、每站 50 個整點小時年齡及既有抽樣 policy。支援的兩種 backtrack mode
    必須同時登錄，實際執行只由 backtrack_mode 選擇一種。runtime_horizon_support_days
    可在尚未建立 horizon suite 的 template 設為 null；suite 產生的正式設定須填正整數。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    backtrack_mode: Literal[
        "fixed_calendar_window",
        "full_horizon_from_deposition",
    ]
    supported_backtrack_modes: tuple[
        Literal["fixed_calendar_window", "full_horizon_from_deposition"], ...
    ]
    maximum_age_days: StrictInt
    sample_count_per_site: StrictInt
    sampling_policy: Literal["discrete_hourly_stratified_uniform_v1"]
    # 新版缺口條件式母體要求每一分層只接受五站均有 exact deposition hour 的
    # 年齡；欄位 optional 是為了載入舊 1.1.0 artifact，normalized_payload 會保留
    # 舊 hash。新版 CURRENT_DESIGN_VERSION 則在 ProjectConfig 層強制明示 exact ID。
    availability_conditioning_policy: Literal[
        "reject_unavailable_deposition_hour_within_stratum_v1"
    ] | None = None
    sampling_seed: StrictInt
    shared_age_offsets_across_sites: StrictBool
    pre_window_policy: Literal[
        "record_pre_window_deposition_without_transport"
    ]
    runtime_horizon_support_days: StrictInt | None

    @model_validator(mode="after")
    def validate_bed_residence_contract(self) -> BedResidenceTimeConfig:
        """拒絕任何未經定案的抽樣範圍、模式清單或 pre-window 處置方式。"""

        required_modes = {
            BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
            BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
        }
        if len(self.supported_backtrack_modes) != 2 or set(
            self.supported_backtrack_modes
        ) != required_modes:
            raise ValueError(
                "scenarios.bed_residence_time.supported_backtrack_modes 必須恰含兩種已定案模式"
            )
        if self.backtrack_mode not in self.supported_backtrack_modes:
            raise ValueError(
                "scenarios.bed_residence_time.backtrack_mode 必須列於 supported_backtrack_modes"
            )
        if self.maximum_age_days != 90:
            raise ValueError(
                "scenarios.bed_residence_time.maximum_age_days 必須固定為 90"
            )
        if self.sample_count_per_site != 50:
            raise ValueError(
                "scenarios.bed_residence_time.sample_count_per_site 必須固定為 50"
            )
        if self.sampling_policy != BED_RESIDENCE_SAMPLING_POLICY_ID:
            raise ValueError(
                "scenarios.bed_residence_time.sampling_policy 不支援；不得靜默採用其他算法"
            )
        if self.availability_conditioning_policy not in {
            None,
            REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID,
        }:
            raise ValueError(
                "scenarios.bed_residence_time.availability_conditioning_policy 不支援"
            )
        if self.sampling_seed < 0:
            raise ValueError(
                "scenarios.bed_residence_time.sampling_seed 必須是非負嚴格整數"
            )
        if self.shared_age_offsets_across_sites is not True:
            raise ValueError(
                "scenarios.bed_residence_time.shared_age_offsets_across_sites 必須為 true"
            )
        if (
            self.runtime_horizon_support_days is not None
            and self.runtime_horizon_support_days < 1
        ):
            raise ValueError(
                "scenarios.bed_residence_time.runtime_horizon_support_days 必須為正整數或 null"
            )
        return self


class ArrivalTimeSelectionConfig(StrictModel):
    """到達時次母體的分層、年份範圍與 replicate 契約。

    舊 YAML 已存在 ``core_design``、``core_count`` 等說明欄位，但尚未把
    observation 年份從 forcing 年份分離；因此新增欄位全部採 optional，只有 YAML
    明示新版 ``policy`` 時才啟用新版跨欄位 gate。新版正式母體以
    ``observation_years`` 指定可作為到達錨點的年份，``inputs.years`` 則仍代表實際
    讀取的 forcing 年份。``replicates=2`` 代表每個「季節 × spring/neap × 相位」
    stratum 選兩筆不同 UTC，總數固定為 4×2×2×3=48，再加兩個事件。

    ``extra=allow`` 是為了讀取舊版文件型 policy 欄位；本類別只對目前已定案的欄位
    提供型別與數值驗證，未知欄位不會被拿來改變 selector 行為。未明示新版欄位時，
    selector 仍使用既有兩年份各四季的 48+2 行為，且 canonical config hash 不會
    因 Pydantic 補出 None 而漂移。
    """

    policy: str | None = None
    core_design: str
    core_count: StrictInt
    observation_years: list[StrictInt] | None = None
    # 舊 typed selection block 沒有新版 replicate 欄位時，對外解析必須仍明確呈現
    # 「每個兩年份 stratum 一筆」的 legacy 預設；normalized_payload 會依
    # model_fields_set 移除這個補出的 1，故不改變舊 hash。
    replicates: StrictInt = 1
    tidal_phase_proxies: list[str]
    event_supplement_count: StrictInt
    event_supplements: list[str]
    deterministic_tie_break: str
    northeast_pair_utc_when_coverage_allows: StrictBool
    decision_status: str

    @model_validator(mode="after")
    def validate_selection_contract(self) -> ArrivalTimeSelectionConfig:
        """驗證新版 policy 的固定分層數量，避免設定與 selector 靜默分歧。"""

        if self.core_count < 1:
            raise ValueError("arrival_time_selection.core_count 必須是正整數")
        if self.event_supplement_count < 0:
            raise ValueError("arrival_time_selection.event_supplement_count 不得為負數")
        if self.observation_years is not None:
            years = [int(year) for year in self.observation_years]
            if not years or len(set(years)) != len(years):
                raise ValueError(
                    "arrival_time_selection.observation_years 必須是非空且不重複的年份"
                )
        if self.policy == ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1:
            if self.core_count != 48:
                raise ValueError(
                    "新版 observation arrival policy 的 core_count 必須固定為 48"
                )
            if self.event_supplement_count != 2:
                raise ValueError(
                    "新版 observation arrival policy 的 event_supplement_count 必須固定為 2"
                )
            if self.replicates != 2:
                raise ValueError(
                    "新版 observation arrival policy 的 replicates 必須固定為 2"
                )
            if self.observation_years is None:
                raise ValueError(
                    "新版 observation arrival policy 必須明示 observation_years"
                )
        elif self.policy is not None and self.policy != ARRIVAL_SELECTION_POLICY_LEGACY_TWO_YEAR_V1:
            raise ValueError(f"不支援的 arrival_time_selection.policy：{self.policy!r}")
        return self


class ScenarioConfig(StrictModel):
    """五站完整交叉、member/seed、沉底時間與受體 random policy 契約。

    受體 policy 欄位在舊 YAML 中仍可為 ``None``，以保持歷史 loader 的相容性；現行
    formal design 則必須明示水平、垂向與 seed policy，避免 extra 欄位拼字錯誤後悄悄
    回到固定座標或固定水層。``master_seed`` 在設計樣板可先留空，正式 input-build
    之前的 release gate 會再要求實際整數 seed。
    """

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
    horizontal_receptor_selection_policy: str | None = None
    vertical_receptor_selection_policy: str | None = None
    receptor_selection_seed_policy: str | None = None
    bed_residence_time: BedResidenceTimeConfig | None = None


class ExecutionConfig(StrictModel):
    """CPU 數值後端、分片、checkpoint cadence 與 forcing cache 的工程欄位。

    ``checkpoint_interval_sweeps`` 的單位是完整批次 sweep，不是輸出觀測點；兩者在
    adaptive time-step 下不等價。``active_chunk_size`` 控制一次散射／回寫的粒子數，
    ``max_resident_forcing_months`` 控制單一 process 的月份 cache 上限。
    ``ocm_interpolation_backend`` 單獨選擇 OCM 垂向／水平／時間插值的 NumPy 或 Numba
    kernel；``physics_kernel_backend`` 選擇有限水深 Stokes、步長候選、RK4 最後 scalar
    組合／時間更新與擴散位移的參考或 Numba CPU primitive。後者不改變 Scenario、seed、
    RK4 stage 查詢順序、品質檢查旗標（QC）、事件、邊界、checkpoint 或缺值政策。兩欄
    未出現在舊 YAML 時分別沿用 NumPy 語意；canonical payload 會省略 Pydantic 補上的預設，
    維持舊 run 的 config hash。明示任一後端時，版本 token 會進入 hash 供重現與追溯。
    """

    reference_backend: str
    production_backend: str
    ocm_interpolation_backend: OcmInterpolationBackend = OCM_INTERPOLATION_BACKEND_NUMPY_V1
    physics_kernel_backend: PhysicsKernelBackend = PHYSICS_KERNEL_BACKEND_NUMPY_V1
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
    """保存局部／外層邊界及資料缺口的停止政策。

    ``stop_at_forcing_start`` 與 ``stop_at_data_gap`` 原先由 YAML 額外欄位保存；將它們
    明列為嚴格布林欄位，讓 A 區 v3 設定能驗證確實採用停止語意。欄位保持 optional，
    以免改變未使用 v3 政策的舊設定載入與 canonical hash。
    """

    local_domain_first_exit: str
    other_site_local_domain_enter: str
    other_site_local_domain_exit: str
    other_site_local_domain_changes_study_site: bool
    flow_domain_open_boundary: str
    stop_at_forcing_start: StrictBool | None = None
    stop_at_data_gap: StrictBool | None = None
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
    arrival_time_selection: ArrivalTimeSelectionConfig | None = None
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
        if self.design_version == CURRENT_DESIGN_VERSION:
            # 現行正式母體的四區 bbox 是 OCM-SVD-Analysis 的上游空間契約。C 區
            # 改為南灣研究站只會改變 site 層 anchor，不得把 houwan forcing domain
            # 平移成另一個 nanwan bbox；任何一個區域不符便在讀取大型產品前停止。
            expected_bboxes = FORMAL_FLOW_DOMAIN_BBOXES_LON_LAT
            for domain in self.domains:
                expected_bbox = expected_bboxes.get(domain.analysis_region_id)
                actual_bbox = tuple(float(value) for value in domain.bbox_lon_lat)
                if expected_bbox is None or actual_bbox != expected_bbox:
                    raise ValueError(
                        f"{domain.analysis_region_id} flow-domain bbox 必須 exact 沿用 OCM-SVD-Analysis"
                    )
                expected_flow_id = FORMAL_FLOW_DOMAIN_IDS_BY_REGION[domain.analysis_region_id]
                if domain.flow_domain_id != expected_flow_id:
                    raise ValueError(
                        f"{domain.analysis_region_id} flow-domain ID 必須是 exact v3／{expected_flow_id}"
                    )
        if self.design_version == CURRENT_DESIGN_VERSION and set(site_ids) != set(
            CURRENT_FORMAL_STUDY_SITE_IDS
        ):
            raise ValueError(
                "目前正式設計的 study_site_id 必須恰含 gongliao、guishan、hsinchu、nanwan、lienchiang；"
                "A 區必須恰含 gongliao 與 guishan"
            )
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
        if self.design_version == CURRENT_DESIGN_VERSION:
            sites_by_id = {site.study_site_id: site for site in self.study_sites}
            for site_id, expected_anchor in FORMAL_STUDY_SITE_ANCHORS_LON_LAT.items():
                site = sites_by_id[site_id]
                actual_anchor = (
                    tuple(float(value) for value in site.anchor_lonlat)
                    if site.anchor_lonlat is not None
                    else None
                )
                if actual_anchor != expected_anchor:
                    raise ValueError(f"{site_id} anchor 必須 exact 沿用四區五站核對圖")
            nanwan = sites_by_id["nanwan"]
            if nanwan.study_site_name_zh != "南灣":
                raise ValueError("C 區目前 study_site_name_zh 必須是南灣")
            if nanwan.analysis_region_id != "C" or nanwan.flow_domain_id != "houwan_nmmba_cache_v3":
                raise ValueError("南灣必須位於 C 區並沿用 houwan_nmmba_cache_v3 forcing")
            if any(
                value is not None
                for value in (
                    nanwan.receptor_candidate_regions,
                    nanwan.receptor_candidate_selection,
                    nanwan.receptor_candidate_regions_provenance,
                )
            ):
                raise ValueError("現行南灣正式設定不得保留後灣紅框 2+3 候選或其 provenance")
            # 目前正式母體已裁決五站都以 seeded random 建立水平面；固定座標與其
            # 來源 hash/tolerance 只可留在歷史 24 小時試跑 loader。欄位保留在 schema
            # 是為了能唯讀讀取舊文件，但一旦進入 current design 就 fail closed，避免
            # B 區五點被誤當成正式母體或其他站點複製使用。
            fixed_site_fields = (
                "horizontal_receptor_coordinates",
                "horizontal_receptor_source_manifest_sha256",
                "horizontal_receptor_selection_policy",
                "horizontal_receptor_coordinate_tolerance_m",
            )
            for site in self.study_sites:
                if any(getattr(site, field_name) is not None for field_name in fixed_site_fields):
                    raise ValueError(
                        f"current formal 的 {site.study_site_id} 不得設定固定水平受體；"
                        "固定五點僅屬歷史 24 小時 pilot"
                    )
            expected_receptor_policies = {
                "horizontal_receptor_selection_policy": (
                    HORIZONTAL_RECEPTOR_SELECTION_POLICY_ID
                ),
                "vertical_receptor_selection_policy": VERTICAL_RECEPTOR_SELECTION_POLICY_ID,
                "receptor_selection_seed_policy": RECEPTOR_SELECTION_SEED_POLICY_ID,
            }
            for field_name, expected_policy in expected_receptor_policies.items():
                actual_policy = getattr(self.scenarios, field_name)
                if actual_policy != expected_policy:
                    raise ValueError(
                        f"scenarios.{field_name} 必須是 current formal random policy "
                        f"{expected_policy!r}"
                    )
            # ``physics.vertical_targets`` 是舊版 10/40/70/near-bed 固定水層設定；即使
            # 新欄位已明示，也不能讓 extra=allow 把兩套垂向語意同時帶入正式建置。
            if self.scenarios.model_extra and "vertical_targets" in self.scenarios.model_extra:
                raise ValueError(
                    "current formal 不得保留 scenarios.vertical_targets 固定垂向 target；"
                    "請使用 seeded random open-interval policy"
                )
            legacy_selector = (
                self.scenarios.model_extra.get("horizontal_selector")
                if self.scenarios.model_extra
                else None
            )
            if isinstance(legacy_selector, str) and (
                "maximin" in legacy_selector.lower() or "fixed" in legacy_selector.lower()
            ):
                raise ValueError(
                    "current formal 不得使用 deterministic maximin/fixed horizontal selector"
                )
            guishan = sites_by_id["guishan"]
            actual_priority_polygon = tuple(
                tuple(float(value) for value in coordinate)
                for coordinate in (guishan.receptor_priority_polygon or ())
            )
            if actual_priority_polygon != GUISHAN_SOFT_PRIORITY_POLYGON_LON_LAT:
                raise ValueError("龜山島 soft priority polygon 必須 exact 沿用核對契約")
            if guishan.receptor_priority_selection is None or guishan.receptor_priority_selection.get(
                "policy_id"
            ) != GUISHAN_SOFT_PRIORITY_POLICY_ID:
                raise ValueError("龜山島 receptor priority policy 不符")
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
        bed_residence = counts.bed_residence_time
        if (
            bed_residence is not None
            and bed_residence.sample_count_per_site
            != counts.expected_arrival_time_count_per_site
        ):
            raise ValueError(
                "scenarios.bed_residence_time.sample_count_per_site 必須等於 "
                "expected_arrival_time_count_per_site"
            )
        _validate_non_rising_material_contract(
            self.physics.get("settling"),
            expected_count=counts.expected_material_count,
        )
        if self.boundaries.other_site_local_domain_changes_study_site:
            raise ValueError("foreign-local crossing 不得改變 study_site_id")
        self.assert_research_domain_policy()
        self._validate_arrival_time_selection_contract()
        self._validate_gap_censoring_contract()
        return self

    def _validate_arrival_time_selection_contract(self) -> None:
        """驗證 observation 年份是 forcing 子集，並綁定新版設計身分。

        ``inputs.years`` 是 input builder 需要讀取的完整 forcing 聯集；它可以包含
        為回溯窗口提供前置資料的 2024。新版 selector 只允許
        ``arrival_time_selection.observation_years`` 中的年份作 arrival anchor，且
        這些年份必須是 forcing 聯集的非空子集。此 gate 放在 config 層，能在讀取
        SERVER 大型產品前拒絕把不存在的 observation 年份送入 selector。
        """

        selection = self.arrival_time_selection
        if selection is None:
            # 舊 YAML 未宣告 typed selection block 時，保留舊 selector 的兩年份行為與
            # canonical hash；輸入 builder 會由實際 forcing 軸依 legacy policy 運作。
            if self.design_version == CURRENT_DESIGN_VERSION:
                raise ValueError(
                    f"{CURRENT_DESIGN_VERSION} 必須明示 arrival_time_selection"
                )
            return
        forcing_years = [int(year) for year in self.inputs.years]
        if len(set(forcing_years)) != len(forcing_years) or not forcing_years:
            raise ValueError("inputs.years 必須是非空且不重複的年份")
        observation_years = selection.observation_years
        if observation_years is not None:
            observation_set = {int(year) for year in observation_years}
            if not observation_set.issubset(set(forcing_years)):
                raise ValueError(
                    "arrival_time_selection.observation_years 必須是 inputs.years 的子集"
                )
        if selection.policy == ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1:
            if self.design_version not in {
                CURRENT_DESIGN_VERSION,
                LEGACY_DESIGN_VERSION_OBSERVATION_2025,
            }:
                raise ValueError(
                    "新版 observation arrival policy 必須搭配目前 gap-censored design 或其唯讀 legacy"
                )
            if observation_years is None or not observation_years:
                raise ValueError("新版 observation arrival policy 必須有 observation_years")
        elif self.design_version == CURRENT_DESIGN_VERSION:
            raise ValueError(
                f"{CURRENT_DESIGN_VERSION} 必須使用 "
                f"{ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1}"
            )

    def _validate_gap_censoring_contract(self) -> None:
        """驗證新版母體確實把已知 OCM 缺口當成可稽核的截尾政策。

        新版不宣稱 2024--2025 forcing 已連續，也不要求輸入建置把缺口補齊；它只
        要求設定把「遇第一個缺口停止」、「統計分母如何處理截尾成員」與沉底起點
        exact-hour 可用性條件寫入 canonical config。舊 observation-2025 config
        可以唯讀載入而不補寫欄位，但一旦宣稱新 design，三者缺一即在任何大型輸入
        I/O 前 fail closed。
        """

        time_contract = self.inputs.time_axis_contract
        if not isinstance(time_contract, dict):
            raise ValueError("inputs.time_axis_contract 必須是 mapping")
        if self.design_version == CURRENT_DESIGN_VERSION:
            configured_reconstruction_policy = time_contract.get("reconstruction_policy")
            if configured_reconstruction_policy == APPROVED_OCM_HYBRID_RECONSTRUCTION_POLICY_ID:
                # 範例設定可先宣告未來要採用的正式 policy，但在 design_pending 階段
                # 尚未產出 manifest 時仍須能被工具載入，讓 input-build 能依同一份
                # canonical config 產生 artifact。只有 approved 且連 gap-safe 替代品也
                # 沒有時，才在 schema 層拒絕；正式 release 仍由下方 formal gate 重驗。
                if (
                    self.config_status == "approved"
                    and not self.inputs.ocm_gap_reconstruction_manifest
                    and not self.inputs.ocm_gap_safe_arrival_manifest
                ):
                    raise ValueError(
                        f"{CURRENT_DESIGN_VERSION} reconstruction baseline 必須明示 "
                        "inputs.ocm_gap_reconstruction_manifest 或 "
                        "inputs.ocm_gap_safe_arrival_manifest"
                    )
                if not self.inputs.ocm_reconstruction_root_env:
                    raise ValueError(
                        f"{CURRENT_DESIGN_VERSION} reconstruction baseline 必須明示 "
                        "inputs.ocm_reconstruction_root_env"
                    )
                if time_contract.get("gap_policy") != OBSERVED_GAP_CENSORED_STOP_AT_FIRST_GAP_POLICY_ID:
                    raise ValueError(
                        f"{CURRENT_DESIGN_VERSION} reconstruction baseline 仍須保留 "
                        "inputs.time_axis_contract.gap_policy=observed_gap_censored_stop_at_first_gap_v1"
                    )
                if time_contract.get("stop_at_first_gap") is not True:
                    raise ValueError(
                        f"{CURRENT_DESIGN_VERSION} reconstruction baseline 必須明示 "
                        "inputs.time_axis_contract.stop_at_first_gap=true"
                    )
                if (
                    time_contract.get("denominator_policy")
                    != EXCLUDE_DATA_GAP_NUMERICAL_FAILURE_AND_PRE_WINDOW_DEPOSITION_DENOMINATOR_POLICY_ID
                ):
                    raise ValueError(
                        f"{CURRENT_DESIGN_VERSION} reconstruction baseline 必須明示 "
                        "inputs.time_axis_contract.denominator_policy="
                        f"{EXCLUDE_DATA_GAP_NUMERICAL_FAILURE_AND_PRE_WINDOW_DEPOSITION_DENOMINATOR_POLICY_ID}"
                    )
                if self.boundaries.stop_at_data_gap is not True:
                    raise ValueError(
                        f"{CURRENT_DESIGN_VERSION} reconstruction baseline 必須設定 "
                        "boundaries.stop_at_data_gap=true"
                    )
                bed = self.scenarios.bed_residence_time
                if bed is not None and bed.availability_conditioning_policy != (
                    REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID
                ):
                    raise ValueError(
                        f"{CURRENT_DESIGN_VERSION} reconstruction baseline 必須明示 "
                        "scenarios.bed_residence_time.availability_conditioning_policy="
                        f"{REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID}"
                    )
                return
            if time_contract.get("gap_policy") != OBSERVED_GAP_CENSORED_STOP_AT_FIRST_GAP_POLICY_ID:
                raise ValueError(
                    f"{CURRENT_DESIGN_VERSION} 必須明示 inputs.time_axis_contract.gap_policy="
                    f"{OBSERVED_GAP_CENSORED_STOP_AT_FIRST_GAP_POLICY_ID}"
                )
            if time_contract.get("stop_at_first_gap") is not True:
                raise ValueError(
                    f"{CURRENT_DESIGN_VERSION} 必須明示 inputs.time_axis_contract.stop_at_first_gap=true"
                )
            if (
                time_contract.get("denominator_policy")
                != EXCLUDE_DATA_GAP_NUMERICAL_FAILURE_AND_PRE_WINDOW_DEPOSITION_DENOMINATOR_POLICY_ID
            ):
                raise ValueError(
                    f"{CURRENT_DESIGN_VERSION} 必須明示 inputs.time_axis_contract.denominator_policy="
                    f"{EXCLUDE_DATA_GAP_NUMERICAL_FAILURE_AND_PRE_WINDOW_DEPOSITION_DENOMINATOR_POLICY_ID}"
                )
            if self.boundaries.stop_at_data_gap is not True:
                raise ValueError(
                    f"{CURRENT_DESIGN_VERSION} 必須設定 boundaries.stop_at_data_gap=true"
                )
            bed = self.scenarios.bed_residence_time
            if bed is not None and bed.availability_conditioning_policy != (
                REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID
            ):
                raise ValueError(
                    f"{CURRENT_DESIGN_VERSION} 必須明示 scenarios.bed_residence_time."
                    "availability_conditioning_policy="
                    f"{REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID}"
                )
            return

        # 舊版本只能以舊 contract 載入。若有人把新政策欄位塞進舊 design，拒絕
        # 「舊 hash + 新語意」的混合 artifact，而不是替它自動升版。
        if self.design_version == LEGACY_DESIGN_VERSION_OBSERVATION_2025:
            if any(
                time_contract.get(field) is not None
                for field in ("gap_policy", "stop_at_first_gap", "denominator_policy")
            ):
                raise ValueError("唯讀 observation-2025 legacy 不得宣稱新版 gap-censored policy")
            bed = self.scenarios.bed_residence_time
            if bed is not None and bed.availability_conditioning_policy is not None:
                raise ValueError("唯讀 observation-2025 legacy 不得宣稱新版沉底可用性條件")

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
        與固定 bbox，且貢寮、龜山島各自保留 20,000 m local、12,500 m receptor core、
        空的本期 local sensitivity，以及版本化的逐 stage 空間支援政策。設定不得保留
        未量測的共同 forcing margin 宣告；既有受體候選邊界幾何篩選仍由各站的
        ``minimum_flow_domain_margin_local_grid_scales`` 控制，兩者用途不同。runtime
        policy 要求無效 forcing 在每個速度取樣階段 fail closed，並要求資料缺口、forcing
        起點與 flow-domain 開放邊界均採停止語意；其他正式產品／時間／hash 閘門仍由
        input derivation 與 release validator 驗證。

        此方法會在 ``ProjectConfig`` schema 驗證及 input derivation 開始時呼叫，讓未知
        policy、A 區錯誤 ID、半徑、核心或 runtime controls 在任何 source I/O 前被拒絕。缺少
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
        elif self.design_version in {
            LEGACY_DESIGN_VERSION_OBSERVATION_2025,
            LEGACY_DESIGN_VERSION_V3_LOCAL20_20260909,
        }:
            if region_a_policy != FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1:
                raise ValueError(
                    "v3-family legacy design 必須搭配 A 區 "
                    f"{FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1}"
                )
        elif region_a_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1:
            raise ValueError(
                f"A 區 {FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1} 只能搭配 v3-family design"
            )
        elif self.design_version != LEGACY_DESIGN_VERSION_V2:
            raise ValueError(
                "design_version 必須是目前 gap-censored CURRENT_DESIGN_VERSION 或已登錄的 "
                f"observation/v3/v2 legacy：{CURRENT_DESIGN_VERSION}、"
                f"{LEGACY_DESIGN_VERSION_OBSERVATION_2025}、"
                f"{LEGACY_DESIGN_VERSION_V3_LOCAL20_20260909}、{LEGACY_DESIGN_VERSION_V2}"
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
            if any(domain.runtime_spatial_support_policy is not None for domain in self.domains):
                raise ValueError(
                    "runtime_spatial_support_policy 只能搭配 A 區 "
                    "v3_local20km_20260909_v1"
                )
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
        if (
            region_a.formal_release_domain_status
            != FORMAL_RELEASE_DOMAIN_STATUS_V3_FAIL_CLOSED_NO_EXPANSION
        ):
            raise ValueError(
                "v3_local20km_20260909_v1 的 A 區 formal_release_domain_status 必須是 "
                f"{FORMAL_RELEASE_DOMAIN_STATUS_V3_FAIL_CLOSED_NO_EXPANSION}"
            )
        if (
            region_a.runtime_spatial_support_policy
            != RUNTIME_SPATIAL_SUPPORT_POLICY_V3_FAIL_CLOSED_NO_EXPANSION_V1
        ):
            raise ValueError(
                "v3_local20km_20260909_v1 必須明示 "
                "runtime_spatial_support_policy=runtime_stage_fail_closed_no_expansion_v1"
            )
        if self.boundaries.stop_at_data_gap is not True:
            raise ValueError("A 區 v3 runtime-stage policy 必須設定 boundaries.stop_at_data_gap=true")
        if self.boundaries.stop_at_forcing_start is not True:
            raise ValueError(
                "A 區 v3 runtime-stage policy 必須設定 boundaries.stop_at_forcing_start=true"
            )
        if self.boundaries.flow_domain_open_boundary != "stop_at_first_crossing":
            raise ValueError(
                "A 區 v3 runtime-stage policy 必須設定 "
                "boundaries.flow_domain_open_boundary=stop_at_first_crossing"
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
        unmeasured_margin_fields = {
            "minimum_common_forcing_margin_grid_cells",
            "margin_required_for_forcings",
            "common_forcing_support_status",
        }
        stale_margin_fields = sorted(unmeasured_margin_fields & set(domain_extra))
        if stale_margin_fields:
            raise ValueError(
                "v3_local20km_20260909_v1 不得宣告未量測的共同 forcing margin："
                + ", ".join(stale_margin_fields)
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
        if (
            isinstance(inputs_payload, dict)
            and "ocm_reconstruction_root_env" in inputs_payload
            and (
                "ocm_reconstruction_root_env" not in self.inputs.model_fields_set
                or self.design_version != CURRENT_DESIGN_VERSION
            )
        ):
            # 舊 YAML 未提供 reconstruction root 時，Pydantic 的 None default 不應改變
            # 既有 config hash；legacy design 即使由新版範例複製欄位，也不能把新 root
            # 名稱誤寫入舊 run identity。正式 reconstruction baseline 必須在來源 YAML 明示名稱。
            del inputs_payload["ocm_reconstruction_root_env"]
        time_axis_payload = (
            inputs_payload.get("time_axis_contract") if isinstance(inputs_payload, dict) else None
        )
        if (
            isinstance(time_axis_payload, dict)
            and "reconstruction_policy" in time_axis_payload
            and self.design_version != CURRENT_DESIGN_VERSION
        ):
            # 舊 v2／legacy fixture 可能由目前範例複製後移除新版 gap 欄位；重建 policy
            # 只屬於目前 v3 canonical design，不應讓相容性 hash 因殘留的 extra key 漂移。
            del time_axis_payload["reconstruction_policy"]
        scenarios_payload = payload.get("scenarios")
        if (
            isinstance(scenarios_payload, dict)
            and "bed_residence_time" in scenarios_payload
            and "bed_residence_time" not in self.scenarios.model_fields_set
        ):
            # 舊 YAML 未提供沉底年齡模型時，schema 的 optional None 只是型別完整性預設，
            # 不能改變舊設定的 canonical config hash 或 checkpoint identity。只有 YAML
            # 明示此區塊時才將其納入 payload；明示 null 則仍保留 operator 的停用意圖。
            del scenarios_payload["bed_residence_time"]
        elif isinstance(scenarios_payload, dict) and isinstance(
            scenarios_payload.get("bed_residence_time"), dict
        ):
            # availability_conditioning_policy 是為缺口條件式抽樣新增的 optional
            # 欄位。舊 1.1.0 YAML 沒有它時，Pydantic 的 None default 不可改變既有
            # config hash；只有來源 YAML 明示欄位（包括明示 null）才進 canonical。
            bed_payload = scenarios_payload["bed_residence_time"]
            bed_model = self.scenarios.bed_residence_time
            if (
                bed_model is not None
                and "availability_conditioning_policy" in bed_payload
                and "availability_conditioning_policy" not in bed_model.model_fields_set
            ):
                del bed_payload["availability_conditioning_policy"]
        if isinstance(scenarios_payload, dict) and self.design_version != CURRENT_DESIGN_VERSION:
            # random receptor policy 是本期 current formal 的新契約；舊 v2／v3 設定即使
            # 由新版範例複製後殘留這些 extra key，也不能讓歷史 config hash 改變，或讓
            # legacy loader 誤以為已完成新的 random receptor release。current 版本則
            # 保留並由上方 scientific contract 強制 exact policy。
            for field_name in (
                "horizontal_receptor_selection_policy",
                "vertical_receptor_selection_policy",
                "receptor_selection_seed_policy",
            ):
                scenarios_payload.pop(field_name, None)
        selection_payload = payload.get("arrival_time_selection")
        if selection_payload is None and "arrival_time_selection" not in self.model_fields_set:
            # 舊 YAML 根本沒有 typed selection block 時，Pydantic 的 None default
            # 不能進入 canonical payload；否則不相關的 schema 擴充會改變舊 run hash。
            payload.pop("arrival_time_selection", None)
        elif isinstance(selection_payload, dict) and self.arrival_time_selection is not None:
            # 舊版兩年份 block 沒有新版 policy 欄位；只有來源 YAML 明示新版欄位時才
            # 將它們寫入 hash。這同時保留 legacy config 的 canonical bytes，並使新版
            # observation 年份／replicate policy 成為可稽核的設計邊界。
            for field_name in ("policy", "observation_years", "replicates"):
                if (
                    field_name in selection_payload
                    and field_name not in self.arrival_time_selection.model_fields_set
                ):
                    del selection_payload[field_name]
        execution_payload = payload.get("execution")
        if (
            isinstance(execution_payload, dict)
            and "ocm_interpolation_backend" in execution_payload
            and "ocm_interpolation_backend" not in self.execution.model_fields_set
        ):
            # 新欄位的實際 runtime 預設是 NumPy；但舊 YAML 沒有這個選項，不能因為
            # Pydantic 補上 ``numpy_v1`` 就讓既有 run／checkpoint 的 canonical hash 漂移。
            # 只有 YAML 明示 backend（包括明示 ``numpy_v1``）時，才把選擇寫入 normalized
            # config，形成可追溯且可驗證的 execution binding。
            del execution_payload["ocm_interpolation_backend"]
        if (
            isinstance(execution_payload, dict)
            and "physics_kernel_backend" in execution_payload
            and "physics_kernel_backend" not in self.execution.model_fields_set
        ):
            # 舊 YAML 未提供新增的 CPU primitive 後端時固定採純 NumPy；省略 default 可
            # 保護既有 run/checkpoint config hash。若 YAML 明示 ``numpy_v1`` 或
            # ``numba_cpu_v1``，則把選擇保存於 canonical payload，形成可稽核的版本邊界。
            del execution_payload["physics_kernel_backend"]
        domain_payloads = payload.get("domains")
        if isinstance(domain_payloads, list):
            for domain, domain_payload in zip(self.domains, domain_payloads, strict=False):
                if not isinstance(domain_payload, dict):
                    continue
                for field_name in ("formal_domain_policy", "runtime_spatial_support_policy"):
                    if field_name in domain_payload and field_name not in domain.model_fields_set:
                        # 新的空間支援欄位只在 YAML 明示時入 hash；舊 v2／expanded 設定
                        # 不因 Pydantic 補出的 null 發生無科學差異的 checkpoint identity 漂移。
                        del domain_payload[field_name]
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
                "horizontal_receptor_coordinates",
                "horizontal_receptor_source_manifest_sha256",
                "horizontal_receptor_selection_policy",
                "horizontal_receptor_coordinate_tolerance_m",
                "receptor_priority_polygon",
                "receptor_priority_selection",
            )
            for site, site_payload in zip(self.study_sites, site_payloads, strict=False):
                if not isinstance(site_payload, dict):
                    continue
                for field_name in candidate_fields:
                    if field_name in site_payload and field_name not in site.model_fields_set:
                        del site_payload[field_name]
        boundaries_payload = payload.get("boundaries")
        if isinstance(boundaries_payload, dict):
            for field_name in ("stop_at_forcing_start", "stop_at_data_gap"):
                if field_name in boundaries_payload and field_name not in self.boundaries.model_fields_set:
                    # 這兩個 key 在新版 schema 成為嚴格布林欄位；未曾出現於舊 YAML 時
                    # 仍從 canonical payload 移除，以維持舊設定 hash。
                    del boundaries_payload[field_name]
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
        通過前，CLI 不得啟動正式 run。對 ``v3_local20km_20260909_v1``，設定層只接受
        已鎖定的 no-expansion runtime-stage fail-closed policy 與停止型邊界；這不表示
        三套 forcing 的共同網格 margin 已量測，實際每站時間支援、accepted schema、產品
        fingerprint、manifest 與正式輸入 status 仍必須由 input validator 通過。舊的
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
        if region_a.formal_domain_policy != FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1:
            # v3 的逐 stage 空間檢查已由 assert_research_domain_policy 鎖定；這裡只保留
            # legacy expanded policy 原本的正式來源檢查。accepted products、完整研究時間
            # 支援與 artifact/hash gate 仍由 input validator 重放，不因移除舊 margin blocker
            # 而放寬。
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
