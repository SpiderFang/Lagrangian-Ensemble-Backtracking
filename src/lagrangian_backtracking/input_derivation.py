"""SERVER v3 輸入衍生 manifest 與 release config 的可重建 orchestration。

本模組是 Slice 1 的單一職責入口。它只讀取已驗收的 OCM schema 3
``ocm_native``、``ocm_surface`` 與 NWW3 schema 1 ``nww3_analysis``；不回讀 raw
NetCDF、不修改上游資料，也不執行粒子軌跡。時間軸、水平受體、到達時次、幾何與
dynamic receptor×arrival 初始條件均以既有模組的演算法為基礎，再把結果寫成固定檔名、
可驗證且不可變的 JSON component。

輸出目錄的每一份 JSON 都有相鄰 ``.sha256`` binding，並由 ``artifact_index.json``
再次列出原始檔 hash、canonical JSON hash 與大小。發布使用 hidden partial directory
加上同檔案系統的 ``os.replace``；目標 final 已存在、目標路徑或其父層是 symbolic link
時一律拒絕。這些限制讓中斷、重跑、人工替換與 SERVER 多程序發布都不會靜默改寫已驗收
的輸入。

本模組保存「條件式來源足跡」所需的資料來源識別與 fingerprint；它不把 A 區的實際
flow-domain 改名成公開標籤。legacy ``expanded_domain_v1`` 若使用南擴產品，manifest
仍會保存真實 domain ID、bbox、schema 與檔案 hash；新的 ``v3_local20km_20260909_v1``
則只綁定 ``northeast_taiwan_common_cache_v3``，並記錄 no-expansion、逐 stage
fail-closed 空間支援 policy/status；manifest 不宣稱三套 forcing 的共同格網 margin 已量測。
圖面可使用 ``A 區分析域`` 作為公開顯示文字。Smagorinsky、正式軌跡執行、報告
renderer 與互動式架構地圖不在本 Slice 範圍。
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import tempfile
from calendar import monthrange
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from shapely.geometry import LineString, Point, Polygon, mapping, shape
from shapely.ops import unary_union

from .arrival_times import select_arrival_times
from .bed_residence import (
    BED_RESIDENCE_POLICY_ID,
    BED_RESIDENCE_SAMPLING_POLICY_ID,
    apply_bed_residence_sampling,
    sample_bed_residence_age_hours,
)
from .config import (
    ARRIVAL_SELECTION_POLICY_LEGACY_TWO_YEAR_V1,
    ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1,
    FORMAL_DOMAIN_POLICY_EXPANDED_V1,
    FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1,
    FORMAL_RELEASE_DOMAIN_STATUS_V3_FAIL_CLOSED_NO_EXPANSION,
    NORTHEAST_V3_BBOX_LON_LAT,
    NORTHEAST_V3_FLOW_DOMAIN_ID,
    RUNTIME_SPATIAL_SUPPORT_POLICY_V3_FAIL_CLOSED_NO_EXPANSION_V1,
    BedResidenceTimeConfig,
    DomainConfig,
    ProjectConfig,
    StudySiteConfig,
    load_config,
    resolve_flow_domain_id,
)
from .geometry import DomainProjection, build_anchor_local_domain, densified_bbox_polygon
from .input_horizon import (
    BED_RESIDENCE_HORIZON_METHOD_ID,
    BED_RESIDENCE_HORIZON_POLICY_ID,
    BED_RESIDENCE_INPUT_SCHEMA_VERSION,
    GENERIC_HORIZON_METHOD_ID,
    GENERIC_HORIZON_POLICY_ID,
    LEGACY_INPUT_SCHEMA_VERSION,
    HorizonContractError,
    HorizonSettings,
    build_horizon_window,
    compute_horizon_coverage,
    resolve_configured_horizon,
    validate_generic_gap_payload,
    validate_support_against_period,
)
from .manifests import (
    load_arrival_time_manifest,
    load_boundary_geometries,
    load_material_manifest,
    load_receptor_arrival_initial_condition_manifest,
    load_receptor_manifest,
)
from .mesh import NativeMesh
from .preflight import PreflightReport, run_preflight
from .receptors import (
    VerticalTarget,
    build_vertical_targets,
    prepare_horizontal_receptor_candidates,
    select_horizontal_receptors_from_pool,
)
from .scenarios import BASELINE_BEHAVIORS, ArrivalTime, Receptor, stable_identifier
from .time_axis import CanonicalTimeAxis, TimeChunk, canonicalize_time_chunks

DERIVED_INPUT_SCHEMA_VERSION = LEGACY_INPUT_SCHEMA_VERSION
"""未啟用隨機沉底時的 Slice 1 衍生輸入 schema；bed 文件版本另由 input_horizon 定義。"""

DEFAULT_MAX_BACKTRACK_DAYS = 7
"""缺時安全基線的完整 backward horizon；單位為日。"""

EXPECTED_NWW_HOURLY_STEPS = 17_544
"""2024-01-01 至 2025-12-31 的完整逐時 UTC 筆數。"""

EXPECTED_FLOW_DOMAIN_COUNT = 4
EXPECTED_STUDY_SITE_COUNT = 5
EXPECTED_RECEPTOR_COUNT = 100
EXPECTED_ARRIVAL_COUNT = 250
EXPECTED_DYNAMIC_INITIAL_CONDITION_COUNT = 5_000

ARTIFACT_FILENAMES: dict[str, str] = {
    "forcing_inventory": "forcing_inventory.json",
    "ocm_gap_safe_arrival_horizon": "ocm_gap_safe_arrival.json",
    "nww_full_hourly": "nww_full_hourly.json",
    "domain_geometry": "domain.json",
    "local_geometry": "local.json",
    "open_boundary": "open.json",
    "material": "material.json",
    "receptor": "receptor.json",
    "arrival": "arrival.json",
    "initial_condition": "initial_conditions.json",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_UTC_HOUR_NS = 3_600_000_000_000
_WETDRY_SEMANTICS_ID = "schism_wetdry_elem_0_wet_1_dry"

# arrival scalar 的 metric location 不是受體位置，也不是 runtime 粒子取樣位置；它只
# 是 NWW 雙線性波浪指標在 anchor 無法通過完整 selector 時的可追溯代理位置。最大 snap
# 距離必須由實際 NWW lon/lat 軸在 anchor 附近投影後推導，不能在此寫死公尺數。
NWW_METRIC_LOCATION_POLICY_ID = "anchor_first_nearest_runtime_supported_nww_cell_center_v1"
NWW_METRIC_LOCATION_MAX_GRID_SCALES = 2.0
ARRIVAL_SELECTION_METHOD_ID = "server_v3_48_strata_plus_two_events_gap_safe_nww_metric_location_v2"
ARRIVAL_SELECTION_METHOD_OBSERVATION_YEAR_ID = (
    "server_v3_observation_year_48_strata_plus_two_events_gap_safe_nww_metric_location_v1"
)

# 這組 pilot policy 只提供固定 24 小時工程視窗；它不是正式五站 48+2 arrival
# selection 的替代品。registry 以「可接受站點組合」集中描述入口：B／C／D 各可
# 單站執行，A 區則必須一次明示 gongliao 與 guishan 的 exact pair。所有登錄時次、
# horizon 與 inclusive 節點固定在同一版本，避免 CLI／Python API 各自散落站點判斷。
PILOT_EXPLICIT_WINDOW_POLICY_ID = "site_scoped_explicit_24h_window_replacement_v1"
# B 區 r5 已發布 immutable artifact 仍使用舊 policy；它只供唯讀相容驗證，不能由
# 新建流程套用至其他站點。這個常數不能刪除，否則既有 B 成果將無法被驗收器重讀。
PILOT_EXPLICIT_LEGACY_WINDOW_POLICY_ID = "hsinchu_explicit_24h_window_replacement_v1"
PILOT_EXPLICIT_LEGACY_SITE_ID = "hsinchu"
# 舊版測試／唯讀 loader 仍可能引用這個名稱；它只代表 legacy artifact 的站點，不會
# 限制新版 registry 的 B/C/D/A pilot 入口。
PILOT_EXPLICIT_SITE_ID = PILOT_EXPLICIT_LEGACY_SITE_ID
PILOT_EXPLICIT_ARRIVAL_UTC = "2024-01-02T01:00:00Z"
PILOT_EXPLICIT_MAX_BACKTRACK_DAYS = 1.0
PILOT_EXPLICIT_TIDE_CLASS = "pilot_explicit_window"
PILOT_EXPLICIT_PHASE_OR_EVENT = "explicit_24h_window"
PILOT_ARRIVAL_SELECTION_METHOD_ID = (
    "server_v3_48_strata_plus_two_events_gap_safe_nww_metric_location_explicit_pilot_window_v1"
)

# C 區紅框候選是研究者明示的兩個近岸子區，不得再退回舊版 12.5 km 圓形核心。
# policy 只描述選點契約；實際 EPSG:4326 座標由 pilot config 的
# ``receptor_candidate_regions`` 保存，並在 input derivation 時重新投影及驗證。
RED_FRAME_CANDIDATE_POLICY_ID = "houwan_red_frame_two_subregions_anchor_first_maximin_2plus3_v1"
RED_FRAME_CANDIDATE_CONFIG_KEY = "receptor_candidate_regions"
RED_FRAME_SELECTION_CONFIG_KEY = "receptor_candidate_selection"

# registry 的 key 是單站或 A 區 exact pair；value 保存研究區域、站點集合與新建政策
# 是否可用。站點組合以 tuple 排序後比對，故 A 區不能只傳一站，也不能混入 B/C/D。
# 所有組合共用同一個 2024-01-02T01:00:00Z／24 h／25 nodes policy，讓 provenance
# 能明確說明「相同設定試跑」而不是任意替換時刻。
PILOT_EXPLICIT_REGISTRY: dict[tuple[str, ...], dict[str, Any]] = {
    ("gongliao", "guishan"): {
        "analysis_region_id": "A",
        "policy_id": PILOT_EXPLICIT_WINDOW_POLICY_ID,
        "selection_scope": "A_pair_only",
    },
    ("hsinchu",): {
        "analysis_region_id": "B",
        "policy_id": PILOT_EXPLICIT_WINDOW_POLICY_ID,
        "selection_scope": "hsinchu_only",
    },
    ("houwan",): {
        "analysis_region_id": "C",
        "policy_id": PILOT_EXPLICIT_WINDOW_POLICY_ID,
        "selection_scope": "houwan_only",
    },
    ("lienchiang",): {
        "analysis_region_id": "D",
        "policy_id": PILOT_EXPLICIT_WINDOW_POLICY_ID,
        "selection_scope": "lienchiang_only",
    },
}
_PILOT_EXPLICIT_POLICY_IDS = frozenset(
    {PILOT_EXPLICIT_WINDOW_POLICY_ID, PILOT_EXPLICIT_LEGACY_WINDOW_POLICY_ID}
)
_PILOT_EXPLICIT_EXPECTED_STEP_COUNT = 25

# 這四個識別碼是既有 receptor schema 的固定垂向類別；任何新的垂向類別都必須先
# 更新資料契約與 loader，不能在這個 helper 內以隱式的最近 layer 或外插方式擴充。
_VERTICAL_TARGET_IDS = (
    "upper_water_column",
    "mid_upper_water_column",
    "mid_lower_water_column",
    "near_bed",
)

# OCM surface 是由上游 OCM schema 3 產生的規則格網摘要。這裡只要求 arrival selector
# 需要的座標、遮罩與時間序列欄位；每個月仍依 metadata 的 arrays contract 逐一比對
# NPY header，避免把 surface cache 當成無契約的便利檔案。
_OCM_SURFACE_REQUIRED_GRID_ARRAYS = (
    "lon.npy",
    "lat.npy",
    "mask_static.npy",
)
_OCM_SURFACE_REQUIRED_MONTH_ARRAYS = (
    "u_surface_mps.npy",
    "v_surface_mps.npy",
    "surface_z.npy",
    "eta_m.npy",
    "valid_mask_surface.npy",
    "qc_flags.npy",
)


class InputDerivationError(ValueError):
    """衍生輸入無法安全完成時使用的明確例外類別。"""


@dataclass(frozen=True, slots=True)
class _CandidateRegionSpec:
    """一個明示候選子區的幾何、配額與可公開追溯欄位。

    ``geometry_lonlat`` 保持設定檔中的 WGS84 Polygon；選點時才由 caller 的
    ``DomainProjection`` 投影為公尺制。``allocation_count`` 是該子區必須產生的
    水平位置數，不足時直接 fail closed，不得從另一子區借點或以最近 face 補足。
    """

    region_id: str
    name_zh: str
    allocation_count: int
    geometry_lonlat: Polygon


def _candidate_region_specs(site: StudySiteConfig) -> tuple[_CandidateRegionSpec, ...]:
    """解析站點設定中的明示 EPSG:4326 候選 Polygon，並驗證 2+3 配額契約。

    候選區是設定的一級輸入，不從附件像素或既有 receptor 座標臨時推測。每個 geometry
    必須是有效、非退化且座標順序為經度／緯度的 Polygon；紅框子區總配額必須恰為五個
    水平位置，且選點 policy 必須要求所有子區各自完成。未設定時回傳空 tuple，保留
    A、B、D 與舊 C caller 的原有全域 selector 行為。這裡只驗證候選描述，不宣稱
    forcing 支援；後續 caller 仍會與 approved local/flow geometry 交集並通過 OCM/NWW gate。
    """

    extras = site.model_extra or {}
    raw_regions = site.receptor_candidate_regions
    if raw_regions is None:
        # 舊版 caller 可能以 extra 欄位載入尚未升級的 model；保留唯讀相容讀取，
        # 但新 YAML 會由 StudySiteConfig 的明示欄位納入 normalized config hash。
        raw_regions = extras.get("receptor_candidate_regions")
    if raw_regions is None:
        return ()
    if not isinstance(raw_regions, list) or not raw_regions:
        raise InputDerivationError(f"{site.study_site_id} 的 receptor_candidate_regions 必須是非空 list")
    specs: list[_CandidateRegionSpec] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_regions):
        if not isinstance(raw, Mapping):
            raise InputDerivationError(f"候選子區[{index}] 必須是 mapping")
        region_id = raw.get("region_id")
        name_zh = raw.get("name_zh")
        allocation = raw.get("allocation_count")
        crs = raw.get("coordinate_reference")
        geometry_payload = raw.get("geometry")
        if (
            not isinstance(region_id, str)
            or not region_id.strip()
            or region_id != region_id.strip()
            or region_id in seen_ids
        ):
            raise InputDerivationError(f"候選子區[{index}].region_id 必須唯一且非空")
        if not isinstance(name_zh, str) or not name_zh.strip() or name_zh != name_zh.strip():
            raise InputDerivationError(f"候選子區[{index}].name_zh 必須是非空文字")
        if isinstance(allocation, bool) or not isinstance(allocation, int) or allocation < 1:
            raise InputDerivationError(f"候選子區[{index}].allocation_count 必須是正整數")
        if crs != "EPSG:4326":
            raise InputDerivationError(f"候選子區[{index}] 必須明示 coordinate_reference=EPSG:4326")
        if not isinstance(geometry_payload, Mapping):
            raise InputDerivationError(f"候選子區[{index}].geometry 必須是 GeoJSON mapping")
        try:
            geometry = shape(dict(geometry_payload))
        except (TypeError, ValueError, AttributeError) as exc:
            raise InputDerivationError(f"候選子區[{index}].geometry 無法解析") from exc
        if not isinstance(geometry, Polygon) or not geometry.is_valid or geometry.area <= 0.0:
            raise InputDerivationError(f"候選子區[{index}].geometry 必須是有效、非退化 Polygon")
        min_lon, min_lat, max_lon, max_lat = geometry.bounds
        if not (-180.0 <= min_lon <= max_lon <= 180.0 and -90.0 <= min_lat <= max_lat <= 90.0):
            raise InputDerivationError(f"候選子區[{index}].geometry 超出 EPSG:4326 bounds")
        seen_ids.add(region_id)
        specs.append(
            _CandidateRegionSpec(
                region_id=region_id,
                name_zh=name_zh,
                allocation_count=allocation,
                geometry_lonlat=geometry,
            )
        )
    policy = site.receptor_candidate_selection
    if policy is None:
        policy = extras.get("receptor_candidate_selection")
    if not isinstance(policy, Mapping):
        raise InputDerivationError(f"{site.study_site_id} 缺少 receptor_candidate_selection policy")
    if policy.get("policy_id") != RED_FRAME_CANDIDATE_POLICY_ID:
        raise InputDerivationError("候選子區 policy_id 未登錄")
    if policy.get("require_each_region") is not True:
        raise InputDerivationError("候選子區必須要求每個子區各自有有效取得點")
    expected_total = policy.get("total_horizontal_count")
    if isinstance(expected_total, bool) or not isinstance(expected_total, int) or expected_total != 5:
        raise InputDerivationError("候選子區 total_horizontal_count 必須為 5")
    actual_total = sum(item.allocation_count for item in specs)
    if actual_total != expected_total:
        raise InputDerivationError(f"候選子區配額總數不符：{actual_total} != {expected_total}")
    return tuple(specs)


def _candidate_region_provenance(
    specs: Sequence[_CandidateRegionSpec],
    *,
    source_region_records: Mapping[str, Mapping[str, Any]],
    source_provenance: Mapping[str, Any] | None = None,
    selection_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """將候選子區的幾何、來源影像與每階段計數整理成 JSON-safe provenance。

    ``source_provenance`` 只保存設定檔已登錄的數位化來源；``source_region_records``
    來自實際 persistent-wet／static mask／垂向 gate 選點結果。兩者並列可區分「紅框
    的來源」與「forcing 支援的實測篩選」，避免圖面紅框被誤讀為有效流場範圍。
    """

    regions: list[dict[str, Any]] = []
    for spec in specs:
        base = source_region_records.get(spec.region_id, {})
        regions.append(
            {
                "region_id": spec.region_id,
                "name_zh": spec.name_zh,
                "allocation_count": spec.allocation_count,
                "coordinate_reference": "EPSG:4326",
                "geometry": mapping(spec.geometry_lonlat),
                **dict(base),
            }
        )
    return {
        "policy_id": RED_FRAME_CANDIDATE_POLICY_ID,
        "selection_policy": dict(selection_policy or {}),
        "source_digitization_provenance": dict(source_provenance or {}),
        "coordinate_reference": "EPSG:4326",
        "require_each_region": True,
        "total_horizontal_count": 5,
        "regions": regions,
    }


def _reject_json_constant(value: str) -> None:
    """拒絕非標準 JSON 的 NaN、Infinity 與 -Infinity。"""

    raise ValueError(f"JSON 不允許非有限常數：{value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """保留重複欄位錯誤，避免解析器靜默採用最後一筆資料。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object 不可重複欄位：{key}")
        result[key] = value
    return result


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """將 JSON object 轉為固定 UTF-8 bytes，供語意 hash 與 binding 使用。

    排序 key 與緊湊分隔符會消除 YAML/JSON 排版差異；``allow_nan=False`` 則把缺值、
    陸地或資料缺口錯誤留在明確的狀態欄位，而不是讓 NaN 混入可驗證文件。函式不會
    自動把 numpy scalar 或 array 轉換，呼叫端必須先產生 JSON-safe payload。
    """

    if not isinstance(payload, Mapping):
        raise TypeError("canonical JSON root 必須是 mapping")
    try:
        return json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload 不是可 canonicalize 的 JSON object：{exc}") from exc


def _absolute_without_resolving(path: str | Path) -> Path:
    """建立絕對 lexical path，不呼叫 ``resolve``，以免先跟隨 symbolic link。"""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _is_known_system_alias(path: Path) -> bool:
    """只放行 macOS 的固定 ``/var``、``/tmp`` 系統別名，不放行使用者 symlink。

    本機 macOS 將 ``/var`` 與 ``/tmp`` 以作業系統管理的 symbolic link 指向
    ``/private/var`` 與 ``/private/tmp``；pytest 與 Python ``tempfile`` 會自然使用這些
    lexical 路徑。若一概拒絕，安全的測試暫存目錄與一般原子寫入都無法使用。因此這裡
    只比對兩個固定 alias 的實際 link target，其他任何 symlink component 仍 fail closed。
    """

    aliases = {
        Path("/var"): Path("/private/var"),
        Path("/tmp"): Path("/private/tmp"),
    }
    expected = aliases.get(path)
    if expected is None or not path.is_symlink():
        return False
    link_target = path.readlink()
    if not link_target.is_absolute():
        link_target = Path(path.anchor) / link_target
    return link_target == expected


def _assert_no_symlink_components(path: str | Path, *, allow_missing_leaf: bool = True) -> None:
    """拒絕路徑本身或既有父層的 symbolic link。

    輸入與輸出都需要這項保護：若只檢查 final leaf，攻擊者仍可把父目錄換成 link，
    使 atomic writer 將結果送到未預期的位置。不存在的中間目錄可由 caller 建立，
    但其後再以同一檢查確認；本函式不會替 caller 猜測或改寫目標。
    """

    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        if current.is_symlink():
            if _is_known_system_alias(current):
                continue
            is_leaf = index == len(parts) - 1
            if not (is_leaf and allow_missing_leaf):
                raise InputDerivationError(f"路徑不可經由 symbolic link：{path}")
            raise InputDerivationError(f"目標 final 不可為 symbolic link：{path}")


def _assert_regular_file(path: str | Path) -> Path:
    """確認檔案存在、是普通檔案且沒有 symbolic link。"""

    candidate = Path(path)
    _assert_no_symlink_components(candidate, allow_missing_leaf=False)
    try:
        mode = candidate.stat().st_mode
    except OSError as exc:
        raise FileNotFoundError(f"找不到普通輸入檔：{candidate}") from exc
    if not stat.S_ISREG(mode):
        raise InputDerivationError(f"輸入必須是普通檔案：{candidate}")
    return candidate


def _assert_regular_directory(path: str | Path) -> Path:
    """確認目錄存在且不透過 symbolic link 進入。"""

    candidate = Path(path)
    _assert_no_symlink_components(candidate, allow_missing_leaf=False)
    if not candidate.is_dir():
        raise NotADirectoryError(f"輸入必須是普通目錄：{candidate}")
    return candidate


def _fsync_file(path: Path) -> None:
    """同步已寫入的普通檔案；某些測試檔案系統不支援時保留原始例外。"""

    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _atomic_write_bytes(path: str | Path, data: bytes, *, suffix: str = ".tmp") -> Path:
    """以同檔案系統暫存檔加 ``os.replace`` 發布 bytes，既有 final 一律拒絕。"""

    destination = Path(path)
    _assert_no_symlink_components(destination.parent, allow_missing_leaf=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"不可覆寫既有 final：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(destination.parent, allow_missing_leaf=False)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=suffix, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        _fsync_file(destination)
        return destination
    finally:
        if temporary is not None:
            with suppress(FileNotFoundError):
                temporary.unlink()


def _atomic_write_text(path: str | Path, text: str) -> Path:
    """以 UTF-8 bytes 原子寫入文字文件；呼叫端負責保證內容已有固定格式。"""

    return _atomic_write_bytes(path, text.encode("utf-8"), suffix=".text")


def write_canonical_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """原子寫入 immutable canonical JSON 與相鄰 SHA-256 binding。

    回傳值包含實際 bytes 大小、raw hash 與 canonical hash。sidecar 使用固定 JSON 形狀，
    使 validator 可以在不信任主文件內部宣告的前提下先確認檔案位元組未被替換。若 final
    或 sidecar 已存在，函式不支援 overwrite；重建時應使用新的 release 目錄。
    """

    destination = Path(path)
    canonical = canonical_json_bytes(payload)
    rendered = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    raw_hash = sha256(rendered).hexdigest()
    canonical_hash = sha256(canonical).hexdigest()
    _atomic_write_bytes(destination, rendered, suffix=".json")
    binding_path = Path(f"{destination}.sha256")
    binding = {
        "algorithm": "sha256",
        "canonical_sha256": canonical_hash,
        "sha256": raw_hash,
        "size_bytes": len(rendered),
    }
    try:
        _atomic_write_bytes(
            binding_path,
            json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            suffix=".binding",
        )
    except Exception:
        # 主文件已經發布時不能以 unlink 廣泛清理；sidecar 若失敗會讓 caller 收到例外，
        # validator 也會拒絕這個不完整 artifact。保留它可供人工診斷，而不是誤稱成功。
        raise
    return {
        "path": destination.name,
        "size_bytes": len(rendered),
        "sha256": raw_hash,
        "canonical_sha256": canonical_hash,
    }


def read_canonical_json(
    path: str | Path, *, verify_binding: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """嚴格讀取 immutable JSON，並可選擇驗證相鄰 binding。

    回傳 ``(payload, fingerprint)``；fingerprint 同時保存 raw 與 canonical hash。讀取器不
    接受 JSON root array、重複 key、NaN、Infinity、symbolic link 或 sidecar hash 不一致。
    ``verify_binding=False`` 僅供建立 sidecar 前的低階測試，不應用於 release validator。
    """

    source = _assert_regular_file(path)
    raw = source.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"無法讀取嚴格 canonical JSON：{source}；{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root 必須是 object：{source}")
    canonical = canonical_json_bytes(payload)
    fingerprint = {
        "path": source.name,
        "size_bytes": len(raw),
        "sha256": sha256(raw).hexdigest(),
        "canonical_sha256": sha256(canonical).hexdigest(),
    }
    if verify_binding:
        binding_path = Path(f"{source}.sha256")
        binding, _ = read_canonical_json(binding_path, verify_binding=False)
        if (
            binding.get("algorithm") != "sha256"
            or binding.get("size_bytes") != fingerprint["size_bytes"]
            or binding.get("sha256") != fingerprint["sha256"]
            or binding.get("canonical_sha256") != fingerprint["canonical_sha256"]
        ):
            raise ValueError(f"JSON SHA-256 binding 不一致：{source}")
    return payload, fingerprint


def _sha256_file(path: str | Path) -> str:
    """以固定 chunk 大小計算普通檔案 SHA-256，不把大型 forcing 整檔載入 RAM。"""

    source = _assert_regular_file(path)
    digest = sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe_scalar(value: Any) -> Any:
    """把 numpy scalar 轉成 Python scalar，拒絕非有限數值與其他隱式型別。"""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON provenance 不可含 NaN 或 Infinity")
        return value
    raise TypeError(f"無法轉換 JSON scalar：{type(value).__name__}")


def _utc_string(time_ns: int) -> str:
    """將 epoch nanoseconds 轉為固定 UTC ISO8601 字串。"""

    return datetime.fromtimestamp(int(time_ns) / 1_000_000_000, tz=UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class _ExplicitPilotArrival:
    """保存一筆已通過格式 gate 的 pilot 明示 arrival 視窗。

    ``time_utc_ns`` 是 UTC epoch nanoseconds，供 OCM、NWW3 與 gap-safe mask 以整數
    時間軸比對；``time_utc`` 是同一時刻的固定 ISO8601 ``Z`` 字串，供 provenance 與
    metadata 交換。這個容器只代表本次 pilot 的單一替換候選，不會改寫 config 中正式
    48+2 分層的年份、季節或潮汐定義。
    """

    study_site_id: str
    time_utc_ns: int
    time_utc: str
    max_backtrack_days: float
    selection_scope: str


def _parse_explicit_pilot_arrivals(
    value: Mapping[str, str] | None,
) -> dict[str, _ExplicitPilotArrival]:
    """依版本化 registry 解析 pilot-only 的 ``study_site_id -> exact UTC`` 入口。

    B／C／D 只接受各自單站；A 區必須同時明示 ``gongliao`` 與 ``guishan``，不得傳入
    單一 A 站或混合不同研究區。所有登錄組合共用 ``2024-01-02T01:00:00Z``、一日
    backward horizon 與 25 個 inclusive exact-hour 節點。這不是一般 arrival selector
    的自由時刻參數，而是為相同設定試跑保存的版本化入口；回傳 epoch nanoseconds 後，仍
    會交給產品時間軸、NWW 空間支撐及 gap-safe gate 再次驗證。
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping) or not value:
        raise InputDerivationError("pilot_arrival_utc 必須是非空的 site=UTC mapping")
    if not all(type(site_id) is str and site_id.strip() == site_id for site_id in value):
        raise InputDerivationError("pilot_arrival_utc 的站點鍵必須是沒有空白的文字")
    selected_sites = tuple(sorted(value))
    registry_entry = PILOT_EXPLICIT_REGISTRY.get(selected_sites)
    if registry_entry is None:
        allowed = ", ".join("+".join(key) for key in sorted(PILOT_EXPLICIT_REGISTRY))
        raise InputDerivationError(
            "pilot_arrival_utc 站點組合未登錄；僅允許 B/C/D 各自單站或 A 區 "
            f"gongliao+guishan exact pair（可用組合：{allowed}）"
        )
    parsed_by_site: dict[str, _ExplicitPilotArrival] = {}
    for site_id in selected_sites:
        raw = value[site_id]
        if type(raw) is not str or not raw or raw != raw.strip() or not raw.endswith("Z"):
            raise InputDerivationError("pilot_arrival_utc 必須是沒有空白的 UTC Z 字串")
        try:
            parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
        except ValueError as exc:
            raise InputDerivationError(f"pilot_arrival_utc 無法解析：{raw!r}") from exc
        if parsed.utcoffset() != timedelta(0):
            raise InputDerivationError("pilot_arrival_utc 必須使用 UTC offset 0")
        if parsed.minute or parsed.second or parsed.microsecond:
            raise InputDerivationError("pilot_arrival_utc 必須落在 exact-hour UTC")
        canonical = parsed.isoformat().replace("+00:00", "Z")
        if canonical != raw:
            raise InputDerivationError("pilot_arrival_utc 必須使用固定 ISO8601 格式 2024-01-02T01:00:00Z")
        if raw != PILOT_EXPLICIT_ARRIVAL_UTC:
            raise InputDerivationError(
                f"{PILOT_EXPLICIT_WINDOW_POLICY_ID} 只登錄 {PILOT_EXPLICIT_ARRIVAL_UTC}"
            )
        time_ns = int(parsed.timestamp()) * 1_000_000_000 + parsed.microsecond * 1_000
        parsed_by_site[site_id] = _ExplicitPilotArrival(
            study_site_id=site_id,
            time_utc_ns=time_ns,
            time_utc=canonical,
            max_backtrack_days=PILOT_EXPLICIT_MAX_BACKTRACK_DAYS,
            selection_scope=str(registry_entry["selection_scope"]),
        )
    return parsed_by_site


def _explicit_pilot_window_times(explicit: _ExplicitPilotArrival) -> np.ndarray:
    """建立 pilot 明示視窗的 25 個 inclusive exact-hour UTC 節點。

    視窗終點由版本化 policy 固定為 arrival，起點則是 arrival 減去一日；因此這裡
    產生的是 ``2024-01-01T01:00:00Z`` 至 ``2024-01-02T01:00:00Z`` 的整數 epoch
    nanoseconds，而不是由經過 rounding 的浮點日期或鄰近資料列推導。呼叫端會再用
    OCM／surface／NWW 各自的 canonical axis 驗證每一個節點，任何一小時缺失都會
    fail closed，不會被這個 helper 補出來。
    """

    horizon_steps = int(round(explicit.max_backtrack_days * 24.0))
    if horizon_steps != 24 or not math.isclose(
        horizon_steps / 24.0,
        explicit.max_backtrack_days,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise InputDerivationError("explicit pilot window 必須是固定一日、逐時視窗")
    start_ns = explicit.time_utc_ns - horizon_steps * _UTC_HOUR_NS
    window = np.arange(
        start_ns,
        explicit.time_utc_ns + _UTC_HOUR_NS,
        _UTC_HOUR_NS,
        dtype=np.int64,
    )
    if window.size != 25 or int(window[0]) != start_ns or int(window[-1]) != explicit.time_utc_ns:
        raise InputDerivationError("explicit pilot window 未建立 25 個 inclusive exact-hour 節點")
    return window


def _parse_month(month: str) -> tuple[int, int]:
    """驗證並解析 YYYYMM，避免以字串切片接受不存在的月份。"""

    if not isinstance(month, str) or re.fullmatch(r"[0-9]{6}", month) is None:
        raise ValueError(f"月份必須是 YYYYMM：{month!r}")
    year, number = int(month[:4]), int(month[4:])
    monthrange(year, number)
    return year, number


def _months_for_config(config: ProjectConfig) -> list[str]:
    """依 config years 產生唯一、按時間排序的曆月清單。"""

    years = [int(year) for year in config.inputs.years]
    if len(set(years)) != len(years):
        raise ValueError("config.inputs.years 不可重複")
    return [f"{year}{month:02d}" for year in sorted(years) for month in range(1, 13)]


def _expected_hourly_axis(months: Sequence[str], *, step_hours: float = 1.0) -> np.ndarray:
    """建立指定月份聯集的規則 UTC 參考軸，排除月份 halo。"""

    step_ns = int(round(step_hours * _UTC_HOUR_NS))
    if step_ns <= 0 or not math.isclose(step_ns / _UTC_HOUR_NS, step_hours, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("expected timestep 必須是可精確換算的正值")
    pieces: list[np.ndarray] = []
    for label in sorted(months):
        year, month = _parse_month(label)
        start = int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
        end = start + monthrange(year, month)[1] * 24 * _UTC_HOUR_NS
        pieces.append(np.arange(start, end, step_ns, dtype=np.int64))
    if not pieces:
        raise ValueError("至少需要一個月份建立時間軸")
    return np.concatenate(pieces)


def _provenance(*, method_id: str, source_hashes: Mapping[str, str], **extra: Any) -> dict[str, Any]:
    """建立所有 generated component 共用的 provenance object。

    ``source_hashes`` 的 key 是不含絕對路徑的語意名稱；值是 accepted-product fingerprint
    record 的 SHA-256，該 record 內再區分 metadata/time 軸的實際內容 hash 與大型 NPY 的
    size/header structural binding。extra 可保存 flow-domain ID、bbox、schema、root token
    與 public label 政策，但不得放入密碼、SERVER 絕對路徑或大型陣列。
    """

    if not method_id.strip() or not source_hashes:
        raise ValueError("provenance method_id 與 source_hashes 不可為空")
    normalized = {}
    for key, digest in source_hashes.items():
        if not isinstance(key, str) or not key.strip() or _SHA256_RE.fullmatch(str(digest)) is None:
            raise ValueError(f"provenance source hash 不合法：{key!r}")
        normalized[key] = str(digest)
    created = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return {
        "method_id": method_id,
        "created_at_utc": created,
        "source_hashes": dict(sorted(normalized.items())),
        **extra,
    }


def _env_root(value: str | Path | None, env_name: str, *, required: bool = True) -> Path | None:
    """以 CLI 明示 root 優先、其次讀 config 指定的 env；不猜測 SERVER 絕對路徑。"""

    if value is not None:
        return _assert_regular_directory(value)
    if not _ENV_NAME_RE.fullmatch(env_name):
        raise ValueError(f"root environment name 不合法：{env_name}")
    raw = os.environ.get(env_name)
    if not raw:
        if required:
            raise ValueError(f"缺少 root 參數或環境變數：{env_name}")
        return None
    return _assert_regular_directory(raw)


def _read_metadata(path: Path) -> dict[str, Any]:
    """嚴格讀取上游 metadata；source metadata 不要求 component sidecar。"""

    source = _assert_regular_file(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise InputDerivationError(f"無法讀取上游 metadata：{source}；{exc}") from exc
    if not isinstance(value, dict):
        raise InputDerivationError(f"上游 metadata root 必須是 object：{source}")
    return value


def _schema_major(value: Any) -> int | None:
    """取 schema version 的 major number；非法值回傳 None 以便 fail closed。"""

    try:
        return int(str(value).split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _load_npy(path: Path, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    """唯讀開啟 NPY，拒絕 pickle 與 symbolic link；大型陣列保持 memory-map。"""

    _assert_regular_file(path)
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise InputDerivationError(f"無法讀取 NPY：{path}；{exc}") from exc
    if dtype is not None and array.dtype != dtype:
        raise InputDerivationError(f"NPY dtype 不符：{path}；{array.dtype} != {dtype}")
    return array


@dataclass(frozen=True, slots=True)
class _MonthData:
    """單一 flow-domain/month 的唯讀檔案與時間來源索引。"""

    label: str
    directory: Path
    metadata: Mapping[str, Any]
    time_ns: np.ndarray


@dataclass(frozen=True, slots=True)
class _ProductData:
    """單一產品/domain 的 metadata、月份分片與 canonical UTC 軸。"""

    product: str
    flow_domain_id: str
    root: Path
    root_token: str
    grid_dir: Path
    grid_metadata: Mapping[str, Any]
    months: tuple[_MonthData, ...]
    canonical: CanonicalTimeAxis
    required_arrays: tuple[str, ...]
    source_file_records: tuple[Mapping[str, Any], ...]

    @property
    def month_by_label(self) -> Mapping[str, _MonthData]:
        """提供 immutable-friendly 月份索引，避免 caller 依檔案順序猜測來源。"""

        return {item.label: item for item in self.months}

    def source_for_canonical_index(self, index: int) -> tuple[_MonthData, int]:
        """回傳 canonical index 所使用的月份與 local index。"""

        chunk_index = int(self.canonical.source_chunk_index[index])
        local_index = int(self.canonical.source_local_index[index])
        return self.months[chunk_index], local_index


@dataclass(frozen=True, slots=True)
class _OCMMonthPairArrays:
    """單一月份供 OCM receptor／dynamic pair 共用的唯讀陣列檢視。

    ``zcor`` 的座標軸是 ``(time, node, layer)``、``elev`` 是
    ``(time, node)`` 海面高程、``wetdry`` 是 ``(time, face)`` 的元素濕乾旗標；三者
    都由已驗收的 OCM native schema 3 NPY 以唯讀 memory-map 開啟。這個小型容器把三個
    陣列綁在同一月份，避免 receptor 與 dynamic 兩條路徑以裸 tuple 傳遞時把不同月份
    或不同產品的資料錯接。``__iter__`` 只為既有工程 adapter 的私有 tuple 解構保留
    相容性，新的跨階段程式應使用具名欄位。
    """

    zcor: np.ndarray
    elev: np.ndarray
    wetdry: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        """依舊有順序提供 tuple 解構；不改變三個陣列的來源或內容。"""

        yield self.zcor
        yield self.elev
        yield self.wetdry


@dataclass(frozen=True, slots=True)
class _OCMPairCache:
    """同一 flow domain 的 OCM pair 取樣共同 cache。

    ``utc_prefer_last_index`` 將 exact UTC 綁到月份與 local row；遇到跨月 halo 的重複
    UTC 時，最後一個依既有月份排序寫入的來源會勝出，與 canonical time-axis 契約一致。
    ``depth`` 是 source node 的正值向下水深，``monthly_arrays`` 則保存每月 zcor、elev
    與 wetdry 的唯讀 memory-map。``flow_domain_id`` 是所有取樣資料的 binding；任何
    caller 將 cache 套到不同產品前都必須重新核對，錯綁時直接 fail closed。

    這個型別的生命週期通常是一個 ``build_input_derivatives`` 呼叫，讓 receptor 選點
    和 5,000 筆 dynamic pair 共用同一組檔案檢視；它不複製大型陣列，也不把資料缺口
    轉成零值或最近值。``__iter__`` 僅支援既有 ``prepare_engineering_window`` 的
    私有 tuple 解構，實際共享一律使用具名欄位或整個 cache 物件。
    """

    flow_domain_id: str
    utc_prefer_last_index: Mapping[int, tuple[_MonthData, int]]
    depth: np.ndarray
    monthly_arrays: Mapping[str, _OCMMonthPairArrays]

    @property
    def by_time(self) -> Mapping[int, tuple[_MonthData, int]]:
        """舊有 helper 使用的 UTC index 別名；內容仍是 prefer-last mapping。"""

        return self.utc_prefer_last_index

    @property
    def month_arrays(self) -> Mapping[str, _OCMMonthPairArrays]:
        """舊有 helper 使用的月份陣列別名；每項仍可 tuple 解構。"""

        return self.monthly_arrays

    def __iter__(self) -> Iterator[object]:
        """依既有順序回傳 UTC index、depth、月份 arrays，維持私有 caller 相容。"""

        yield self.utc_prefer_last_index
        yield self.depth
        yield self.monthly_arrays


@dataclass(frozen=True, slots=True)
class _NativeMeshBinding:
    """把已建立的 NativeMesh 與其 flow domain 綁定，供 dynamic 安全重用。

    ``NativeMesh`` 本身只保存幾何陣列，沒有 flow-domain 識別碼；若只把裸 mesh 由
    receptor 傳給 dynamic，誤把另一區網格套入目前產品時無法從物件本身察覺。此 binding
    保存產品 ID 與 immutable mesh，dynamic 會先驗證 ID 再取用，獨立 caller 未提供
    binding 時仍會自行載入 mesh。
    """

    flow_domain_id: str
    mesh: NativeMesh


# 垂向支撐 cache 的 key 依序是 analysis region、study site、arrival UTC ns 與 source
# face local index。相同 horizontal face 的四個 vertical receptor 因而只需建立一次
# 全部 target/bracket；不同站點即使共享 flow domain 也不會誤用另一站 arrival 的結果。
_VerticalSupportCacheKey = tuple[str, str, int, int]


@dataclass(frozen=True, slots=True)
class _NWWRuntimeMonth:
    """單一 NWW 月份的 runtime-equivalent exact-hour 陣列檢視。

    陣列仍由 ``_load_npy`` 以唯讀 memory-map 開啟；這個容器只保存月份時間軸與四個
    需要參與支撐 gate 的欄位，不建立逐時 ``WaveSample`` 物件，也不複製整個網格。
    ``significant_wave_height`` 的單位是 m、``peak_frequency`` 的單位是 Hz、方向是
    原始角度（degree），而 ``valid_mask_wave`` 代表該 UTC／格點是否可供 runtime 使用。
    """

    source: _MonthData
    significant_wave_height: np.ndarray
    peak_frequency: np.ndarray
    peak_direction_raw_deg: np.ndarray
    valid_mask_wave: np.ndarray


@dataclass(frozen=True, slots=True)
class _NWWRuntimeCache:
    """一套 NWW3 規則格網的唯讀 runtime-equivalent 取樣 cache。

    runtime 的 ``NWWAnalysisMonth`` 只接受一維、嚴格遞增的 lon／lat 軸與
    ``(time, lat, lon)`` 欄位；input derivation 不可因測試或上游資料是二維座標陣列
    就默許一個 runtime 無法重現的插值契約。``static_mask`` 的 shape 是
    ``(lat, lon)``，每次 exact-hour sample 仍會檢查雙線性四角，而不是尋找最近有效點。
    """

    lon_axis: np.ndarray
    lat_axis: np.ndarray
    static_mask: np.ndarray
    months: tuple[_NWWRuntimeMonth, ...]


@dataclass(frozen=True, slots=True)
class NWWMetricLocationBinding:
    """保存 arrival NWW metric proxy 的位置、距離與 runtime cell binding。

    ``location_kind`` 只有 ``anchor`` 或 ``cell_center``：前者表示站點 anchor 本身
    通過 runtime-equivalent NWW exact-hour selector，後者表示 anchor 失敗後選到的規則
    格網雙線性 cell center。``lon``／``lat`` 僅是資料交換與 NWW metric 指標定位，真正
    的距離判定使用該 flow domain 的 ``DomainProjection`` 公尺座標；它們不能取代
    receptor 的實際經緯度或 runtime trajectory sample。

    ``cell_x0``／``cell_x1``／``cell_y0``／``cell_y1`` 是與 runtime
    ``searchsorted`` 相同的四角索引，固定代表 ``(y0,x0)、(y0,x1)、(y1,x0)、(y1,x1)``。
    ``representative_grid_scale_m`` 是 anchor 附近實際 NWW lon/lat 軸投影後的局地水平
    格網尺度中位數，``maximum_snap_distance_m`` 固定等於其兩倍。所有浮點欄位均須有限，
    所有 cell 索引均須是嚴格遞增的整數，讓每一筆 ArrivalTime metadata 可被重建與驗證。
    """

    policy_id: str
    location_kind: str
    lon: float
    lat: float
    anchor_distance_m: float
    representative_grid_scale_m: float
    maximum_snap_distance_m: float
    cell_x0: int
    cell_x1: int
    cell_y0: int
    cell_y1: int

    def __post_init__(self) -> None:
        """在建立 binding 時拒絕不可重建的 policy、座標、距離或 cell 索引。"""

        if self.policy_id != NWW_METRIC_LOCATION_POLICY_ID:
            raise ValueError(f"未知 NWW metric location policy：{self.policy_id}")
        if self.location_kind not in {"anchor", "cell_center"}:
            raise ValueError(f"NWW metric location_kind 無效：{self.location_kind}")
        floating_values = (
            self.lon,
            self.lat,
            self.anchor_distance_m,
            self.representative_grid_scale_m,
            self.maximum_snap_distance_m,
        )
        if not all(np.isfinite(float(value)) for value in floating_values):
            raise ValueError("NWW metric location binding 的浮點欄位必須有限")
        if self.representative_grid_scale_m <= 0.0 or self.maximum_snap_distance_m <= 0.0:
            raise ValueError("NWW metric location grid scale 與最大 snap 距離必須大於零")
        if self.anchor_distance_m < 0.0 or self.anchor_distance_m > self.maximum_snap_distance_m + 1e-9:
            raise ValueError("NWW metric location anchor distance 超出最大 snap 距離")
        if not math.isclose(
            self.maximum_snap_distance_m,
            NWW_METRIC_LOCATION_MAX_GRID_SCALES * self.representative_grid_scale_m,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("NWW metric location maximum snap 距離未依 policy 由 grid scale 推導")
        index_values = (self.cell_x0, self.cell_x1, self.cell_y0, self.cell_y1)
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in index_values
        ):
            raise ValueError("NWW metric location cell index 必須是整數")
        if (
            self.cell_x0 < 0
            or self.cell_y0 < 0
            or self.cell_x1 <= self.cell_x0
            or self.cell_y1 <= self.cell_y0
        ):
            raise ValueError("NWW metric location cell index 必須是有效的四角範圍")

    def to_metadata(self) -> dict[str, float | int | str]:
        """轉成 ArrivalTime 可保存的扁平 JSON-safe metric binding 欄位。"""

        return {
            "metric_location_policy_id": self.policy_id,
            "metric_location_kind": self.location_kind,
            "metric_location_lon": float(self.lon),
            "metric_location_lat": float(self.lat),
            "metric_location_anchor_distance_m": float(self.anchor_distance_m),
            "metric_location_representative_grid_scale_m": float(self.representative_grid_scale_m),
            "metric_location_maximum_snap_distance_m": float(self.maximum_snap_distance_m),
            "metric_location_cell_x0": int(self.cell_x0),
            "metric_location_cell_x1": int(self.cell_x1),
            "metric_location_cell_y0": int(self.cell_y0),
            "metric_location_cell_y1": int(self.cell_y1),
        }


@dataclass(frozen=True, slots=True)
class _NWWMetricLocationCandidate:
    """經幾何、靜態四角與距離 gate 篩過的 NWW bilinear cell center 候選。"""

    lon: float
    lat: float
    distance_m: float
    cell_x0: int
    cell_x1: int
    cell_y0: int
    cell_y1: int


@dataclass(frozen=True, slots=True)
class _SiteArrivalSelectionContext:
    """保存單站 arrival selector 的 OCM/NWW 原始序列與 metric binding。

    A 區 paired UTC clone 需要用龜山島自己的 OCM elevation/current、NWW metric series
    與 gap-safe mask 重驗貢寮選出的 50 個 UTC；若只保存已選 ArrivalTime，就會把貢寮
    的 physical metadata 誤複製到龜山島。因此這個 context 僅在一次 build 內保存小型
    dictionary，不能被當成 runtime forcing cache 或受體實際位置。
    """

    elevation: Mapping[int, float]
    speed: Mapping[int, float]
    nww_series: Mapping[int, float]
    nww_valid: Mapping[int, bool]
    metric_binding: NWWMetricLocationBinding


@dataclass(frozen=True, slots=True)
class _FaceVerticalSupport:
    """保存單一 OCM face 垂向支撐與代表性 bracket 的共同結果。

    ``zcor_node_layer`` 的資料結構是 ``(face node, vertical layer)``，每一個 node
    都必須能以有限的原生 zcor 同時包住所需 target；這個條件比把多個 node 先取
    中位數更嚴格，能阻止陡峭海床下某一節點被不相容的代表水柱掩蓋。代表性 bed、
    eta、target 與上下 layer 則沿用既有 face median 與 10/40/70/near-bed 設計，
    供 receptor template 與 dynamic pair 共用，避免兩條程式路徑各自發明內插契約。

    屬性中的 layer 與 bracket 都是 immutable tuple；實際 NPY 仍維持 memory-map，
    只把單一 face 的小型代表結果保存於本次函式呼叫的生命週期內。
    """

    bed_z_m_positive_up: float
    eta_z_m_positive_up: float
    representative_zcor_m: tuple[float, ...]
    targets: tuple[VerticalTarget, ...]
    brackets: tuple[tuple[str, float, float], ...]

    def target_for(self, vertical_id: str) -> VerticalTarget:
        """依固定垂向識別碼取回已通過逐 node 支撐檢查的 target。"""

        for target in self.targets:
            if target.vertical_id == vertical_id:
                return target
        raise InputDerivationError(f"未知或未要求的 vertical_id：{vertical_id}")

    def bracket_for(self, vertical_id: str) -> tuple[float, float]:
        """取回 target 對應的代表性 lower／upper zcor layer。"""

        for candidate_id, lower, upper in self.brackets:
            if candidate_id == vertical_id:
                return lower, upper
        raise InputDerivationError(f"找不到 vertical_id 的代表性 bracket：{vertical_id}")


def _build_face_vertical_support(
    *,
    zcor_node_layer: np.ndarray,
    node_elev_m: np.ndarray,
    node_depth_m: np.ndarray,
    vertical_ids: Sequence[str],
) -> _FaceVerticalSupport:
    """以同一個 fail-closed 規則建立 face 的垂向 target 與代表性 bracket。

    輸入的 ``zcor_node_layer`` 必須是單一 OCM source face 的
    ``(node, layer)`` positive-up z 座標；``node_elev_m`` 是同一批 node 在目前 UTC
    的海面高程，``node_depth_m`` 是靜態正值向下水深。每個 node 的 eta 與 depth 都
    必須有限，且每個 depth 必須嚴格大於零；這與正式 OCM runtime 要求全 node
    海面／海床支撐一致，不能忽略缺值或非物理的 node 再以其他 node 的中位數掩蓋。
    face 的代表性 bed／eta 仍維持既有定義，分別對全數通過檢查的 ``-depth`` 與
    node eta 取中位數；因此中位數不會再觸發 ``nanmedian`` 全 NaN 警告，也絕不以
    零或其他 fallback 取代缺值。

    代表性水柱先把 ``bed <= zcor <= eta`` 以外的 layer 排除，再呼叫既有
    ``build_vertical_targets`` 產生 10%、40%、70% 及 near-bed 四個固定類別。每一個
    caller 指定的 target 另外逐一檢查每個 face node：必須存在有限 ``zcor <= target``
    與有限 ``zcor >= target``；任一 node 缺少任一側支撐就 fail closed。這是 runtime
    三節點 sampler 所需的共同支撐前置條件，不搜尋最近有效 layer、不夾到海床或海面、
    不做單側外插，也不改變 vertical class。回傳的 bracket 是代表性水柱中最近的
    lower／upper layer，且要求 ``upper > lower``，讓 loader 的嚴格不等式能直接成立。

    回傳的 ``_FaceVerticalSupport`` 同時被 receptor template 與 dynamic
    receptor×arrival pair 使用；因此模板通過但某一實際 UTC 垂向支撐改變時，dynamic
    建置會明確失敗，而不會靜默沿用模板 z 或把缺值變成零。
    """

    zcor = np.asarray(zcor_node_layer, dtype=np.float64)
    node_eta = np.asarray(node_elev_m, dtype=np.float64)
    node_depth = np.asarray(node_depth_m, dtype=np.float64)
    if zcor.ndim != 2 or zcor.shape[0] == 0 or zcor.shape[1] == 0:
        raise InputDerivationError("OCM face zcor 必須是非空的 (node, layer) 二維陣列")
    if node_eta.ndim != 1 or node_depth.ndim != 1 or node_eta.size != zcor.shape[0]:
        raise InputDerivationError("OCM face node eta／depth 維度與 zcor node 數不符")
    if node_depth.size != zcor.shape[0]:
        raise InputDerivationError("OCM face node depth 維度與 zcor node 數不符")
    requested_ids = tuple(str(value) for value in vertical_ids)
    if not requested_ids or len(set(requested_ids)) != len(requested_ids):
        raise InputDerivationError("垂向支撐檢查必須指定唯一且非空的 vertical_id")
    unknown_ids = set(requested_ids).difference(_VERTICAL_TARGET_IDS)
    if unknown_ids:
        raise InputDerivationError(f"未知 vertical_id：{sorted(unknown_ids)}")

    if not np.all(np.isfinite(node_eta)):
        raise InputDerivationError("OCM face 每個 node 的 eta 都必須是有限值")
    if not np.all(np.isfinite(node_depth)) or np.any(node_depth <= 0.0):
        raise InputDerivationError("OCM face 每個 node 的 source depth 都必須有限且大於零")
    bed = float(np.median(-node_depth))
    eta = float(np.median(node_eta))
    if not np.isfinite(bed) or not np.isfinite(eta) or eta <= bed:
        raise InputDerivationError("OCM face 代表性水柱必須有限且 eta 高於 bed")

    # 先以代表性 bed／eta 篩掉水柱外 layer；這一步只建立 template/actual pair 的
    # 代表 column，逐 node support 仍會使用各 node 的全部有限 zcor 來檢查兩側，
    # 以免把深節點本身合法的 below-target layer 錯誤刪掉。
    representative_layers = np.full(zcor.shape[1], np.nan, dtype=np.float64)
    for layer_index in range(zcor.shape[1]):
        layer_values = zcor[:, layer_index]
        finite_values = layer_values[np.isfinite(layer_values)]
        if finite_values.size:
            representative_layers[layer_index] = float(np.median(finite_values))
    in_water = np.isfinite(representative_layers)
    in_water &= representative_layers >= bed
    in_water &= representative_layers <= eta
    valid_layers = np.sort(np.unique(representative_layers[in_water]))
    if valid_layers.size < 4:
        raise InputDerivationError("OCM face 代表性 water column 的有效 zcor layer 少於四層")

    try:
        all_targets = tuple(
            build_vertical_targets(
                surface_z_m=eta,
                bed_z_m=bed,
                valid_layer_z_m=valid_layers,
            )
        )
    except ValueError as exc:
        raise InputDerivationError(f"OCM face 垂向 target／bracket 無法建立：{exc}") from exc
    target_by_id = {target.vertical_id: target for target in all_targets}
    selected_targets = tuple(target_by_id[vertical_id] for vertical_id in requested_ids)

    brackets: list[tuple[str, float, float]] = []
    for target in selected_targets:
        below = valid_layers[valid_layers <= target.z_m_positive_up]
        above = valid_layers[valid_layers >= target.z_m_positive_up]
        if below.size == 0 or above.size == 0:
            raise InputDerivationError(f"OCM face vertical_id={target.vertical_id} 沒有代表性雙側 zcor 支撐")
        lower = float(below[-1])
        upper = float(above[0])
        if upper <= lower:
            raise InputDerivationError(
                f"OCM face vertical_id={target.vertical_id} representative bracket 寬度不正"
            )
        brackets.append((target.vertical_id, lower, upper))

        # 這個檢查故意逐 node 進行；把 node 先取中位數會將 steep-bed/deep-node 的
        # 無支撐情況遮掉，正是正式三節點 sampler 可能回傳全 NaN 的來源。
        for node_index, node_layers in enumerate(zcor):
            finite_node_layers = node_layers[np.isfinite(node_layers)]
            if not (
                np.any(finite_node_layers <= target.z_m_positive_up)
                and np.any(finite_node_layers >= target.z_m_positive_up)
            ):
                raise InputDerivationError(
                    f"OCM face node={node_index} vertical_id={target.vertical_id} "
                    "缺少 target 的雙側有限 zcor 支撐"
                )

    return _FaceVerticalSupport(
        bed_z_m_positive_up=bed,
        eta_z_m_positive_up=eta,
        representative_zcor_m=tuple(float(value) for value in valid_layers),
        targets=selected_targets,
        brackets=tuple(brackets),
    )


def _metadata_provenance(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """擷取上游 metadata 的 source/config provenance，並排除絕對路徑。

    上游 metadata 可能同時含 source commit、設定版本、產製器與操作環境欄位；這些小型
    追溯資訊應進入 inventory，但原始 root、工作目錄或暫存目錄不應被複製到 release。
    因此只保留名稱含 source/config/provenance/commit/version 的欄位，並遞迴移除 path、
    root、directory 等定位欄位。實際 metadata bytes 另以 source record 的 SHA-256 綁定，
    此函式不把 structural fingerprint 誤稱為內容 hash。
    """

    def clean(value: Any, *, key_name: str = "") -> Any:
        """遞迴清理 provenance 值中的絕對定位資訊。"""

        lowered = key_name.lower()
        if any(token in lowered for token in ("path", "root", "directory", "filename")):
            return None
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key in sorted(value, key=str):
                if not isinstance(key, str):
                    continue
                cleaned = clean(value[key], key_name=key)
                if cleaned is not None:
                    result[key] = cleaned
            return result
        if isinstance(value, (list, tuple)):
            result = [clean(item, key_name=key_name) for item in value]
            return [item for item in result if item is not None]
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and Path(value).is_absolute():
                return None
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        return None

    result: dict[str, Any] = {}
    for key in sorted(metadata):
        lowered = str(key).lower()
        if not any(token in lowered for token in ("source", "config", "provenance", "commit", "version")):
            continue
        cleaned = clean(metadata[key], key_name=str(key))
        if cleaned is not None:
            result[str(key)] = cleaned
    return result


def _source_file_record(
    source: Path,
    *,
    relative_path: Path,
    file_kind: str,
    include_content_hash: bool,
) -> Mapping[str, Any]:
    """建立單一 source record；大型 NPY 預設只讀 header，不掃描 payload。

    metadata JSON 與 `time_utc_ns.npy` 都是小型且直接影響契約的檔案，因此保留實際
    SHA-256；其餘 required NPY 只保存檔案大小與 NPY header 的 shape/dtype。這種設計可
    在數 TB forcing 上快速偵測替換、截斷與結構錯誤，但不能偵測「相同大小且 header 不變
    的 NPY payload 內容替換」；那屬於另行明示的 deep content audit，不是正式建置必要步驟。
    """

    _assert_regular_file(source)
    record: dict[str, Any] = {
        "path": relative_path.as_posix(),
        "file_kind": file_kind,
        "size_bytes": int(source.stat().st_size),
    }
    if source.suffix == ".npy":
        array = _load_npy(source)
        record["fingerprint_kind"] = "npy_header_structural"
        record["npy_header"] = {
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
        }
        if include_content_hash:
            record["sha256"] = _sha256_file(source)
    else:
        if not include_content_hash:
            raise ValueError(f"非 NPY source 必須明示 content hash：{source}")
        record["fingerprint_kind"] = "metadata_content_sha256"
        record["sha256"] = _sha256_file(source)
    return record


def _list_source_files(
    root: Path,
    *,
    relative_prefix: Path,
    grid_arrays: Sequence[str],
    month_arrays: Sequence[str],
    months: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    """建立 accepted product 的 structural fingerprint，不遞迴掃描非必要檔案。

    ``relative_prefix`` 會帶有 `$ROOT_TOKEN/<flow_domain_id>`；列入的集合固定由
    schema contract 的 grid/month required arrays 與 metadata/time axis 組成。每個大型
    NPY 僅讀取 header、size 與必要時的 time-axis content hash，所以正式 inventory 不會
    對數 TB payload 做逐 byte SHA-256；未知的額外檔案不影響這份契約，也不會被誤當成已
    驗證的內容。檔案存在、普通檔案與 symbolic link 檢查仍由 caller 及 record builder 完成。
    """

    _assert_regular_directory(root)
    records: list[Mapping[str, Any]] = []

    def add(relative: Path, *, file_kind: str, include_content_hash: bool) -> None:
        """以單一明示相對路徑加入 source record。"""

        source = root / relative
        records.append(
            _source_file_record(
                source,
                relative_path=relative_prefix / relative,
                file_kind=file_kind,
                include_content_hash=include_content_hash,
            )
        )

    add(Path("grid") / "metadata.json", file_kind="grid_metadata", include_content_hash=True)
    for name in sorted(dict.fromkeys(grid_arrays)):
        add(Path("grid") / name, file_kind="grid_npy", include_content_hash=False)
    for month in sorted(months):
        month_path = Path("months") / month
        add(month_path / "metadata.json", file_kind="month_metadata", include_content_hash=True)
        add(month_path / "time_utc_ns.npy", file_kind="time_axis", include_content_hash=True)
        for name in sorted(dict.fromkeys(month_arrays)):
            # OCM/NWW config 的 required array 清單為相容既有契約而包含 time 軸；它已
            # 在上一行以 time_axis kind 加入，這裡排除重複，避免同一檔案同時有「內容
            # hash」與「month NPY structural」兩種互相矛盾的 inventory record。
            if name == "time_utc_ns.npy":
                continue
            add(month_path / name, file_kind="month_npy", include_content_hash=False)
    return tuple(sorted(records, key=lambda item: str(item["path"])))


def _validate_grid_metadata(product: str, flow_domain_id: str, metadata: Mapping[str, Any]) -> None:
    """驗證 OCM/NWW grid 的 schema 與 domain ID，避免空間格網錯配。"""

    if product in {"ocm_native", "ocm_surface"}:
        if _schema_major(metadata.get("cache_schema_version")) != 3:
            raise InputDerivationError("OCM grid schema major 必須是 3")
        domain = metadata.get("domain")
        actual = domain.get("domain_id") if isinstance(domain, Mapping) else None
    else:
        if _schema_major(metadata.get("schema_version")) != 1:
            raise InputDerivationError("NWW3 analysis grid schema major 必須是 1")
        actual = metadata.get("flow_domain_id")
    if actual != flow_domain_id:
        raise InputDerivationError(f"grid flow_domain_id 不符：{actual} != {flow_domain_id}")


def _validate_month_metadata(
    product: str,
    flow_domain_id: str,
    month: str,
    metadata: Mapping[str, Any],
    config: ProjectConfig,
) -> None:
    """驗證月份 schema/status/cache kind；partial OCM 由時間缺口另行處理。"""

    contract = (
        config.inputs.ocm_contract if product in {"ocm_native", "ocm_surface"} else config.inputs.nww_contract
    )
    schema_value = (
        metadata.get("cache_schema_version")
        if product in {"ocm_native", "ocm_surface"}
        else metadata.get("schema_version")
    )
    if _schema_major(schema_value) != int(contract["required_schema_major"]):
        raise InputDerivationError(f"{product} {month} schema major 不符")
    expected_domain = (
        metadata.get("domain", {}).get("domain_id")
        if product in {"ocm_native", "ocm_surface"} and isinstance(metadata.get("domain"), Mapping)
        else metadata.get("flow_domain_id")
    )
    if expected_domain != flow_domain_id:
        raise InputDerivationError(f"{product} {month} flow_domain_id 不符")
    accepted_statuses = {str(value) for value in contract.get("accepted_statuses", [])}
    if metadata.get("status") not in accepted_statuses:
        raise InputDerivationError(f"{product} {month} status 不在 accepted_statuses")
    accepted_kinds = {str(value) for value in contract.get("accepted_cache_kinds", [])}
    if accepted_kinds and metadata.get("cache_kind") not in accepted_kinds:
        raise InputDerivationError(f"{product} {month} cache_kind 不在 accepted_cache_kinds")
    declared_month = metadata.get("month") or metadata.get("year_month")
    if declared_month is not None and str(declared_month) != month:
        raise InputDerivationError(f"{product} {month} metadata month 不符")


def _validate_month_array_headers(
    month_dir: Path,
    metadata: Mapping[str, Any],
    *,
    required_arrays: Sequence[str],
) -> None:
    """比對月份 metadata 宣告與每個 NPY header 的 shape／dtype。

    NPY header 可用 memory-map 讀取，無需把大型 `hvel` 或波浪欄位載入記憶體；但仍能
    發現 metadata 與被截斷、錯置或原地替換檔案不一致。`time_utc_ns.npy` 也納入同一
    檢查，因為其第一軸決定所有 forcing 欄位的 UTC 對位。accepted schema 若缺少陣列
    contract 便不能證明資料結構，直接拒絕而不使用猜測的 shape／dtype。
    """

    declarations = metadata.get("arrays")
    if not isinstance(declarations, Mapping):
        raise InputDerivationError(f"月份 metadata 缺少 arrays contract：{month_dir}")
    names = ("time_utc_ns.npy", *tuple(required_arrays))
    for name in dict.fromkeys(names):
        declaration = declarations.get(name)
        if not isinstance(declaration, Mapping):
            raise InputDerivationError(f"月份 metadata 缺少陣列 contract：{month_dir}/{name}")
        raw_shape = declaration.get("shape")
        raw_dtype = declaration.get("dtype")
        if (
            not isinstance(raw_shape, list)
            or not raw_shape
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in raw_shape)
            or not isinstance(raw_dtype, str)
            or not raw_dtype.strip()
        ):
            raise InputDerivationError(f"月份 metadata 陣列 contract 不合法：{month_dir}/{name}")
        actual = _load_npy(month_dir / name)
        expected_shape = tuple(int(item) for item in raw_shape)
        if tuple(actual.shape) != expected_shape or str(actual.dtype) != raw_dtype:
            raise InputDerivationError(
                f"月份陣列 shape/dtype 與 metadata 不符：{month_dir}/{name}；"
                f"宣告={expected_shape}/{raw_dtype}，實際={tuple(actual.shape)}/{actual.dtype}"
            )


def _load_product(
    *,
    product: str,
    root: Path,
    root_token: str,
    flow_domain_id: str,
    months: Sequence[str],
    config: ProjectConfig,
) -> _ProductData:
    """低記憶體載入一套產品的 grid/header、月份時間軸與 structural fingerprint。

    陣列資料用 memory-map 只讀 NPY header；產品級 binding 不代表大型 NPY 的逐 byte
    內容 hash。metadata 與 time axis 的實際 hash、必要 NPY 的 size/shape/dtype 會在
    `_list_source_files` 形成可重建的 accepted-product fingerprint。
    """

    _assert_regular_directory(root)
    domain_root = root / flow_domain_id
    _assert_regular_directory(domain_root)
    grid_dir = domain_root / "grid"
    _assert_regular_directory(grid_dir)
    grid_metadata = _read_metadata(grid_dir / "metadata.json")
    _validate_grid_metadata(product, flow_domain_id, grid_metadata)
    contract = config.inputs.ocm_contract if product == "ocm_native" else config.inputs.nww_contract
    if product == "ocm_native":
        required_arrays = tuple(str(value) for value in contract["required_month_arrays"])
        required_grid_arrays = tuple(str(value) for value in contract["required_grid_arrays"])
    elif product == "ocm_surface":
        required_arrays = _OCM_SURFACE_REQUIRED_MONTH_ARRAYS
        required_grid_arrays = _OCM_SURFACE_REQUIRED_GRID_ARRAYS
    else:
        required_arrays = tuple(str(value) for value in contract["required_arrays"])
        required_grid_arrays = ("lon.npy", "lat.npy", "mask_static.npy")
    for name in required_grid_arrays:
        _assert_regular_file(grid_dir / name)

    month_data: list[_MonthData] = []
    chunks: list[TimeChunk] = []
    for month in sorted(months):
        month_dir = domain_root / "months" / month
        _assert_regular_directory(month_dir)
        metadata = _read_metadata(month_dir / "metadata.json")
        _validate_month_metadata(product, flow_domain_id, month, metadata, config)
        _validate_month_array_headers(month_dir, metadata, required_arrays=required_arrays)
        for name in required_arrays:
            _assert_regular_file(month_dir / name)
        time_ns = _load_npy(month_dir / "time_utc_ns.npy", dtype=np.dtype("int64"))
        if time_ns.ndim != 1 or time_ns.size < 2 or np.any(np.diff(time_ns) <= 0):
            raise InputDerivationError(f"{product} {month} time_utc_ns 必須是嚴格遞增一維 int64")
        item = _MonthData(month, month_dir, metadata, time_ns)
        month_data.append(item)
        chunks.append(TimeChunk(month, time_ns))
    time_contract = config.inputs.time_axis_contract
    canonical = canonicalize_time_chunks(
        chunks,
        policy=str(time_contract["canonicalization_policy"]),  # type: ignore[arg-type]
        expected_timestep_hours=float(time_contract["expected_timestep_hours"]),
    )
    source_records = _list_source_files(
        domain_root,
        relative_prefix=Path(f"${root_token}") / flow_domain_id,
        grid_arrays=required_grid_arrays,
        month_arrays=required_arrays,
        months=months,
    )
    return _ProductData(
        product=product,
        flow_domain_id=flow_domain_id,
        root=root,
        root_token=root_token,
        grid_dir=grid_dir,
        grid_metadata=grid_metadata,
        months=tuple(month_data),
        canonical=canonical,
        required_arrays=required_arrays,
        source_file_records=source_records,
    )


def _array_record_hash(product: _ProductData) -> str:
    """以 canonical UTC 與 structural source records 建立產品級 fingerprint。

    產品 hash 是 metadata/time content hash 與所有 required NPY header/size 的 canonical
    binding，不是大型 NPY payload 的內容摘要；payload deep audit 必須另行明示開啟。
    """

    payload = {
        "flow_domain_id": product.flow_domain_id,
        "product": product.product,
        "time_sha256": sha256(np.asarray(product.canonical.time_utc_ns, dtype="<i8").tobytes()).hexdigest(),
        "files": list(product.source_file_records),
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _canonical_values(product: _ProductData, name: str, *, reducer: str = "mean") -> dict[int, float]:
    """從每月 array 產生 ``UTC ns -> scalar`` 摘要，不對缺值用零替代。

    OCM 的 ``elev``／``hvel`` 需要站點空間索引時由 caller 先提供專用 helper；本函式只
    處理 NWW 全域代表序列。``reducer`` 目前支援 mean、median、max，所有非有限元素
    都被排除，整個時間切片無有效值時保留 NaN。
    """

    if name not in product.required_arrays:
        raise InputDerivationError(f"產品未宣告必要欄位：{product.product}/{name}")
    result: dict[int, float] = {}
    for month in product.months:
        array = _load_npy(month.directory / name)
        if array.shape[0] != month.time_ns.size:
            raise InputDerivationError(f"{product.product} {month.label}/{name} time shape 不符")
        for local_index, time_ns in enumerate(month.time_ns):
            values = np.asarray(array[local_index], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                scalar = float("nan")
            elif reducer == "median":
                scalar = float(np.median(finite))
            elif reducer == "max":
                scalar = float(np.max(finite))
            else:
                scalar = float(np.mean(finite))
            # 同 UTC 的跨月 halo 採 canonical 軸最後來源；直接覆寫的順序是月份排序，
            # 與 canonicalize_time_chunks 的 stable/prefer-last 契約一致。
            result[int(time_ns)] = scalar
    return result


def _source_hashes_for_products(products: Iterable[_ProductData]) -> dict[str, str]:
    """將產品級 fingerprint 壓成 provenance 可用的 source hash mapping。"""

    return {
        f"{product.product}:{product.flow_domain_id}": _array_record_hash(product) for product in products
    }


def _load_native_mesh(product: _ProductData, projection: DomainProjection) -> NativeMesh:
    """由 OCM grid 靜態陣列建立既有 NativeMesh，所有水平計算落在公尺座標。"""

    return NativeMesh.from_directory(product.grid_dir, projection=projection)


def _face_polygon_union(mesh: NativeMesh) -> Polygon:
    """由原生 SCHISM faces 建立靜態海域支撐 polygon。

    union 優先保留網格的實際 face 支撐；若 source mesh 有碎片，取包含最多面積的連通
    polygon；若 union 退化則回傳 node convex hull。這是固定 geometry 候選，不把逐時
    wet/dry mask 改成動態 boundary；逐時濕乾仍會在 receptor 與 forcing stage gate。
    """

    polygons = []
    for row, count in zip(mesh.face_nodes_local, mesh.face_node_count, strict=True):
        nodes = row[: int(count)]
        coordinates = [tuple(mesh.node_xy[int(node)]) for node in nodes]
        candidate = Polygon(coordinates)
        if candidate.is_valid and candidate.area > 0:
            polygons.append(candidate)
    if not polygons:
        hull = Polygon(mesh.node_xy).convex_hull
        if not isinstance(hull, Polygon) or hull.area <= 0:
            raise InputDerivationError("OCM mesh 無法建立靜態海域 polygon")
        return hull
    union = unary_union(polygons)
    if isinstance(union, Polygon):
        return union
    candidates = [item for item in getattr(union, "geoms", ()) if isinstance(item, Polygon)]
    if candidates:
        return max(candidates, key=lambda item: item.area)
    hull = union.convex_hull
    if not isinstance(hull, Polygon) or hull.area <= 0:
        raise InputDerivationError("OCM face union 無法建立 polygon")
    return hull


def _authoritative_flow_domain_bbox_lon_lat(
    domain: DomainConfig,
    actual_flow_domain_id: str,
) -> tuple[float, float, float, float]:
    """解析實際 accepted flow-domain 所註冊的權威 WGS84 bbox。

    ``expanded_domain_v1`` 的舊設定仍可由 base、formal 或明示 expanded candidate
    綁定；函式不依目錄名稱或 ID 字串推測空間範圍。對
    ``v3_local20km_20260909_v1``，accepted ID 必須是唯一的
    ``northeast_taiwan_common_cache_v3``，只使用固定 base bbox，絕不讀取同根目錄的
    v4 candidate。expanded bbox 會檢查四個有限數值的順序與設定中心是否位於範圍內，讓
    舊版 source ID、outer geometry、inventory 與 release config 不可能各自綁定不同的
    空間版本。回傳順序固定為 ``(lon_min, lon_max, lat_min, lat_max)``，單位為度。
    """

    if not isinstance(actual_flow_domain_id, str) or not actual_flow_domain_id.strip():
        raise InputDerivationError("actual flow-domain ID 必須是非空字串")
    if domain.formal_domain_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1:
        # ProjectConfig 已在整體 schema gate 驗證新 policy；此處仍重做 source-level
        # exact check，防止 caller 以 model_copy 或人工建立的 DomainConfig 繞過整體 gate。
        # 20 km policy 尚未有共同有效網格證據，這個 resolver 只允許準備性 strict input
        # 讀取原始 v3，不把任何 candidate 目錄升格成 formal source。
        if (
            domain.analysis_region_id != "A"
            or domain.flow_domain_id != NORTHEAST_V3_FLOW_DOMAIN_ID
            or domain.formal_release_flow_domain_id != NORTHEAST_V3_FLOW_DOMAIN_ID
            or tuple(float(value) for value in domain.bbox_lon_lat) != NORTHEAST_V3_BBOX_LON_LAT
            or actual_flow_domain_id != NORTHEAST_V3_FLOW_DOMAIN_ID
        ):
            raise InputDerivationError(
                "v3_local20km_20260909_v1 僅接受 A 區 exact northeast_taiwan_common_cache_v3 及固定 v3 bbox"
            )
        return tuple(float(value) for value in domain.bbox_lon_lat)
    if domain.formal_domain_policy != FORMAL_DOMAIN_POLICY_EXPANDED_V1:
        raise InputDerivationError(f"未知 formal_domain_policy：{domain.formal_domain_policy!r}")
    base_id = domain.flow_domain_id
    formal_id = domain.formal_release_flow_domain_id
    candidate_id = domain.model_extra.get("expanded_domain_candidate_id") if domain.model_extra else None
    allowed_ids = {
        value for value in (base_id, formal_id, candidate_id) if isinstance(value, str) and value.strip()
    }
    if actual_flow_domain_id not in allowed_ids:
        raise InputDerivationError(
            f"{domain.analysis_region_id} source ID 未在 config 明示 base/formal/candidate 中："
            f"{actual_flow_domain_id}"
        )
    if actual_flow_domain_id == base_id:
        bbox = tuple(float(value) for value in domain.bbox_lon_lat)
    else:
        raw_bbox = domain.model_extra.get("expanded_bbox_lon_lat") if domain.model_extra else None
        if (
            not isinstance(raw_bbox, (list, tuple))
            or len(raw_bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_bbox)
        ):
            raise InputDerivationError(
                f"{domain.analysis_region_id} expanded source 缺少合法 expanded_bbox_lon_lat"
            )
        bbox = tuple(float(value) for value in raw_bbox)
    lon_min, lon_max, lat_min, lat_max = bbox
    if (
        len(bbox) != 4
        or not all(math.isfinite(value) for value in bbox)
        or not (lon_min < lon_max and lat_min < lat_max)
    ):
        raise InputDerivationError(
            f"{domain.analysis_region_id} authoritative bbox 必須是有限且嚴格遞增的四元組"
        )
    center_lon, center_lat = (float(value) for value in domain.center_lonlat)
    if not (lon_min <= center_lon <= lon_max and lat_min <= center_lat <= lat_max):
        raise InputDerivationError(f"{domain.analysis_region_id} authoritative bbox 必須涵蓋 config center")
    return bbox


def _flow_domain_bbox_registration(domain: DomainConfig, actual_flow_domain_id: str) -> str:
    """回傳已由權威 bbox resolver 驗證的 base／expanded provenance 標籤。"""

    _authoritative_flow_domain_bbox_lon_lat(domain, actual_flow_domain_id)
    return "base" if actual_flow_domain_id == domain.flow_domain_id else "expanded_registered"


def _geometry_payloads(
    *,
    config: ProjectConfig,
    products_by_region: Mapping[str, _ProductData],
    source_hashes: Mapping[str, str],
    strict: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Polygon], dict[str, Polygon]]:
    """由四個 flow domain 與五個 study site 生成三份 strict geometry manifest。

    flow polygon 來自 authoritative resolver 選出的 base 或 expanded registered bbox，
    再建立加密 WGS84 boundary；A 區 local polygon 依 policy 使用 legacy 25 km 或新 v3
    20 km 的 anchor-centered buffer，再與 OCM face-derived static ocean polygon 求交。
    B-D local 等於 flow。這樣 geometry 不會因某一個 arrival 的 wet/dry 狀態改變，同時
    保留 domain/local/open-boundary 的來源 provenance、真實 flow-domain ID 與其共同註冊
    的 bbox 版本。新 v3 local policy 會把 20 km 幾何版本 token 寫入 identity，避免沿用
    舊 25 km candidate 的 manifest 識別碼。
    """

    domain_records: list[dict[str, Any]] = []
    local_records: list[dict[str, Any]] = []
    open_records: list[dict[str, Any]] = []
    flow_polygons: dict[str, Polygon] = {}
    local_polygons: dict[str, Polygon] = {}
    site_by_id = {site.study_site_id: site for site in config.study_sites}
    domain_by_region = {domain.analysis_region_id: domain for domain in config.domains}
    projections: dict[str, DomainProjection] = {}
    flow_id_by_region: dict[str, str] = {}
    bbox_by_region: dict[str, tuple[float, float, float, float]] = {}
    bbox_registration_by_region: dict[str, str] = {}
    has_v3_policy = any(
        domain.formal_domain_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1
        for domain in config.domains
    )

    for region in sorted(domain_by_region):
        domain = domain_by_region[region]
        product = products_by_region[region]
        flow_id = product.flow_domain_id
        geometry_policy_token = (
            "v3_local20km_20260909_v1"
            if domain.formal_domain_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1
            else "v1"
        )
        flow_id_by_region[region] = flow_id
        bbox_by_region[region] = _authoritative_flow_domain_bbox_lon_lat(domain, flow_id)
        bbox_registration_by_region[region] = _flow_domain_bbox_registration(domain, flow_id)
        projection = DomainProjection(*domain.center_lonlat)
        projections[flow_id] = projection
        polygon = densified_bbox_polygon(bbox_by_region[region])
        flow_polygons[flow_id] = polygon
        domain_records.append(
            {
                "analysis_region_id": region,
                "flow_domain_id": flow_id,
                "geometry": mapping(polygon),
                "source_geometry_id": (
                    f"{product.product}_{flow_id}_{bbox_registration_by_region[region]}_bbox_"
                    f"{geometry_policy_token}"
                ),
            }
        )
        open_records.append(
            {
                "owner_kind": "flow_domain",
                "owner_id": flow_id,
                "analysis_region_id": region,
                "segment_id": f"{flow_id}_flow_open_boundary",
                "geometry": mapping(LineString(polygon.exterior.coords)),
                "source_geometry_id": (
                    f"{flow_id}_{bbox_registration_by_region[region]}_bbox_exterior_open_boundary_"
                    f"{geometry_policy_token}"
                ),
            }
        )

    for site_id in sorted(site_by_id):
        site = site_by_id[site_id]
        region = site.analysis_region_id
        flow_id = flow_id_by_region[region]
        domain = domain_by_region[region]
        polygon = flow_polygons[flow_id]
        local_equals_flow = region != "A"
        geometry_policy_token = (
            "v3_local20km_20260909_v1"
            if domain.formal_domain_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1
            else "v1"
        )
        if local_equals_flow:
            local = polygon
        else:
            projection = projections[flow_id]
            product = products_by_region[region]
            mesh = _load_native_mesh(product, projection)
            static_ocean = _face_polygon_union(mesh)
            anchor = site.anchor_lonlat
            radius = site.local_domain_baseline_radius_m
            if anchor is None or radius is None:
                raise InputDerivationError(f"A 區 study site 缺少 anchor/radius：{site_id}")
            flow_metric = projection.project_geometry(polygon)
            anchor_x, anchor_y = projection.project(*anchor)
            local_buffer = Point(float(anchor_x), float(anchor_y)).buffer(float(radius), quad_segs=64)
            # 先驗證完整 local 圓是否落在 authoritative flow bbox，再與靜態海域求交。
            # 若先 clip，越界會被悄悄裁掉，後續看似有 geometry 卻沒有證明 20 km 邊界
            # 的 forcing 支援；這裡只建立準備性 strict input，不替未完成的共同格網
            # validator 宣稱通過。
            if (
                domain.formal_domain_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1
                and not flow_metric.covers(local_buffer)
            ):
                raise InputDerivationError(
                    f"{site_id} local radius {float(radius):g} m 超出 authoritative flow-domain "
                    "bbox；不得以 clip 隱藏越界"
                )
            # candidate geometry 先與 flow bbox 交集，避免 static mesh convex hull 因邊界
            # 三角形數量很少而意外越過設定 outer domain。
            local = build_anchor_local_domain(
                projection=projection,
                anchor_lonlat=anchor,
                radius_m=float(radius),
                static_ocean_polygon_metric=static_ocean,
            )
            clipped = local.intersection(flow_metric)
            if clipped.is_empty or not isinstance(clipped, Polygon):
                raise InputDerivationError(f"local geometry 與 flow geometry 無有效交集：{site_id}")
            local_metric = clipped
            local = projection.unproject_geometry(local_metric)
            if not isinstance(local, Polygon) or not local.is_valid or local.area <= 0:
                raise InputDerivationError(f"local geometry 反投影失效：{site_id}")
        local_polygons[site_id] = local
        local_records.append(
            {
                "study_site_id": site_id,
                "analysis_region_id": region,
                "flow_domain_id": flow_id,
                "local_equals_flow": local_equals_flow,
                "geometry": mapping(local),
                "source_geometry_id": (
                    f"{flow_id}_{bbox_registration_by_region[region]}_{site_id}_local_domain_"
                    f"{geometry_policy_token}"
                ),
            }
        )
        if not local_equals_flow:
            open_records.append(
                {
                    "owner_kind": "local_domain",
                    "owner_id": site_id,
                    "analysis_region_id": region,
                    "segment_id": f"{site_id}_local_open_boundary",
                    "geometry": mapping(LineString(local.exterior.coords)),
                    "source_geometry_id": (
                        f"{flow_id}_{bbox_registration_by_region[region]}_"
                        f"{site_id}_local_exterior_open_boundary_{geometry_policy_token}"
                    ),
                }
            )

    public_policy = {
        "A": "A 區分析域",
        "note": "公開圖表可使用 A 區分析域；內部 records/provenance 保留實際 flow-domain ID。",
    }
    common_extra = {
        "public_analysis_label_policy": public_policy,
        "resolved_flow_domain_ids": dict(sorted(flow_id_by_region.items())),
        "resolved_flow_domain_bboxes": {
            region: {
                "flow_domain_id": flow_id_by_region[region],
                "bbox_lon_lat": list(bbox_by_region[region]),
                "registration": bbox_registration_by_region[region],
            }
            for region in sorted(bbox_by_region)
        },
    }
    if has_v3_policy:
        # 新 policy 才需要在 geometry provenance 明示版本；legacy payload 保持原欄位
        # 語意，避免預設值改變既有 artifact canonical fingerprint。
        common_extra["formal_domain_policy_by_region"] = {
            region: domain_by_region[region].formal_domain_policy for region in sorted(domain_by_region)
        }
    domain_payload = {
        "manifest_kind": "domain_geometry_manifest",
        "schema_version": DERIVED_INPUT_SCHEMA_VERSION,
        "status": "approved" if strict else "generated",
        "design_version": config.design_version,
        "coordinate_reference": "EPSG:4326",
        "provenance": _provenance(
            method_id="server_v3_domain_bbox_geometry_v1",
            source_hashes=source_hashes,
            **common_extra,
        ),
        "records": domain_records,
    }
    local_payload = {
        "manifest_kind": "local_geometry_manifest",
        "schema_version": DERIVED_INPUT_SCHEMA_VERSION,
        "status": "approved" if strict else "generated",
        "design_version": config.design_version,
        "coordinate_reference": "EPSG:4326",
        "provenance": _provenance(
            method_id=(
                "server_v3_anchor_local_geometry_v3_local20km_20260909_v1"
                if has_v3_policy
                else "server_v3_anchor_local_geometry_v1"
            ),
            source_hashes=source_hashes,
            **common_extra,
        ),
        "records": local_records,
    }
    open_payload = {
        "manifest_kind": "open_boundary_manifest",
        "schema_version": DERIVED_INPUT_SCHEMA_VERSION,
        "status": "approved" if strict else "generated",
        "design_version": config.design_version,
        "coordinate_reference": "EPSG:4326",
        "provenance": _provenance(
            method_id=(
                "server_v3_open_boundary_geometry_v3_local20km_20260909_v1"
                if has_v3_policy
                else "server_v3_open_boundary_geometry_v1"
            ),
            source_hashes=source_hashes,
            **common_extra,
        ),
        "records": open_records,
    }
    return domain_payload, local_payload, open_payload, flow_polygons, local_polygons


def _nww_summary_series(product: _ProductData) -> tuple[dict[int, float], dict[int, bool]]:
    """讀取 NWW Hs 與 valid mask，產生每 UTC 的有限代表波高／共同有效旗標。"""

    result: dict[int, float] = {}
    valid_result: dict[int, bool] = {}
    for month in product.months:
        hs = _load_npy(month.directory / "significant_wave_height.npy")
        valid = _load_npy(month.directory / "valid_mask_wave.npy")
        if hs.shape != valid.shape or hs.shape[0] != month.time_ns.size:
            raise InputDerivationError(f"NWW3 {month.label} Hs/valid_mask shape 不符")
        for local, time_ns in enumerate(month.time_ns):
            frame = np.asarray(hs[local], dtype=np.float64)
            mask = np.asarray(valid[local], dtype=bool) & np.isfinite(frame)
            values = frame[mask]
            result[int(time_ns)] = float(np.median(values)) if values.size else float("nan")
            valid_result[int(time_ns)] = bool(values.size)
    return result, valid_result


def _grid_spatial_support(
    *,
    grid_dir: Path,
    product_label: str,
    lon: float,
    lat: float,
) -> tuple[tuple[int, ...], tuple[int, ...], int, bool]:
    """找出規則／曲線格網的保守支撐索引與最近格點。

    回傳 `(spatial_shape, support_indices, nearest_index, static_valid)`。規則一維經緯度
    軸會依 runtime 的雙線性 sampler 取四角；站點在軸外時回傳空支撐。二維曲線格網
    沒有既有四角拓撲，則只允許最近格點作為保守支撐。所有 mask 與座標都在此先驗證，
    後續 surface selector 只能使用明確的 common-valid 結果，不會以最近有效值偷換
    域外或陸地資料；NWW selector 另由 runtime-equivalent 四角 helper 處理，不使用
    這裡的曲線格網 nearest fallback。
    """

    lon_values = np.asarray(_load_npy(grid_dir / "lon.npy"), dtype=np.float64)
    lat_values = np.asarray(_load_npy(grid_dir / "lat.npy"), dtype=np.float64)
    static_mask = np.asarray(_load_npy(grid_dir / "mask_static.npy"), dtype=bool)
    regular_axes = lon_values.ndim == 1 and lat_values.ndim == 1
    if regular_axes:
        if lon_values.size < 2 or lat_values.size < 2:
            raise InputDerivationError(f"{product_label} 一維 lon/lat 軸至少需要兩個格點")
        if np.any(np.diff(lon_values) <= 0) or np.any(np.diff(lat_values) <= 0):
            raise InputDerivationError(f"{product_label} 一維 lon/lat 軸必須嚴格遞增")
        lon_axis = lon_values
        lat_axis = lat_values
        lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis, indexing="xy")
    elif lon_values.ndim == 2 and lat_values.shape == lon_values.shape:
        lon_grid, lat_grid = lon_values, lat_values
    else:
        raise InputDerivationError(f"{product_label} grid lon/lat 必須是相同形狀的一維軸或二維座標網格")
    if static_mask.shape != lon_grid.shape:
        raise InputDerivationError(f"{product_label} grid mask_static 形狀與 lon/lat 不符")
    finite_coordinates = np.isfinite(lon_grid) & np.isfinite(lat_grid)
    if not np.any(finite_coordinates):
        raise InputDerivationError(f"{product_label} grid 沒有有限的分析格點座標")
    distance_squared = (lon_grid - float(lon)) ** 2 + (lat_grid - float(lat)) ** 2
    distance_squared[~finite_coordinates] = np.inf
    nearest_index = int(np.argmin(distance_squared))
    if regular_axes:
        if not (
            float(lon_axis[0]) <= float(lon) <= float(lon_axis[-1])
            and float(lat_axis[0]) <= float(lat) <= float(lat_axis[-1])
        ):
            support_indices: tuple[int, ...] = ()
        else:
            x_after = min(max(int(np.searchsorted(lon_axis, lon, side="right")), 1), lon_axis.size - 1)
            y_after = min(max(int(np.searchsorted(lat_axis, lat, side="right")), 1), lat_axis.size - 1)
            x0, x1 = x_after - 1, x_after
            y0, y1 = y_after - 1, y_after
            support_indices = tuple(
                int(y_index * lon_axis.size + x_index)
                for y_index, x_index in ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
            )
    else:
        support_indices = (nearest_index,)
    static_valid = bool(support_indices) and all(
        bool(static_mask.ravel()[index]) for index in support_indices
    )
    return tuple(int(value) for value in lon_grid.shape), support_indices, nearest_index, static_valid


def _load_nww_runtime_cache(product: _ProductData) -> _NWWRuntimeCache:
    """載入可由 ``NWWAnalysisMonth.sample`` 完全重現的 NWW 規則格網檢視。

    accepted NWW3 analysis 的座標軸必須是有限、嚴格遞增的一維 lon／lat，波浪欄位則
    必須是 ``(time, lat, lon)``。這裡保留 memory-map，不複製大型資料；但會先驗證
    每個月份的 shape 與 UTC 是否為整點，讓後續 exact-hour gate 不會以時間內插或錯位
    欄位代替缺少的支撐。二維曲線格網雖可被舊的共用 grid helper 讀取，runtime 的
    NWW sampler 無法對它建立同一套 searchsorted／四角契約，因此在此明確拒絕。
    """

    if product.product != "nww3_analysis":
        raise InputDerivationError(f"NWW runtime cache 收到非 NWW3 product：{product.product}")
    lon_axis = np.asarray(_load_npy(product.grid_dir / "lon.npy"), dtype=np.float64)
    lat_axis = np.asarray(_load_npy(product.grid_dir / "lat.npy"), dtype=np.float64)
    if lon_axis.ndim != 1 or lat_axis.ndim != 1:
        raise InputDerivationError("NWW runtime sampler 只支援一維 lon/lat 軸，不接受二維座標網格")
    if lon_axis.size < 2 or lat_axis.size < 2:
        raise InputDerivationError("NWW runtime lon/lat 軸至少需要兩個嚴格遞增格點")
    if (
        not np.all(np.isfinite(lon_axis))
        or not np.all(np.isfinite(lat_axis))
        or np.any(np.diff(lon_axis) <= 0.0)
        or np.any(np.diff(lat_axis) <= 0.0)
    ):
        raise InputDerivationError("NWW runtime lon/lat 軸必須是有限且嚴格遞增的一維座標")

    static_mask_values = np.asarray(_load_npy(product.grid_dir / "mask_static.npy"), dtype=np.float64)
    expected_grid_shape = (lat_axis.size, lon_axis.size)
    if static_mask_values.shape != expected_grid_shape:
        raise InputDerivationError("NWW mask_static shape 必須是 (lat,lon) 且與 runtime grid 相符")
    if not np.all(np.isfinite(static_mask_values)):
        raise InputDerivationError("NWW mask_static 不得含非有限值")
    static_mask = static_mask_values != 0.0

    months: list[_NWWRuntimeMonth] = []
    expected_shape = (lat_axis.size, lon_axis.size)
    for source_month in product.months:
        hs = _load_npy(source_month.directory / "significant_wave_height.npy")
        fp = _load_npy(source_month.directory / "peak_frequency.npy")
        direction = _load_npy(source_month.directory / "peak_direction_raw_deg.npy")
        valid_wave = _load_npy(source_month.directory / "valid_mask_wave.npy")
        expected_array_shape = (source_month.time_ns.size, *expected_shape)
        arrays = (hs, fp, direction, valid_wave)
        if any(tuple(array.shape) != expected_array_shape for array in arrays):
            raise InputDerivationError(f"NWW3 {source_month.label} runtime array shape 不符")
        if np.any(np.mod(source_month.time_ns, _UTC_HOUR_NS) != 0):
            raise InputDerivationError(f"NWW3 {source_month.label} time_utc_ns 必須是 exact-hour UTC")
        months.append(
            _NWWRuntimeMonth(
                source=source_month,
                significant_wave_height=hs,
                peak_frequency=fp,
                peak_direction_raw_deg=direction,
                valid_mask_wave=valid_wave,
            )
        )
    return _NWWRuntimeCache(
        lon_axis=lon_axis,
        lat_axis=lat_axis,
        static_mask=static_mask,
        months=tuple(months),
    )


def _nww_runtime_spatial_support(
    cache: _NWWRuntimeCache,
    *,
    lon: float,
    lat: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    """依 runtime 的 searchsorted 與邊界 clamp 回傳四角索引、權重與 static gate。

    回傳四角順序固定為 ``(y0,x0)、(y0,x1)、(y1,x0)、(y1,x1)``，與
    ``NWWAnalysisMonth.sample`` 相同；權重也是先由 x 再由 y 的雙線性權重。座標域外
    不回傳 nearest fallback，而是以 ``None`` 表示所有 exact-hour 時次均不支援。
    """

    if not np.isfinite(float(lon)) or not np.isfinite(float(lat)):
        return None
    if (
        float(lon) < cache.lon_axis[0]
        or float(lon) > cache.lon_axis[-1]
        or float(lat) < cache.lat_axis[0]
        or float(lat) > cache.lat_axis[-1]
    ):
        return None
    x_after = min(
        max(int(np.searchsorted(cache.lon_axis, lon, side="right")), 1),
        cache.lon_axis.size - 1,
    )
    y_after = min(
        max(int(np.searchsorted(cache.lat_axis, lat, side="right")), 1),
        cache.lat_axis.size - 1,
    )
    x0, x1 = x_after - 1, x_after
    y0, y1 = y_after - 1, y_after
    wx = float((lon - cache.lon_axis[x0]) / (cache.lon_axis[x1] - cache.lon_axis[x0]))
    wy = float((lat - cache.lat_axis[y0]) / (cache.lat_axis[y1] - cache.lat_axis[y0]))
    corner_y = np.asarray([y0, y0, y1, y1], dtype=np.int64)
    corner_x = np.asarray([x0, x1, x0, x1], dtype=np.int64)
    spatial_weights = np.asarray(
        [(1.0 - wy) * (1.0 - wx), (1.0 - wy) * wx, wy * (1.0 - wx), wy * wx],
        dtype=np.float64,
    )
    static_valid = bool(np.all(cache.static_mask[corner_y, corner_x]))
    return corner_y, corner_x, spatial_weights, static_valid


def _nww_runtime_cell_indices(
    cache: _NWWRuntimeCache,
    *,
    lon: float,
    lat: float,
) -> tuple[int, int, int, int] | None:
    """回傳 runtime-equivalent 四角 cell index；域外或 static 四角失敗時回傳 ``None``。

    這個 helper 與 ``_nww_runtime_spatial_support`` 共用同一套 ``searchsorted``／邊界
    clamp，避免 metric-location binding 記錄的 cell 與實際 runtime sample 不一致。它只
    做座標與靜態 mask gate，不讀取逐時波浪陣列；dynamic exact-hour 可用性仍須由
    ``_nww_exact_hour_samples`` 逐 UTC 驗證。
    """

    support = _nww_runtime_spatial_support(cache, lon=lon, lat=lat)
    if support is None or not support[3]:
        return None
    corner_y, corner_x, _weights, _static_valid = support
    if corner_x.shape != (4,) or corner_y.shape != (4,):
        raise InputDerivationError("NWW runtime cell 四角索引形狀不符")
    return int(corner_x[0]), int(corner_x[1]), int(corner_y[0]), int(corner_y[2])


def _nww_runtime_cell_axis_indices(axis: np.ndarray, coordinate: float) -> tuple[int, int]:
    """依 runtime 邊界 clamp 取得一維座標軸上包住位置的相鄰索引。"""

    if axis.ndim != 1 or axis.size < 2:
        raise InputDerivationError("NWW runtime 座標軸至少需要兩個一維格點")
    after = min(max(int(np.searchsorted(axis, coordinate, side="right")), 1), axis.size - 1)
    return after - 1, after


def _nww_representative_grid_scale_m(
    cache: _NWWRuntimeCache,
    *,
    projection: DomainProjection,
    anchor_lon: float,
    anchor_lat: float,
) -> float:
    """以 anchor 所在 NWW cell 的實際投影邊長推導局地代表格網尺度。

    經度方向邊長在 anchor latitude 投影，緯度方向邊長在 anchor longitude 投影；兩者
    都是實際 NWW 軸相鄰節點的公尺距離，最後取中位數作單一代表尺度。這個尺度只用來
    定義 metric proxy 的兩格搜尋半徑，不是 OCM mesh 尺度、受體半徑或粒子步長，且不
    寫死公里常數。anchor 超出 NWW 軸時仍依 runtime 邊界 clamp 使用最近 cell 的軸距離，
    但候選中心最後仍須通過實際投影距離與 local polygon gate。
    """

    x0, x1 = _nww_runtime_cell_axis_indices(cache.lon_axis, float(anchor_lon))
    y0, y1 = _nww_runtime_cell_axis_indices(cache.lat_axis, float(anchor_lat))
    lon_pair = np.asarray(cache.lon_axis[[x0, x1]], dtype=np.float64)
    lat_pair = np.asarray(cache.lat_axis[[y0, y1]], dtype=np.float64)
    lon_x, lon_y = projection.project(lon_pair, np.full(2, float(anchor_lat), dtype=np.float64))
    lat_x, lat_y = projection.project(np.full(2, float(anchor_lon), dtype=np.float64), lat_pair)
    lon_spacing = float(np.hypot(lon_x[1] - lon_x[0], lon_y[1] - lon_y[0]))
    lat_spacing = float(np.hypot(lat_x[1] - lat_x[0], lat_y[1] - lat_y[0]))
    spacings = np.asarray([lon_spacing, lat_spacing], dtype=np.float64)
    if not np.all(np.isfinite(spacings)) or np.any(spacings <= 0.0):
        raise InputDerivationError("NWW anchor 附近無法由實際 lon/lat 軸推導正的 grid scale")
    return float(np.median(spacings))


def _nww_metric_axis_candidate_starts(
    axis: np.ndarray,
    projected_axis: np.ndarray,
    *,
    anchor_projected: float,
    maximum_distance_m: float,
    representative_grid_scale_m: float,
) -> tuple[int, ...]:
    """先以投影軸距離縮小可行 cell start，避免逐一掃描整個 NWW cell 網格。"""

    nearest = int(np.argmin(np.abs(projected_axis - float(anchor_projected))))
    axis_margin = float(maximum_distance_m + representative_grid_scale_m)
    nearby_nodes = np.flatnonzero(np.abs(projected_axis - float(anchor_projected)) <= axis_margin)
    # 若 anchor 落在稀疏軸或軸外，保留 nearest 附近少量 cell 作保守候選；後續仍會以
    # 真正 cell center 的投影距離再次篩選，因此這裡不會放寬兩格尺度的最終 gate。
    nodes = set(int(value) for value in nearby_nodes)
    nodes.add(nearest)
    starts: set[int] = set()
    for node in nodes:
        if node > 0:
            starts.add(node - 1)
        if node < axis.size - 1:
            starts.add(node)
    # 這個小窗口只處理 anchor 恰位於 cell 節點或 projection 軸距離在極端曲率下
    # 不完全單調的情況；它仍受後續 local polygon／metric distance gate 約束。
    for start in range(max(0, nearest - 2), min(axis.size - 1, nearest + 3)):
        starts.add(start)
    return tuple(sorted(starts))


def _nww_metric_location_candidates(
    cache: _NWWRuntimeCache,
    *,
    projection: DomainProjection,
    anchor_lon: float,
    anchor_lat: float,
    local_polygon_lonlat: Polygon,
    representative_grid_scale_m: float,
) -> tuple[_NWWMetricLocationCandidate, ...]:
    """建立通過 local polygon、static 四角與兩格尺度距離 gate 的 cell-center 候選。

    候選生成先只掃描投影軸附近的小範圍，再對 cell center 做 Shapely 與公尺距離判定；
    因此不會對大型 NWW 規則格網的每一個 cell 先執行 17,544 小時 dynamic sample。
    回傳排序固定為投影距離、``y0``、``x0``，後續 caller 才按此順序逐個執行完整
    exact-hour 48+2 selector。
    """

    anchor_x_array, anchor_y_array = projection.project(float(anchor_lon), float(anchor_lat))
    anchor_x = float(anchor_x_array)
    anchor_y = float(anchor_y_array)
    maximum_distance_m = NWW_METRIC_LOCATION_MAX_GRID_SCALES * float(representative_grid_scale_m)
    lon_projected, _ = projection.project(
        cache.lon_axis,
        np.full(cache.lon_axis.shape, float(anchor_lat), dtype=np.float64),
    )
    _, lat_projected = projection.project(
        np.full(cache.lat_axis.shape, float(anchor_lon), dtype=np.float64),
        cache.lat_axis,
    )
    x_starts = _nww_metric_axis_candidate_starts(
        cache.lon_axis,
        lon_projected,
        anchor_projected=anchor_x,
        maximum_distance_m=maximum_distance_m,
        representative_grid_scale_m=representative_grid_scale_m,
    )
    y_starts = _nww_metric_axis_candidate_starts(
        cache.lat_axis,
        lat_projected,
        anchor_projected=anchor_y,
        maximum_distance_m=maximum_distance_m,
        representative_grid_scale_m=representative_grid_scale_m,
    )
    candidates: list[_NWWMetricLocationCandidate] = []
    distance_tolerance = max(1e-6, maximum_distance_m * 1e-12)
    for y0 in y_starts:
        y1 = y0 + 1
        for x0 in x_starts:
            x1 = x0 + 1
            if not bool(
                np.all(
                    cache.static_mask[
                        np.asarray([y0, y0, y1, y1], dtype=np.int64),
                        np.asarray([x0, x1, x0, x1], dtype=np.int64),
                    ]
                )
            ):
                continue
            lon = float((cache.lon_axis[x0] + cache.lon_axis[x1]) / 2.0)
            lat = float((cache.lat_axis[y0] + cache.lat_axis[y1]) / 2.0)
            if not local_polygon_lonlat.covers(Point(lon, lat)):
                continue
            cell_x_array, cell_y_array = projection.project(lon, lat)
            distance_m = float(np.hypot(float(cell_x_array) - anchor_x, float(cell_y_array) - anchor_y))
            if distance_m > maximum_distance_m + distance_tolerance:
                continue
            candidates.append(
                _NWWMetricLocationCandidate(
                    lon=lon,
                    lat=lat,
                    distance_m=distance_m,
                    cell_x0=int(x0),
                    cell_x1=int(x1),
                    cell_y0=int(y0),
                    cell_y1=int(y1),
                )
            )
    return tuple(sorted(candidates, key=lambda item: (item.distance_m, item.cell_y0, item.cell_x0)))


def _nww_exact_hour_sample_rows(
    month: _NWWRuntimeMonth,
    row_indices: np.ndarray,
    *,
    corner_y: np.ndarray,
    corner_x: np.ndarray,
    spatial_weights: np.ndarray,
    static_valid: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """向量化計算指定 UTC rows 的四角 support 與雙線性 Hs。

    這裡一次處理一批 exact-hour rows，避免對每個小時建立 ``WaveSample`` 物件。四角
    dynamic mask 必須全部為 True；Hs、fp、原始方向四角先逐一有限檢查，再使用與
    runtime 相同的雙線性 scalar 與 sin/cos 方向合成。方向向量不重正規化，因為 runtime
    的非退化判定只要求 ``abs(direction_x)+abs(direction_y) > 1e-15``。
    """

    rows = np.asarray(row_indices, dtype=np.int64)
    invalid_values = np.full(rows.size, np.nan, dtype=np.float64)
    invalid_flags = np.zeros(rows.size, dtype=bool)
    if rows.size == 0 or not static_valid:
        return invalid_values, invalid_flags

    # 三個 spatial advanced-index 維度以 (row, corner) 直接 broadcast；不可先寫
    # ``array[rows]``，否則會把整個 (rows, lat, lon) 月份切片 materialize，破壞 accepted
    # 大型 NPY 的低記憶體／memory-map 契約。這裡每個欄位最多只產生 (rows, 4) 四角小矩陣。
    row_grid = rows[:, None]
    corner_y_grid = corner_y[None, :]
    corner_x_grid = corner_x[None, :]
    hs_values = np.asarray(
        month.significant_wave_height[row_grid, corner_y_grid, corner_x_grid], dtype=np.float64
    )
    fp_values = np.asarray(month.peak_frequency[row_grid, corner_y_grid, corner_x_grid], dtype=np.float64)
    direction_values = np.asarray(
        month.peak_direction_raw_deg[row_grid, corner_y_grid, corner_x_grid], dtype=np.float64
    )
    dynamic_valid = np.all(
        np.asarray(month.valid_mask_wave[row_grid, corner_y_grid, corner_x_grid], dtype=bool),
        axis=1,
    )
    directions = np.deg2rad(direction_values)
    finite_corners = (
        np.all(np.isfinite(hs_values), axis=1)
        & np.all(np.isfinite(fp_values), axis=1)
        & np.all(np.isfinite(directions), axis=1)
    )
    hs = hs_values @ spatial_weights
    fp = fp_values @ spatial_weights
    direction_x = np.sin(directions) @ spatial_weights
    direction_y = np.cos(directions) @ spatial_weights
    physical = (
        np.isfinite(hs)
        & np.isfinite(fp)
        & np.isfinite(direction_x)
        & np.isfinite(direction_y)
        & (hs >= 0.0)
        & (fp > 0.0)
        & ((np.abs(direction_x) + np.abs(direction_y)) > 1e-15)
    )
    valid = dynamic_valid & finite_corners & physical
    return np.where(valid, hs, np.nan), valid


def _nww_exact_hour_samples(
    cache: _NWWRuntimeCache,
    *,
    lon: float,
    lat: float,
    requested_times: Sequence[int] | None = None,
) -> tuple[dict[int, float], dict[int, bool]]:
    """以 exact-hour 四角 gate 建立 Hs／availability mapping，供 arrival 與 receptor 共用。

    ``requested_times`` 為 ``None`` 時，函式掃描每個月份的完整逐時軸，供 arrival
    selector 建立 site series；指定時只用 ``searchsorted(..., side="left")`` 找 exact
    UTC，供一個 horizontal face 的 50 個 arrival 重複使用。跨月份重複 UTC 保留既有
    prefer-last precedence。找不到 exact UTC、座標域外、四角 mask／數值／物理條件任一
    失敗時，回傳 ``nan`` 與 ``False``，不以最近時間、最近格點或零值補齊。
    """

    support = _nww_runtime_spatial_support(cache, lon=lon, lat=lat)
    if requested_times is None:
        result: dict[int, float] = {}
        valid_result: dict[int, bool] = {}
        requested_values: tuple[int, ...] | None = None
    else:
        requested_values = tuple(int(value) for value in requested_times)
        result = {value: float("nan") for value in requested_values}
        valid_result = {value: False for value in requested_values}
    if support is None:
        if requested_values is None:
            for month in cache.months:
                for value in month.source.time_ns:
                    result[int(value)] = float("nan")
                    valid_result[int(value)] = False
        return result, valid_result

    corner_y, corner_x, spatial_weights, static_valid = support
    for month in cache.months:
        if requested_values is None:
            row_indices = np.arange(month.source.time_ns.size, dtype=np.int64)
            sampled_times = np.asarray(month.source.time_ns, dtype=np.int64)
            sampled_values, sampled_valid = _nww_exact_hour_sample_rows(
                month,
                row_indices,
                corner_y=corner_y,
                corner_x=corner_x,
                spatial_weights=spatial_weights,
                static_valid=static_valid,
            )
        else:
            requested_array = np.asarray(requested_values, dtype=np.int64)
            positions = np.searchsorted(month.source.time_ns, requested_array, side="left")
            exact = (positions < month.source.time_ns.size) & (
                month.source.time_ns[np.minimum(positions, month.source.time_ns.size - 1)] == requested_array
            )
            output_indices = np.flatnonzero(exact)
            row_indices = positions[output_indices].astype(np.int64, copy=False)
            sampled_times = requested_array[output_indices]
            sampled_values, sampled_valid = _nww_exact_hour_sample_rows(
                month,
                row_indices,
                corner_y=corner_y,
                corner_x=corner_x,
                spatial_weights=spatial_weights,
                static_valid=static_valid,
            )
        for time_ns, value, available in zip(sampled_times, sampled_values, sampled_valid, strict=True):
            time_key = int(time_ns)
            result[time_key] = float(value) if bool(available) else float("nan")
            valid_result[time_key] = bool(available)
    return result, valid_result


def _nww_series_for_location(
    product: _ProductData,
    *,
    lon: float,
    lat: float,
    cache: _NWWRuntimeCache | None = None,
) -> tuple[dict[int, float], dict[int, bool]]:
    """依 runtime-equivalent exact-hour 四角雙線性 gate 建立站點 NWW Hs 序列。

    NWW 的靜態 mask 與逐時 ``valid_mask_wave`` 都必須在同一組四角全部有效；四角的
    Hs、peak frequency（Hz）與原始波向（degree）必須有限，並以 runtime 同樣的權重
    計算 Hs、fp 與 sin/cos 方向向量。只有 Hs 非負、fp 正值且方向向量非退化時，回傳
    的 Hs 才標記為可用。函式不讀最近有效點、不補零、不重正規化，也不對時間做線性
    內插；二維 NWW 座標網格直接拒絕，因為正式 runtime 只支援一維嚴格遞增軸。``cache``
    可由同一次 build 的 receptor gate 重用 memory-map 與座標檢視，但不改變結果。
    """

    runtime_cache = cache if cache is not None else _load_nww_runtime_cache(product)
    return _nww_exact_hour_samples(runtime_cache, lon=lon, lat=lat)


def _surface_series_for_location(
    product: _ProductData,
    *,
    lon: float,
    lat: float,
) -> tuple[dict[int, float], dict[int, float]]:
    """由 OCM surface cache 建立站點潮位與水平流速序列。

    OCM schema 3 surface cache 提供 arrival selector 所需的靜態分析格網與逐時
    ``eta_m``、``u_surface_mps``、``v_surface_mps``；規則格網採四角 static／dynamic
    valid 且有限的保守支撐，曲線格網則採最近格點。回傳值是最近格點的 surface
    elevation 與水平速度模長；native 不在這裡掃描全域節點或垂向 hvel，只在 geometry、
    wetdry、receptor 與 dynamic pair 的必要切片中讀取。``surface_z`` 與 ``qc_flags``
    雖不直接進 scalar，仍逐月驗證 shape，確保 surface cache 的完整資料契約未被錯配。
    取值時先依列優先順序攤平（C-order flatten）契約換算必要格點的多維座標，再對
    少量四角與最近格點做 ``float64`` 轉換；這保留列優先順序結果，也避免將每個逐時
    整張 float32 網格複製成 float64。索引只使用多維座標 tuple，因此對 C-order、欄
    優先順序（Fortran-order）及非連續陣列檢視（non-contiguous ndarray view）都不依賴
    ``ravel`` 是否能建立大型連續副本。
    """

    spatial_shape, spatial_indices, flat_index, static_valid = _grid_spatial_support(
        grid_dir=product.grid_dir,
        product_label="OCM surface",
        lon=lon,
        lat=lat,
    )
    # ``_grid_spatial_support`` 的整數索引是依列優先順序攤平（C-order）契約產生；
    # 先轉成多維座標後，後續每個逐時網格只會擷取四角及最近格點。這個轉換放在月份
    # 迴圈外，既保持原本支撐點的順序，也不要求來源 NPY 的記憶體布局必須是連續排列。
    support_coordinates = np.unravel_index(
        np.asarray(spatial_indices, dtype=np.intp), spatial_shape, order="C"
    )
    nearest_coordinates = np.unravel_index(np.asarray(flat_index, dtype=np.intp), spatial_shape, order="C")
    elevation: dict[int, float] = {}
    speed: dict[int, float] = {}
    for month in product.months:
        u_surface = _load_npy(month.directory / "u_surface_mps.npy")
        v_surface = _load_npy(month.directory / "v_surface_mps.npy")
        surface_z = _load_npy(month.directory / "surface_z.npy")
        eta = _load_npy(month.directory / "eta_m.npy")
        valid_surface = _load_npy(month.directory / "valid_mask_surface.npy")
        qc_flags = _load_npy(month.directory / "qc_flags.npy")
        arrays = (u_surface, v_surface, surface_z, eta, valid_surface, qc_flags)
        if any(
            array.shape[0] != month.time_ns.size or tuple(array.shape[1:]) != spatial_shape
            for array in arrays
        ):
            raise InputDerivationError(f"OCM surface {month.label} 陣列時間／空間 shape 不符")
        for local, time_ns in enumerate(month.time_ns):
            # 先以多維座標擷取（gather）必要格點，再轉成計算所需資料型別。這種以座標
            # 陣列取值的索引結果最多只有四個值，故不會把整個逐時網格轉成 float64；
            # 直接座標索引也能正確處理欄優先順序或非連續的逐時網格檢視。
            u_support = np.asarray(u_surface[local][support_coordinates], dtype=np.float64)
            v_support = np.asarray(v_surface[local][support_coordinates], dtype=np.float64)
            surface_z_support = np.asarray(surface_z[local][support_coordinates], dtype=np.float64)
            eta_support = np.asarray(eta[local][support_coordinates], dtype=np.float64)
            valid_support = np.asarray(valid_surface[local][support_coordinates], dtype=bool)
            available = static_valid and bool(np.all(valid_support))
            if available:
                support_values = np.concatenate(
                    (
                        u_support,
                        v_support,
                        surface_z_support,
                        eta_support,
                    )
                )
                available = bool(np.all(np.isfinite(support_values)))
            if available:
                # 最近格點單一值必須沿用既有 flat_index，而不是把四角值做雙線性
                # 平均；只把這一個 scalar 轉成 float64，保持 arrival selector 的既有
                # 最近點語意與速度模長數值。
                u_nearest = np.asarray(u_surface[local][nearest_coordinates], dtype=np.float64)
                v_nearest = np.asarray(v_surface[local][nearest_coordinates], dtype=np.float64)
                eta_nearest = np.asarray(eta[local][nearest_coordinates], dtype=np.float64)
                elevation[int(time_ns)] = float(eta_nearest)
                speed[int(time_ns)] = float(np.hypot(u_nearest, v_nearest))
            else:
                elevation[int(time_ns)] = float("nan")
                speed[int(time_ns)] = float("nan")
    return elevation, speed


def _load_ocm_pair_cache(
    product: _ProductData,
) -> _OCMPairCache:
    """一次載入 OCM pair 所需月份陣列，並建立 prefer-last UTC lookup。

    回傳物件明確綁定 ``flow_domain_id``、UTC 到 ``(month, local index)`` 的
    prefer-last 索引、靜態正值向下水深，以及每月 zcor/elev/wetdry memory-map。receptor
    的水平重選需要對少量候選 face 檢查全部 arrival；dynamic manifest 也需要同一批
    切片。兩條路徑共用同一個 cache 建立契約，避免每次重選或每個 pair 重複開啟同一個
    月份檔案，同時保留月份重複 UTC 的既有 prefer-last precedence。這裡只驗證時間第一
    軸；face/node 的局部維度仍在實際取樣時按 mesh 與 face 檢查，避免為了 cache 掃描
    完整 zcor。
    """

    by_time: dict[int, tuple[_MonthData, int]] = {}
    month_arrays: dict[str, _OCMMonthPairArrays] = {}
    for month in product.months:
        zcor = _load_npy(month.directory / "zcor.npy")
        elev = _load_npy(month.directory / "elev.npy")
        wetdry = _load_npy(month.directory / "wetdry_elem.npy")
        if (
            zcor.ndim < 2
            or elev.ndim < 1
            or wetdry.ndim < 1
            or zcor.shape[0] != month.time_ns.size
            or elev.shape[0] != month.time_ns.size
            or wetdry.shape[0] != month.time_ns.size
        ):
            raise InputDerivationError(f"OCM {month.label} dynamic array time shape 不符")
        month_arrays[month.label] = _OCMMonthPairArrays(zcor=zcor, elev=elev, wetdry=wetdry)
        for local, value in enumerate(month.time_ns):
            # 月份依既有排序寫入；後來的重複 UTC 覆蓋前者，與 canonical source precedence
            # 一致，不能以 dict 建立順序以外的鄰近時間替代缺少的 exact UTC。
            by_time[int(value)] = (month, local)
    depth = _load_npy(product.grid_dir / "source_depth_m.npy")
    if depth.ndim != 1:
        raise InputDerivationError("OCM source_depth_m 必須是一維 node 陣列")
    return _OCMPairCache(
        flow_domain_id=product.flow_domain_id,
        utc_prefer_last_index=by_time,
        depth=depth,
        monthly_arrays=month_arrays,
    )


def _assert_ocm_pair_cache(product: _ProductData, cache: _OCMPairCache) -> _OCMPairCache:
    """核對 OCM pair cache 與產品的 flow-domain binding，錯綁時 fail closed。

    cache 雖然只保存唯讀 memory-map，但其內容仍代表特定 OCM native flow domain。若
    caller 以 analysis region 或字典位置誤把另一域的 cache 套用到目前產品，所有 face
    index 與垂向資料都可能在數值上看似可讀卻失去物理來源。這裡在任何取樣前比對
    ``flow_domain_id``，維持獨立 helper 的明確錯誤，而不是靜默退回重新載入或補值。
    """

    if not isinstance(cache, _OCMPairCache):
        raise InputDerivationError("OCM pair cache 必須是具 flow-domain binding 的 _OCMPairCache")
    if cache.flow_domain_id != product.flow_domain_id:
        raise InputDerivationError(
            "OCM pair cache flow_domain_id 錯綁："
            f"cache={cache.flow_domain_id}，product={product.flow_domain_id}"
        )
    return cache


def _face_times(
    product: _ProductData,
    arrivals: Sequence[ArrivalTime],
    face_count: int,
    *,
    pair_cache: _OCMPairCache | None = None,
    by_time: Mapping[int, tuple[_MonthData, int]] | None = None,
    month_arrays: Mapping[str, _OCMMonthPairArrays] | None = None,
) -> np.ndarray:
    """依 arrival UTC 取出 OCM wetdry face matrix，維持 0=wet 的原值語意。

    若 caller 已建立帶 flow-domain binding 的 OCM pair cache，這裡只切取其 memory-map，
    不再重複開啟月份檔案；沒有提供 cache 時則保留原本的獨立使用行為，方便純測試或
    其他小型 caller 使用。``by_time``／``month_arrays`` 參數只保留既有私有 helper
    的相容路徑；新的跨階段 caller 應傳入完整 ``pair_cache``，避免拆散 binding。
    """

    rows: list[np.ndarray] = []
    if pair_cache is not None:
        if by_time is not None or month_arrays is not None:
            raise InputDerivationError("OCM pair cache 不得與拆散的 by_time/month_arrays 同時提供")
        checked_cache = _assert_ocm_pair_cache(product, pair_cache)
        by_time = checked_cache.utc_prefer_last_index
        month_arrays = checked_cache.monthly_arrays
    elif by_time is None or month_arrays is None:
        checked_cache = _load_ocm_pair_cache(product)
        by_time = checked_cache.utc_prefer_last_index
        month_arrays = checked_cache.monthly_arrays
    for arrival in arrivals:
        source = by_time.get(int(arrival.time_utc_ns))
        if source is None:
            raise InputDerivationError(f"arrival UTC 不在 OCM source time：{arrival.time_utc_ns}")
        month, local = source
        month_values = month_arrays.get(month.label)
        if month_values is None:
            raise InputDerivationError(f"OCM cache 缺少月份 wetdry：{month.label}")
        wetdry = month_values.wetdry
        frame = np.asarray(wetdry[local], dtype=np.float64)
        if frame.shape != (face_count,):
            raise InputDerivationError("OCM wetdry_elem face 維度與網格不符")
        rows.append(frame)
    return np.asarray(rows, dtype=np.float64)


def _backward_window_mask(
    time_ns: np.ndarray,
    *,
    expected_axis: np.ndarray,
    available_axis: np.ndarray,
    max_backtrack_days: float,
) -> np.ndarray:
    """判斷每個候選 UTC 的 inclusive backward horizon 是否完全沒有 gap。

    計算先以規則 hourly expected axis 建立 available mask，再用 prefix sum 判斷
    ``[arrival-horizon, arrival]`` 的每一個 expected time 都存在。這不會把最近值、零值
    或跨缺口外插當成支援；若 arrival 前方不是整點邊界，直接拒絕該 candidate。
    """

    if max_backtrack_days <= 0:
        raise ValueError("max_backtrack_days 必須為正")
    expected = np.asarray(expected_axis, dtype=np.int64)
    available = np.isin(expected, np.asarray(available_axis, dtype=np.int64))
    missing_prefix = np.concatenate(([0], np.cumsum(~available, dtype=np.int64)))
    horizon_steps = int(round(max_backtrack_days * 24.0))
    if horizon_steps < 1 or not math.isclose(
        horizon_steps / 24.0, max_backtrack_days, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("目前 gap-safe baseline 只接受整日 horizon")
    positions = {int(value): index for index, value in enumerate(expected)}
    result = np.zeros(np.asarray(time_ns).shape, dtype=bool)
    for output_index, value in enumerate(np.asarray(time_ns, dtype=np.int64)):
        position = positions.get(int(value))
        if position is None or position - horizon_steps < 0:
            continue
        start = position - horizon_steps
        stop = position + 1
        result[output_index] = missing_prefix[stop] == missing_prefix[start]
    return result


def _select_arrivals_for_site(
    *,
    site_id: str,
    product: _ProductData,
    nww_series: Mapping[int, float],
    nww_valid_for_site: Mapping[int, bool] | None = None,
    ocm_elevation: Mapping[int, float],
    ocm_speed: Mapping[int, float],
    max_backtrack_days: float,
    design_version: str,
    expected_axis: np.ndarray,
    strict: bool,
    observation_years: Sequence[int] | None = None,
    replicates: int | None = None,
    selection_policy: str | None = None,
) -> list[ArrivalTime]:
    """依版本化 observation policy 產生單站 48+2 arrival。

    ``product.canonical.time_utc_ns`` 仍是 forcing 的完整時間軸；新版 policy 只把
    ``observation_years`` 傳給 selector，故 2024 能繼續參與 180 日
    ``backward_window_available``，卻不會被選成 2025 observation anchor。非 strict
    synthetic fixture 也沿用同一 policy 產生 metadata；它只服務測試，不代表正式資料
    通過 accepted-product 或 gap-safe gate。
    """

    times = np.asarray(product.canonical.time_utc_ns, dtype=np.int64)
    values = np.asarray(
        [float(ocm_elevation.get(int(value), float("nan"))) for value in times],
        dtype=np.float64,
    )
    waves = np.asarray([float(nww_series.get(int(value), float("nan"))) for value in times], dtype=np.float64)
    speeds = np.asarray([float(ocm_speed.get(int(value), float("nan"))) for value in times], dtype=np.float64)
    if nww_valid_for_site is None:
        nww_available = np.asarray(
            [np.isfinite(nww_series.get(int(value), float("nan"))) for value in times], dtype=bool
        )
    else:
        nww_available = np.asarray(
            [bool(nww_valid_for_site.get(int(value), False)) for value in times], dtype=bool
        )
    backward = _backward_window_mask(
        times,
        expected_axis=expected_axis,
        available_axis=product.canonical.time_utc_ns,
        max_backtrack_days=max_backtrack_days,
    )
    valid = np.isfinite(values) & np.isfinite(waves) & np.isfinite(speeds) & nww_available
    valid &= np.isin(times, expected_axis)
    try:
        return select_arrival_times(
            study_site_id=site_id,
            time_utc_ns=times,
            elevation_m=values,
            significant_wave_height_m=waves,
            current_speed_mps=speeds,
            valid_forcing=valid,
            backward_window_available=backward,
            design_version=design_version,
            observation_years=observation_years,
            replicates=replicates,
            selection_policy=selection_policy,
        )
    except ValueError:
        if strict:
            raise
        # 低維 synthetic fixture 常用 constant elevation，原 selector 的 spring/neap
        # 中位數可能讓其中一類沒有候選。這個 fallback 只用於開發產物，仍依呼叫端
        # policy 建立固定數量與 metadata，並將 status 留在 generated；正式 release
        # validator 不接受以 fallback 取代 accepted forcing 的 selector 證據。
        candidate_indices = np.flatnonzero(valid)
        if candidate_indices.size < 50:
            raise
        datetimes = {
            int(index): datetime.fromtimestamp(int(times[index]) / 1_000_000_000, tz=UTC)
            for index in candidate_indices
        }
        years = sorted({item.year for item in datetimes.values()})
        is_observation_policy = selection_policy == "observation_year_stratified_48_plus_2_v1"
        if is_observation_policy:
            if observation_years is None:
                raise
            selected_years = [int(value) for value in observation_years]
            if len(selected_years) != 1 or not set(selected_years).issubset(set(years)):
                raise
            if replicates not in (None, 2):
                raise
            years = selected_years
        elif len(years) != 2:
            raise
        season_by_month = {
            12: "DJF",
            1: "DJF",
            2: "DJF",
            3: "MAM",
            4: "MAM",
            5: "MAM",
            6: "JJA",
            7: "JJA",
            8: "JJA",
            9: "SON",
            10: "SON",
            11: "SON",
        }
        selected: set[int] = set()
        records: list[ArrivalTime] = []
        if is_observation_policy:
            selection_label = "observation_year_stratified_48_plus_2_v1"
            for year in years:
                for season in ("DJF", "MAM", "JJA", "SON"):
                    cell = [
                        int(i)
                        for i in candidate_indices
                        if datetimes[int(i)].year == year
                        and season_by_month[datetimes[int(i)].month] == season
                    ]
                    if len(cell) < 24:
                        raise
                    half = len(cell) // 2
                    for tide_class, class_cell in (
                        ("spring_proxy", cell[:half]),
                        ("neap_proxy", cell[half:]),
                    ):
                        if len(class_cell) < 12:
                            raise
                        phases = ("fastest_rising", "fastest_falling", "slack_proxy")
                        for phase_index, phase in enumerate(phases):
                            for replicate_rank in range(2):
                                offset = phase_index * 2 + replicate_rank
                                index = next(
                                    candidate
                                    for candidate in class_cell[offset:]
                                    if candidate not in selected
                                )
                                selected.add(index)
                                value = int(times[index])
                                records.append(
                                    ArrivalTime(
                                        arrival_time_id=stable_identifier(
                                            "arr",
                                            [
                                                site_id,
                                                str(value),
                                                tide_class,
                                                phase,
                                                str(replicate_rank),
                                                selection_label,
                                                design_version,
                                            ],
                                        ),
                                        study_site_id=site_id,
                                        time_utc_ns=value,
                                        year=year,
                                        season=season,
                                        tide_class=tide_class,
                                        phase_or_event=phase,
                                        metadata={
                                            "selection_method": "synthetic_constant_field_fallback",
                                            "selection_rank": len(records),
                                            "selection_policy": selection_label,
                                            "observation_year": year,
                                            "replicate_rank": replicate_rank,
                                            "replicate_rank_one_based": replicate_rank + 1,
                                            "stratum_id": f"{year}/{season}/{tide_class}/{phase}",
                                        },
                                    )
                                )
        else:
            for year in years:
                for season in ("DJF", "MAM", "JJA", "SON"):
                    cell = [
                        i
                        for i in candidate_indices
                        if (
                            datetimes[int(i)].year == year
                            and season_by_month[datetimes[int(i)].month] == season
                        )
                    ]
                    if len(cell) < 6:
                        raise
                    for tide_class, phase_index in (
                        ("spring_proxy", 0),
                        ("neap_proxy", 3),
                    ):
                        for phase, offset in (
                            ("fastest_rising", 0),
                            ("fastest_falling", 1),
                            ("slack_proxy", 2),
                        ):
                            index = cell[(phase_index + offset) % len(cell)]
                            while index in selected:
                                index = cell[(cell.index(index) + 1) % len(cell)]
                            selected.add(index)
                            value = int(times[index])
                            records.append(
                                ArrivalTime(
                                    arrival_time_id=stable_identifier(
                                        "arr", [site_id, str(value), tide_class, phase, design_version]
                                    ),
                                    study_site_id=site_id,
                                    time_utc_ns=value,
                                    year=year,
                                    season=season,
                                    tide_class=tide_class,
                                    phase_or_event=phase,
                                    metadata={
                                        "selection_method": "synthetic_constant_field_fallback",
                                        "selection_rank": len(records),
                                    },
                                )
                            )
        remaining = [
            int(i)
            for i in candidate_indices
            if int(i) not in selected
            and (not is_observation_policy or datetimes[int(i)].year in years)
        ]
        for event, score in (("high_wave_event", waves), ("strong_current_event", speeds)):
            if not remaining:
                raise
            index = max(remaining, key=lambda i: (float(score[i]), -int(times[i])))
            remaining.remove(index)
            value = int(times[index])
            dt = datetime.fromtimestamp(value / 1_000_000_000, tz=UTC)
            records.append(
                ArrivalTime(
                    arrival_time_id=stable_identifier("arr", [site_id, str(value), event, design_version]),
                    study_site_id=site_id,
                    time_utc_ns=value,
                    year=dt.year,
                    season={
                        12: "DJF",
                        1: "DJF",
                        2: "DJF",
                        3: "MAM",
                        4: "MAM",
                        5: "MAM",
                        6: "JJA",
                        7: "JJA",
                        8: "JJA",
                        9: "SON",
                        10: "SON",
                        11: "SON",
                    }[dt.month],
                    tide_class="event",
                    phase_or_event=event,
                    metadata={
                        "selection_method": "synthetic_constant_field_fallback",
                        "selection_rank": len(records),
                        **(
                            {
                                "selection_policy": "observation_year_stratified_48_plus_2_v1",
                                "observation_year": dt.year,
                                "stratum_id": f"event/{event}",
                            }
                            if is_observation_policy
                            else {}
                        ),
                    },
                )
            )
        if len(records) != 50 or len({item.time_utc_ns for item in records}) != 50:
            raise InputDerivationError("synthetic arrival fallback 未產生 50 個唯一 UTC") from None
        return records


def _nww_metric_location_binding(
    cache: _NWWRuntimeCache,
    *,
    projection: DomainProjection,
    anchor_lon: float,
    anchor_lat: float,
    location_kind: str,
    lon: float,
    lat: float,
    representative_grid_scale_m: float,
) -> NWWMetricLocationBinding:
    """把已通過 runtime spatial gate 的位置轉成可保存的 metric binding。"""

    cell_indices = _nww_runtime_cell_indices(cache, lon=lon, lat=lat)
    if cell_indices is None:
        raise InputDerivationError("NWW metric location 未通過 static 四角 runtime gate")
    if location_kind == "anchor" and (float(lon) != float(anchor_lon) or float(lat) != float(anchor_lat)):
        raise InputDerivationError("anchor metric location 的座標不得被靜默改寫")
    if location_kind == "anchor":
        anchor_distance_m = 0.0
    else:
        anchor_x, anchor_y = projection.project(float(anchor_lon), float(anchor_lat))
        location_x, location_y = projection.project(float(lon), float(lat))
        anchor_distance_m = float(
            np.hypot(float(location_x) - float(anchor_x), float(location_y) - float(anchor_y))
        )
    x0, x1, y0, y1 = cell_indices
    return NWWMetricLocationBinding(
        policy_id=NWW_METRIC_LOCATION_POLICY_ID,
        location_kind=location_kind,
        lon=float(lon),
        lat=float(lat),
        anchor_distance_m=anchor_distance_m,
        representative_grid_scale_m=float(representative_grid_scale_m),
        maximum_snap_distance_m=(NWW_METRIC_LOCATION_MAX_GRID_SCALES * float(representative_grid_scale_m)),
        cell_x0=x0,
        cell_x1=x1,
        cell_y0=y0,
        cell_y1=y1,
    )


def _merge_arrival_metric_location_binding(
    arrivals: Sequence[ArrivalTime],
    binding: NWWMetricLocationBinding,
) -> list[ArrivalTime]:
    """在保留既有潮汐／事件欄位後，為每筆 arrival 加入扁平 metric binding。"""

    binding_metadata = binding.to_metadata()
    result: list[ArrivalTime] = []
    for item in arrivals:
        # 只移除相同命名空間的舊 binding，避免在 paired clone 或重試時殘留另一個
        # metric location；elevation、derivative、Hs、current 等既有 selector metadata
        # 則原樣保留，除非 A 區 paired clone 另有明確的站點物理值重寫政策。
        metadata = {
            key: value for key, value in item.metadata.items() if not key.startswith("metric_location_")
        }
        metadata.update(binding_metadata)
        result.append(
            ArrivalTime(
                arrival_time_id=item.arrival_time_id,
                study_site_id=item.study_site_id,
                time_utc_ns=item.time_utc_ns,
                year=item.year,
                season=item.season,
                tide_class=item.tide_class,
                phase_or_event=item.phase_or_event,
                metadata=metadata,
            )
        )
    return result


def _validate_explicit_pilot_arrival_support(
    *,
    explicit: _ExplicitPilotArrival,
    product: _ProductData,
    surface_product: _ProductData,
    nww_product: _ProductData,
    nww_cache: _NWWRuntimeCache,
    context: _SiteArrivalSelectionContext,
    expected_axis: np.ndarray,
) -> None:
    """逐時驗證展示視窗的三套 exact UTC 軸與 NWW 四角空間支撐。

    ``product`` 是 OCM native，``surface_product`` 是只供 arrival scalar 的 OCM
    surface，``nww_product`` 是 NWW3 analysis，``nww_cache`` 則是同一產品的 runtime-
    equivalent 規則格網檢視。四者都必須逐筆支援同一組 25 個 UTC 節點；NWW 另以
    ``_nww_exact_hour_samples`` 重做 metric location 的四角 static／dynamic／數值／物理
    gate，而不是只信任 arrival endpoint 的 metadata。最後以 OCM native 的可得時間軸對
    ``[arrival-24 h, arrival]`` 做 inclusive、逐時、不可跨缺口檢查，因此不會用 00Z
    最近值、零值或時間內插補足 2024-01-01T01:00:00Z 起點。
    """

    time_ns = int(explicit.time_utc_ns)
    window_times = _explicit_pilot_window_times(explicit)
    expected_window = np.isin(window_times, np.asarray(expected_axis, dtype=np.int64))
    if not bool(np.all(expected_window)):
        missing = window_times[~expected_window]
        raise InputDerivationError(
            f"{PILOT_EXPLICIT_WINDOW_POLICY_ID} configured expected axis 缺少逐時節點："
            f"{_utc_string(int(missing[0]))}"
        )
    axis_labels = (
        ("OCM native", product.canonical.time_utc_ns),
        ("OCM surface", surface_product.canonical.time_utc_ns),
        ("NWW3 analysis", nww_product.canonical.time_utc_ns),
    )
    for label, axis in axis_labels:
        available_window = np.isin(window_times, np.asarray(axis, dtype=np.int64))
        if not bool(np.all(available_window)):
            missing = window_times[~available_window]
            raise InputDerivationError(
                f"{PILOT_EXPLICIT_WINDOW_POLICY_ID} {label} exact UTC 視窗缺少逐時節點："
                f"{_utc_string(int(missing[0]))}"
            )

    # OCM surface selector 的 mapping 已由 anchor 的 static／逐時 valid 四角 gate 建立；
    # 這裡仍逐時重查有限 scalar，避免只因 arrival endpoint 有值就放行中間缺時。
    for value in window_times:
        time_key = int(value)
        elevation = float(context.elevation.get(time_key, float("nan")))
        speed = float(context.speed.get(time_key, float("nan")))
        if not np.isfinite(elevation) or not np.isfinite(speed):
            raise InputDerivationError(
                f"{PILOT_EXPLICIT_WINDOW_POLICY_ID} OCM surface exact UTC 支援失敗：{_utc_string(time_key)}"
            )

    # 重新以選定 metric location 的四角資料取樣 25 個 UTC；這個旗標同時包含
    # static mask、四角 valid_mask_wave、四角有限值與 Hs/fp/方向物理條件。不能只看
    # context.nww_valid 的 arrival endpoint，否則中間一小時的 NWW spatial invalid 會漏過。
    nww_series, nww_valid = _nww_exact_hour_samples(
        nww_cache,
        lon=float(context.metric_binding.lon),
        lat=float(context.metric_binding.lat),
        requested_times=tuple(int(value) for value in window_times),
    )
    for value in window_times:
        time_key = int(value)
        nww_value = float(nww_series.get(time_key, float("nan")))
        if (
            not bool(nww_valid.get(time_key, False))
            or not bool(context.nww_valid.get(time_key, False))
            or not np.isfinite(nww_value)
            or not np.isfinite(float(context.nww_series.get(time_key, float("nan"))))
        ):
            raise InputDerivationError(
                f"{PILOT_EXPLICIT_WINDOW_POLICY_ID} NWW metric location 四角 exact UTC "
                f"支援失敗：{_utc_string(time_key)}"
            )

    # 端點呼叫會把整段 [arrival-24h, arrival] 交給 prefix-sum gap gate；上方逐筆軸檢查
    # 則提供能指出缺失產品與 UTC 的診斷。兩者都保留，避免 future expected-axis 變更時
    # 只有「逐筆存在」卻沒有「連續 backward horizon」的錯誤放行。
    supported = _backward_window_mask(
        np.asarray([time_ns], dtype=np.int64),
        expected_axis=expected_axis,
        available_axis=product.canonical.time_utc_ns,
        max_backtrack_days=explicit.max_backtrack_days,
    )
    if supported.shape != (1,) or not bool(supported[0]):
        start_utc = _utc_string(int(window_times[0]))
        raise InputDerivationError(
            f"{PILOT_EXPLICIT_WINDOW_POLICY_ID} inclusive gap-safe window 不完整："
            f"{start_utc} 至 {explicit.time_utc}"
        )


def _replace_with_explicit_pilot_arrival(
    *,
    site_id: str,
    arrivals: Sequence[ArrivalTime],
    explicit: _ExplicitPilotArrival,
    product: _ProductData,
    surface_product: _ProductData,
    nww_product: _ProductData,
    nww_cache: _NWWRuntimeCache,
    context: _SiteArrivalSelectionContext,
    expected_axis: np.ndarray,
    design_version: str,
) -> list[ArrivalTime]:
    """以固定 policy 替換一筆站點 baseline arrival，保留 50 筆站點 coverage。

    替換規則是：若 baseline 已經選到同一 UTC，就替換該 UTC；否則在其餘 49 筆候選
    中依 ``(time_utc_ns, arrival_time_id)`` 取排序最後一筆。這個 deterministic rule
    不依軌跡結果或亂數，並把原始／替換 identity、25 個 inclusive hourly nodes、OCM／
    NWW 支撐與非潮汐標籤寫進新 ArrivalTime metadata。新標籤刻意不是 spring、neap 或
    event，故正式 48+2 loader 會拒絕它；pilot loader 則可在保持 250 arrivals／5,000
    dynamic pairs 的前提下使用。
    """

    if site_id != explicit.study_site_id:
        raise InputDerivationError("explicit pilot arrival 只能套用 registry 登錄的相同站點")
    _validate_explicit_pilot_arrival_support(
        explicit=explicit,
        product=product,
        surface_product=surface_product,
        nww_product=nww_product,
        nww_cache=nww_cache,
        context=context,
        expected_axis=expected_axis,
    )
    by_time = [item for item in arrivals if int(item.time_utc_ns) == explicit.time_utc_ns]
    if len(by_time) > 1:
        raise InputDerivationError("hsinchu baseline arrival 的 UTC 已重複，不能 deterministic 替換")
    if by_time:
        original = by_time[0]
    else:
        ordered = sorted(arrivals, key=lambda item: (int(item.time_utc_ns), item.arrival_time_id))
        if not ordered:
            raise InputDerivationError("hsinchu baseline arrival 不可為空")
        original = ordered[-1]

    replacement_id = stable_identifier(
        "arr",
        [
            explicit.study_site_id,
            explicit.time_utc,
            PILOT_EXPLICIT_WINDOW_POLICY_ID,
            design_version,
        ],
    )
    if any(item.arrival_time_id == replacement_id for item in arrivals if item is not original):
        raise InputDerivationError("explicit pilot replacement arrival_time_id 與既有 identity 衝突")
    window_times = _explicit_pilot_window_times(explicit)
    start_ns = int(window_times[0])
    parsed = datetime.fromtimestamp(explicit.time_utc_ns / 1_000_000_000, tz=UTC)
    metadata: dict[str, float | int | str] = {
        "selection_method": PILOT_EXPLICIT_WINDOW_POLICY_ID,
        "pilot_selection_scope": explicit.selection_scope,
        "explicit_pilot_window": f"{_utc_string(start_ns)}/{explicit.time_utc}/inclusive_1h",
        "explicit_pilot_window_start_utc": _utc_string(start_ns),
        "explicit_pilot_window_end_utc": explicit.time_utc,
        "explicit_pilot_window_expected_step_count": 25,
        "explicit_pilot_window_time_support": "ocm_native_surface_nww_exact_utc_and_gap_safe_v1",
        "pilot_replaced_arrival_time_id": original.arrival_time_id,
        "pilot_replaced_arrival_time_utc": _utc_string(original.time_utc_ns),
        "pilot_replacement_arrival_time_id": replacement_id,
        "pilot_replacement_arrival_time_utc": explicit.time_utc,
        "pilot_replacement_policy_id": PILOT_EXPLICIT_WINDOW_POLICY_ID,
        "pilot_replacement_max_backtrack_days": explicit.max_backtrack_days,
        "tide_phase_semantics": "non_tidal_explicit_pilot_window",
        "elevation_m": float(context.elevation[explicit.time_utc_ns]),
        "significant_wave_height_m": float(context.nww_series[explicit.time_utc_ns]),
        "current_speed_mps": float(context.speed[explicit.time_utc_ns]),
    }
    metadata.update(context.metric_binding.to_metadata())
    replacement = ArrivalTime(
        arrival_time_id=replacement_id,
        study_site_id=explicit.study_site_id,
        time_utc_ns=explicit.time_utc_ns,
        year=parsed.year,
        season="DJF",
        tide_class=PILOT_EXPLICIT_TIDE_CLASS,
        phase_or_event=PILOT_EXPLICIT_PHASE_OR_EVENT,
        metadata=metadata,
    )
    result = [item for item in arrivals if item.arrival_time_id != original.arrival_time_id]
    if any(item.time_utc_ns == explicit.time_utc_ns for item in result):
        raise InputDerivationError("explicit pilot replacement 會造成同站 duplicate UTC")
    result.append(replacement)
    if len(result) != len(arrivals) or len({item.time_utc_ns for item in result}) != len(result):
        raise InputDerivationError("explicit pilot replacement 未保留既有 arrival count／唯一 UTC")
    return sorted(result, key=lambda item: (int(item.time_utc_ns), item.arrival_time_id))


def _pilot_selection_summary(arrivals: Sequence[ArrivalTime]) -> dict[str, Any] | None:
    """由 arrival metadata 建立 pilot replacement 的小型 provenance snapshot。

    summary 不保存 source path 或大型資料，只保存可由 arrival manifest 重算的 policy、站點
    集合、原／替換 identity 與 24 小時 horizon。A 區 exact pair 會保留兩筆 arrival ID，
    讓 gap-safe component 對兩站各自套用一日 override；B／C／D 單站則保留既有
    ``arrival_time_id`` 欄位以維持舊 artifact／下游讀取相容性。沒有明示 pilot arrival 時
    回傳 ``None``，使一般 48+2 artifact 的 root/provenance 保持既有語意。
    """

    explicit = [
        item
        for item in arrivals
        if item.metadata.get("pilot_replacement_policy_id") in _PILOT_EXPLICIT_POLICY_IDS
    ]
    if not explicit:
        return None
    policies = {item.metadata.get("pilot_replacement_policy_id") for item in explicit}
    if len(policies) != 1:
        raise InputDerivationError("pilot explicit replacement 不可混用新舊 policy")
    policy_id = str(next(iter(policies)))
    site_ids = tuple(sorted(item.study_site_id for item in explicit))
    if len(site_ids) != len(set(site_ids)):
        raise InputDerivationError("pilot explicit replacement 不可重複同一站點")
    if policy_id == PILOT_EXPLICIT_LEGACY_WINDOW_POLICY_ID:
        if site_ids != (PILOT_EXPLICIT_LEGACY_SITE_ID,):
            raise InputDerivationError("legacy pilot artifact 只允許 hsinchu 單站")
        registry_entry: Mapping[str, Any] = {
            "analysis_region_id": "B",
            "selection_scope": "hsinchu_only",
        }
    else:
        registry_entry = PILOT_EXPLICIT_REGISTRY.get(site_ids, {})
        if not registry_entry:
            raise InputDerivationError("pilot explicit replacement 站點組合不在 registry")

    records: list[dict[str, Any]] = []
    for item in sorted(explicit, key=lambda value: (value.study_site_id, int(value.time_utc_ns))):
        metadata = item.metadata
        required = (
            "pilot_replaced_arrival_time_id",
            "pilot_replacement_arrival_time_id",
            "explicit_pilot_window_start_utc",
            "explicit_pilot_window_end_utc",
        )
        if any(not isinstance(metadata.get(key), str) or not str(metadata[key]).strip() for key in required):
            raise InputDerivationError("pilot explicit arrival metadata 缺少原／替換 identity 或視窗")
        if _utc_string(item.time_utc_ns) != PILOT_EXPLICIT_ARRIVAL_UTC:
            raise InputDerivationError("pilot explicit arrival UTC 不符合 registry 固定時次")
        if metadata.get("explicit_pilot_window_expected_step_count") != _PILOT_EXPLICIT_EXPECTED_STEP_COUNT:
            raise InputDerivationError("pilot explicit arrival 必須保存 25 個 inclusive 節點")
        if metadata.get("pilot_selection_scope") != registry_entry["selection_scope"]:
            raise InputDerivationError("pilot explicit arrival selection scope 與 registry 不一致")
        records.append(
            {
                "study_site_id": item.study_site_id,
                "arrival_time_id": item.arrival_time_id,
                "arrival_time_utc": _utc_string(item.time_utc_ns),
                "replaced_arrival_time_id": metadata["pilot_replaced_arrival_time_id"],
                "replaced_arrival_time_utc": metadata.get("pilot_replaced_arrival_time_utc"),
                "replacement_arrival_time_id": metadata["pilot_replacement_arrival_time_id"],
                "replacement_arrival_time_utc": metadata.get("pilot_replacement_arrival_time_utc"),
                "window_start_utc": metadata["explicit_pilot_window_start_utc"],
                "window_end_utc": metadata["explicit_pilot_window_end_utc"],
            }
        )
    arrival_times = {record["arrival_time_utc"] for record in records}
    if arrival_times != {PILOT_EXPLICIT_ARRIVAL_UTC}:
        raise InputDerivationError("pilot explicit replacement 的站點 UTC 必須完全相同")
    result: dict[str, Any] = {
        "policy_id": policy_id,
        "site_ids": list(site_ids),
        "arrival_time_ids": [record["arrival_time_id"] for record in records],
        "arrival_times": records,
        "window_start_utc": records[0]["window_start_utc"],
        "window_end_utc": records[0]["window_end_utc"],
        "max_backtrack_days": float(PILOT_EXPLICIT_MAX_BACKTRACK_DAYS),
        "expected_step_count": _PILOT_EXPLICIT_EXPECTED_STEP_COUNT,
        "selection_scope": registry_entry["selection_scope"],
    }
    if len(records) == 1:
        # 舊下游只讀這些單筆欄位；保留它們不影響新的 A pair schema。
        result.update(
            {
                "study_site_id": records[0]["study_site_id"],
                "arrival_time_id": records[0]["arrival_time_id"],
                "arrival_time_utc": records[0]["arrival_time_utc"],
                "replaced_arrival_time_id": records[0]["replaced_arrival_time_id"],
                "replaced_arrival_time_utc": records[0]["replaced_arrival_time_utc"],
                "replacement_arrival_time_id": records[0]["replacement_arrival_time_id"],
                "replacement_arrival_time_utc": records[0]["replacement_arrival_time_utc"],
            }
        )
    elif site_ids != ("gongliao", "guishan") or len(records) != 2:
        raise InputDerivationError("pilot explicit replacement 多站只允許 A 區 exact pair")
    return result


def _site_arrival_support_by_time(
    *,
    product: _ProductData,
    ocm_elevation: Mapping[int, float],
    ocm_speed: Mapping[int, float],
    nww_series: Mapping[int, float],
    nww_valid: Mapping[int, bool],
    expected_axis: np.ndarray,
    max_backtrack_days: float,
) -> dict[int, bool]:
    """建立單站每 UTC 的 OCM／NWW／gap-safe 共同有效旗標。

    這個 mapping 不重新選 arrival，只用來在 A 區 clone 前驗證貢寮選出的每一個 shared
    UTC 也能由龜山島自己的 OCM surface、NWW metric series 與 inclusive backward window
    支援。任何欄位缺失、非有限、未在 exact expected axis 或跨 OCM gap 都會標成 False，
    不以最近值、零值或另一站 metadata 代替。
    """

    times = np.asarray(product.canonical.time_utc_ns, dtype=np.int64)
    backward = _backward_window_mask(
        times,
        expected_axis=expected_axis,
        available_axis=product.canonical.time_utc_ns,
        max_backtrack_days=max_backtrack_days,
    )
    expected = np.isin(times, np.asarray(expected_axis, dtype=np.int64))
    result: dict[int, bool] = {}
    for index, value in enumerate(times):
        time_key = int(value)
        result[time_key] = bool(
            expected[index]
            and backward[index]
            and np.isfinite(float(ocm_elevation.get(time_key, float("nan"))))
            and np.isfinite(float(ocm_speed.get(time_key, float("nan"))))
            and np.isfinite(float(nww_series.get(time_key, float("nan"))))
            and bool(nww_valid.get(time_key, False))
        )
    return result


def _select_arrivals_with_nww_metric_location(
    *,
    site_id: str,
    anchor_lon: float,
    anchor_lat: float,
    local_polygon_lonlat: Polygon,
    projection: DomainProjection,
    product: _ProductData,
    nww_product: _ProductData,
    nww_cache: _NWWRuntimeCache,
    ocm_elevation: Mapping[int, float],
    ocm_speed: Mapping[int, float],
    max_backtrack_days: float,
    design_version: str,
    expected_axis: np.ndarray,
    strict: bool,
    observation_years: Sequence[int] | None = None,
    replicates: int | None = None,
    selection_policy: str | None = None,
) -> tuple[list[ArrivalTime], _SiteArrivalSelectionContext]:
    """以 anchor-first policy 選出 arrival 並保存 NWW metric location binding。

    strict build 先在站點 anchor 執行與 runtime 等價的完整 48+2 selector；只有 selector
    失敗才由實際 NWW lon/lat 軸組成 cell-center 候選。候選先通過 local polygon、四角
    static mask 與兩格局地尺度距離，再依公尺距離、``y0``、``x0`` 穩定排序，逐個執行
    同一個 strict selector。所有失敗都在本函式內 fail closed；只有 ``strict=False`` 的
    內部小型 fixture 才保留既有 anchor synthetic fallback，且不會被 CLI 使用。

    ``observation_years``、``replicates`` 與 ``selection_policy`` 會原樣傳給單站
    selector；metric location 的 anchor/cell-center 重試不得把新版 2025-only policy
    降級成舊兩年 policy。
    """

    representative_scale_m = _nww_representative_grid_scale_m(
        nww_cache,
        projection=projection,
        anchor_lon=anchor_lon,
        anchor_lat=anchor_lat,
    )

    def attempt(
        *,
        location_kind: str,
        lon: float,
        lat: float,
        selector_strict: bool,
    ) -> tuple[list[ArrivalTime], _SiteArrivalSelectionContext]:
        """在單一 metric location 執行完整 selector，成功後建立 binding/context。"""

        nww_series, nww_valid = _nww_series_for_location(
            nww_product,
            lon=lon,
            lat=lat,
            cache=nww_cache,
        )
        selected = _select_arrivals_for_site(
            site_id=site_id,
            product=product,
            nww_series=nww_series,
            nww_valid_for_site=nww_valid,
            ocm_elevation=ocm_elevation,
            ocm_speed=ocm_speed,
            max_backtrack_days=max_backtrack_days,
            design_version=design_version,
            expected_axis=expected_axis,
            strict=selector_strict,
            observation_years=observation_years,
            replicates=replicates,
            selection_policy=selection_policy,
        )
        binding = _nww_metric_location_binding(
            nww_cache,
            projection=projection,
            anchor_lon=anchor_lon,
            anchor_lat=anchor_lat,
            location_kind=location_kind,
            lon=lon,
            lat=lat,
            representative_grid_scale_m=representative_scale_m,
        )
        context = _SiteArrivalSelectionContext(
            elevation=ocm_elevation,
            speed=ocm_speed,
            nww_series=nww_series,
            nww_valid=nww_valid,
            metric_binding=binding,
        )
        return _merge_arrival_metric_location_binding(selected, binding), context

    try:
        # 非 strict 的 internal fixture 仍可使用原有 anchor fallback；正式／CLI 路徑則
        # 由 caller 傳入 True，這裡的 anchor 嘗試與後續 cell 嘗試都不會進 fallback。
        return attempt(
            location_kind="anchor",
            lon=float(anchor_lon),
            lat=float(anchor_lat),
            selector_strict=bool(strict),
        )
    except ValueError as anchor_error:
        if not strict:
            raise
        anchor_error_text = str(anchor_error)

    candidates = _nww_metric_location_candidates(
        nww_cache,
        projection=projection,
        anchor_lon=anchor_lon,
        anchor_lat=anchor_lat,
        local_polygon_lonlat=local_polygon_lonlat,
        representative_grid_scale_m=representative_scale_m,
    )
    for candidate in candidates:
        try:
            # 這是 production strict selector；任何 candidate 只要不能完整產生 48+2，
            # 就跳到下一個 deterministic cell，絕不呼叫 strict=False synthetic fallback。
            return attempt(
                location_kind="cell_center",
                lon=candidate.lon,
                lat=candidate.lat,
                selector_strict=True,
            )
        except ValueError:
            continue
    raise InputDerivationError(
        f"{site_id} NWW metric location 無法完成 strict 48+2 selector；"
        f"anchor_error={anchor_error_text};候選數={len(candidates)}"
    )


def _clone_paired_a_arrivals(
    *,
    source_arrivals: Sequence[ArrivalTime],
    guishan_context: _SiteArrivalSelectionContext,
    guishan_product: _ProductData,
    expected_axis: np.ndarray,
    max_backtrack_days: float,
    design_version: str,
    strict: bool,
) -> list[ArrivalTime]:
    """以 guishan 自己的序列重建 A 區 paired UTC arrival metadata。

    貢寮只提供 shared UTC、潮汐類別與潮內 phase label；龜山島的 elevation、Hs、current
    與 NWW metric location binding 必須從自己的序列取得。strict pipeline 會先逐一驗證
    每個 shared UTC 的 OCM finite、NWW exact-hour validity 與 inclusive gap-safe window；
    任一時次不支援就直接 fail closed。非 strict internal fixture 保留既有小型資料的相容
    行為，但仍不把貢寮的物理 metadata 複製給龜山島。
    """

    if strict:
        guishan_supported = _site_arrival_support_by_time(
            product=guishan_product,
            ocm_elevation=guishan_context.elevation,
            ocm_speed=guishan_context.speed,
            nww_series=guishan_context.nww_series,
            nww_valid=guishan_context.nww_valid,
            expected_axis=expected_axis,
            max_backtrack_days=max_backtrack_days,
        )
        unsupported_shared = [
            int(item.time_utc_ns)
            for item in source_arrivals
            if not guishan_supported.get(int(item.time_utc_ns), False)
        ]
        if unsupported_shared:
            raise InputDerivationError(
                f"A 區 paired UTC 的 guishan OCM/NWW/gap-safe 支援失敗：utc={unsupported_shared[0]}"
            )

    cloned: list[ArrivalTime] = []
    for item in source_arrivals:
        time_key = int(item.time_utc_ns)
        try:
            guishan_elevation = float(guishan_context.elevation[time_key])
            guishan_wave_height = float(guishan_context.nww_series[time_key])
            guishan_current_speed = float(guishan_context.speed[time_key])
        except KeyError as exc:
            raise InputDerivationError(
                f"A 區 paired UTC 缺少 guishan 自己的 arrival metric：utc={time_key}"
            ) from exc
        metadata = {
            key: value
            for key, value in item.metadata.items()
            if key
            not in {
                "elevation_m",
                "elevation_derivative_mps",
                "tidal_strength_proxy_m",
                "significant_wave_height_m",
                "current_speed_mps",
            }
            and not key.startswith("metric_location_")
        }
        # paired design 只繼承 gongliao 選出的 UTC、潮汐類別與 phase label；所有
        # 指標數值改由 guishan 自己的 OCM anchor 與 NWW metric proxy 重算，避免把
        # gongliao 的 elevation/Hs/current 或 derivative/strength 冒充成 guishan 值。
        metadata.update(
            {
                "elevation_m": guishan_elevation,
                "significant_wave_height_m": guishan_wave_height,
                "current_speed_mps": guishan_current_speed,
                **guishan_context.metric_binding.to_metadata(),
                "shared_A_forcing_utc": "true",
                "shared_A_forcing_reference_site": "gongliao",
                "shared_A_forcing_policy": "gongliao_paired_utc_reference_v1",
                "tide_phase_label_source": "gongliao_paired_design_inherited",
            }
        )
        selection_policy = metadata.get("selection_policy")
        if isinstance(selection_policy, str) and selection_policy:
            # 新 observation policy 的 paired 站點仍須各自擁有可重算的 arrival ID；
            # core identity 包含 replicate rank，event identity 則包含 policy。舊兩年份
            # metadata 沒有 selection_policy，維持下方原有五欄 identity。
            if item.tide_class == "event":
                paired_identity_fields = [
                    "guishan",
                    str(time_key),
                    item.phase_or_event,
                    selection_policy,
                    design_version,
                ]
            else:
                replicate_rank = metadata.get("replicate_rank")
                if type(replicate_rank) is not int:
                    raise InputDerivationError(
                        f"新版 paired arrival 缺少 replicate_rank：{item.arrival_time_id}"
                    )
                paired_identity_fields = [
                    "guishan",
                    str(time_key),
                    item.tide_class,
                    item.phase_or_event,
                    str(replicate_rank),
                    selection_policy,
                    design_version,
                ]
        else:
            paired_identity_fields = [
                "guishan",
                str(time_key),
                item.tide_class,
                item.phase_or_event,
                design_version,
            ]
        cloned.append(
            ArrivalTime(
                arrival_time_id=stable_identifier(
                    "arr",
                    paired_identity_fields,
                ),
                study_site_id="guishan",
                time_utc_ns=item.time_utc_ns,
                year=item.year,
                season=item.season,
                tide_class=item.tide_class,
                phase_or_event=item.phase_or_event,
                metadata=metadata,
            )
        )
    return cloned


def _face_node_indices(mesh: NativeMesh, *, polygon: Polygon) -> np.ndarray:
    """找出落在 candidate polygon 內的 source face node 聯集。"""

    indices: set[int] = set()
    for row, count in zip(mesh.face_nodes_local, mesh.face_node_count, strict=True):
        face = Polygon([tuple(mesh.node_xy[int(node)]) for node in row[: int(count)]])
        if polygon.intersects(face):
            indices.update(int(node) for node in row[: int(count)])
    return np.asarray(sorted(indices), dtype=np.int64)


def _receptor_candidate_polygon_metric(
    *,
    site: StudySiteConfig,
    projection: DomainProjection,
    local_polygon_lonlat: Polygon,
) -> Polygon:
    """建立單一站點的公尺制受體候選區，必要時套用明示的核心圓限制。

    ``local_polygon_lonlat`` 是 geometry builder 已建立的固定候選區：A 區已是 anchor
    與 OCM 靜態海域 polygon 的 local-domain 交集，B--D 則是各自 flow domain。這個
    helper 只在受體選點前縮小候選範圍，不會回寫或改變 local geometry manifest。
    若設定同時提供 ``anchor_lonlat`` 與 ``receptor_core_radius_m``，兩者會先以該 flow
    domain 的既有 AEQD 投影轉成公尺，再求核心圓與上述候選區的交集；因此半徑是公尺制
    幾何距離，不是以經緯度差近似的角度距離。未明示核心的站點直接回傳原本投影後的
    local/static-ocean 候選區，保留既有 horizontal maximin 的選點結果。

    ``StudySiteConfig`` 的核心欄位必須成對出現。交集若為空、退化或不是單一 Polygon，
    便 fail closed，避免以最近 face、凸包或其他未登錄的空間補值擴張候選範圍。
    """

    candidate_metric = projection.project_geometry(local_polygon_lonlat)
    if not isinstance(candidate_metric, Polygon):
        raise InputDerivationError(f"receptor candidate polygon 無效：{site.study_site_id}")

    anchor_lonlat = site.anchor_lonlat
    radius_m = site.receptor_core_radius_m
    if anchor_lonlat is None and radius_m is None:
        # 沒有核心設定的站點必須沿用原本 local/flow candidate polygon；這是 B--D
        # 既有行為的相容分支，不能因其他站點新增核心而連帶縮小候選範圍。
        return candidate_metric
    if anchor_lonlat is None or radius_m is None:
        raise InputDerivationError(
            f"{site.study_site_id} 的 receptor core 必須同時明示 anchor_lonlat 與 receptor_core_radius_m"
        )
    radius_value = float(radius_m)
    if not math.isfinite(radius_value) or radius_value <= 0.0:
        raise InputDerivationError(f"{site.study_site_id} 的 receptor_core_radius_m 必須是有限正數")

    anchor_x, anchor_y = projection.project(*anchor_lonlat)
    anchor_point = Point(float(anchor_x), float(anchor_y))
    # 以既有幾何 helper 相同的 64 段圓周近似建立公尺制核心；這只影響固定候選邊界，
    # 不會把逐時 wet/dry 狀態帶入 local geometry 或改變 flow-domain 支撐。
    core_polygon = anchor_point.buffer(radius_value, quad_segs=64)
    clipped = candidate_metric.intersection(core_polygon)
    if clipped.is_empty or not isinstance(clipped, Polygon) or not clipped.is_valid or clipped.area <= 0.0:
        raise InputDerivationError(
            f"{site.study_site_id} 的 receptor core 與 local/static ocean 候選區沒有有效交集"
        )
    return clipped


def _receptor_payload(
    *,
    config: ProjectConfig,
    products_by_region: Mapping[str, _ProductData],
    nww_products_by_region: Mapping[str, _ProductData],
    flow_polygons: Mapping[str, Polygon],
    local_polygons: Mapping[str, Polygon],
    arrivals_by_site: Mapping[str, Sequence[ArrivalTime]],
    nww_runtime_caches: Mapping[str, _NWWRuntimeCache] | None = None,
    ocm_pair_caches: Mapping[str, _OCMPairCache] | None = None,
    vertical_support_cache: MutableMapping[_VerticalSupportCacheKey, _FaceVerticalSupport] | None = None,
    source_hashes: Mapping[str, str],
    strict: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[Receptor, ...],
    Mapping[str, NativeMesh],
    Mapping[str, DomainProjection],
]:
    """產生每站 5×4 receptors，並以 OCM／NWW runtime support gate 篩選。

    水平候選先以既有 local／static-ocean geometry 建立；若站點同時明示 receptor core
    anchor 與半徑，這裡只把候選 polygon 與同一 AEQD 公尺投影中的核心圓求交，並不改寫
    local geometry。之後仍由既有 persistent-wet、boundary margin 與 deterministic
    anchor-first maximin 產生 5 個 face；每一輪只對被選出的少量 face 執行一次 NWW exact-hour
    四角 support gate，再切取所有 arrival 的 ``(node, layer)`` zcor。NWW cache 由
    build handler 依 analysis region 建立並與 arrival selector 共用；一個 horizontal
    face 的四個 vertical receptor 只會共用同一次 50-arrival 判定。任何 NWW 或垂向 gate
    失敗的 face 會在候選 wetdry copy 中永久
    blacklist，再重新執行同一 maximin；這使水平選擇保持站點獨立、可重現且有限終止，
    不會刪除 arrival/receptor/scenario 或以最近有效值補填。

    若站點設定了 ``receptor_candidate_regions``，候選面會先依每個紅框子區的
    GeoJSON 多邊形與已驗收 local／static-ocean flow polygon 求交，再按
    ``receptor_candidate_selection`` 的明示配額分區執行相同的 wet/dry、邊界距離與
    NWW 垂向支撐檢查。紅框只代表受體候選選擇的數位化來源與空間限制，不代表 OCM
    或 NWW forcing 支援；所有 forcing、靜態海域、持續濕點與垂向遮罩 gate 仍必須
    通過，且選出的 face 會在 metadata 與 provenance 保留其 ``candidate_region_id``。
    """

    site_by_id = {site.study_site_id: site for site in config.study_sites}
    domain_by_region = {domain.analysis_region_id: domain for domain in config.domains}
    receptors: list[Receptor] = []
    rows: list[dict[str, Any]] = []
    meshes: dict[str, NativeMesh] = {}
    projections: dict[str, DomainProjection] = {}
    # 只有明示紅框候選的站點才會填入這份索引；它同時供 receptor metadata 與
    # provenance 使用，讓每個最終 face 能回溯到指定的研究子區，而不是靠座標事後猜測。
    candidate_region_selection_by_site: dict[str, dict[str, Any]] = {}
    # build_input_derivatives 會把每個 analysis region 的 cache 傳進來；直接使用此
    # internal helper 的小型 caller 若未提供 mapping，才保留 lazy 建立的相容行為。
    shared_nww_runtime_caches = dict(nww_runtime_caches or {})
    # 同一 flow domain 可能服務兩個站點；pair cache 以 analysis region 作外層 key，
    # 內部再核對實際 source ID。A 區兩站的 wetdry 與 zcor/elev 因而只開檔一次；站點
    # 本身仍各自使用自己的 arrival、candidate geometry 與 deterministic maximin，不能
    # 因 cache 共用而合併受體選擇。
    pair_caches: dict[str, _OCMPairCache] = dict(ocm_pair_caches or {})
    # 這份 cache 由 build handler 傳入時會延伸到 dynamic 階段；獨立 helper 未傳入時
    # 建立本次呼叫專用字典，仍維持原本 lazy、無全域狀態的行為。key 同時含 region、
    # site、arrival UTC 與 face，避免兩站共用 flow domain 時錯拿另一站的垂向結果。
    shared_vertical_support_cache = vertical_support_cache if vertical_support_cache is not None else {}
    for site_id in sorted(site_by_id):
        site = site_by_id[site_id]
        product = products_by_region[site.analysis_region_id]
        nww_product = nww_products_by_region.get(site.analysis_region_id)
        if nww_product is None:
            raise InputDerivationError(f"site 缺少 NWW product：{site_id}")
        domain = domain_by_region[site.analysis_region_id]
        projection = DomainProjection(*domain.center_lonlat)
        projections[product.flow_domain_id] = projection
        nww_cache = shared_nww_runtime_caches.get(site.analysis_region_id)
        if nww_cache is None:
            nww_cache = _load_nww_runtime_cache(nww_product)
            shared_nww_runtime_caches[site.analysis_region_id] = nww_cache
        # 不使用 dict.setdefault 的 eager default；同一 flow domain 的第二個站點不應
        # 因為 Python 先求值 default 而重新開啟整套 mesh NPY。A 區兩站仍各自建立
        # candidate geometry，但共用 immutable mesh object。
        mesh = meshes.get(product.flow_domain_id)
        if mesh is None:
            mesh = _load_native_mesh(product, projection)
            meshes[product.flow_domain_id] = mesh
        polygon_lonlat = local_polygons[site_id]
        candidate_specs = _candidate_region_specs(site)
        if candidate_specs:
            # 紅框候選是新的研究空間契約：先與 approved local/flow geometry 求交，
            # 但不再套用舊版 12,500 m 圓形核心。核心欄位仍由 config 保留作舊版
            # provenance，避免把研究者明示的兩個子區偷偷縮回舊中心附近。
            candidate_metric = projection.project_geometry(polygon_lonlat)
            if (
                not isinstance(candidate_metric, Polygon)
                or candidate_metric.is_empty
                or not candidate_metric.is_valid
                or candidate_metric.area <= 0.0
            ):
                raise InputDerivationError(f"{site_id} approved local/flow 候選區無效")
        else:
            candidate_metric = _receptor_candidate_polygon_metric(
                site=site,
                projection=projection,
                local_polygon_lonlat=polygon_lonlat,
            )
        anchor_lonlat = site.anchor_lonlat or domain.center_lonlat
        anchor_xy_array = projection.project(*anchor_lonlat)
        anchor_xy = (float(anchor_xy_array[0]), float(anchor_xy_array[1]))
        arrival_records = list(arrivals_by_site[site_id])
        if not arrival_records:
            raise InputDerivationError(f"site 沒有 arrival，無法建立 receptor：{site_id}")
        analysis_region_id = site.analysis_region_id
        pair_cache = pair_caches.get(analysis_region_id)
        if pair_cache is None:
            pair_cache = _load_ocm_pair_cache(product)
            pair_caches[analysis_region_id] = pair_cache
        pair_cache = _assert_ocm_pair_cache(product, pair_cache)
        by_time = pair_cache.utc_prefer_last_index
        depth_array = pair_cache.depth
        month_arrays = pair_cache.monthly_arrays
        arrival_sources: list[tuple[_MonthData, int]] = []
        for arrival in arrival_records:
            source = by_time.get(int(arrival.time_utc_ns))
            if source is None:
                raise InputDerivationError(f"arrival UTC 不在 OCM source time：{arrival.time_utc_ns}")
            arrival_sources.append(source)
        wetdry = _face_times(
            product,
            arrival_records,
            mesh.face_nodes_local.shape[0],
            pair_cache=pair_cache,
        )
        # arrival selector 已先完成逐 UTC 的 NWW exact-hour 四角 gate；此處對每個
        # horizontal face 再以其實際 lon/lat 重做同一套空間支撐，確保 receptor 與
        # runtime 的四角 mask/物理契約一致。若某一 arrival 失敗，整個 face 會被本站
        # persistent blacklist 淘汰，而不是刪除 arrival 或以 nearest wave 補值。
        if wetdry.shape[0] != len(arrival_records):
            raise InputDerivationError("receptor wetdry rows 與 arrival count 不一致")
        margin = float(
            site.model_extra.get("minimum_flow_domain_margin_local_grid_scales", 0.0)
            if site.model_extra
            else 0.0
        )
        # config 的 margin 單位是網格尺度；以網格中位邊長估計公尺數，沒有猜測固定 1 km。
        if margin > 0:
            triangle_edges = np.linalg.norm(
                mesh.node_xy[mesh.triangle_nodes] - np.roll(mesh.node_xy[mesh.triangle_nodes], -1, axis=1),
                axis=2,
            )
            margin *= (
                float(np.median(triangle_edges[triangle_edges > 0])) if np.any(triangle_edges > 0) else 0.0
            )
        # 這份 cache 只屬於目前 study site；不同站點可能有不同 arrival UTC 集合，
        # 即使共用同一個 flow-domain/NWW memory-map，也不能跨站點誤用 support 結果。
        # value=None 代表此 face 已知因 NWW 或 OCM 垂向 gate 失敗，error message 只供
        # 重新被 selector 選到時維持相同的 fail-closed 行為，不會重新讀取 50 個時次。
        face_support_cache: dict[int, _FaceVerticalSupport | None] = {}
        face_support_errors: dict[int, str] = {}

        def face_support(
            horizontal_item: Any,
            *,
            mesh: NativeMesh = mesh,
            depth_array: np.ndarray = depth_array,
            analysis_region_id: str = analysis_region_id,
            site_id: str = site_id,
            arrival_records: Sequence[ArrivalTime] = arrival_records,
            arrival_sources: Sequence[tuple[_MonthData, int]] = arrival_sources,
            month_arrays: Mapping[str, _OCMMonthPairArrays] = month_arrays,
            nww_cache: _NWWRuntimeCache = nww_cache,
            face_support_cache: dict[int, _FaceVerticalSupport | None] = face_support_cache,
            face_support_errors: dict[int, str] = face_support_errors,
        ) -> _FaceVerticalSupport:
            """檢查一個候選 face 的全部 arrival 與四個垂向類別。

            這個局部檢查先以 face 的水平 lon/lat 對 NWW runtime-equivalent 四角 gate
            一次檢查全部 arrival，再切出目前候選 face 的 node/layer 小矩陣，不會為了
            重選水平 receptor 掃描整個網格 zcor。第一個 arrival 的結果回傳作 template；
            其餘 arrival 只要任一 NWW 或 OCM 垂向支撐失敗，整個 face 就進 deterministic
            blacklist，下一輪沿用原本的 maximin 重新選擇。
            """

            face_index = int(horizontal_item.source_face_local_index)
            if face_index in face_support_cache:
                cached_support = face_support_cache[face_index]
                if cached_support is None:
                    raise InputDerivationError(face_support_errors[face_index])
                return cached_support

            _nww_hs_by_time, nww_valid_by_time = _nww_exact_hour_samples(
                nww_cache,
                lon=float(horizontal_item.lon),
                lat=float(horizontal_item.lat),
                requested_times=tuple(int(item.time_utc_ns) for item in arrival_records),
            )
            unsupported_times = [
                int(item.time_utc_ns)
                for item in arrival_records
                if not nww_valid_by_time.get(int(item.time_utc_ns), False)
            ]
            if unsupported_times:
                raise InputDerivationError(
                    f"{site_id} receptor face NWW runtime 四角支撐失敗："
                    f"face={horizontal_item.source_face_local_index}, utc={unsupported_times[0]}"
                )

            node_count = int(mesh.face_node_count[horizontal_item.source_face_local_index])
            nodes = mesh.face_nodes_local[horizontal_item.source_face_local_index, :node_count]
            if np.any(nodes < 0) or np.any(nodes >= depth_array.size):
                raise InputDerivationError(
                    f"{site_id} receptor face node 超出 source depth：{horizontal_item}"
                )
            node_depth = np.asarray(depth_array[nodes], dtype=np.float64)
            if node_depth.size != node_count or not np.all(np.isfinite(node_depth)):
                raise InputDerivationError(f"{site_id} receptor face 水深無有效值")
            first_support: _FaceVerticalSupport | None = None
            for arrival, (source_month, source_local) in zip(
                arrival_records,
                arrival_sources,
                strict=True,
            ):
                month_values = month_arrays.get(source_month.label)
                if month_values is None:
                    raise InputDerivationError(
                        f"{site_id} receptor cache 缺少 OCM 月份：{source_month.label}"
                    )
                zcor = month_values.zcor
                elev = month_values.elev
                support_key: _VerticalSupportCacheKey = (
                    analysis_region_id,
                    site_id,
                    int(arrival.time_utc_ns),
                    face_index,
                )
                support = shared_vertical_support_cache.get(support_key)
                if support is None:
                    try:
                        # 同一個 horizontal face×arrival 一次建立完整四個 vertical
                        # target/bracket；後面四個 receptor 只取對應 target，不再各自掃描
                        # node/layer。所有逐 node 雙側支撐、wet/dry 與缺值 gate 均維持原規則。
                        zcor_slice = np.asarray(zcor[source_local, nodes], dtype=np.float64)
                        eta_values = np.asarray(elev[source_local, nodes], dtype=np.float64)
                    except (IndexError, TypeError, ValueError) as exc:
                        raise InputDerivationError(
                            f"{site_id} receptor face 無法切取 OCM zcor／eta：{arrival.time_utc_ns}"
                        ) from exc
                    support = _build_face_vertical_support(
                        zcor_node_layer=zcor_slice,
                        node_elev_m=eta_values,
                        node_depth_m=node_depth,
                        vertical_ids=_VERTICAL_TARGET_IDS,
                    )
                    shared_vertical_support_cache[support_key] = support
                if first_support is None:
                    first_support = support
            if first_support is None:
                raise InputDerivationError(f"{site_id} receptor face 沒有可用的 OCM arrival")
            return first_support

        # geometry、persistent-wet 與 boundary margin 只在本站（或本站的每一個紅框
        # 子區）建立一次 immutable pool；後續 blacklist rounds 只對 pool 內的公尺制候選
        # 重跑 maximin，不再逐輪建立 Shapely Point 或掃描全部 source face。紅框分支
        # 對每一個子區獨立配額，禁止跨區借點補足。
        blacklisted_faces: set[int] = set()
        template_supports: dict[int, _FaceVerticalSupport] = {}
        region_id_by_face: dict[int, str] = {}

        if candidate_specs:
            horizontal = []
            used_faces: set[int] = set()
            region_records: dict[str, dict[str, Any]] = {}
            for spec in candidate_specs:
                region_metric = projection.project_geometry(spec.geometry_lonlat)
                scoped_metric = candidate_metric.intersection(region_metric)
                if (
                    not isinstance(scoped_metric, Polygon)
                    or scoped_metric.is_empty
                    or not scoped_metric.is_valid
                    or scoped_metric.area <= 0.0
                ):
                    raise InputDerivationError(
                        f"{site_id}/{spec.region_id} 紅框候選與 approved local/flow geometry 無有效交集"
                    )
                # representative_point 保證落在子區內，作為每子區 anchor-first 的
                # 公尺制 anchor；不使用舊站點 anchor，避免西岸子區被遠端中心牽引。
                region_anchor = scoped_metric.representative_point()
                region_anchor_xy = (float(region_anchor.x), float(region_anchor.y))
                region_pool = prepare_horizontal_receptor_candidates(
                    study_site_id=f"{site_id}:{spec.region_id}",
                    mesh=mesh,
                    candidate_polygon_metric=scoped_metric,
                    anchor_xy=region_anchor_xy,
                    wetdry_at_arrivals=wetdry,
                    boundary_margin_m=margin,
                    wet_value=0.0,
                )
                candidate_count = int(region_pool.candidate_face_local_indices.size)
                scope_blacklist: set[int] = set()
                scope_template_supports: dict[int, _FaceVerticalSupport] = {}
                for _attempt in range(candidate_count + 1):
                    try:
                        selected_scope = select_horizontal_receptors_from_pool(
                            region_pool,
                            count=spec.allocation_count,
                            excluded_face_indices=scope_blacklist | used_faces,
                        )
                    except ValueError as exc:
                        raise InputDerivationError(
                            f"{site_id}/{spec.region_id} 紅框子區在垂向支撐淘汰後候選不足：{exc}"
                        ) from exc
                    failed_faces: list[int] = []
                    for horizontal_item in selected_scope:
                        face_index = int(horizontal_item.source_face_local_index)
                        try:
                            support = face_support(horizontal_item)
                        except InputDerivationError as exc:
                            # NWW exact-hour 或 OCM 四垂向層位任一失敗都永久列入
                            # 此站點 blacklist；下一輪仍用相同 tie-break 與 anchor。
                            face_support_cache[face_index] = None
                            face_support_errors[face_index] = str(exc)
                            failed_faces.append(face_index)
                        else:
                            face_support_cache[face_index] = support
                            scope_template_supports[face_index] = support
                    if not failed_faces:
                        break
                    scope_blacklist.update(failed_faces)
                else:
                    raise InputDerivationError(
                        f"{site_id}/{spec.region_id} 紅框子區垂向支撐重選超過有限迭代次數"
                    )
                anchor_lon, anchor_lat = projection.unproject(float(region_anchor.x), float(region_anchor.y))
                for horizontal_item in selected_scope:
                    face_index = int(horizontal_item.source_face_local_index)
                    if face_index in used_faces:
                        raise InputDerivationError(
                            f"{site_id}/{spec.region_id} 選到其他子區已使用 face：{face_index}"
                        )
                    used_faces.add(face_index)
                    region_id_by_face[face_index] = spec.region_id
                    support = scope_template_supports.get(face_index)
                    if support is None:
                        raise InputDerivationError(
                            f"{site_id}/{spec.region_id} 最終 receptor face 未通過垂向 gate：{face_index}"
                        )
                    template_supports[face_index] = support
                blacklisted_faces.update(scope_blacklist)
                region_records[spec.region_id] = {
                    "candidate_face_count_persistent_wet_margin": candidate_count,
                    "blacklist_face_count": len(scope_blacklist),
                    "blacklist_face_local_indices": sorted(scope_blacklist),
                    "selected_face_local_indices": [
                        int(item.source_face_local_index) for item in selected_scope
                    ],
                    "selected_face_global_indices": [
                        int(item.source_face_global_index) for item in selected_scope
                    ],
                    "selected_positions_lonlat": [
                        {"lon": float(item.lon), "lat": float(item.lat)} for item in selected_scope
                    ],
                    "anchor_lonlat": [float(anchor_lon), float(anchor_lat)],
                    "anchor_xy_m": [float(region_anchor.x), float(region_anchor.y)],
                }
                horizontal.extend(selected_scope)
            candidate_region_selection_by_site[site_id] = _candidate_region_provenance(
                candidate_specs,
                source_region_records=region_records,
                source_provenance=(
                    site.receptor_candidate_regions_provenance
                    or (site.model_extra or {}).get("receptor_candidate_regions_provenance")
                ),
                selection_policy=(
                    site.receptor_candidate_selection
                    or (site.model_extra or {}).get("receptor_candidate_selection")
                ),
            )
        else:
            candidate_pool = prepare_horizontal_receptor_candidates(
                study_site_id=site_id,
                mesh=mesh,
                candidate_polygon_metric=candidate_metric,
                anchor_xy=anchor_xy,
                wetdry_at_arrivals=wetdry,
                boundary_margin_m=margin,
                wet_value=0.0,
            )
            candidate_count = int(candidate_pool.candidate_face_local_indices.size)
            # 任一候選的任一 arrival NWW 或垂向支撐失敗，就以 source face local index
            # 加入本站 persistent blacklist，再用完全相同的 deterministic pool selector
            # 重選。每次至少排除一個 face，故重選輪次有限且不會回到已知壞 face。
            for _attempt in range(candidate_count + 1):
                try:
                    horizontal = select_horizontal_receptors_from_pool(
                        candidate_pool,
                        count=5,
                        excluded_face_indices=blacklisted_faces,
                    )
                except ValueError as exc:
                    raise InputDerivationError(
                        f"{site_id} 垂向支撐淘汰後 persistent-wet 候選不足：{exc}"
                    ) from exc
                failed_faces = []
                for horizontal_item in horizontal:
                    face_index = int(horizontal_item.source_face_local_index)
                    try:
                        support = face_support(horizontal_item)
                    except InputDerivationError as exc:
                        # 將 combined gate 的失敗也快取；下一輪若 deterministic maximin
                        # 仍因其他 face 淘汰而碰到同一個 face，不能重做相同的 NWW／OCM I/O。
                        face_support_cache[face_index] = None
                        face_support_errors[face_index] = str(exc)
                        failed_faces.append(face_index)
                    else:
                        face_support_cache[face_index] = support
                        template_supports[face_index] = support
                if not failed_faces:
                    break
                for face_index in sorted(set(failed_faces)):
                    blacklisted_faces.add(face_index)
            else:
                raise InputDerivationError(f"{site_id} 垂向支撐重選超過有限迭代次數")

        for horizontal_item in horizontal:
            face_index = int(horizontal_item.source_face_local_index)
            support = template_supports.get(face_index)
            if support is None or face_index in blacklisted_faces:
                raise InputDerivationError(f"{site_id} 最終 receptor face 未通過垂向 gate：{face_index}")
            for target in support.targets:
                receptor_id = stable_identifier(
                    "rec",
                    [
                        site_id,
                        horizontal_item.horizontal_receptor_id,
                        target.vertical_id,
                        config.design_version,
                    ],
                )
                metadata: dict[str, float | int | str] = {
                    "source_face_local_index": horizontal_item.source_face_local_index,
                    "source_face_global_index": horizontal_item.source_face_global_index,
                    "horizontal_receptor_id": horizontal_item.horizontal_receptor_id,
                    "target_fraction_below_surface": target.target_fraction_below_surface
                    if target.target_fraction_below_surface is not None
                    else "near_bed_lowest_layer_center",
                    "template_bracket_span_m": target.bracket_span_m,
                }
                candidate_region_id = region_id_by_face.get(face_index)
                if candidate_region_id is not None:
                    metadata["candidate_region_id"] = candidate_region_id
                receptors.append(
                    Receptor(
                        receptor_id=receptor_id,
                        study_site_id=site_id,
                        analysis_region_id=site.analysis_region_id,
                        lon=horizontal_item.lon,
                        lat=horizontal_item.lat,
                        z_m_positive_up=target.z_m_positive_up,
                        vertical_id=target.vertical_id,
                        metadata=metadata,
                    )
                )
                rows.append(asdict(receptors[-1]))
    if len(receptors) != EXPECTED_RECEPTOR_COUNT:
        raise InputDerivationError(f"receptor 應有 100 筆，實際 {len(receptors)}")
    receptor_method_id = (
        "server_v3_persistent_wet_face_maximin_5x4_ocm_nww_runtime_support_red_frame_regions_v1"
        if candidate_region_selection_by_site
        else "server_v3_persistent_wet_face_maximin_5x4_ocm_nww_runtime_support_core_intersection_v3"
    )
    provenance = _provenance(
        method_id=receptor_method_id,
        source_hashes=source_hashes,
        counts={"study_sites": EXPECTED_STUDY_SITE_COUNT, "receptors": EXPECTED_RECEPTOR_COUNT},
        public_analysis_label_policy={"A": "A 區分析域"},
    )
    if candidate_region_selection_by_site:
        # provenance 以 site id 分層保存紅框 polygon、子區配額、persistent-wet 候選數、
        # blacklist 與最終 source face；JSON validator 會再檢查 hash closure，這裡不保存
        # 大型 mesh 或逐時 wetdry 陣列。
        provenance["candidate_regions_by_site"] = candidate_region_selection_by_site
    payload = {
        "manifest_kind": "receptor_manifest",
        "schema_version": DERIVED_INPUT_SCHEMA_VERSION,
        "status": "approved" if strict else "generated",
        "design_version": config.design_version,
        "coordinate_reference": "EPSG:4326",
        "vertical_reference": "z_m_positive_up",
        "generation_method_id": receptor_method_id,
        "provenance": provenance,
        "records": rows,
    }
    return payload, {item.receptor_id: item for item in receptors}, tuple(receptors), meshes, projections


def _dynamic_initial_payload(
    *,
    config: ProjectConfig,
    products_by_region: Mapping[str, _ProductData],
    receptors: Sequence[Receptor],
    arrivals: Sequence[ArrivalTime],
    source_hashes: Mapping[str, str],
    strict: bool,
    pilot_selection: Mapping[str, Any] | None = None,
    expected_pair_count: int | None = None,
    generation_method_id: str = "server_v3_ocm_dynamic_receptor_arrival_initial_condition_v1",
    provenance_extra: Mapping[str, Any] | None = None,
    ocm_pair_caches: Mapping[str, _OCMPairCache] | None = None,
    native_mesh_bindings: Mapping[str, _NativeMeshBinding] | None = None,
    vertical_support_cache: MutableMapping[_VerticalSupportCacheKey, _FaceVerticalSupport] | None = None,
) -> dict[str, Any]:
    """建立每個 receptor×arrival 一筆的 OCM-derived dynamic manifest。

    建立前先按 analysis region 建立／接收帶 flow-domain binding 的唯讀 cache，將網格
    拓撲、月份時間索引與 OCM 的 zcor、elev、wetdry 陣列各載入一次。正式 builder 會
    把 receptor 階段已建立的 cache 與 NativeMesh 直接傳入，因此不會在 dynamic 階段
    再次開啟同一組月份檔案或重建網格；沒有傳入時則保留獨立 helper 的 lazy 行為。
    時間索引沿用月份排序後的 prefer-last 語意。每列仍保存實際月份、source time index、
    面索引與 observed origin，因而不會把模板 receptor 的 z 值誤當成所有 arrival 共用的
    固定深度。每個 unique ``(analysis region, site, arrival UTC, face)`` 只以
    ``_VERTICAL_TARGET_IDS`` 建立一次完整垂向支撐；四個 vertical receptor 後續只取
    對應 target/bracket。若 receptor gate 與實際 zcor／eta 不一致，仍直接 fail closed。
    ``expected_pair_count`` 預設為既有正式流程的 5,000 筆；只有明確傳入時才允許
    工程性小範圍 adapter 使用另一個已驗收的 receptor×arrival 數量，因此不會改變正式
    wrapper 的固定計數。``generation_method_id`` 與 ``provenance_extra`` 讓工程 artifact
    能標示自己的來源與限制；它們只會加入新 manifest 的 provenance，不會把舊 accepted
    hash 或正式狀態重新綁定到新設定。
    """

    if expected_pair_count is not None and (
        isinstance(expected_pair_count, bool)
        or not isinstance(expected_pair_count, int)
        or expected_pair_count < 1
    ):
        raise InputDerivationError("expected_pair_count 必須是正整數或 None")
    if not isinstance(generation_method_id, str) or not generation_method_id.strip():
        raise InputDerivationError("generation_method_id 不可為空白")
    if provenance_extra is not None and not isinstance(provenance_extra, Mapping):
        raise InputDerivationError("provenance_extra 必須是 mapping 或 None")

    site_by_id = {site.study_site_id: site for site in config.study_sites}
    domain_by_region = {domain.analysis_region_id: domain for domain in config.domains}
    arrival_by_site: dict[str, list[ArrivalTime]] = {}
    for arrival in arrivals:
        arrival_by_site.setdefault(arrival.study_site_id, []).append(arrival)

    # 每個 region 的 tuple 內容依序為 NativeMesh 與帶 flow-domain binding 的 OCM cache。
    # 陣列仍是 mmap view，不會因建立 5,000 rows 把完整四維 forcing 複製進記憶體；
    # cache 的生命週期通常涵蓋同一個 builder call 的 receptor 與 dynamic 階段。
    shared_pair_caches: dict[str, _OCMPairCache] = dict(ocm_pair_caches or {})
    shared_mesh_bindings: dict[str, _NativeMeshBinding] = dict(native_mesh_bindings or {})
    shared_vertical_support_cache = vertical_support_cache if vertical_support_cache is not None else {}
    # 先按唯一 pair key 收集實際 caller 要求的 vertical 類別。正式 builder 的每個
    # horizontal face 會有四類，因此仍按固定 ``_VERTICAL_TARGET_IDS`` 順序一次建立
    # 全部支撐；engineering adapter 可能只保留 near-bed，不能因未要求的類別缺少
    # 支撐而改變原本的單站相容行為。
    requested_vertical_ids_by_key: dict[tuple[str, str, int, int], set[str]] = {}
    for receptor in receptors:
        site_arrivals = arrival_by_site.get(receptor.study_site_id, ())
        node_face = receptor.metadata.get("source_face_local_index")
        if not isinstance(node_face, int) or isinstance(node_face, bool):
            # 保留後面依 receptor 的既有明確錯誤；此處不先猜測 face key。
            continue
        for arrival in site_arrivals:
            key = (
                receptor.analysis_region_id,
                receptor.study_site_id,
                int(arrival.time_utc_ns),
                int(node_face),
            )
            requested_vertical_ids_by_key.setdefault(key, set()).add(receptor.vertical_id)
    runtime_cache: dict[
        str,
        tuple[NativeMesh, _OCMPairCache],
    ] = {}
    for region, product in sorted(products_by_region.items()):
        domain = domain_by_region[region]
        projection = DomainProjection(*domain.center_lonlat)
        pair_cache = shared_pair_caches.get(region)
        if pair_cache is None:
            # 獨立 engineering／測試 caller 未提供共同 cache 時，仍在此處 lazy 建立，
            # 但建立後以 analysis region 保存，避免同一 helper 內後續 receptor 取樣重開。
            pair_cache = _load_ocm_pair_cache(product)
            shared_pair_caches[region] = pair_cache
        pair_cache = _assert_ocm_pair_cache(product, pair_cache)
        mesh_binding = shared_mesh_bindings.get(region)
        if mesh_binding is None:
            mesh_binding = _NativeMeshBinding(
                flow_domain_id=product.flow_domain_id,
                mesh=_load_native_mesh(product, projection),
            )
            shared_mesh_bindings[region] = mesh_binding
        if not isinstance(mesh_binding, _NativeMeshBinding):
            raise InputDerivationError(
                f"dynamic NativeMesh binding 必須是具 flow-domain binding 的 _NativeMeshBinding：{region}"
            )
        if mesh_binding.flow_domain_id != product.flow_domain_id:
            raise InputDerivationError(
                "dynamic NativeMesh flow_domain_id 錯綁："
                f"cache={mesh_binding.flow_domain_id}，product={product.flow_domain_id}"
            )
        runtime_cache[region] = (mesh_binding.mesh, pair_cache)

    records: list[dict[str, Any]] = []
    for receptor in sorted(receptors, key=lambda item: item.receptor_id):
        site = site_by_id[receptor.study_site_id]
        product = products_by_region[site.analysis_region_id]
        mesh, pair_cache = runtime_cache[site.analysis_region_id]
        by_time = pair_cache.utc_prefer_last_index
        depth = pair_cache.depth
        month_arrays = pair_cache.monthly_arrays
        node_face = receptor.metadata.get("source_face_local_index")
        if not isinstance(node_face, int) or isinstance(node_face, bool):
            raise InputDerivationError(f"receptor 缺少 source_face_local_index：{receptor.receptor_id}")
        if node_face < 0 or node_face >= mesh.face_nodes_local.shape[0]:
            raise InputDerivationError(f"receptor source_face_local_index 超出網格：{receptor.receptor_id}")
        node_count = int(mesh.face_node_count[node_face])
        nodes = mesh.face_nodes_local[node_face, :node_count]
        depth_values = np.asarray(depth[nodes], dtype=np.float64)
        if depth_values.size != node_count or not np.all(np.isfinite(depth_values)):
            raise InputDerivationError(f"dynamic pair source face 水深無效：{receptor.receptor_id}")
        for arrival in sorted(arrival_by_site[receptor.study_site_id], key=lambda item: item.time_utc_ns):
            source = by_time.get(int(arrival.time_utc_ns))
            if source is None:
                raise InputDerivationError(f"dynamic pair UTC 不在 OCM：{arrival.time_utc_ns}")
            month, local = source
            month_values = month_arrays.get(month.label)
            if month_values is None:
                raise InputDerivationError(f"dynamic pair cache 缺少 OCM 月份：{month.label}")
            zcor = month_values.zcor
            elev = month_values.elev
            wetdry = month_values.wetdry
            # dynamic 只需要目前 receptor 所在 face 的濕乾狀態；直接以二維索引取一個
            # scalar，避免先把整個 time row materialize 成 float64。receptor 階段仍需
            # 整列 wetdry 建立 horizontal candidate pool，兩者資料需求不同，不能以此
            # 優化放寬逐 face 的 wet/dry gate。
            wetdry_value = int(round(float(np.asarray(wetdry[local, node_face], dtype=np.float64))))
            if wetdry_value != 0:
                raise InputDerivationError(
                    f"dynamic pair 遇到 dry face：{receptor.receptor_id}/{arrival.arrival_time_id}"
                )
            support_key: _VerticalSupportCacheKey = (
                receptor.analysis_region_id,
                receptor.study_site_id,
                int(arrival.time_utc_ns),
                int(node_face),
            )
            pair_support = shared_vertical_support_cache.get(support_key)
            if pair_support is None:
                # 不能沿用 receptor template 的 z 或只對 face node 先取 median；actual
                # pair 必須依此 UTC 的每個 node/layer zcor 重新證明 caller 要求的
                # vertical target 雙側支撐。正式四類支撐一旦建立，四個 receptor 只取
                # 自己的 target/bracket，不會為每個 vertical class 重複掃描相同資料。
                requested_vertical_ids = tuple(
                    vertical_id
                    for vertical_id in _VERTICAL_TARGET_IDS
                    if vertical_id in requested_vertical_ids_by_key.get(support_key, set())
                )
                if not requested_vertical_ids:
                    raise InputDerivationError(
                        f"dynamic pair 找不到目前 face／arrival 的 vertical 要求：{support_key}"
                    )
                pair_support = _build_face_vertical_support(
                    zcor_node_layer=np.asarray(zcor[local, nodes], dtype=np.float64),
                    node_elev_m=np.asarray(elev[local, nodes], dtype=np.float64),
                    node_depth_m=depth_values,
                    vertical_ids=requested_vertical_ids,
                )
                shared_vertical_support_cache[support_key] = pair_support
            target = pair_support.target_for(receptor.vertical_id)
            lower, upper = pair_support.bracket_for(receptor.vertical_id)
            alpha = float((target.z_m_positive_up - lower) / (upper - lower))
            z = float(lower + alpha * (upper - lower))
            eta = pair_support.eta_z_m_positive_up
            bed = pair_support.bed_z_m_positive_up
            water = float(eta - bed)
            record = {
                "receptor_id": receptor.receptor_id,
                "arrival_time_id": arrival.arrival_time_id,
                "study_site_id": receptor.study_site_id,
                "analysis_region_id": receptor.analysis_region_id,
                "flow_domain_id": product.flow_domain_id,
                "time_utc_ns": int(arrival.time_utc_ns),
                "vertical_id": receptor.vertical_id,
                "z_m_positive_up": z,
                "eta_m_positive_up": eta,
                "bed_z_m_positive_up": bed,
                "water_column_height_m": water,
                "height_above_bed_m": z - bed,
                "zcor_lower_m_positive_up": lower,
                "zcor_upper_m_positive_up": upper,
                "vertical_bracket_alpha": alpha,
                "source_face_local_index": int(node_face),
                "source_face_global_index": int(receptor.metadata.get("source_face_global_index", node_face)),
                "wetdry_elem_value": wetdry_value,
                "wetdry_semantics_id": _WETDRY_SEMANTICS_ID,
                "ocm_month_yyyymm": month.label,
                "ocm_source_time_index": int(local),
                "ocm_time_origin": "observed",
            }
            records.append(record)
    # 診斷訊息也依 site 配對計算，避免工程單站只傳一筆 arrival 時因完整五站 config
    # 而顯示 0；這只影響 mismatch 的說明文字，不改正式 5,000-row gate。
    expected_product = sum(
        sum(1 for receptor in receptors if receptor.study_site_id == site_id) * len(site_arrivals)
        for site_id, site_arrivals in arrival_by_site.items()
    )
    expected_count = (
        EXPECTED_DYNAMIC_INITIAL_CONDITION_COUNT if expected_pair_count is None else expected_pair_count
    )
    if len(records) != expected_count:
        raise InputDerivationError(
            f"dynamic initial conditions 應有 {expected_count} 筆，實際 {len(records)}；"
            f"expected_product={expected_product}"
        )
    provenance_payload: dict[str, Any] = {
        "counts": {"receptors": len(receptors), "arrivals": len(arrivals), "pairs": len(records)},
        "wetdry_semantics_id": _WETDRY_SEMANTICS_ID,
    }
    if pilot_selection is not None:
        provenance_payload["pilot_selection_scope"] = dict(pilot_selection)
    if provenance_extra is not None:
        # caller 的 mapping 可能是 immutable proxy；只複製第一層，避免 builder 在建立
        # provenance 時意外改寫呼叫端保存的工程 lineage。巢狀資料會由 canonical JSON
        # writer 在輸出前檢查是否可序列化。
        provenance_payload.update(dict(provenance_extra))
    return {
        "manifest_kind": "receptor_arrival_initial_condition_manifest",
        "schema_version": DERIVED_INPUT_SCHEMA_VERSION,
        "status": "approved" if strict else "generated",
        "design_version": config.design_version,
        "vertical_reference": "z_m_positive_up",
        "time_standard": "UTC",
        "generation_method_id": generation_method_id,
        "provenance": _provenance(
            method_id=generation_method_id,
            source_hashes=source_hashes,
            **provenance_payload,
        ),
        "records": records,
    }


def _material_payload(config: ProjectConfig, *, source_hashes: Mapping[str, str]) -> dict[str, Any]:
    """由既有十類 baseline behavior 生成 legacy-compatible material schema 2。"""

    # material loader 有意保留既有 schema 的 exact root keys；source hash 由 artifact index
    # 與 release binding 保存，不能直接把 provenance/status 加進 root 破壞相容性。
    del source_hashes
    return {
        "schema_version": "2.0.0",
        "design_version": config.design_version,
        "classification_source": "海洋保育署 iOcean 分類；僅作臺灣海廢分類命名，不提供單體物性",
        "velocity_unit": "m s-1; z positive-up; all values must be strictly negative",
        "velocity_source": "design_sensitivity_grid_not_oca_measurement",
        "positive_or_zero_velocity_policy": "reject_config",
        "calibration_scope": "provisional_material_shape_proxy_pending_local_measurement",
        "records": [asdict(item) for item in BASELINE_BEHAVIORS],
    }


def _arrival_payload(
    config: ProjectConfig,
    arrivals: Sequence[ArrivalTime],
    *,
    source_hashes: Mapping[str, str],
    strict: bool,
    horizon_settings: HorizonSettings | None = None,
    pilot_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """將 ArrivalTime dataclass 轉成 strict schema 1 manifest。

    pilot replacement 仍保存相同 250 筆 records 與既有 schema root；差異只寫入版本化
    ``selection_method_id`` 與 provenance，並由單筆 record metadata 保存原／替換 identity。
    因此 artifact hash、release binding 與 run-create 仍可沿用既有 loader，而 formal
    arrival strata loader 會因 pilot-only 非潮汐標籤明確拒絕升格。
    """

    bed_config = config.scenarios.bed_residence_time
    selection_config = config.arrival_time_selection
    observation_policy = (
        selection_config.policy
        if selection_config is not None and selection_config.policy is not None
        else "two_years_stratified_48_plus_2_v1"
    )
    method_id = (
        PILOT_ARRIVAL_SELECTION_METHOD_ID
        if pilot_selection is not None
        else (
            # bed-residence 1.1 loader 的 component method identity 已凍結；新版
            # observation policy 另由 provenance 的 arrival_time_selection 保存，不能
            # 只改 root method_id 而使既有 bed loader 無法讀取同一份母體。
            "server_v3_48_strata_plus_two_observation_anchors_then_random_deposition_v1"
            if bed_config is not None
            else (
                ARRIVAL_SELECTION_METHOD_OBSERVATION_YEAR_ID
                if observation_policy == "observation_year_stratified_48_plus_2_v1"
                else ARRIVAL_SELECTION_METHOD_ID
            )
        )
    )
    provenance_extra: dict[str, Any] = {
        "counts": {"study_sites": EXPECTED_STUDY_SITE_COUNT, "arrivals": len(arrivals)},
        "public_analysis_label_policy": {"A": "A 區分析域"},
        "forcing_years": [int(year) for year in config.inputs.years],
        "observation_years": (
            [int(year) for year in selection_config.observation_years]
            if selection_config is not None and selection_config.observation_years is not None
            else [int(year) for year in config.inputs.years]
        ),
        "arrival_selection_policy": observation_policy,
    }
    if selection_config is not None:
        provenance_extra["arrival_selection_contract"] = {
            "policy": observation_policy,
            "core_count": int(selection_config.core_count),
            "observation_years": (
                [int(year) for year in selection_config.observation_years]
                if selection_config.observation_years is not None
                else [int(year) for year in config.inputs.years]
            ),
            "replicates": (
                int(selection_config.replicates)
                if selection_config.replicates is not None
                else None
            ),
            "event_supplement_count": int(selection_config.event_supplement_count),
        }
        if observation_policy == "observation_year_stratified_48_plus_2_v1":
            # manifests loader 會從這個 nested object 重播 forcing／observation 年份、
            # policy 與 replicate 契約；root 仍保留平坦欄位供舊工具讀取。
            provenance_extra["arrival_time_selection"] = {
                "policy": observation_policy,
                "forcing_years": [int(year) for year in config.inputs.years],
                "observation_years": (
                    [int(year) for year in selection_config.observation_years]
                    if selection_config.observation_years is not None
                    else [int(year) for year in config.inputs.years]
                ),
                "replicates": (
                    int(selection_config.replicates)
                    if selection_config.replicates is not None
                    else None
                ),
                "core_count": int(selection_config.core_count),
                "event_supplement_count": int(selection_config.event_supplement_count),
            }
    if pilot_selection is not None:
        provenance_extra["pilot_selection_scope"] = dict(pilot_selection)
    if horizon_settings is not None and horizon_settings.is_generic:
        # arrival 的 48+2 分層方法仍然不變；這裡額外保存它使用哪一套共同回溯母體，
        # 讓 7／30 執行設定能追溯到同一批通過 support gate 的 arrival，而不是只看
        # selection method 名稱猜測窗口長度。
        provenance_extra["shared_arrival_horizon"] = {
            "policy": GENERIC_HORIZON_POLICY_ID,
            "method_id": GENERIC_HORIZON_METHOD_ID,
            "support_days": horizon_settings.support_days,
            "requested_max_backtrack_days": horizon_settings.requested_days,
        }
    if bed_config is not None:
        if horizon_settings is None or not horizon_settings.bed_residence_enabled:
            raise InputDerivationError("bed residence arrival manifest 缺少共同 horizon 設定")
        offsets = sample_bed_residence_age_hours(
            maximum_age_days=bed_config.maximum_age_days,
            sample_count=bed_config.sample_count_per_site,
            seed=bed_config.sampling_seed,
        )
        provenance_extra["bed_residence_sampling"] = {
            "policy_id": BED_RESIDENCE_POLICY_ID,
            "sampling_policy_id": BED_RESIDENCE_SAMPLING_POLICY_ID,
            "maximum_age_days": bed_config.maximum_age_days,
            "sample_count_per_site": bed_config.sample_count_per_site,
            "sampling_seed": bed_config.sampling_seed,
            "shared_age_offsets_across_sites": bed_config.shared_age_offsets_across_sites,
            "age_offsets_hours_sha256": sha256(canonical_json_bytes(list(offsets))).hexdigest(),
            "selection_support_days": horizon_settings.selection_support_days,
            "runtime_support_days": horizon_settings.runtime_support_days,
            "pre_window_policy": bed_config.pre_window_policy,
            "observation_anchor_selection_method_id": ARRIVAL_SELECTION_METHOD_ID,
        }
    return {
        "manifest_kind": "arrival_time_manifest",
        "schema_version": (
            BED_RESIDENCE_INPUT_SCHEMA_VERSION
            if bed_config is not None
            else DERIVED_INPUT_SCHEMA_VERSION
        ),
        "status": "approved" if strict else "generated",
        "design_version": config.design_version,
        "time_standard": "UTC",
        "selection_method_id": method_id,
        "provenance": _provenance(
            method_id=method_id,
            source_hashes=source_hashes,
            **provenance_extra,
        ),
        "records": [asdict(item) for item in arrivals],
    }


def _gap_safe_payload(
    *,
    config: ProjectConfig,
    ocm_by_region: Mapping[str, _ProductData],
    arrivals: Sequence[ArrivalTime],
    expected_axis: np.ndarray,
    source_hashes: Mapping[str, str],
    max_backtrack_days: float,
    strict: bool,
    horizon_settings: HorizonSettings | None = None,
    pilot_horizon_overrides: Mapping[str, float] | None = None,
    pilot_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """建立每一 arrival 的 inclusive backward horizon 與 gap crossing 證據。

    一般 arrival 沿用 config／baseline 的 ``max_backtrack_days``；pilot 明示 replacement
    可由 arrival identity 指定較短的 1 日視窗。每筆 record 都保存實際 horizon，故
    2024-01-02T01:00:00Z 會明確呈現 2024-01-01T01:00:00Z 至自身共 25 個節點，不能
    把 global 7 日預設誤讀成該展示案例的回推期。
    """

    records: list[dict[str, Any]] = []
    expected_set = {int(value) for value in expected_axis}
    overrides = {str(key): float(value) for key, value in (pilot_horizon_overrides or {}).items()}

    if horizon_settings is not None and horizon_settings.is_generic:
        # generic shared horizon 的每筆窗口都用共同 support 上限計算；pilot replacement
        # 是另一個版本化流程，若混入此母體會讓「7 日與 30 日只差執行設定」失去同一
        # arrival 分母，因此在任何資料列建立前直接拒絕。expected_axis 只提供全域期別，
        # 實際可用節點由每個 OCM canonical axis 的 bounds／gaps 重新計算，避免相信
        # gap manifest 自己宣稱的 missing 清單。
        if overrides or pilot_selection is not None:
            raise InputDerivationError("generic shared horizon 不得混入 pilot arrival window")
        support_days = (
            horizon_settings.runtime_support_days
            if horizon_settings.bed_residence_enabled
            else horizon_settings.support_days
        )
        if support_days is None:
            raise InputDerivationError("generic shared horizon 缺少 runtime 正整日支援")
        if not math.isclose(
            float(max_backtrack_days), float(support_days), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise InputDerivationError(
                "generic shared horizon 的 max_backtrack_days 必須等於 runtime support_days"
            )
        selection_support_days = horizon_settings.selection_support_days
        if selection_support_days is None:
            raise InputDerivationError("generic shared horizon 缺少 selection support")
        expected_start_ns = int(expected_axis[0])
        expected_end_ns = int(expected_axis[-1])
        site_region = {site.study_site_id: site.analysis_region_id for site in config.study_sites}
        for arrival in sorted(arrivals, key=lambda item: (item.study_site_id, item.time_utc_ns)):
            region = site_region.get(arrival.study_site_id)
            if region is None or region not in ocm_by_region:
                raise InputDerivationError(
                    f"generic shared horizon 找不到 arrival 對應的 OCM region：{arrival.study_site_id}"
                )
            product = ocm_by_region[region]
            try:
                window = build_horizon_window(
                    arrival.time_utc_ns,
                    support_days,
                    expected_start_ns=expected_start_ns,
                    expected_end_ns=expected_end_ns,
                    context=f"arrival[{arrival.arrival_time_id}]",
                )
                coverage = compute_horizon_coverage(
                    window,
                    expected_start_ns=expected_start_ns,
                    expected_end_ns=expected_end_ns,
                    canonical_start_ns=int(product.canonical.time_utc_ns[0]),
                    canonical_end_ns=int(product.canonical.time_utc_ns[-1]),
                    canonical_gaps=product.canonical.gaps,
                )
                selection_window = None
                selection_coverage = None
                if horizon_settings.bed_residence_enabled:
                    observation_ns = arrival.metadata.get("observation_time_utc_ns")
                    selection_window = build_horizon_window(
                        observation_ns,
                        selection_support_days,
                        expected_start_ns=expected_start_ns,
                        expected_end_ns=expected_end_ns,
                        context=f"observation[{arrival.arrival_time_id}]",
                    )
                    selection_coverage = compute_horizon_coverage(
                        selection_window,
                        expected_start_ns=expected_start_ns,
                        expected_end_ns=expected_end_ns,
                        canonical_start_ns=int(product.canonical.time_utc_ns[0]),
                        canonical_end_ns=int(product.canonical.time_utc_ns[-1]),
                        canonical_gaps=product.canonical.gaps,
                    )
            except (HorizonContractError, IndexError, ValueError) as exc:
                raise InputDerivationError(
                    f"generic shared horizon 無法建立 {arrival.arrival_time_id} 的 support window：{exc}"
                ) from exc
            if (
                coverage.crossed_gap
                or (selection_coverage is not None and selection_coverage.crossed_gap)
            ) and strict:
                raise InputDerivationError(
                    f"generic shared/runtime horizon 遇到未支援節點：{arrival.arrival_time_id}"
                )
            missing = coverage.missing_time_ns
            record = {
                    "arrival_time_id": arrival.arrival_time_id,
                    "study_site_id": arrival.study_site_id,
                    "analysis_region_id": region,
                    "flow_domain_id": product.flow_domain_id,
                    "arrival_time_utc": _utc_string(window.arrival_time_ns),
                    "horizon_start_utc": _utc_string(window.start_time_ns),
                    "horizon_end_utc": _utc_string(window.end_time_ns),
                    "max_backtrack_days": int(support_days),
                    "support_days": int(support_days),
                    "expected_step_count": window.expected_step_count,
                    "supported_step_count": coverage.supported_step_count,
                    "crossed_gap": coverage.crossed_gap,
                    "missing_utc": [_utc_string(value) for value in missing],
                    "time_support_policy": (
                        BED_RESIDENCE_HORIZON_POLICY_ID
                        if horizon_settings.bed_residence_enabled
                        else GENERIC_HORIZON_POLICY_ID
                    ),
                }
            if horizon_settings.bed_residence_enabled:
                assert selection_window is not None and selection_coverage is not None
                record.update(
                    {
                        "observation_time_utc": _utc_string(selection_window.arrival_time_ns),
                        "selection_horizon_start_utc": _utc_string(selection_window.start_time_ns),
                        "selection_horizon_end_utc": _utc_string(selection_window.end_time_ns),
                        "selection_support_days": int(selection_support_days),
                        "selection_expected_step_count": selection_window.expected_step_count,
                        "selection_supported_step_count": selection_coverage.supported_step_count,
                        "selection_crossed_gap": selection_coverage.crossed_gap,
                        "selection_missing_utc": [
                            _utc_string(value) for value in selection_coverage.missing_time_ns
                        ],
                    }
                )
            records.append(record)
        crossed_any = any(
            item["crossed_gap"]
            or item.get("selection_crossed_gap", False)
            for item in records
        )
        status = (
            "generated"
            if crossed_any
            else ("approved" if strict else "generated")
        )
        policy_id = (
            BED_RESIDENCE_HORIZON_POLICY_ID
            if horizon_settings.bed_residence_enabled
            else GENERIC_HORIZON_POLICY_ID
        )
        method_id = (
            BED_RESIDENCE_HORIZON_METHOD_ID
            if horizon_settings.bed_residence_enabled
            else GENERIC_HORIZON_METHOD_ID
        )
        root_payload: dict[str, Any] = {
            "manifest_kind": "ocm_gap_safe_arrival_horizon_manifest",
            "schema_version": (
                BED_RESIDENCE_INPUT_SCHEMA_VERSION
                if horizon_settings.bed_residence_enabled
                else DERIVED_INPUT_SCHEMA_VERSION
            ),
            "status": status,
            "design_version": config.design_version,
            "time_standard": "UTC",
            "policy": policy_id,
            "max_backtrack_days": int(support_days),
            "support_days": int(support_days),
            "requested_max_backtrack_days": horizon_settings.requested_days,
            "provenance": _provenance(
                method_id=method_id,
                source_hashes=source_hashes,
                expected_time_count=int(expected_axis.size),
                support_days=int(support_days),
                requested_max_backtrack_days=horizon_settings.requested_days,
                gap_policy="canonical OCM bounds and gaps; no nearest/zero fill",
            ),
            "records": records,
        }
        if horizon_settings.bed_residence_enabled:
            root_payload["selection_support_days"] = int(selection_support_days)
            root_payload["runtime_support_days"] = int(support_days)
            root_payload["provenance"].update(
                {
                    "selection_support_days": int(selection_support_days),
                    "runtime_support_days": int(support_days),
                    "selection_anchor": "observation_time_utc_ns",
                    "runtime_anchor": "deposition_time_utc_ns",
                    "age_sampling_policy_id": BED_RESIDENCE_SAMPLING_POLICY_ID,
                }
            )
        return root_payload

    def horizon_steps_for(value: float) -> int:
        """把日數轉成正整數逐時步數，拒絕半日或非有限輸入。"""

        steps = int(round(value * 24.0))
        if steps < 1 or not math.isclose(steps / 24.0, value, rel_tol=0.0, abs_tol=1e-12):
            raise InputDerivationError("gap-safe baseline 目前只接受整日 max_backtrack_days")
        return steps

    horizon_steps_for(float(max_backtrack_days))
    for override in overrides.values():
        horizon_steps_for(override)
    site_region = {site.study_site_id: site.analysis_region_id for site in config.study_sites}
    available_by_region = {
        region: {int(value) for value in product.canonical.time_utc_ns}
        for region, product in ocm_by_region.items()
    }
    for arrival in sorted(arrivals, key=lambda item: (item.study_site_id, item.time_utc_ns)):
        region = site_region[arrival.study_site_id]
        product = ocm_by_region[region]
        effective_days = overrides.get(arrival.arrival_time_id, float(max_backtrack_days))
        horizon_steps = horizon_steps_for(effective_days)
        start_ns = int(arrival.time_utc_ns) - horizon_steps * _UTC_HOUR_NS
        horizon = np.arange(start_ns, int(arrival.time_utc_ns) + _UTC_HOUR_NS, _UTC_HOUR_NS, dtype=np.int64)
        available_set = available_by_region[region]
        missing = [
            int(value)
            for value in horizon
            if int(value) not in expected_set or int(value) not in available_set
        ]
        records.append(
            {
                "arrival_time_id": arrival.arrival_time_id,
                "study_site_id": arrival.study_site_id,
                "analysis_region_id": region,
                "flow_domain_id": product.flow_domain_id,
                "arrival_time_utc": _utc_string(arrival.time_utc_ns),
                "horizon_start_utc": _utc_string(start_ns),
                "horizon_end_utc": _utc_string(arrival.time_utc_ns),
                "max_backtrack_days": effective_days,
                "expected_step_count": int(horizon.size),
                "supported_step_count": int(horizon.size - len(missing)),
                "crossed_gap": bool(missing),
                "missing_utc": [_utc_string(value) for value in missing],
                "time_support_policy": "inclusive_observed_ocm_only_no_cross_gap",
            }
        )
    if any(item["crossed_gap"] for item in records):
        status = "generated"
    else:
        status = "approved" if strict else "generated"
    provenance_extra: dict[str, Any] = {
        "expected_time_count": int(expected_axis.size),
        "gap_policy": "known OCM gap cannot be crossed; no nearest/zero fill",
    }
    if pilot_selection is not None:
        provenance_extra["pilot_selection_scope"] = dict(pilot_selection)
    return {
        "manifest_kind": "ocm_gap_safe_arrival_horizon_manifest",
        "schema_version": DERIVED_INPUT_SCHEMA_VERSION,
        "status": status,
        "design_version": config.design_version,
        "time_standard": "UTC",
        "policy": "7_day_gap_safe_baseline",
        "max_backtrack_days": max_backtrack_days,
        "provenance": _provenance(
            method_id="server_v3_ocm_gap_safe_arrival_horizon_v1",
            source_hashes=source_hashes,
            **provenance_extra,
        ),
        "records": records,
    }


def _runtime_spatial_support_contracts_by_region(config: ProjectConfig) -> dict[str, dict[str, Any]]:
    """建立 A 區 v3 runtime spatial-support policy 的可追溯設定快照。

    這份快照只記錄研究範圍、no-expansion 狀態與既有停止型 runtime controls；它不計算
    也不聲稱三套產品有任何共同格網 margin。config schema 已先驗證 domain ID、固定
    bbox、兩站幾何半徑、每 stage fail-closed policy、資料缺口與外層邊界設定，這裡將
    同一組值複製到 forcing inventory、artifact source bindings 與 release trace。
    未採用 A 區 v3 policy 的舊設定回傳空 mapping，避免改變 legacy artifact 語意。
    """

    domains = [
        domain
        for domain in config.domains
        if domain.formal_domain_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1
    ]
    if not domains:
        return {}
    if len(domains) != 1 or domains[0].analysis_region_id != "A":
        raise InputDerivationError("A 區 runtime spatial-support policy 的 domain 綁定不唯一")
    domain = domains[0]
    return {
        "A": {
            "analysis_region_id": "A",
            "formal_domain_policy": domain.formal_domain_policy,
            "flow_domain_id": domain.formal_release_flow_domain_id,
            "bbox_lon_lat": list(domain.bbox_lon_lat),
            "runtime_spatial_support_policy": domain.runtime_spatial_support_policy,
            "formal_release_domain_status": domain.formal_release_domain_status,
            "stop_at_data_gap": config.boundaries.stop_at_data_gap,
            "stop_at_forcing_start": config.boundaries.stop_at_forcing_start,
            "flow_domain_open_boundary": config.boundaries.flow_domain_open_boundary,
        }
    }


def _runtime_spatial_support_contract_errors(
    config: ProjectConfig,
    *,
    forcing_inventory: Mapping[str, Any],
    source_bindings: Mapping[str, Any] | None,
) -> list[str]:
    """確認 v3 inventory、產品記錄與 artifact source binding 共用同一停止政策。

    比對只檢查設定宣告與不可變 provenance；它不等價於預先建立共同空間 margin 證據。
    A 區三項產品都必須記錄同一 policy/status，舊的兩格共同 margin key 一律拒絕；
    runtime 仍需在每個粒子速度 stage 依實際位置與時刻驗證 forcing。
    """

    expected = _runtime_spatial_support_contracts_by_region(config)
    if not expected:
        return []
    errors: list[str] = []
    if forcing_inventory.get("runtime_spatial_support_contracts_by_region") != expected:
        errors.append("forcing_inventory_runtime_spatial_support_contract_mismatch")
    if source_bindings is None:
        errors.append("artifact_runtime_spatial_support_source_binding_missing")
    elif source_bindings.get("runtime_spatial_support_contracts_by_region") != expected:
        errors.append("artifact_runtime_spatial_support_source_binding_mismatch")

    products = forcing_inventory.get("products")
    a_products = [
        item
        for item in products
        if isinstance(item, Mapping) and item.get("analysis_region_id") == "A"
    ] if isinstance(products, list) else []
    if len(a_products) != 3:
        errors.append("forcing_inventory_runtime_spatial_support_product_count_invalid")
        return errors
    a_contract = expected["A"]
    old_margin_fields = {
        "minimum_common_forcing_margin_grid_cells",
        "margin_required_for_forcings",
        "common_forcing_support_status",
    }
    for item in a_products:
        grid_metadata = item.get("grid_metadata")
        if not isinstance(grid_metadata, Mapping):
            errors.append("forcing_inventory_runtime_spatial_support_grid_metadata_missing")
            continue
        if (
            grid_metadata.get("formal_domain_policy") != a_contract["formal_domain_policy"]
            or grid_metadata.get("runtime_spatial_support_policy")
            != a_contract["runtime_spatial_support_policy"]
            or grid_metadata.get("formal_release_domain_status")
            != a_contract["formal_release_domain_status"]
        ):
            errors.append("forcing_inventory_runtime_spatial_support_product_binding_mismatch")
        if old_margin_fields & set(grid_metadata):
            errors.append("forcing_inventory_unmeasured_common_margin_claim_present")
    return errors


def _artifact_source_bindings(
    config: ProjectConfig,
    *,
    pilot_selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """建立 artifact index 的來源設定、A 區空間政策與 pilot 範圍綁定。

    ``config_hash`` 是來源 YAML 的正規化語意摘要；A 區 v3 則額外保存固定 flow-domain、
    bbox、站點來源 ID、no-expansion 狀態與逐 stage 停止控制。正式 release 的既有
    ``source_config_hash`` 與 artifact-index SHA-256 會鎖定這筆 source binding，不再另建
    一份可能與來源設定分歧的 release-binding map。pilot 選取範圍只作 provenance，不能
    把 generated artifact 轉成 formal approved，也不會升格 24 小時 DEMO 到正式母體。
    """

    source_bindings: dict[str, Any] = {
        "config_hash": config.config_hash(),
        "ocm_native_root_token": config.inputs.ocm_native_root_env,
        "ocm_surface_root_token": config.inputs.ocm_surface_root_env,
        "nww_analysis_root_token": config.inputs.nww_analysis_root_env,
    }
    runtime_support_contracts = _runtime_spatial_support_contracts_by_region(config)
    if runtime_support_contracts:
        # 將設定中的版本化契約原樣放入 artifact index；它說明 runtime 必須怎麼停止，
        # 不聲稱 accepted products 的共同 forcing margin 曾經被預先量測或核准。
        region_a = next(domain for domain in config.domains if domain.analysis_region_id == "A")
        source_bindings.update(
            {
                "formal_domain_policy": region_a.formal_domain_policy,
                "formal_domain_id": region_a.formal_release_flow_domain_id,
                "formal_domain_bbox_lon_lat": list(region_a.bbox_lon_lat),
                "formal_domain_status": region_a.formal_release_domain_status,
                "runtime_spatial_support_policy": region_a.runtime_spatial_support_policy,
                "runtime_spatial_support_contracts_by_region": runtime_support_contracts,
                "formal_site_ids": {
                    site.study_site_id: site.formal_release_flow_domain_id
                    for site in config.study_sites
                    if site.analysis_region_id == "A"
                },
            }
        )
    if pilot_selection is not None:
        source_bindings["pilot_selection_scope"] = dict(pilot_selection)
    return source_bindings


def _forcing_inventory_payload(
    *,
    config: ProjectConfig,
    products_by_region: Mapping[str, tuple[_ProductData, ...]],
    preflight_reports: Sequence[PreflightReport],
    expected_axis: np.ndarray,
    strict: bool,
) -> dict[str, Any]:
    """將 preflight 與檔案 fingerprint 合併成 release 可引用的 forcing inventory。"""

    products: list[dict[str, Any]] = []
    domain_by_region = {domain.analysis_region_id: domain for domain in config.domains}
    has_v3_policy = any(
        domain.formal_domain_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1
        for domain in config.domains
    )
    runtime_support_contracts = _runtime_spatial_support_contracts_by_region(config)
    for region in sorted(products_by_region):
        # products_by_region 的 tuple 固定包含 ``ocm_native``、``ocm_surface`` 與
        # ``nww3_analysis``；逐一展開可保留三套產品各自的時間軸、grid schema、來源
        # provenance 與 structural fingerprint。surface 雖不供 runtime 三維取樣，仍是
        # arrival selector 的正式 accepted input，不能只在檔案系統中存在而不入 inventory。
        region_products = products_by_region[region]
        actual_ids = {product.flow_domain_id for product in region_products}
        if len(actual_ids) != 1:
            raise InputDerivationError(
                f"forcing inventory 的三套產品必須共用同一 actual flow-domain ID：{region}"
            )
        domain = domain_by_region.get(region)
        if domain is None:
            raise InputDerivationError(f"forcing inventory 找不到 config domain：{region}")
        actual_flow_domain_id = next(iter(actual_ids))
        resolved_bbox = _authoritative_flow_domain_bbox_lon_lat(domain, actual_flow_domain_id)
        bbox_registration = _flow_domain_bbox_registration(domain, actual_flow_domain_id)
        for product in region_products:
            axis = product.canonical
            expected = np.asarray(expected_axis, dtype=np.int64)
            available = np.isin(expected, axis.time_utc_ns)
            grid_metadata = {
                "schema_version": product.grid_metadata.get("cache_schema_version")
                or product.grid_metadata.get("schema_version"),
                "bbox_lon_lat": list(resolved_bbox),
                "bbox_registration": bbox_registration,
            }
            if domain.formal_domain_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1:
                # 每項 accepted product 都攜帶同一個 no-expansion/runtime-stage policy
                # 與設定狀態，方便逐產品追溯。這些欄位不是空間支援量測結果，也不提升
                # generated/approved inventory 自身的 status。
                grid_metadata.update(
                    {
                        "formal_domain_policy": domain.formal_domain_policy,
                        "runtime_spatial_support_policy": domain.runtime_spatial_support_policy,
                        "formal_release_domain_status": domain.formal_release_domain_status,
                    }
                )
            products.append(
                {
                    "analysis_region_id": region,
                    "flow_domain_id": product.flow_domain_id,
                    "product": product.product,
                    "schema_major": 1 if product.product == "nww3_analysis" else 3,
                    "root_token": product.root_token,
                    "fingerprint_policy": {
                        "kind": "accepted_product_structural_fingerprint",
                        "large_npy": "size_and_npy_header_shape_dtype_only",
                        "content_hashed_files": [
                            "grid/metadata.json",
                            "months/*/metadata.json",
                            "months/*/time_utc_ns.npy",
                        ],
                        "deep_content_audit": "optional_not_run_by_default",
                    },
                    "grid_metadata": grid_metadata,
                    "metadata_provenance": {
                        "grid": _metadata_provenance(product.grid_metadata),
                        "months": {
                            month.label: _metadata_provenance(month.metadata) for month in product.months
                        },
                    },
                    "months": [
                        {
                            "month": month.label,
                            "path": f"${product.root_token}/{product.flow_domain_id}/months/{month.label}",
                            "time_count": int(month.time_ns.size),
                            "time_start_utc": _utc_string(int(month.time_ns[0])),
                            "time_end_utc": _utc_string(int(month.time_ns[-1])),
                        }
                        for month in product.months
                    ],
                    "canonical_time": {
                        "policy": axis.policy,
                        "expected_timestep_hours": axis.expected_timestep_hours,
                        "input_time_count": axis.input_time_count,
                        "canonical_time_count": int(axis.time_utc_ns.size),
                        "reordered_time_step_count": axis.reordered_time_step_count,
                        "dropped_duplicate_time_step_count": axis.dropped_duplicate_time_step_count,
                        "expected_period_time_count": int(expected.size),
                        "available_period_time_count": int(np.count_nonzero(available)),
                        "missing_period_time_count": int(expected.size - np.count_nonzero(available)),
                        "continuous_hourly": bool(np.array_equal(axis.time_utc_ns, expected)),
                        "time_start_utc": _utc_string(int(axis.time_utc_ns[0])),
                        "time_end_utc": _utc_string(int(axis.time_utc_ns[-1])),
                        "time_sha256": sha256(
                            np.asarray(axis.time_utc_ns, dtype="<i8").tobytes()
                        ).hexdigest(),
                        "gaps": [
                            {
                                "before_utc": _utc_string(item.before_utc_ns),
                                "after_utc": _utc_string(item.after_utc_ns),
                                "gap_hours": item.gap_hours,
                                "missing_step_count": item.missing_step_count,
                            }
                            for item in axis.gaps
                        ],
                    },
                    "files": list(product.source_file_records),
                    "product_fingerprint_sha256": _array_record_hash(product),
                }
            )
    source_hashes = {
        f"preflight:{index}": sha256(canonical_json_bytes(report.to_dict())).hexdigest()
        for index, report in enumerate(preflight_reports)
    }
    provenance_extra: dict[str, Any] = {
        # 這是建置時載入的 config 語意 hash；release config 之後會增加 immutable
        # manifest path 與 approval 欄位，因此不把它誤當成 release YAML bytes hash。
        "config_hash": config.config_hash(),
        "public_analysis_label_policy": {"A": "A 區分析域"},
    }
    if has_v3_policy:
        provenance_extra["formal_domain_policy_by_region"] = {
            region: domain_by_region[region].formal_domain_policy for region in sorted(domain_by_region)
        }
        provenance_extra["runtime_spatial_support_contracts_by_region"] = runtime_support_contracts
    return {
        "manifest_kind": "forcing_inventory",
        "schema_version": DERIVED_INPUT_SCHEMA_VERSION,
        # OCM 的全期缺時可由同一份 release 中的 gap-safe arrival/horizon component
        # 合法承接；因此 inventory 的核准條件是 schema、grid、月份與 fingerprint 已
        # 完成驗證，而不是要求 OCM canonical 軸盲目補成無缺口。NWW 全期完整性另由
        # nww_full_hourly component gate，arrival horizon 是否跨缺口則由 gap component gate。
        "status": "approved" if strict else "generated",
        "design_version": config.design_version,
        "time_standard": "UTC",
        "scope": "ocm_native_schema_3_ocm_surface_schema_3_and_nww3_analysis_schema_1_four_flow_domains",
        "expected_flow_domain_count": EXPECTED_FLOW_DOMAIN_COUNT,
        "expected_study_site_count": EXPECTED_STUDY_SITE_COUNT,
        "expected_period": {
            "start_utc": _utc_string(int(expected_axis[0])),
            "end_utc": _utc_string(int(expected_axis[-1])),
            "hourly_step_count": int(expected_axis.size),
        },
        **(
            {"runtime_spatial_support_contracts_by_region": runtime_support_contracts}
            if runtime_support_contracts
            else {}
        ),
        "products": products,
        "preflight_reports": [report.to_dict() for report in preflight_reports],
        "provenance": _provenance(
            method_id="server_v3_forcing_inventory_and_fingerprint_v1",
            source_hashes=source_hashes,
            **provenance_extra,
        ),
    }


def _nww_full_hourly_payload(
    *,
    config: ProjectConfig,
    nww_by_region: Mapping[str, _ProductData],
    expected_axis: np.ndarray,
    source_hashes: Mapping[str, str],
    strict: bool,
) -> dict[str, Any]:
    """建立 NWW manifest，明確證明四域是否有連續 17,544 個 UTC。"""

    domains: list[dict[str, Any]] = []
    for region in sorted(nww_by_region):
        product = nww_by_region[region]
        equal = bool(np.array_equal(product.canonical.time_utc_ns, expected_axis))
        domains.append(
            {
                "analysis_region_id": region,
                "flow_domain_id": product.flow_domain_id,
                "product": "nww3_analysis",
                "schema_major": 1,
                "expected_time_count": int(expected_axis.size),
                "actual_canonical_time_count": int(product.canonical.time_utc_ns.size),
                "continuous_hourly_utc": equal,
                "is_complete_17544_hourly_utc": equal
                and int(expected_axis.size) == EXPECTED_NWW_HOURLY_STEPS,
                "time_start_utc": _utc_string(int(product.canonical.time_utc_ns[0])),
                "time_end_utc": _utc_string(int(product.canonical.time_utc_ns[-1])),
                "time_sha256": sha256(
                    np.asarray(product.canonical.time_utc_ns, dtype="<i8").tobytes()
                ).hexdigest(),
                "grid_metadata": {
                    "flow_domain_id": product.grid_metadata.get("flow_domain_id"),
                    "schema_version": product.grid_metadata.get("schema_version"),
                },
                "files": list(product.source_file_records),
            }
        )
    all_complete = all(item["is_complete_17544_hourly_utc"] for item in domains)
    return {
        "manifest_kind": "nww_full_hourly_analysis_manifest",
        "schema_version": DERIVED_INPUT_SCHEMA_VERSION,
        "status": "approved" if all_complete and strict else "generated",
        "design_version": config.design_version,
        "time_standard": "UTC",
        "native_time_policy": "nww3_analysis_native_available_valid_time_no_statistical_fill",
        "required_hourly_step_count": EXPECTED_NWW_HOURLY_STEPS,
        "all_domains_complete_17544": all_complete,
        "domains": domains,
        "provenance": _provenance(
            method_id="server_v3_nww3_complete_hourly_analysis_binding_v1",
            source_hashes=source_hashes,
            required_time_count=EXPECTED_NWW_HOURLY_STEPS,
            no_wave_time_imputation=True,
        ),
    }


def _write_artifact_directory(
    destination: Path,
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    config_path: str | Path,
    formal: bool,
    ocm_native_root: str | Path,
    ocm_surface_root: str | Path,
    nww_analysis_root: str | Path,
    extra_source_bindings: Mapping[str, Any] | None = None,
) -> InputDerivationResult:
    """先完成 partial 目錄與唯讀 cross-reference validator，再原子發布 immutable artifact。

    ``config_path``、``formal`` 與三個 accepted-product root 都是刻意的顯式參數；
    validator 會以它們重新檢查 component loader、source fingerprint 與 formal gate。
    驗證對象永遠是尚未 ``os.replace`` 的 hidden partial directory，因此 invalid
    component、錯配 root 或 cross-reference 不會留下可被誤用的 final release。任何
    例外都會清除 partial；成功後 final directory 仍只建立一次，維持既有不可覆寫與
    status semantics。
    """

    _assert_no_symlink_components(destination.parent, allow_missing_leaf=True)
    config_file = _assert_regular_file(config_path)
    accepted_native_root = _assert_regular_directory(ocm_native_root)
    accepted_surface_root = _assert_regular_directory(ocm_surface_root)
    accepted_nww_root = _assert_regular_directory(nww_analysis_root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"不可覆寫既有 input release 目錄：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(destination.parent, allow_missing_leaf=False)
    partial = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    bindings: dict[str, dict[str, Any]] = {}
    try:
        for kind in sorted(payloads):
            filename = ARTIFACT_FILENAMES[kind]
            record = write_canonical_json(partial / filename, payloads[kind])
            record = {"kind": kind, **record}
            bindings[filename] = record
        index_payload: dict[str, Any] = {
            "manifest_kind": "derived_input_artifact_index",
            "schema_version": DERIVED_INPUT_SCHEMA_VERSION,
            "status": "immutable",
            "artifacts": [bindings[name] for name in sorted(bindings)],
        }
        if extra_source_bindings:
            index_payload["source_bindings"] = dict(extra_source_bindings)
        index_record = write_canonical_json(partial / "artifact_index.json", index_payload)
        bindings["artifact_index.json"] = {"kind": "artifact_index", **index_record}
        # index 是目錄內容的一部分，但不把它自己的 hash 回寫進自己，避免遞迴；其 sidecar
        # 與 read_canonical_json 仍能獨立驗證。artifact_bindings.json 提供整個目錄的外部
        # closure，讓 release config 可精確綁定所有 component。
        closure = {
            "manifest_kind": "derived_input_artifact_closure",
            "schema_version": DERIVED_INPUT_SCHEMA_VERSION,
            "artifacts": [bindings[name] for name in sorted(bindings)],
        }
        closure_record = write_canonical_json(partial / "artifact_bindings.json", closure)
        bindings["artifact_bindings.json"] = {"kind": "artifact_bindings", **closure_record}
        # immutable publish 前必須驗證 partial 自身；把 validator 放在 os.replace 之後
        # 會讓失敗留下看似完成的 final 目錄，也無法滿足「驗證後才可發布」的 release
        # 契約。三個 root 明確傳入，避免 validator 回到環境變數或猜測 SERVER 路徑。
        validation = validate_input_derivatives(
            partial,
            config_path=config_file,
            formal=formal,
            ocm_native_root=accepted_native_root,
            ocm_surface_root=accepted_surface_root,
            nww_analysis_root=accepted_nww_root,
        )
        if validation.get("valid") is not True:
            errors = validation.get("errors", [])
            detail = ";".join(str(item) for item in errors[:8])
            if len(errors) > 8:
                detail += f";...共{len(errors)}項"
            raise InputDerivationError(
                "input artifact partial publish 前 cross-reference validator 失敗：" + detail
            )
        os.replace(partial, destination)
        partial = Path()
        return InputDerivationResult(
            destination=destination,
            artifact_bindings=tuple(bindings[name] for name in sorted(bindings)),
        )
    except Exception:
        if partial and str(partial) not in {"", "."}:
            shutil.rmtree(partial, ignore_errors=True)
        raise


@dataclass(frozen=True, slots=True)
class InputDerivationResult:
    """衍生輸入發布結果；只保存 destination 與 immutable artifact fingerprints。"""

    destination: Path
    artifact_bindings: tuple[Mapping[str, Any], ...]

    @property
    def paths(self) -> Mapping[str, Path]:
        """依 artifact kind 提供 component 路徑，供 release config orchestration 使用。"""

        result: dict[str, Path] = {}
        for item in self.artifact_bindings:
            kind = item.get("kind")
            if isinstance(kind, str) and kind in ARTIFACT_FILENAMES:
                result[kind] = self.destination / str(item["path"])
        return result

    def to_dict(self) -> dict[str, Any]:
        """回傳不含絕對 SERVER root 的 CLI-safe summary。"""

        return {
            "destination": str(self.destination),
            "artifacts": [dict(item) for item in self.artifact_bindings],
        }


def _resolve_flow_product_id(config: ProjectConfig, region: str, root: Path, *, formal: bool) -> str:
    """依 config resolver 找 source domain，並依 versioned policy 決定是否可 fallback。

    ``expanded_domain_v1`` 的舊設定仍可在非 formal build 讀取 config 明示的
    ``expanded_domain_candidate_id``，解決 release config 尚未填 formal ID 但候選目錄已
    存在的建置順序問題；它不從路徑名稱猜測任意 domain。對
    ``v3_local20km_20260909_v1``，resolver 結果固定是 v3，v3 缺失時保留該 ID 交由
    product loader 報缺產品，絕不讀取同一 root 下的 v4 candidate。
    """

    expected = resolve_flow_domain_id(config, region, formal=formal)
    if (root / expected).is_dir():
        return expected
    domain = next(item for item in config.domains if item.analysis_region_id == region)
    if domain.formal_domain_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1:
        return expected
    if domain.formal_domain_policy != FORMAL_DOMAIN_POLICY_EXPANDED_V1:
        raise InputDerivationError(f"未知 formal_domain_policy：{domain.formal_domain_policy!r}")
    candidate = domain.model_extra.get("expanded_domain_candidate_id") if domain.model_extra else None
    if not formal and isinstance(candidate, str) and candidate and (root / candidate).is_dir():
        return candidate
    return expected


def _apply_bed_residence_sampling_by_site(
    arrivals_by_site: Mapping[str, Sequence[ArrivalTime]],
    *,
    bed_residence: BedResidenceTimeConfig,
    design_version: str,
) -> dict[str, list[ArrivalTime]]:
    """把五站 observation anchors 轉為 deposition UTC，回傳供後續 builder 共用的新 mapping。

    每個 mapping value 必須是一站恰好 50 筆的 ArrivalTime；此函式只在 A 區 paired UTC
    已完成後呼叫。抽樣器只產生一次共用的 50 個整點小時 age vector，再依站點逐一套用，
    因而保留各站 identity 並讓同一 observation 時序使用同一組沉底年齡。回傳值會取代
    builder 後續使用的 arrivals_by_site，故 receptor 選點、dynamic OCM 初始條件、arrival
    manifest 與 gap evidence 都以沉底時刻為 runtime 起點；本函式不宣稱 forcing 已通過
    180/90 日 window，該證據仍由 selector 和 gap validator 建立。
    """

    age_offsets = sample_bed_residence_age_hours(
        maximum_age_days=bed_residence.maximum_age_days,
        sample_count=bed_residence.sample_count_per_site,
        seed=bed_residence.sampling_seed,
    )
    transformed: dict[str, list[ArrivalTime]] = {}
    for site_id in sorted(arrivals_by_site):
        site_arrivals = tuple(arrivals_by_site[site_id])
        if any(arrival.study_site_id != site_id for arrival in site_arrivals):
            raise InputDerivationError(
                f"bed-residence site mapping key mismatch: {site_id}"
            )
        transformed[site_id] = list(
            apply_bed_residence_sampling(
                site_arrivals,
                age_hours=age_offsets,
                sampling_seed=bed_residence.sampling_seed,
                maximum_age_days=bed_residence.maximum_age_days,
                design_version=design_version,
            )
        )
    return transformed


def build_input_derivatives(
    *,
    config_path: str | Path,
    destination: str | Path,
    ocm_native_root: str | Path | None = None,
    ocm_surface_root: str | Path | None = None,
    nww_analysis_root: str | Path | None = None,
    formal: bool = False,
    strict: bool | None = None,
    pilot_arrival_utc: Mapping[str, str] | None = None,
) -> InputDerivationResult:
    """從四域 OCM/NWW3 v3 產品建立全部 Slice 1 immutable manifests。

    參數的 root 必須由 CLI 明示或由 config 指定的環境變數注入；缺少 root 時拒絕猜測
    SERVER 路徑。OCM native、OCM surface 與 NWW analysis 三套 accepted product 都是
    必需輸入；surface cache 專供 arrival selector，native 不再掃描全域 hvel 產生 arrival
    scalar。``strict`` 預設等於 ``formal``；非 strict 允許小型 synthetic fixture 的
    constant-field arrival fallback，但仍會把 manifest status 標成 ``generated``，不會誤稱
    為 approved formal data。正式資料需有 4 domains、5 sites、100 receptors、250 arrivals、
    5,000 dynamic pairs，且 NWW manifest 必須證明完整 17,544 小時。

    ``pilot_arrival_utc`` 是唯一版本化的 pilot-only 明示入口。B、C、D 各接受一個
    已登錄站點鍵（分別為 ``hsinchu``、``houwan``、``lienchiang``），A 只能以
    ``{"gongliao": ..., "guishan": ...}`` 的 exact pair 方式提交；本次登錄值固定為
    ``2024-01-02T01:00:00Z``，並由同一版本化窗口產生 24 小時、含首尾共 25 個節點。
    它在任何 source 或 destination I/O 前拒絕 ``formal=True``；非正式 build 則先驗證
    OCM native／surface、NWW3 exact UTC、NWW 四角空間支撐與 OCM native 的 1 日
    inclusive gap-safe horizon，再 deterministic 替換對應站點的 arrival。替換不改
    五站 250 arrivals／5,000 dynamic pairs 契約，且 provenance 明示 pilot selection
    scope，供 release config 保持 ``generated``；舊 Hsinchu artifact 的 policy ID 仍可
    由唯讀驗證路徑相容讀取。
    """

    pilot_arrivals = _parse_explicit_pilot_arrivals(pilot_arrival_utc)
    if formal and pilot_arrivals:
        raise InputDerivationError(
            "formal input build 禁止 pilot-only explicit arrival；destination 尚未寫入"
        )
    config_file = _assert_regular_file(config_path)
    config = load_config(config_file, formal_release=False)
    try:
        horizon_settings = resolve_configured_horizon(config)
    except HorizonContractError as exc:
        raise InputDerivationError(f"回溯 horizon 設定無效：{exc}") from exc
    if horizon_settings.is_generic and pilot_arrivals:
        raise InputDerivationError(
            "generic shared horizon 不得同時使用 pilot_arrival_utc；請以同一母體選取 arrival"
        )
    bed_residence = config.scenarios.bed_residence_time
    if bed_residence is not None and not horizon_settings.bed_residence_enabled:
        raise InputDerivationError(
            "bed residence config 必須明示 inputs.backtrack_support_days 才能建立共同包絡"
        )
    if bed_residence is not None and pilot_arrivals:
        raise InputDerivationError(
            "random bed-residence formal population 不得混入 explicit pilot arrival replacement"
        )
    # 先在任何 forcing root 讀取前鎖定研究範圍與 source binding；新 v3 policy 可用
    # ``formal=False, strict=True`` 產生準備性 geometry/input，但不能因目錄中另有 v4
    # 就改變來源版本。
    config.assert_research_domain_policy()
    strict_mode = formal if strict is None else bool(strict)
    # generic shared 母體的 arrival 分層必須以 support 上限驗證；即使 caller 只是建立
    # generated 準備資料，也不能因日期不足而走 legacy synthetic fallback，否則 7／30
    # 比較會共用一批其實只通過短窗的 arrival。legacy 非 strict fixture 才保留 fallback。
    selection_strict = strict_mode or horizon_settings.is_generic
    native_root = _env_root(ocm_native_root, config.inputs.ocm_native_root_env)
    nww_root = _env_root(nww_analysis_root, config.inputs.nww_analysis_root_env)
    surface_root = _env_root(ocm_surface_root, config.inputs.ocm_surface_root_env)
    assert native_root is not None and nww_root is not None
    assert surface_root is not None
    months = _months_for_config(config)
    if formal and [int(year) for year in config.inputs.years] != [2024, 2025]:
        raise InputDerivationError("formal input build 的 years 必須 exact 為 [2024, 2025]")
    if (
        formal
        and not horizon_settings.is_generic
        and not math.isclose(
            float(config.boundaries.max_backtrack_days or 0.0),
            float(DEFAULT_MAX_BACKTRACK_DAYS),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise InputDerivationError("formal input build 的 gap-safe horizon 必須是 7 日")
    if (
        len(config.domains) != EXPECTED_FLOW_DOMAIN_COUNT
        or len(config.study_sites) != EXPECTED_STUDY_SITE_COUNT
    ):
        raise InputDerivationError("Slice 1 必須有四個 flow domains 與五個 study sites")
    # expected period 只由 config 月份與 hourly contract 決定；先建立這個小型 metadata 軸
    # 並檢查 generic support 的基本容量，再讀取三套大型產品，讓不可能的超長日數早退。
    expected_axis = _expected_hourly_axis(
        months, step_hours=float(config.inputs.time_axis_contract["expected_timestep_hours"])
    )
    if horizon_settings.is_generic:
        support_days = horizon_settings.support_days
        if support_days is None:
            raise InputDerivationError("generic shared horizon 缺少 support_days")
        try:
            validate_support_against_period(
                support_days,
                expected_start_ns=int(expected_axis[0]),
                expected_end_ns=int(expected_axis[-1]),
            )
        except HorizonContractError as exc:
            raise InputDerivationError(f"generic shared horizon 超出 config 資料期：{exc}") from exc
    products_by_region: dict[str, tuple[_ProductData, _ProductData, _ProductData]] = {}
    ocm_by_region: dict[str, _ProductData] = {}
    nww_by_region: dict[str, _ProductData] = {}
    preflight_reports: list[PreflightReport] = []
    for domain in sorted(config.domains, key=lambda item: item.analysis_region_id):
        region = domain.analysis_region_id
        ocm_flow_id = _resolve_flow_product_id(config, region, native_root, formal=formal)
        surface_flow_id = _resolve_flow_product_id(config, region, surface_root, formal=formal)
        nww_flow_id = _resolve_flow_product_id(config, region, nww_root, formal=formal)
        if len({ocm_flow_id, surface_flow_id, nww_flow_id}) != 1:
            raise InputDerivationError(f"OCM native/surface/NWW source flow-domain 不一致：{region}")
        ocm = _load_product(
            product="ocm_native",
            root=native_root,
            root_token=config.inputs.ocm_native_root_env,
            flow_domain_id=ocm_flow_id,
            months=months,
            config=config,
        )
        nww = _load_product(
            product="nww3_analysis",
            root=nww_root,
            root_token=config.inputs.nww_analysis_root_env,
            flow_domain_id=nww_flow_id,
            months=months,
            config=config,
        )
        surface = _load_product(
            product="ocm_surface",
            root=surface_root,
            root_token=config.inputs.ocm_surface_root_env,
            flow_domain_id=surface_flow_id,
            months=months,
            config=config,
        )
        products_by_region[region] = (ocm, surface, nww)
        ocm_by_region[region] = ocm
        nww_by_region[region] = nww
    # NWW runtime cache 在所有 accepted products 載入後一次建立，並由 arrival metric
    # location selector 與 receptor face support gate 共用。同一 analysis region 只保留
    # 一份座標軸、static mask 與月份 memory-map，避免兩條流程各自重開 24 個月份檔案。
    nww_runtime_caches = {
        region: _load_nww_runtime_cache(nww_by_region[region]) for region in sorted(nww_by_region)
    }
    # preflight 是既有低記憶體 schema/month gate；它的 finding 與本模組 fingerprint
    # 一起保存，避免只看 derived scalar 而忽略 source metadata。只呼叫一次，避免四個
    # domain loop 產生四份重複報告。當來源 ID 是已核定的 formal expanded domain，而
    # 目前 config 的 pilot resolver 仍指向 v3 時，既有 preflight 無法在不改寫 config 的
    # 情況下解析該 ID；此時保留一份明確的 ``source_id_resolved_by_input_builder`` finding，
    # 並由本模組自己的 product loader 完成實際 domain/schema/time gate。
    base_ids = {
        domain.analysis_region_id: resolve_flow_domain_id(config, domain.analysis_region_id, formal=False)
        for domain in config.domains
    }
    actual_ids = {region: product.flow_domain_id for region, product in ocm_by_region.items()}
    if actual_ids == base_ids:
        try:
            preflight_reports.append(
                run_preflight(
                    config,
                    ocm_native_root=native_root,
                    nww_analysis_root=nww_root,
                    months=months,
                    formal_release=False,
                )
            )
        except Exception:
            if strict_mode:
                raise
            preflight_reports.append(
                PreflightReport(
                    created_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    config_hash=config.config_hash(),
                    mode="development",
                    findings=[],
                )
            )
    else:
        preflight_reports.append(
            PreflightReport(
                created_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                config_hash=config.config_hash(),
                mode="formal" if strict_mode else "development",
                findings=[],
            )
        )
    # 來源 hash 只取第一個 loop 的整套產品；四域各自保留 key，供所有 component closure。
    all_products = [item for pair in products_by_region.values() for item in pair]
    source_hashes = _source_hashes_for_products(all_products)
    geometry_sources = dict(source_hashes)
    domain_payload, local_payload, open_payload, flow_polygons, local_polygons = _geometry_payloads(
        config=config,
        products_by_region=ocm_by_region,
        source_hashes=geometry_sources,
        strict=strict_mode,
    )
    # arrival metrics 以 OCM surface 的既有保守支撐建立潮位／流速，NWW 則以與 runtime
    # 相同的 exact-hour 四角 mask-aware 雙線性 gate 建立 Hs。NWW metric location 只用於
    # arrival 分層與事件指標，不是受體實際位置或整條軌跡的 runtime forcing；OCM native
    # 不在這一步讀取 hvel，避免為選 250 個 arrival 掃描全域四維陣列。
    arrivals_by_site: dict[str, list[ArrivalTime]] = {}
    arrival_contexts: dict[str, _SiteArrivalSelectionContext] = {}
    arrival_selection = config.arrival_time_selection
    observation_years = (
        tuple(int(year) for year in arrival_selection.observation_years)
        if arrival_selection is not None and arrival_selection.observation_years is not None
        else None
    )
    selection_policy = arrival_selection.policy if arrival_selection is not None else None
    selection_replicates = arrival_selection.replicates if arrival_selection is not None else None
    sites_by_region: dict[str, list[str]] = {}
    for site in config.study_sites:
        sites_by_region.setdefault(site.analysis_region_id, []).append(site.study_site_id)
    for region, site_ids in sorted(sites_by_region.items()):
        nww = nww_by_region[region]
        ocm = ocm_by_region[region]
        domain = next(item for item in config.domains if item.analysis_region_id == region)
        projection = DomainProjection(*domain.center_lonlat)
        for site_id in sorted(site_ids):
            site = next(item for item in config.study_sites if item.study_site_id == site_id)
            site_lon, site_lat = site.anchor_lonlat or domain.center_lonlat
            surface = products_by_region[region][1]
            elevation, speed = _surface_series_for_location(
                surface,
                lon=float(site_lon),
                lat=float(site_lat),
            )
            selected, context = _select_arrivals_with_nww_metric_location(
                site_id=site_id,
                anchor_lon=float(site_lon),
                anchor_lat=float(site_lat),
                local_polygon_lonlat=local_polygons[site_id],
                projection=projection,
                product=ocm,
                nww_product=nww,
                nww_cache=nww_runtime_caches[region],
                ocm_elevation=elevation,
                ocm_speed=speed,
                max_backtrack_days=horizon_settings.selection_days,
                design_version=config.design_version,
                expected_axis=expected_axis,
                strict=selection_strict,
                observation_years=observation_years,
                replicates=selection_replicates,
                selection_policy=selection_policy,
            )
            explicit = pilot_arrivals.get(site_id)
            # A 區 exact pair 必須先建立共同 UTC，再各自套用明示 replacement；若在
            # gongliao loop 內先替換，後面的 guishan paired clone 會把 gongliao 的
            # replacement metadata 當成來源，造成兩站 provenance 互相指向。B/C/D
            # 單站則在各自 selector 完成後立即替換。
            a_pair_pilot = set(pilot_arrivals) == {"gongliao", "guishan"}
            if explicit is not None and not (a_pair_pilot and site_id in {"gongliao", "guishan"}):
                selected = _replace_with_explicit_pilot_arrival(
                    site_id=site_id,
                    arrivals=selected,
                    explicit=explicit,
                    product=ocm,
                    surface_product=surface,
                    nww_product=nww,
                    nww_cache=nww_runtime_caches[region],
                    context=context,
                    expected_axis=expected_axis,
                    design_version=config.design_version,
                )
            arrivals_by_site[site_id] = selected
            arrival_contexts[site_id] = context
    # A 區共用同一套 forcing 時，只要兩站都能支援，使用同一組 UTC；站點 ID 重新 hash，
    # 但 stratum/時間保持一致，避免兩個相連站點因代表 scalar 微小差異失去可比較性。
    if "gongliao" in arrivals_by_site and "guishan" in arrivals_by_site:
        guishan_context = arrival_contexts["guishan"]
        guishan_product = ocm_by_region[
            next(site.analysis_region_id for site in config.study_sites if site.study_site_id == "guishan")
        ]
        arrivals_by_site["guishan"] = _clone_paired_a_arrivals(
            source_arrivals=arrivals_by_site["gongliao"],
            guishan_context=guishan_context,
            guishan_product=guishan_product,
            expected_axis=expected_axis,
            max_backtrack_days=horizon_settings.selection_days,
            design_version=config.design_version,
            strict=selection_strict,
        )
        if set(pilot_arrivals) == {"gongliao", "guishan"}:
            # paired clone 完成後，兩站都以自己的 baseline arrival identity 建立固定
            # pilot replacement；OCM／NWW support 仍各自重新驗證，不能只複製另一站結果。
            for pilot_site_id in ("gongliao", "guishan"):
                pilot_site = next(item for item in config.study_sites if item.study_site_id == pilot_site_id)
                pilot_region = pilot_site.analysis_region_id
                pilot_surface = products_by_region[pilot_region][1]
                arrivals_by_site[pilot_site_id] = _replace_with_explicit_pilot_arrival(
                    site_id=pilot_site_id,
                    arrivals=arrivals_by_site[pilot_site_id],
                    explicit=pilot_arrivals[pilot_site_id],
                    product=ocm_by_region[pilot_region],
                    surface_product=pilot_surface,
                    nww_product=nww_by_region[pilot_region],
                    nww_cache=nww_runtime_caches[pilot_region],
                    context=arrival_contexts[pilot_site_id],
                    expected_axis=expected_axis,
                    design_version=config.design_version,
                )
    all_arrivals = tuple(item for site in sorted(arrivals_by_site) for item in arrivals_by_site[site])
    if len(all_arrivals) != EXPECTED_ARRIVAL_COUNT:
        raise InputDerivationError(f"arrival 應有 250 筆，實際 {len(all_arrivals)}")
    if bed_residence is not None:
        # 五站先各完成 50 筆 observation-anchor selector，A 區的 paired UTC 也已定案；
        # 現在才依單一 seed 產生共用的 50 個整點沉底年齡，並把 runtime 起點轉為沉底 UTC。
        # 這個位置刻意早於 receptor 與 receptor×arrival dynamic 初始條件建置，使後續
        # OCM/NWW 驗證、實際水深與初始粒子時刻都引用同一個沉底時次。
        try:
            # helper 逐站傳入一樣的年齡向量，並以回傳 mapping 完整取代 observation
            # rows；後續 receptor/dynamic、arrival manifest 與 gap gate 因而共同使用
            # deposition UTC，不會只有 all_arrivals 改了、站點 mapping 卻仍留觀測時刻。
            arrivals_by_site = _apply_bed_residence_sampling_by_site(
                arrivals_by_site,
                bed_residence=bed_residence,
                design_version=config.design_version,
            )
            all_arrivals = tuple(
                item for site in sorted(arrivals_by_site) for item in arrivals_by_site[site]
            )
        except Exception as exc:
            raise InputDerivationError(
                f"random bed-residence observation-to-deposition transformation failed: {type(exc).__name__}"
            ) from exc
        if len(all_arrivals) != EXPECTED_ARRIVAL_COUNT:
            raise InputDerivationError("bed-residence 轉換不得改變正式 250 筆 arrival 母體")
    pilot_selection = _pilot_selection_summary(all_arrivals)
    if pilot_selection is None:
        pilot_horizon_overrides: dict[str, float] = {}
    else:
        pilot_arrival_ids = pilot_selection.get("arrival_time_ids")
        if not isinstance(pilot_arrival_ids, list):
            pilot_arrival_ids = [pilot_selection["arrival_time_id"]]
        pilot_horizon_overrides = {
            str(arrival_id): PILOT_EXPLICIT_MAX_BACKTRACK_DAYS for arrival_id in pilot_arrival_ids
        }
    # OCM native 的 zcor/elev/wetdry 與 source depth 只建立一次帶 flow-domain binding 的
    # pair cache；receptor 需要整列 wetdry 來選水平 face，dynamic 只取 selected face scalar，
    # 但兩階段共用同一批 memory-map 與 prefer-last UTC index。外層 key 統一使用
    # analysis_region_id，避免把不同 region 的產品因 flow-domain 字串碰巧相同而錯接。
    ocm_pair_caches = {
        region: _load_ocm_pair_cache(product) for region, product in sorted(ocm_by_region.items())
    }
    vertical_support_cache: dict[_VerticalSupportCacheKey, _FaceVerticalSupport] = {}
    receptor_payload, receptor_index, receptor_objects, meshes, projections = _receptor_payload(
        config=config,
        products_by_region=ocm_by_region,
        nww_products_by_region=nww_by_region,
        flow_polygons=flow_polygons,
        local_polygons=local_polygons,
        arrivals_by_site=arrivals_by_site,
        nww_runtime_caches=nww_runtime_caches,
        ocm_pair_caches=ocm_pair_caches,
        vertical_support_cache=vertical_support_cache,
        source_hashes=source_hashes,
        strict=strict_mode,
    )
    # receptor helper 以 flow-domain 共用 mesh，先把相同 immutable object 包成每 region
    # 的 binding 供 dynamic 使用。A 區兩站仍指向同一個 NativeMesh；binding 會在 dynamic
    # 取用前核對產品 flow_domain_id，避免裸 mesh 無法辨識錯綁。
    native_mesh_bindings: dict[str, _NativeMeshBinding] = {}
    for region, product in sorted(ocm_by_region.items()):
        mesh = meshes.get(product.flow_domain_id)
        if mesh is None:
            raise InputDerivationError(f"receptor 未回傳 OCM NativeMesh：{region}")
        native_mesh_bindings[region] = _NativeMeshBinding(
            flow_domain_id=product.flow_domain_id,
            mesh=mesh,
        )
    del receptor_index, meshes, projections
    arrival_payload = _arrival_payload(
        config,
        all_arrivals,
        source_hashes=source_hashes,
        strict=strict_mode,
        horizon_settings=horizon_settings,
        pilot_selection=pilot_selection,
    )
    dynamic_payload = _dynamic_initial_payload(
        config=config,
        products_by_region=ocm_by_region,
        receptors=receptor_objects,
        arrivals=all_arrivals,
        source_hashes=source_hashes,
        strict=strict_mode,
        pilot_selection=pilot_selection,
        ocm_pair_caches=ocm_pair_caches,
        native_mesh_bindings=native_mesh_bindings,
        vertical_support_cache=vertical_support_cache,
    )
    max_days = (
        float(horizon_settings.runtime_support_days)
        if horizon_settings.bed_residence_enabled and horizon_settings.runtime_support_days is not None
        else horizon_settings.selection_days
    )
    gap_payload = _gap_safe_payload(
        config=config,
        ocm_by_region=ocm_by_region,
        arrivals=all_arrivals,
        expected_axis=expected_axis,
        source_hashes=source_hashes,
        max_backtrack_days=max_days,
        strict=strict_mode,
        horizon_settings=horizon_settings,
        pilot_horizon_overrides=pilot_horizon_overrides,
        pilot_selection=pilot_selection,
    )
    inventory_payload = _forcing_inventory_payload(
        config=config,
        products_by_region=products_by_region,
        preflight_reports=preflight_reports,
        expected_axis=expected_axis,
        strict=strict_mode,
    )
    nww_payload = _nww_full_hourly_payload(
        config=config,
        nww_by_region=nww_by_region,
        expected_axis=expected_axis,
        source_hashes=source_hashes,
        strict=strict_mode,
    )
    material_payload = _material_payload(config, source_hashes=source_hashes)
    if strict_mode:
        # formal build 必須在發布前 fail closed。這裡不以「目錄已寫出」代替核准；若
        # OCM gap-safe window、NWW 完整逐時或 forcing inventory 尚未達到正式閘門，連
        # partial directory 都不發布，避免 operator 誤把 generated artifact 當 release。
        # A 區 v3 的空間政策由 config schema exact 綁定；空間位置有效性仍在 runtime 每個
        # 速度取樣階段 fail closed，因此不再以未量測的共同網格 margin 當成建置 blocker。
        formal_blockers = [
            f"{kind}_not_approved"
            for kind, payload in (
                ("forcing_inventory", inventory_payload),
                ("ocm_gap_safe_arrival_horizon", gap_payload),
                ("nww_full_hourly", nww_payload),
            )
            if payload.get("status") != "approved"
        ]
        if formal_blockers:
            raise InputDerivationError("formal input build 未通過：" + ";".join(formal_blockers))
    payloads = {
        "forcing_inventory": inventory_payload,
        "ocm_gap_safe_arrival_horizon": gap_payload,
        "nww_full_hourly": nww_payload,
        "domain_geometry": domain_payload,
        "local_geometry": local_payload,
        "open_boundary": open_payload,
        "material": material_payload,
        "receptor": receptor_payload,
        "arrival": arrival_payload,
        "initial_condition": dynamic_payload,
    }
    source_bindings = _artifact_source_bindings(config, pilot_selection=pilot_selection)
    return _write_artifact_directory(
        Path(destination),
        payloads,
        config_path=config_file,
        formal=formal,
        ocm_native_root=native_root,
        ocm_surface_root=surface_root,
        nww_analysis_root=nww_root,
        extra_source_bindings=source_bindings,
    )


def _expected_component_paths(directory: Path) -> dict[str, Path]:
    """由固定檔名建立 component 路徑，不接受 artifact index 的任意 path injection。"""

    return {kind: directory / filename for kind, filename in ARTIFACT_FILENAMES.items()}


def _validate_artifact_directory(directory: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    """驗證 artifact index、sidecar、固定檔名與檔案 hash；不讀 domain source arrays。"""

    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {}
    try:
        _assert_regular_directory(directory)
    except Exception as exc:
        return [f"directory_invalid:{type(exc).__name__}"], [], summary
    try:
        index, index_fp = read_canonical_json(directory / "artifact_index.json")
        closure, closure_fp = read_canonical_json(directory / "artifact_bindings.json")
        summary["artifact_index_sha256"] = index_fp["sha256"]
        summary["artifact_bindings_sha256"] = closure_fp["sha256"]
    except Exception as exc:
        return [f"artifact_index_invalid:{type(exc).__name__}"], [], summary
    if index.get("manifest_kind") != "derived_input_artifact_index":
        errors.append("artifact_index_kind_invalid")
    raw_artifacts = index.get("artifacts")
    if not isinstance(raw_artifacts, list):
        errors.append("artifact_index_artifacts_invalid")
        return errors, warnings, summary
    expected = _expected_component_paths(directory)
    found_kinds: set[str] = set()
    for item in raw_artifacts:
        if not isinstance(item, Mapping):
            errors.append("artifact_index_record_invalid")
            continue
        kind = item.get("kind")
        filename = item.get("path")
        if kind not in expected or filename != expected[str(kind)].name:
            errors.append(f"artifact_index_path_invalid:{kind}")
            continue
        found_kinds.add(str(kind))
        try:
            _, fingerprint = read_canonical_json(expected[str(kind)])
            for field in ("sha256", "canonical_sha256", "size_bytes"):
                if item.get(field) != fingerprint.get(field):
                    errors.append(f"artifact_hash_mismatch:{kind}:{field}")
        except Exception as exc:
            errors.append(f"artifact_unreadable:{kind}:{type(exc).__name__}")
    missing = sorted(set(expected) - found_kinds)
    errors.extend(f"artifact_missing:{kind}" for kind in missing)
    closure_records = closure.get("artifacts")
    if not isinstance(closure_records, list):
        errors.append("artifact_closure_invalid")
    else:
        index_by_path = {item.get("path"): item for item in raw_artifacts if isinstance(item, Mapping)}
        for item in closure_records:
            # artifact_index 自身不會回寫到自己的 ``artifacts`` 清單，否則會形成
            # 自我 hash 遞迴；但 closure 仍需把它列入整個目錄的完整檔案集合。
            if not isinstance(item, Mapping) or (
                item.get("path") not in index_by_path and item.get("path") != "artifact_index.json"
            ):
                errors.append("artifact_closure_mismatch")
                continue
            closure_path = directory / str(item.get("path"))
            try:
                _, closure_fingerprint = read_canonical_json(closure_path)
                for field in ("sha256", "canonical_sha256", "size_bytes"):
                    if item.get(field) != closure_fingerprint.get(field):
                        errors.append(f"artifact_closure_hash_mismatch:{item.get('path')}:{field}")
            except Exception as exc:
                errors.append(f"artifact_closure_unreadable:{item.get('path')}:{type(exc).__name__}")
    summary["component_count"] = len(found_kinds)
    return errors, warnings, summary


def _validate_source_file_bindings(
    inventory: Mapping[str, Any],
    *,
    roots_by_token: Mapping[str, Path | None],
) -> list[str]:
    """依 inventory root token 重算 metadata/time hash 與 NPY structural fingerprint。

    root token 不是固定常數：正式 SERVER 可在 config 中使用不同的環境變數名稱。驗證器
    因此先由 caller 以 config 建立 token→root mapping，再檢查 ``$TOKEN/...`` 的相對路徑。
    未提供對應 root 時仍驗證 record 的 kind/header 契約；提供 root 時只讀 metadata/time
    的實際 SHA-256，並以 memory-map 讀取大型 NPY 的 size、shape、dtype。這能偵測 metadata、
    time axis、截斷、換檔與結構變更，但不宣稱能偵測相同 size/header 的 NPY payload 內容
    替換；deep content audit 必須另行明示，不能成為正式建置的隱含成本。
    """

    errors: list[str] = []
    products = inventory.get("products")
    if not isinstance(products, list):
        return ["forcing_inventory_products_invalid"]
    for product in products:
        if not isinstance(product, Mapping):
            errors.append("forcing_inventory_product_invalid")
            continue
        token = product.get("root_token")
        root = roots_by_token.get(str(token))
        files = product.get("files")
        if not isinstance(files, list):
            errors.append("forcing_inventory_files_invalid")
            continue
        if roots_by_token and token not in roots_by_token:
            errors.append("forcing_inventory_root_token_unbound")
        for record in files:
            if not isinstance(record, Mapping):
                errors.append("forcing_inventory_file_record_invalid")
                continue
            relative = record.get("path")
            if not isinstance(token, str) or _ENV_NAME_RE.fullmatch(token) is None:
                errors.append("forcing_inventory_root_token_invalid")
                continue
            prefix = f"${token}/"
            if not isinstance(relative, str) or not relative.startswith(prefix):
                errors.append("forcing_inventory_path_token_invalid")
                continue
            # 即使 caller 沒有提供實際 root，也要先拒絕絕對路徑與 ``..``；否則一份
            # 只做 lexical 驗證的 inventory 仍可能把後續 hash 重算導向 root 外部檔案。
            relative_suffix = relative[len(prefix) :]
            relative_path = Path(relative_suffix)
            if (
                relative_path.is_absolute()
                or not relative_suffix
                or any(part in {"", ".", ".."} for part in relative_path.parts)
            ):
                errors.append("forcing_inventory_source_path_invalid")
                continue
            # 先檢查 inventory record 自身的 fingerprint schema，即使 caller 沒有提供
            # 上游 root 也不能把任意檔案或不完整 header 當成「尚未驗證但可接受」。這一
            # 層只讀小型 JSON record；真正的檔案大小、NPY header 與 metadata/time hash
            # 重算則在 root 可用時進行。
            product_name = product.get("product")
            file_kind = record.get("file_kind")
            fingerprint_kind = record.get("fingerprint_kind")
            expected_size = record.get("size_bytes")
            if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
                errors.append(f"source_size_record_invalid:{product_name}:{relative}")
                continue
            if relative_path.suffix == ".npy":
                header = record.get("npy_header")
                raw_shape = header.get("shape") if isinstance(header, Mapping) else None
                raw_dtype = header.get("dtype") if isinstance(header, Mapping) else None
                if (
                    file_kind not in {"grid_npy", "month_npy", "time_axis"}
                    or fingerprint_kind != "npy_header_structural"
                    or not isinstance(header, Mapping)
                    or set(header) != {"shape", "dtype"}
                    or not isinstance(raw_shape, list)
                    or not raw_shape
                    or any(
                        isinstance(value, bool) or not isinstance(value, int) or value < 1
                        for value in raw_shape
                    )
                    or not isinstance(raw_dtype, str)
                    or not raw_dtype.strip()
                ):
                    errors.append(f"source_npy_fingerprint_invalid:{product_name}:{relative}")
                    continue
                if file_kind == "time_axis" and (
                    not isinstance(record.get("sha256"), str)
                    or _SHA256_RE.fullmatch(str(record.get("sha256"))) is None
                ):
                    errors.append(f"source_time_fingerprint_invalid:{product_name}:{relative}")
                    continue
            elif relative_path.suffix == ".json":
                if (
                    file_kind not in {"grid_metadata", "month_metadata"}
                    or fingerprint_kind != "metadata_content_sha256"
                    or not isinstance(record.get("sha256"), str)
                    or _SHA256_RE.fullmatch(str(record.get("sha256"))) is None
                ):
                    errors.append(f"source_metadata_fingerprint_invalid:{product_name}:{relative}")
                    continue
            else:
                errors.append(f"source_file_extension_invalid:{product_name}:{relative}")
                continue
            if root is None:
                continue
            source = root / relative_suffix
            try:
                actual_size = int(_assert_regular_file(source).stat().st_size)
                if actual_size != expected_size:
                    errors.append(f"source_size_mismatch:{product_name}:{relative}")
                    continue
                if relative_path.suffix == ".npy":
                    array = _load_npy(source)
                    actual_header = {
                        "shape": [int(value) for value in array.shape],
                        "dtype": str(array.dtype),
                    }
                    if dict(record["npy_header"]) != actual_header:
                        errors.append(f"source_npy_header_mismatch:{product_name}:{relative}")
                    if file_kind == "time_axis":
                        actual_hash = _sha256_file(source)
                        if actual_hash != record.get("sha256"):
                            errors.append(f"source_time_hash_mismatch:{product_name}:{relative}")
                else:
                    actual_hash = _sha256_file(source)
                    if actual_hash != record.get("sha256"):
                        errors.append(f"source_metadata_hash_mismatch:{product_name}:{relative}")
            except Exception as exc:
                errors.append(f"source_unreadable:{product_name}:{type(exc).__name__}")
    return errors


def _validate_canonical_axis_bindings(
    inventory: Mapping[str, Any],
    *,
    roots_by_token: Mapping[str, Path | None],
) -> list[str]:
    """從已綁定月份的 ``time_utc_ns.npy`` 重建 canonical summary。

    `_validate_source_file_bindings` 已驗證每一檔的大小、header 與小檔 hash，但若只相信
    inventory 的 ``canonical_time`` summary，攻擊者仍可只改 summary、重簽 component
    sidecar，讓 gap／count 看似完整。這裡在 caller 提供實際 root 時重新讀取每個產品的
    exact flow-domain／month time axis，依同一 canonicalization policy 重建 count、排序、
    去重、bounds、gap 與 canonical hash，再逐欄對比 inventory。時間軸是小型 metadata，
    因此不會讀取大型 forcing payload；root 未提供時只保留離線結構驗證，不能把它標成
    已完成來源重建。
    """

    errors: list[str] = []
    products = inventory.get("products")
    expected_period = inventory.get("expected_period")
    if not isinstance(products, list) or not isinstance(expected_period, Mapping):
        return ["canonical_axis_rebuild_inventory_period_invalid"]
    try:
        period_start = datetime.fromisoformat(str(expected_period["start_utc"]).replace("Z", "+00:00"))
        period_end = datetime.fromisoformat(str(expected_period["end_utc"]).replace("Z", "+00:00"))
        if period_start.utcoffset() != timedelta(0) or period_end.utcoffset() != timedelta(0):
            raise ValueError("expected period 必須是 UTC")
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        expected_start_ns = int((period_start - epoch).total_seconds()) * 1_000_000_000
        expected_end_ns = int((period_end - epoch).total_seconds()) * 1_000_000_000
        raw_expected_count = expected_period["hourly_step_count"]
        if isinstance(raw_expected_count, bool) or not isinstance(raw_expected_count, int):
            raise ValueError("expected period 筆數必須是整數")
        expected_count = int(raw_expected_count)
        if (
            expected_count < 1
            or expected_end_ns < expected_start_ns
            or (expected_end_ns - expected_start_ns) % _UTC_HOUR_NS
            or (expected_end_ns - expected_start_ns) // _UTC_HOUR_NS + 1 != expected_count
        ):
            raise ValueError("expected period 不合法")
    except (KeyError, TypeError, ValueError, OverflowError):
        return ["canonical_axis_rebuild_expected_period_invalid"]

    for product in products:
        if not isinstance(product, Mapping):
            errors.append("canonical_axis_rebuild_product_invalid")
            continue
        token = product.get("root_token")
        root = roots_by_token.get(str(token))
        # 沒有實際 root 時不能把 metadata-only 的 self-consistency 說成來源驗收；
        # `_validate_source_file_bindings` 仍會檢查 token/path/header/hash 結構。
        if root is None:
            continue
        product_name = product.get("product")
        region = product.get("analysis_region_id")
        flow_domain_id = product.get("flow_domain_id")
        canonical_summary = product.get("canonical_time")
        months_payload = product.get("months")
        files = product.get("files")
        if (
            not isinstance(token, str)
            or not isinstance(product_name, str)
            or not isinstance(region, str)
            or not isinstance(flow_domain_id, str)
            or not isinstance(canonical_summary, Mapping)
            or not isinstance(months_payload, list)
            or not isinstance(files, list)
        ):
            errors.append(f"canonical_axis_rebuild_metadata_invalid:{product_name}:{region}")
            continue

        month_labels: list[str] = []
        for month in months_payload:
            if not isinstance(month, Mapping) or not isinstance(month.get("month"), str):
                errors.append(f"canonical_axis_rebuild_month_record_invalid:{product_name}:{region}")
                continue
            label = str(month["month"])
            try:
                _parse_month(label)
            except (TypeError, ValueError):
                errors.append(f"canonical_axis_rebuild_month_label_invalid:{product_name}:{label}")
                continue
            if label in month_labels:
                errors.append(f"canonical_axis_rebuild_duplicate_month:{product_name}:{label}")
            month_labels.append(label)
            expected_month_path = f"${token}/{flow_domain_id}/months/{label}"
            if month.get("path") != expected_month_path:
                errors.append(f"canonical_axis_rebuild_month_path_mismatch:{product_name}:{label}")
        month_labels.sort()
        if not month_labels:
            errors.append(f"canonical_axis_rebuild_months_empty:{product_name}:{region}")
            continue
        # inventory 的 month list 不能自己縮短成「看起來連續」的小樣本；expected period
        # 由同一份 forcing inventory 宣告，因此直接展開其涵蓋的曆月集合做 exact 比對。
        expected_month_labels: list[str] = []
        cursor = datetime(period_start.year, period_start.month, 1, tzinfo=UTC)
        last_month = datetime(period_end.year, period_end.month, 1, tzinfo=UTC)
        while cursor <= last_month:
            expected_month_labels.append(f"{cursor.year}{cursor.month:02d}")
            if cursor.month == 12:
                cursor = datetime(cursor.year + 1, 1, 1, tzinfo=UTC)
            else:
                cursor = datetime(cursor.year, cursor.month + 1, 1, tzinfo=UTC)
        if month_labels != expected_month_labels:
            errors.append(f"canonical_axis_rebuild_month_set_mismatch:{product_name}:{region}")
            continue

        expected_product_prefix = f"${token}/{flow_domain_id}/"
        for record in files:
            if not isinstance(record, Mapping):
                continue
            raw_path = record.get("path")
            if not isinstance(raw_path, str) or not raw_path.startswith(expected_product_prefix):
                errors.append(f"canonical_axis_rebuild_product_path_mismatch:{product_name}:{region}")
                break

        expected_time_paths = {
            f"${token}/{flow_domain_id}/months/{label}/time_utc_ns.npy" for label in month_labels
        }
        time_records: dict[str, Mapping[str, Any]] = {}
        for record in files:
            if not isinstance(record, Mapping) or record.get("file_kind") != "time_axis":
                continue
            path = record.get("path")
            if not isinstance(path, str):
                errors.append(f"canonical_axis_rebuild_time_path_invalid:{product_name}:{region}")
                continue
            if path in time_records:
                errors.append(f"canonical_axis_rebuild_duplicate_time_record:{product_name}:{path}")
            time_records[path] = record
        if set(time_records) != expected_time_paths:
            errors.append(f"canonical_axis_rebuild_time_month_set_mismatch:{product_name}:{region}")
            continue

        chunks: list[TimeChunk] = []
        invalid_product = False
        for label in month_labels:
            path = f"${token}/{flow_domain_id}/months/{label}/time_utc_ns.npy"
            relative_suffix = Path(flow_domain_id) / "months" / label / "time_utc_ns.npy"
            try:
                values = _load_npy(root / relative_suffix, dtype=np.dtype("int64"))
                if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0):
                    raise InputDerivationError("time axis 必須是嚴格遞增一維 int64")
                month_record = next(
                    item
                    for item in months_payload
                    if isinstance(item, Mapping) and item.get("month") == label
                )
                if (
                    month_record.get("time_count") != int(values.size)
                    or month_record.get("time_start_utc") != _utc_string(int(values[0]))
                    or month_record.get("time_end_utc") != _utc_string(int(values[-1]))
                ):
                    errors.append(f"canonical_axis_rebuild_month_summary_mismatch:{product_name}:{label}")
                chunks.append(TimeChunk(label, np.asarray(values)))
            except Exception as exc:
                errors.append(
                    f"canonical_axis_rebuild_time_unreadable:{product_name}:{path}:{type(exc).__name__}"
                )
                invalid_product = True
        if invalid_product or not chunks:
            continue
        try:
            policy = canonical_summary.get("policy")
            expected_timestep = float(canonical_summary.get("expected_timestep_hours", 1.0))
            axis = canonicalize_time_chunks(
                chunks,
                policy=policy,  # type: ignore[arg-type]
                expected_timestep_hours=expected_timestep,
            )
            actual_hash = sha256(np.asarray(axis.time_utc_ns, dtype="<i8").tobytes()).hexdigest()
            actual_gaps = [
                {
                    "before_utc": _utc_string(item.before_utc_ns),
                    "after_utc": _utc_string(item.after_utc_ns),
                    "gap_hours": item.gap_hours,
                    "missing_step_count": item.missing_step_count,
                }
                for item in axis.gaps
            ]
            period_mask = (axis.time_utc_ns >= expected_start_ns) & (axis.time_utc_ns <= expected_end_ns)
            available_period_count = int(np.count_nonzero(period_mask))
            expected_summary = {
                "policy": axis.policy,
                "expected_timestep_hours": axis.expected_timestep_hours,
                "input_time_count": axis.input_time_count,
                "canonical_time_count": int(axis.time_utc_ns.size),
                "reordered_time_step_count": axis.reordered_time_step_count,
                "dropped_duplicate_time_step_count": axis.dropped_duplicate_time_step_count,
                "expected_period_time_count": expected_count,
                "available_period_time_count": available_period_count,
                "missing_period_time_count": expected_count - available_period_count,
                "continuous_hourly": bool(
                    axis.time_utc_ns.size == expected_count
                    and int(axis.time_utc_ns[0]) == expected_start_ns
                    and int(axis.time_utc_ns[-1]) == expected_end_ns
                    and not axis.gaps
                ),
                "time_start_utc": _utc_string(int(axis.time_utc_ns[0])),
                "time_end_utc": _utc_string(int(axis.time_utc_ns[-1])),
                "time_sha256": actual_hash,
                "gaps": actual_gaps,
            }
            for field, actual in expected_summary.items():
                # expected_timestep_hours 是新 inventory 的明示欄位；legacy artifact 未
                # 保存它時以既有 hourly contract 重建，但不因缺少新欄位而改判舊產物。
                if field == "expected_timestep_hours" and field not in canonical_summary:
                    continue
                if canonical_summary.get(field) != actual:
                    errors.append(f"canonical_axis_summary_mismatch:{product_name}:{region}:{field}")
        except Exception as exc:
            errors.append(f"canonical_axis_rebuild_invalid:{product_name}:{region}:{type(exc).__name__}")
    return errors


def validate_input_derivatives(
    directory: str | Path,
    *,
    config_path: str | Path | None = None,
    config_override: ProjectConfig | None = None,
    formal: bool = False,
    ocm_native_root: str | Path | None = None,
    ocm_surface_root: str | Path | None = None,
    nww_analysis_root: str | Path | None = None,
) -> dict[str, Any]:
    """唯讀驗證 Slice 1 目錄，回傳 JSON-safe ``valid/errors/warnings/summary``。

    validator 不執行重建、不寫回輸入、不接受 manifest 內任意檔案路徑；component loader
    使用固定 config 與固定 filename。CLI 可以傳入相對 artifact directory，但函式入口
    會先以不解析 symbolic link target 的 lexical absolute normalization 建立 root，讓
    `_expected_component_paths` 與 component loader 一律收到絕對 artifact component path；
    這避免相對 manifest 被誤解成 config-relative path，同時保留既有 symlink gate 與
    不存在／非法目錄的 JSON-safe 錯誤契約。formal gate 額外要求四域、五站、100/250/5,000
    計數、NWW 17,544、所有 gap-safe horizon 不跨缺口，以及 artifact exact hash closure。
    """

    # os.path.abspath 只做目前工作目錄下的 lexical 絕對化與 normpath，不呼叫
    # realpath／resolve，因此不會繞過 _validate_artifact_directory 的 symbolic link
    # 檢查；若 directory 不存在或不是目錄，後續既有 validator 仍負責回傳 JSON-safe errors。
    root = Path(os.path.abspath(os.fspath(directory)))
    errors, warnings, summary = _validate_artifact_directory(root)
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "summary": summary}
    if config_override is not None and config_path is not None:
        raise ValueError("config_path 與 config_override 不可同時提供")
    config = config_override or (
        load_config(config_path, formal_release=False) if config_path is not None else None
    )
    paths = _expected_component_paths(root)
    try:
        forcing, _ = read_canonical_json(paths["forcing_inventory"])
        gap, _ = read_canonical_json(paths["ocm_gap_safe_arrival_horizon"])
        nww, _ = read_canonical_json(paths["nww_full_hourly"])
        material, _ = read_canonical_json(paths["material"])
        receptor, _ = read_canonical_json(paths["receptor"])
        arrival, _ = read_canonical_json(paths["arrival"])
        dynamic, _ = read_canonical_json(paths["initial_condition"])
        domain, _ = read_canonical_json(paths["domain_geometry"])
        local, _ = read_canonical_json(paths["local_geometry"])
        open_boundary, _ = read_canonical_json(paths["open_boundary"])
    except Exception as exc:
        return {
            "valid": False,
            "errors": [f"component_read_invalid:{type(exc).__name__}"],
            "warnings": warnings,
            "summary": summary,
        }
    if config is not None and not formal:
        try:
            config = _config_for_inventory_validation(
                config,
                inventory=forcing,
                formal=False,
            )
        except Exception as exc:
            errors.append(f"inventory_flow_domain_binding_invalid:{type(exc).__name__}")
    if config is not None:
        try:
            artifact_index, _ = read_canonical_json(root / "artifact_index.json")
            raw_source_bindings = artifact_index.get("source_bindings")
            source_bindings = raw_source_bindings if isinstance(raw_source_bindings, Mapping) else None
            errors.extend(
                _runtime_spatial_support_contract_errors(
                    config,
                    forcing_inventory=forcing,
                    source_bindings=source_bindings,
                )
            )
        except Exception as exc:
            errors.append(f"runtime_spatial_support_binding_invalid:{type(exc).__name__}")
    horizon_settings: HorizonSettings | None = None
    if config is not None:
        try:
            horizon_settings = resolve_configured_horizon(config)
        except HorizonContractError as exc:
            errors.append(f"horizon_config_invalid:{exc}")
    # generic policy 一旦出現就必須走完整的 shared semantic validator；即使 config 沒有
    # 提供，也不能因 unknown/missing policy 自動套回 legacy 7 日檢查。validator 會只讀
    # forcing inventory 的 expected period 與 OCM canonical bounds/gaps 重算每筆 row。
    gap_provenance = gap.get("provenance")
    gap_rows_for_policy = gap.get("records")
    generic_gap_marked = (
        gap.get("support_days") is not None
        or gap.get("requested_max_backtrack_days") is not None
        or (
            isinstance(gap_provenance, Mapping)
            and gap_provenance.get("method_id") == GENERIC_HORIZON_METHOD_ID
        )
        or (
            isinstance(gap_rows_for_policy, list)
            and any(
                isinstance(row, Mapping) and row.get("support_days") is not None
                for row in gap_rows_for_policy
            )
        )
    )
    if (
        (horizon_settings is not None and horizon_settings.is_generic)
        or gap.get("policy") == GENERIC_HORIZON_POLICY_ID
        or generic_gap_marked
    ):
        generic_result = validate_generic_gap_payload(
            gap,
            arrival,
            forcing,
            config=config,
            # 共同母體的完整支援窗是重用前提，與本次是否正式發布無關。即使只跑
            # 七日工程驗證，三十日母體較早的缺口也不能降為警告而繼續執行。
            strict=True,
        )
        errors.extend(generic_result.errors)
        warnings.extend(generic_result.warnings)
        summary.update(
            {
                "shared_horizon_policy": generic_result.summary.get("policy"),
                "shared_horizon_support_days": generic_result.summary.get("support_days"),
                "shared_horizon_arrival_records_checked": generic_result.summary.get(
                    "arrival_records_checked", 0
                ),
                "shared_horizon_arrival_records_crossing_gap": generic_result.summary.get(
                    "arrival_records_crossing_gap", 0
                ),
            }
        )
    if config is not None:
        try:
            load_material_manifest(paths["material"], config, formal=formal)
            load_receptor_manifest(paths["receptor"], config, formal=formal)
            load_arrival_time_manifest(paths["arrival"], config, formal=formal)
            load_receptor_arrival_initial_condition_manifest(
                paths["initial_condition"],
                config,
                load_receptor_manifest(paths["receptor"], config, formal=formal),
                load_arrival_time_manifest(paths["arrival"], config, formal=formal),
                formal=formal,
            )
            load_boundary_geometries(
                config,
                domain_manifest_path=paths["domain_geometry"],
                local_domain_manifest_path=paths["local_geometry"],
                open_boundary_manifest_path=paths["open_boundary"],
                formal=formal,
            )
        except Exception as exc:
            errors.append(f"component_cross_reference_invalid:{type(exc).__name__}")
    if formal:
        # strict loader 已檢查 receptor/arrival/dynamic/geometry 的 status；這裡補上
        # 沒有既有 loader 的 forcing、NWW 與 gap-safe 根節點，避免有人只把 records
        # 留完整卻把 component 標成 generated 後仍進入正式 release。
        for kind, payload in (
            ("forcing_inventory", forcing),
            ("nww_full_hourly", nww),
            ("ocm_gap_safe_arrival_horizon", gap),
        ):
            if payload.get("status") != "approved":
                errors.append(f"{kind}_not_approved")
        if config is not None:
            # formal config 的年份與 gap-safe baseline 是 release contract，而不是可由
            # inventory 的實際月份數量推測的偏好；若設定被換成別的研究期間，必須明確
            # 重新建立版本化契約，不可沿用 2024–2025 的 17,544 小時聲明。
            if [int(year) for year in config.inputs.years] != [2024, 2025]:
                errors.append("formal_years_must_be_2024_2025")
            if (horizon_settings is None or not horizon_settings.is_generic) and not math.isclose(
                float(config.boundaries.max_backtrack_days or 0.0),
                float(DEFAULT_MAX_BACKTRACK_DAYS),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                errors.append("formal_gap_safe_horizon_must_be_7_days")
    products = forcing.get("products")
    if not isinstance(products, list) or len(products) != EXPECTED_FLOW_DOMAIN_COUNT * 3:
        errors.append("forcing_inventory_product_count_invalid")
    else:
        domain_ids = {
            (item.get("analysis_region_id"), item.get("flow_domain_id"), item.get("product"))
            for item in products
            if isinstance(item, Mapping)
        }
        if len(domain_ids) != EXPECTED_FLOW_DOMAIN_COUNT * 3:
            errors.append("forcing_inventory_domain_product_binding_invalid")
        expected_products = {"ocm_native", "ocm_surface", "nww3_analysis"}
        for region in {item[0] for item in domain_ids}:
            products_for_region = {item[2] for item in domain_ids if item[0] == region}
            if products_for_region != expected_products:
                errors.append(f"forcing_inventory_product_set_invalid:{region}")
    nww_domains = nww.get("domains")
    if not isinstance(nww_domains, list) or len(nww_domains) != EXPECTED_FLOW_DOMAIN_COUNT:
        errors.append("nww_domain_count_invalid")
    elif formal and not all(
        isinstance(item, Mapping) and item.get("is_complete_17544_hourly_utc") is True for item in nww_domains
    ):
        errors.append("nww_not_complete_17544")
    receptor_rows = receptor.get("records")
    arrival_rows = arrival.get("records")
    dynamic_rows = dynamic.get("records")
    material_rows = material.get("records")
    if not isinstance(material_rows, list) or len(material_rows) != 10:
        errors.append("material_count_invalid")
    if not isinstance(receptor_rows, list) or len(receptor_rows) != EXPECTED_RECEPTOR_COUNT:
        errors.append("receptor_count_invalid")
    if not isinstance(arrival_rows, list) or len(arrival_rows) != EXPECTED_ARRIVAL_COUNT:
        errors.append("arrival_count_invalid")
    if not isinstance(dynamic_rows, list) or len(dynamic_rows) != EXPECTED_DYNAMIC_INITIAL_CONDITION_COUNT:
        errors.append("dynamic_initial_condition_count_invalid")
    if isinstance(receptor_rows, list) and isinstance(arrival_rows, list) and isinstance(dynamic_rows, list):
        receptor_ids = {row.get("receptor_id") for row in receptor_rows if isinstance(row, Mapping)}
        arrival_ids = {row.get("arrival_time_id") for row in arrival_rows if isinstance(row, Mapping)}
        pairs = {
            (row.get("receptor_id"), row.get("arrival_time_id"))
            for row in dynamic_rows
            if isinstance(row, Mapping)
        }
        expected_pairs = {
            (row.get("receptor_id"), arrival_row.get("arrival_time_id"))
            for row in receptor_rows
            if isinstance(row, Mapping)
            for arrival_row in arrival_rows
            if isinstance(arrival_row, Mapping)
            and row.get("study_site_id") == arrival_row.get("study_site_id")
        }
        if pairs != expected_pairs:
            errors.append(
                "dynamic_pair_cross_reference_invalid:"
                f"missing={len(expected_pairs - pairs)}:extra={len(pairs - expected_pairs)}"
            )
        if len(receptor_ids) != len(receptor_rows) or len(arrival_ids) != len(arrival_rows):
            errors.append("component_identifier_not_unique")
    gap_rows = gap.get("records")
    if not isinstance(gap_rows, list) or len(gap_rows) != EXPECTED_ARRIVAL_COUNT:
        errors.append("gap_safe_arrival_count_invalid")
    elif any(not isinstance(row, Mapping) for row in gap_rows):
        errors.append("gap_safe_record_invalid")
    else:
        for row in gap_rows:
            assert isinstance(row, Mapping)
            missing = row.get("missing_utc")
            expected_steps = row.get("expected_step_count")
            supported_steps = row.get("supported_step_count")
            crossed_gap = row.get("crossed_gap")
            if (
                not isinstance(missing, list)
                or not isinstance(expected_steps, int)
                or isinstance(expected_steps, bool)
                or not isinstance(supported_steps, int)
                or isinstance(supported_steps, bool)
                or expected_steps < 1
                or supported_steps < 0
                or supported_steps + len(missing) != expected_steps
                or crossed_gap is not (len(missing) > 0)
            ):
                errors.append("gap_safe_record_consistency_invalid")
        if formal and any(
            row.get("crossed_gap") is not False for row in gap_rows if isinstance(row, Mapping)
        ):
            errors.append("gap_safe_horizon_crosses_gap")
    # pilot replacement 維持 250 筆總量，但它的非潮汐 label 不是正式 48+2 strata；
    # 這裡在既有 formal loader 之外再保存一個可搜尋的明確錯誤。非正式 validator 則
    # 檢查 25-node、1 日 gap record 與 arrival metadata 是否彼此一致，避免只因 count
    # 正確就把缺少中間支援的 pilot artifact 當成可用。
    pilot_rows = (
        [
            row
            for row in arrival_rows
            if isinstance(row, Mapping)
            and isinstance(row.get("metadata"), Mapping)
            and row["metadata"].get("pilot_replacement_policy_id") in _PILOT_EXPLICIT_POLICY_IDS
        ]
        if isinstance(arrival_rows, list)
        else []
    )
    if pilot_rows:
        if formal:
            errors.append("formal_pilot_explicit_window_not_48_plus_2")
        policy_ids = {
            str(row["metadata"].get("pilot_replacement_policy_id"))
            for row in pilot_rows
            if isinstance(row.get("metadata"), Mapping)
        }
        if len(policy_ids) != 1:
            errors.append("pilot_explicit_window_policy_mixed")
        else:
            policy_id = next(iter(policy_ids))
            site_ids = tuple(sorted(str(row.get("study_site_id")) for row in pilot_rows))
            expected_scope: str | None = None
            if policy_id == PILOT_EXPLICIT_LEGACY_WINDOW_POLICY_ID:
                expected_sites = (PILOT_EXPLICIT_LEGACY_SITE_ID,)
                expected_scope = "hsinchu_only"
            else:
                registry_entry = PILOT_EXPLICIT_REGISTRY.get(site_ids)
                expected_sites = site_ids if registry_entry is not None else ()
                expected_scope = (
                    str(registry_entry["selection_scope"]) if registry_entry is not None else None
                )
            if site_ids != expected_sites or len(site_ids) != len(set(site_ids)):
                errors.append("pilot_explicit_window_record_count_invalid")
            if isinstance(gap_rows, list):
                for pilot_row in pilot_rows:
                    pilot_id = pilot_row.get("arrival_time_id")
                    matching_gap = [
                        row
                        for row in gap_rows
                        if isinstance(row, Mapping) and row.get("arrival_time_id") == pilot_id
                    ]
                    metadata = pilot_row.get("metadata")
                    if len(matching_gap) != 1 or not isinstance(metadata, Mapping):
                        errors.append("pilot_explicit_window_gap_cross_reference_invalid")
                        continue
                    gap_row = matching_gap[0]
                    raw_gap_days = gap_row.get("max_backtrack_days")
                    try:
                        gap_days_valid = math.isclose(
                            float(raw_gap_days),
                            PILOT_EXPLICIT_MAX_BACKTRACK_DAYS,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    except (TypeError, ValueError):
                        gap_days_valid = False
                    expected_time_valid = False
                    with suppress(TypeError, ValueError, OverflowError):
                        expected_time_valid = (
                            _utc_string(int(pilot_row.get("time_utc_ns"))) == PILOT_EXPLICIT_ARRIVAL_UTC
                        )
                    if (
                        policy_id not in _PILOT_EXPLICIT_POLICY_IDS
                        or pilot_row.get("tide_class") != PILOT_EXPLICIT_TIDE_CLASS
                        or pilot_row.get("phase_or_event") != PILOT_EXPLICIT_PHASE_OR_EVENT
                        or metadata.get("pilot_selection_scope") != expected_scope
                        or metadata.get("explicit_pilot_window_expected_step_count")
                        != _PILOT_EXPLICIT_EXPECTED_STEP_COUNT
                        or not expected_time_valid
                        or not gap_days_valid
                        or gap_row.get("expected_step_count") != _PILOT_EXPLICIT_EXPECTED_STEP_COUNT
                        or gap_row.get("supported_step_count") != _PILOT_EXPLICIT_EXPECTED_STEP_COUNT
                        or gap_row.get("crossed_gap") is not False
                        or gap_row.get("missing_utc") != []
                    ):
                        errors.append("pilot_explicit_window_support_record_invalid")
    roots_by_token: dict[str, Path | None] = {}
    if config is not None:
        roots_by_token[config.inputs.ocm_native_root_env] = _env_root(
            ocm_native_root, config.inputs.ocm_native_root_env, required=False
        )
        roots_by_token[config.inputs.ocm_surface_root_env] = _env_root(
            ocm_surface_root, config.inputs.ocm_surface_root_env, required=False
        )
        roots_by_token[config.inputs.nww_analysis_root_env] = _env_root(
            nww_analysis_root, config.inputs.nww_analysis_root_env, required=False
        )
    inventory_products = forcing.get("products")
    source_tokens = (
        {
            str(item.get("root_token"))
            for item in inventory_products
            if isinstance(item, Mapping) and isinstance(item.get("root_token"), str)
        }
        if isinstance(inventory_products, list)
        else set()
    )
    source_roots_complete = bool(source_tokens) and all(
        roots_by_token.get(token) is not None for token in source_tokens
    )
    summary["source_canonical_axis_rebuilt"] = source_roots_complete
    if not source_roots_complete:
        # 沒有 caller 明示的三套 accepted-product root 時，只能做 artifact／metadata
        # 自洽檢查；保留這個 warning 讓離線報告不會被誤讀成現場來源已重建。
        warnings.append("source_canonical_axis_rebuild_skipped_without_roots")
    errors.extend(
        _validate_source_file_bindings(
            forcing,
            roots_by_token=roots_by_token,
        )
    )
    errors.extend(
        _validate_canonical_axis_bindings(
            forcing,
            roots_by_token=roots_by_token,
        )
    )
    summary.update(
        {
            "material_count": len(material_rows) if isinstance(material_rows, list) else 0,
            "receptor_count": len(receptor_rows) if isinstance(receptor_rows, list) else 0,
            "arrival_count": len(arrival_rows) if isinstance(arrival_rows, list) else 0,
            "dynamic_initial_condition_count": len(dynamic_rows) if isinstance(dynamic_rows, list) else 0,
            "nww_all_domains_complete_17544": bool(nww.get("all_domains_complete_17544")),
        }
    )
    if formal and config is None:
        errors.append("formal_validation_requires_config")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "summary": summary}


def _replace_manifest_references(
    payload: dict[str, Any], *, config_output: Path, input_directory: Path
) -> dict[str, Any]:
    """把範例 config 的 derived path 改成相對於新 config 的 immutable component path。"""

    result = json.loads(json.dumps(payload, ensure_ascii=False))
    relative_directory = os.path.relpath(input_directory, config_output.parent)
    relative_directory = Path(relative_directory)
    paths = {
        kind: (relative_directory / filename).as_posix() for kind, filename in ARTIFACT_FILENAMES.items()
    }
    result.setdefault("inputs", {})["ocm_gap_safe_arrival_manifest"] = paths["ocm_gap_safe_arrival_horizon"]
    result.setdefault("inputs", {})["nww_full_hourly_analysis_manifest"] = paths["nww_full_hourly"]
    result.setdefault("scenarios", {})["material_manifest"] = paths["material"]
    result.setdefault("scenarios", {})["receptor_manifest"] = paths["receptor"]
    result.setdefault("scenarios", {})["arrival_time_manifest"] = paths["arrival"]
    result.setdefault("scenarios", {})["receptor_arrival_initial_condition_manifest"] = paths[
        "initial_condition"
    ]
    result.setdefault("physics", {}).setdefault("settling", {})["material_manifest"] = paths["material"]
    result.setdefault("geometry", {})["domain_manifest"] = paths["domain_geometry"]
    result.setdefault("geometry", {})["local_domain_manifest"] = paths["local_geometry"]
    result.setdefault("geometry", {})["open_boundary_manifest"] = paths["open_boundary"]
    result.setdefault("geometry", {})["receptor_manifest"] = paths["receptor"]
    result["inputs"]["derived_input_artifact_index"] = (relative_directory / "artifact_index.json").as_posix()
    return result


def _flow_domain_ids_from_inventory(inventory: Mapping[str, Any]) -> dict[str, str]:
    """由 forcing inventory 解析每個 region 的唯一三產品 flow-domain ID。

    inventory 已是 builder 產生並由 artifact sidecar 綁定的資料；此函式仍重新檢查
    `ocm_native`、`ocm_surface`、`nww3_analysis` 三產品是否各自恰一列且 ID 一致，避免
    release config 只依第一筆資料寫入而漏掉跨產品錯配。回傳的 ID 來自實際 inventory，
    不是依任意資料夾名稱猜測。
    """

    products = inventory.get("products")
    if not isinstance(products, list):
        raise InputDerivationError("forcing inventory 缺少 products")
    expected_products = {"ocm_native", "ocm_surface", "nww3_analysis"}
    by_region: dict[str, dict[str, set[str]]] = {}
    for item in products:
        if not isinstance(item, Mapping):
            raise InputDerivationError("forcing inventory product record 必須是 object")
        region = item.get("analysis_region_id")
        flow_id = item.get("flow_domain_id")
        product = item.get("product")
        if not all(isinstance(value, str) and value.strip() for value in (region, flow_id, product)):
            raise InputDerivationError("forcing inventory product 缺少 region／flow-domain／product")
        by_region.setdefault(region, {}).setdefault(product, set()).add(flow_id)
    result: dict[str, str] = {}
    for region, products_by_name in by_region.items():
        if set(products_by_name) != expected_products:
            raise InputDerivationError(f"{region} forcing inventory 未完整包含三套產品")
        flow_ids = {next(iter(ids)) for ids in products_by_name.values() if len(ids) == 1}
        if any(len(ids) != 1 for ids in products_by_name.values()) or len(flow_ids) != 1:
            raise InputDerivationError(f"{region} 三套 forcing 的 flow-domain ID 不一致")
        result[region] = next(iter(flow_ids))
    if set(result) != {"A", "B", "C", "D"}:
        raise InputDerivationError("forcing inventory 必須恰有 A-D 四個 analysis region")
    return result


def _bind_inventory_flow_domains(
    config_payload: dict[str, Any],
    *,
    inventory_flow_ids: Mapping[str, str],
    runtime_spatial_support_contracts_by_region: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """把 inventory 的真實 source ID 綁定到 release config 的 formal 欄位。

    ``expanded_domain_v1`` 的每個 ID 必須出現在該 domain 的 base、formal 或明示
    candidate 欄位中；因此舊 expanded A 只能來自 config 已登錄的
    ``expanded_domain_candidate_id``／formal ID，不能從任意目錄名稱推導。對
    ``v3_local20km_20260909_v1``，inventory 必須 exact 綁定 v3；domain/site 的 formal
    ID、bbox、no-expansion runtime policy/status 會維持設定原值，不會被此函式升格為
    空間支援已量測或正式輸入已核准。若呼叫端提供 forcing inventory 的 contract map，
    函式會核對其與 config 相同；正式 release builder 另以 source-config hash 及
    artifact-index hash 不可變地綁定這份來源設定與 inventory。公開分析標籤不在此函式中變更。
    """

    domains = config_payload.get("domains")
    sites = config_payload.get("study_sites")
    if not isinstance(domains, list) or not isinstance(sites, list):
        raise InputDerivationError("config 必須含 domains 與 study_sites array")
    try:
        validated_config = ProjectConfig.model_validate(config_payload)
        validated_config.assert_research_domain_policy()
    except Exception as exc:
        raise InputDerivationError("config 的 formal_domain_policy／研究範圍契約不合法") from exc
    expected_runtime_contracts = _runtime_spatial_support_contracts_by_region(validated_config)
    if (
        runtime_spatial_support_contracts_by_region is not None
        and dict(runtime_spatial_support_contracts_by_region) != expected_runtime_contracts
    ):
        raise InputDerivationError("inventory runtime spatial-support contract 與 config 不一致")
    domain_by_region: dict[str, dict[str, Any]] = {}
    for domain in domains:
        if not isinstance(domain, dict) or not isinstance(domain.get("analysis_region_id"), str):
            raise InputDerivationError("config domain record 不合法")
        region = domain["analysis_region_id"]
        if region in domain_by_region:
            raise InputDerivationError(f"config domain region 重複：{region}")
        domain_by_region[region] = domain
    for region, actual_id in inventory_flow_ids.items():
        domain = domain_by_region.get(region)
        if domain is None:
            raise InputDerivationError(f"config 缺少 inventory region：{region}")
        try:
            domain_model = DomainConfig.model_validate(domain)
        except Exception as exc:
            raise InputDerivationError(f"{region} config domain schema 不合法") from exc
        base_id = domain.get("flow_domain_id")
        formal_id = domain.get("formal_release_flow_domain_id")
        candidate_id = domain.get("expanded_domain_candidate_id")
        if domain_model.formal_domain_policy == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1:
            if any(
                field in domain
                for field in (
                    "expanded_domain_candidate_id",
                    "expanded_bbox_lon_lat",
                    "expanded_south_boundary_at_or_south_of_deg",
                    "radius_25000_formal_requires_expanded_domain",
                    "radius_35000_formal_requires_expanded_domain",
                )
            ):
                raise InputDerivationError(
                    "v3_local20km_20260909_v1 inventory binding 禁止 expanded candidate 欄位"
                )
            if (
                region != "A"
                or actual_id != NORTHEAST_V3_FLOW_DOMAIN_ID
                or base_id != NORTHEAST_V3_FLOW_DOMAIN_ID
                or formal_id != NORTHEAST_V3_FLOW_DOMAIN_ID
                or domain.get("formal_release_domain_status")
                != FORMAL_RELEASE_DOMAIN_STATUS_V3_FAIL_CLOSED_NO_EXPANSION
                or domain.get("runtime_spatial_support_policy")
                != RUNTIME_SPATIAL_SUPPORT_POLICY_V3_FAIL_CLOSED_NO_EXPANSION_V1
            ):
                raise InputDerivationError(
                    "v3_local20km_20260909_v1 inventory binding 必須 exact 使用 v3，"
                    "並保留 no-expansion runtime-stage policy/status"
                )
            resolved_bbox = _authoritative_flow_domain_bbox_lon_lat(domain_model, actual_id)
            if list(domain.get("bbox_lon_lat", ())) != list(resolved_bbox):
                raise InputDerivationError("v3_local20km_20260909_v1 inventory binding 不得改寫 A 區 bbox")
            for site in sites:
                if not isinstance(site, dict) or site.get("analysis_region_id") != region:
                    continue
                if (
                    site.get("flow_domain_id") != NORTHEAST_V3_FLOW_DOMAIN_ID
                    or site.get("formal_release_flow_domain_id") != NORTHEAST_V3_FLOW_DOMAIN_ID
                ):
                    raise InputDerivationError(
                        f"{site.get('study_site_id', region)} 的 v3 policy formal source 必須 exact v3"
                    )
            continue
        if domain_model.formal_domain_policy != FORMAL_DOMAIN_POLICY_EXPANDED_V1:
            raise InputDerivationError(f"未知 formal_domain_policy：{domain_model.formal_domain_policy!r}")
        allowed_ids = {
            value for value in (base_id, formal_id, candidate_id) if isinstance(value, str) and value.strip()
        }
        if actual_id not in allowed_ids:
            raise InputDerivationError(
                f"{region} inventory source ID 未在 config 明示候選／formal/base 中：{actual_id}"
            )
        if isinstance(formal_id, str) and formal_id.strip() and formal_id != actual_id:
            raise InputDerivationError(
                f"{region} config formal source ID 與 inventory 不一致：{formal_id} != {actual_id}"
            )
        resolved_bbox = _authoritative_flow_domain_bbox_lon_lat(domain_model, actual_id)
        if actual_id != base_id:
            # expanded bbox 與 ID 必須同時回寫 release config；base flow_domain_id 與
            # expanded_domain_candidate_id 保留，讓 provenance 仍能追溯「從哪個 pilot
            # 設定升級」以及「實際 accepted source 為哪個明示候選」，不把舊 bbox 留在
            # runtime 設定中造成 ID／空間支撐錯配。
            domain["bbox_lon_lat"] = list(resolved_bbox)
            domain["formal_release_flow_domain_id"] = actual_id
            domain["formal_release_domain_status"] = "approved_source_bound_by_inventory"
        for site in sites:
            if not isinstance(site, dict) or site.get("analysis_region_id") != region:
                continue
            # 若實際來源仍是 pilot base，template 不需要額外的 formal binding；只有
            # expanded／已明示 formal ID 才把同一識別碼寫入 site。否則會產生「site 有
            # formal ID、domain 卻是 null」的自相矛盾設定，讓 Pydantic 跨欄位契約拒絕
            # 原本完全合法的 base artifact。
            if actual_id != base_id or formal_id == actual_id:
                site["formal_release_flow_domain_id"] = actual_id
    return config_payload


def _config_for_inventory_validation(
    config: ProjectConfig,
    *,
    inventory: Mapping[str, Any],
    formal: bool,
) -> ProjectConfig:
    """建立與 inventory source ID 對齊的 validator config view。

    generated／pilot artifact 可能先以 config 明示的 expanded candidate 建置，但原始
    template 尚未把它升為 formal ID；此時 component loader 仍需用實際 artifact ID 做
    cross-reference。formal validation 則刻意不做自動對齊，要求 config 已明示 formal ID，
    以免 validator 替 release config 隱藏遺漏的 formal binding。
    """

    if formal:
        return config
    inventory_flow_ids = _flow_domain_ids_from_inventory(inventory)
    payload = config.model_dump(mode="json", exclude_none=False)
    # ``model_dump`` 會把 Pydantic 為舊 YAML 補出的新欄位也寫回 mapping；若不先移除，
    # 後面的 model_validate 會把「舊設定未明示 support」誤記成「明示 null」，使 legacy
    # artifact 在 inventory flow-domain 對齊後錯誤觸發 generic 未定案 gate。只有原始
    # YAML 真正寫過欄位時才保留它，與 ProjectConfig.normalized_payload 的 hash 相容政策一致。
    inputs_payload = payload.get("inputs")
    if isinstance(inputs_payload, dict) and "backtrack_support_days" not in config.inputs.model_fields_set:
        inputs_payload.pop("backtrack_support_days", None)
    _bind_inventory_flow_domains(payload, inventory_flow_ids=inventory_flow_ids)
    domains = payload["domains"]
    sites = payload["study_sites"]
    for domain in domains:
        region = domain["analysis_region_id"]
        domain["flow_domain_id"] = inventory_flow_ids[region]
    for site in sites:
        site["flow_domain_id"] = inventory_flow_ids[site["analysis_region_id"]]
    return ProjectConfig.model_validate(payload)


def _release_horizon_override(value: Any, *, label: str) -> float:
    """驗證 release 輸出的回溯日數覆寫值，不讓 YAML／CLI 偷換型別。

    release config 的 ``max_backtrack_days`` 仍保留浮點型別，以相容既有 pilot 的
    小於一日視窗；這裡只負責拒絕布林、非有限值與非正值。是否超過母體支援窗，
    會在完整 ``ProjectConfig`` 與 gap-safe artifact 證據都載入後再判定。
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} 必須是有限正數")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{label} 必須是有限正數")
    return normalized


def _release_step_count_override(value: Any, *, label: str) -> int:
    """驗證 release 輸出的最大步數覆寫值，保留原生正整數語意。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} 必須是正整數")
    return int(value)


def _release_step_budget_error(config: ProjectConfig) -> str | None:
    """回傳 requested horizon 與既有 dt／最大步數不相容時的錯誤訊息。

    只有三個欄位都已定案才做數值比較；準備性設定缺少 dt 或 step budget 時，保留
    ``generated`` 與原有 formal blocker，不能由這個 helper 偷填研究參數。最低需求
    以 ``ceil(H 秒 / dt_max)`` 計算，確保單一粒子在允許的最大步長下仍有足夠迭代額度。
    """

    requested_days = config.boundaries.max_backtrack_days
    dt_max = config.integration.dt_max_seconds
    step_count = config.boundaries.maximum_step_count
    if requested_days is None or dt_max is None or step_count is None:
        return None
    if isinstance(dt_max, bool) or not isinstance(dt_max, (int, float)):
        return "integration.dt_max_seconds 必須是有限正數"
    dt_max_float = float(dt_max)
    if not math.isfinite(dt_max_float) or dt_max_float <= 0.0:
        return "integration.dt_max_seconds 必須是有限正數"
    required_steps = math.ceil(float(requested_days) * 86_400.0 / dt_max_float)
    if int(step_count) < required_steps:
        return "boundaries.maximum_step_count 不足以涵蓋 requested max_backtrack_days"
    return None


def _release_support_evidence(
    input_root: Path,
    *,
    source_config: ProjectConfig,
    release_config: ProjectConfig,
    requested_days: float | None,
) -> dict[str, Any]:
    """讀取小型 release artifact 證據，確認母體 hash 與支援窗沒有被偷換。

    這個檢查只讀 ``artifact_index.json`` 與 gap-safe component，不開啟 OCM/NWW 大型陣列。
    一般共同母體要求 gap root 與 selection support 相同；bed-residence 母體則明確分開
    observation selection envelope 與逐沉底時刻 runtime support。實際 component／每筆
    arrival 的完整性仍交由共用 validator 驗證。未明示新版欄位的 legacy config 保留舊契約；
    回傳摘要只寫入 release binding，不修改來源檔案。
    """

    index, _ = read_canonical_json(input_root / "artifact_index.json")
    source_bindings = index.get("source_bindings")
    support_declared = "backtrack_support_days" in source_config.inputs.model_fields_set
    if support_declared:
        if not isinstance(source_bindings, Mapping):
            raise InputDerivationError("input artifact 缺少 source_bindings，不能建立 release config")
        source_hash = source_bindings.get("config_hash")
        if type(source_hash) is not str or _SHA256_RE.fullmatch(source_hash) is None:
            raise InputDerivationError("input artifact source_config_hash 格式不合法")
    else:
        source_hash = source_bindings.get("config_hash") if isinstance(source_bindings, Mapping) else None

    gap, _ = read_canonical_json(input_root / ARTIFACT_FILENAMES["ocm_gap_safe_arrival_horizon"])
    raw_root_days = gap.get("max_backtrack_days")
    if isinstance(raw_root_days, bool) or not isinstance(raw_root_days, (int, float)):
        raise InputDerivationError("gap-safe artifact 缺少有限正數 max_backtrack_days")
    root_days = float(raw_root_days)
    if not math.isfinite(root_days) or root_days <= 0.0:
        raise InputDerivationError("gap-safe artifact max_backtrack_days 必須是有限正數")

    support_days = source_config.inputs.backtrack_support_days if support_declared else None
    bed = release_config.scenarios.bed_residence_time
    if bed is None:
        if support_days is not None and not math.isclose(
            root_days, float(support_days), rel_tol=0.0, abs_tol=1e-12
        ):
            raise InputDerivationError("gap-safe artifact 的母體支援日數與 source config 宣告不一致")
        if support_declared and requested_days is not None and requested_days > root_days:
            raise InputDerivationError("release config requested max_backtrack_days 超過 gap-safe 母體支援窗")
    else:
        runtime_days = bed.runtime_horizon_support_days
        if runtime_days is None:
            runtime_days = (
                int(requested_days)
                if requested_days is not None and requested_days.is_integer()
                else None
            )
        artifact_selection = gap.get("selection_support_days")
        artifact_runtime = gap.get("runtime_support_days")
        if (
            support_days is None
            or type(artifact_selection) is not int
            or artifact_selection != support_days
            or type(runtime_days) is not int
            or type(artifact_runtime) is not int
            or artifact_runtime != runtime_days
            or not math.isclose(root_days, float(runtime_days), rel_tol=0.0, abs_tol=1e-12)
        ):
            raise InputDerivationError(
                "bed-residence gap artifact 的 selection/runtime support 與共同 config 不一致"
            )
        if requested_days is not None and requested_days > runtime_days:
            raise InputDerivationError("release requested horizon 超過 bed-residence runtime support")

    records = gap.get("records")
    if not isinstance(records, list) or not records:
        raise InputDerivationError("gap-safe artifact 缺少 arrival records，不能證明母體支援窗")
    evidence = {
        # 這裡保存 artifact 建置時的來源 hash；它是 provenance，不要求等於後續
        # 補齊 dt／members／step 等執行參數後的 release config hash。
        "source_config_hash": source_hash,
        "source_backtrack_support_days": support_days,
        "artifact_backtrack_support_days": root_days,
        "requested_max_backtrack_days": requested_days,
    }
    if bed is not None:
        runtime_days = bed.runtime_horizon_support_days
        if runtime_days is None:
            runtime_days = (
                int(requested_days)
                if requested_days is not None and requested_days.is_integer()
                else None
            )
        evidence.update(
            {
                "selection_support_days": support_days,
                "runtime_support_days": runtime_days,
                "artifact_selection_support_days": gap.get("selection_support_days"),
                "artifact_runtime_support_days": gap.get("runtime_support_days"),
                "backtrack_mode": bed.backtrack_mode,
            }
        )
    return evidence


def _arrival_selection_release_binding(config: ProjectConfig) -> dict[str, Any]:
    """建立 release binding 使用的 arrival 母體身分快照。

    ``inputs.years`` 是 common input 必須讀取的完整 forcing 年份；新版
    ``arrival_time_selection.observation_years`` 則是允許成為 arrival anchor 的年份。
    這四個欄位與 arrival manifest 的 nested provenance 同步保存，讓每一份 H30/H60/H90
    release 都能證明沿用同一個 2024–2025 forcing、2025 observation population、policy
    與 replicate 數，而不是只靠共用 artifact hash 間接推測。舊設定未宣告新版 policy
    時，沿用兩 forcing 年份、一 replicate 的相容預設；此 fallback 不改變舊 config hash
    或 selector 行為。
    """

    forcing_years = [int(year) for year in config.inputs.years]
    selection = config.arrival_time_selection
    if selection is None:
        observation_years = list(forcing_years)
        policy = ARRIVAL_SELECTION_POLICY_LEGACY_TWO_YEAR_V1
        replicates = 1
    else:
        observation_years = (
            [int(year) for year in selection.observation_years]
            if selection.observation_years is not None
            else list(forcing_years)
        )
        policy = selection.policy or ARRIVAL_SELECTION_POLICY_LEGACY_TWO_YEAR_V1
        replicates = int(selection.replicates)
    return {
        "forcing_years": forcing_years,
        "observation_years": observation_years,
        "policy": policy,
        "replicates": replicates,
    }


def create_release_config(
    *,
    config_template_path: str | Path,
    input_directory: str | Path,
    output_path: str | Path,
    formal: bool = True,
    max_backtrack_days: float | None = None,
    maximum_step_count: int | None = None,
    backtrack_mode_override: str | None = None,
) -> dict[str, Any]:
    """由範例設定建立新的 release config，並以 exact artifact hash 寫入 binding。

    範例 YAML 永遠不覆寫。所有 component 先通過 validator，再把其 raw/canonical hash
    與 artifact index hash 寫進 `release_binding`；binding 若有任何變更，後續 validator
    會 fail closed。只有所有 manifests 已驗證、cross-reference 完整且目前設定的既有
    formal gate 也通過時才標 ``config_status=approved``；否則輸出 ``generated`` 並列出
    blocker，避免把「檔案已產出」誤稱為可啟動的正式批次。
    """

    template_path = _assert_regular_file(config_template_path)
    input_root = _assert_regular_directory(input_directory)
    output = Path(output_path)
    _assert_no_symlink_components(output.parent, allow_missing_leaf=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"不可覆寫既有 release config：{output}")
    source_config = load_config(template_path, formal_release=False)
    normalized_max_days = (
        None
        if max_backtrack_days is None
        else _release_horizon_override(max_backtrack_days, label="max_backtrack_days")
    )
    normalized_max_steps = (
        None
        if maximum_step_count is None
        else _release_step_count_override(maximum_step_count, label="maximum_step_count")
    )
    # 先在 manifest rewriter 之前檢查 requested 是否超過 source config 已明示的共同
    # 母體支援窗。這個順序很重要：rewriter 會重新綁定 flow-domain，若讓不可能的
    # 31 日設定先進入那條路徑，錯誤可能被誤報成 A 區研究範圍契約，而不是實際的
    # horizon 超限。source 欄位未明示時保留 legacy 行為；明示 null 則只能建立尚未
    # 指定 requested 的準備性設定，不能藉 release override 越過「尚未定案」狀態。
    support_declared = "backtrack_support_days" in source_config.inputs.model_fields_set
    if support_declared and normalized_max_days is not None:
        source_support_days = source_config.inputs.backtrack_support_days
        if source_support_days is None:
            raise ValueError("已指定 release max_backtrack_days，但 inputs.backtrack_support_days 尚未定案")
        if normalized_max_days > float(source_support_days):
            raise ValueError(
                "release config requested max_backtrack_days 不得超過 inputs.backtrack_support_days"
            )
    config_payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("config template root 必須是 mapping")
    source_bed = source_config.scenarios.bed_residence_time
    if backtrack_mode_override is not None:
        if source_bed is None:
            raise ValueError("backtrack_mode_override 只適用於 bed-residence config")
        if backtrack_mode_override not in source_bed.supported_backtrack_modes:
            raise ValueError("backtrack_mode_override 不在 supported_backtrack_modes")
        scenarios_payload = config_payload.get("scenarios")
        bed_payload = (
            scenarios_payload.get("bed_residence_time")
            if isinstance(scenarios_payload, dict)
            else None
        )
        if not isinstance(bed_payload, dict):
            raise ValueError("config template 缺少 scenarios.bed_residence_time mapping")
        bed_payload["backtrack_mode"] = backtrack_mode_override
    elif source_bed is not None:
        backtrack_mode_override = source_bed.backtrack_mode
    if normalized_max_days is not None or normalized_max_steps is not None:
        boundaries = config_payload.get("boundaries")
        if not isinstance(boundaries, dict):
            raise ValueError("config template boundaries 必須是 mapping")
        if normalized_max_days is not None:
            boundaries["max_backtrack_days"] = normalized_max_days
        if normalized_max_steps is not None:
            boundaries["maximum_step_count"] = normalized_max_steps
    # forcing inventory 是 builder 對三套 accepted product 的共同來源紀錄。先從它
    # 解析每個 region 的實際 ID，再把 release config 的 formal 欄位與 site-level
    # runtime binding 一次綁定；A 區 v3 另要求 contract map 與 source config 完全一致，
    # 不能只從 template 的 base ID 或任意資料夾名稱猜測。
    forcing_payload, _ = read_canonical_json(input_root / ARTIFACT_FILENAMES["forcing_inventory"])
    inventory_flow_ids = _flow_domain_ids_from_inventory(forcing_payload)
    expected_runtime_contracts = _runtime_spatial_support_contracts_by_region(source_config)
    rewritten = _replace_manifest_references(config_payload, config_output=output, input_directory=input_root)
    inventory_runtime_contracts = forcing_payload.get("runtime_spatial_support_contracts_by_region")
    if expected_runtime_contracts and (
        not isinstance(inventory_runtime_contracts, Mapping)
        or dict(inventory_runtime_contracts) != expected_runtime_contracts
    ):
        raise InputDerivationError(
            "forcing inventory 缺少或錯綁 A 區 runtime spatial-support contract"
        )
    _bind_inventory_flow_domains(
        rewritten,
        inventory_flow_ids=inventory_flow_ids,
        runtime_spatial_support_contracts_by_region=(
            inventory_runtime_contracts if isinstance(inventory_runtime_contracts, Mapping) else None
        ),
    )
    rewritten["config_status"] = "generated"
    candidate_config = ProjectConfig.model_validate(rewritten)
    requested_days = candidate_config.boundaries.max_backtrack_days
    support_evidence = _release_support_evidence(
        input_root,
        source_config=source_config,
        release_config=candidate_config,
        requested_days=None if requested_days is None else float(requested_days),
    )
    support_evidence["requested_maximum_step_count"] = candidate_config.boundaries.maximum_step_count
    step_budget_error = _release_step_budget_error(candidate_config)
    if step_budget_error is not None:
        raise ValueError(step_budget_error)
    # 非 formal validator 要驗證 artifact 內的實際 flow-domain ID；它使用 base ID 作為
    # pilot provenance，但將 component cross-reference 的 view 對齊 inventory 實際 ID。
    # 這個 view 只存在於記憶體，絕不把 template 的 base ID 靜默改寫成 artifact ID。
    artifact_config_payload = json.loads(json.dumps(rewritten, ensure_ascii=False))
    for domain in artifact_config_payload["domains"]:
        domain["flow_domain_id"] = inventory_flow_ids[domain["analysis_region_id"]]
    for site in artifact_config_payload["study_sites"]:
        site["flow_domain_id"] = inventory_flow_ids[site["analysis_region_id"]]
    artifact_validation_config = ProjectConfig.model_validate(artifact_config_payload)
    validation = validate_input_derivatives(
        input_root,
        config_override=artifact_validation_config,
        formal=False,
    )
    if validation.get("valid") is not True:
        raise InputDerivationError("input manifests 未通過 validator，不能建立 release config")
    blockers: list[str] = []
    formal_input_validation: dict[str, Any] | None = None
    if formal:
        # 一般 validator 只確認 artifact 可用；formal validator 另外檢查 approved
        # status、NWW 17,544 小時、gap-safe horizon 與 formal resolver。兩者都通過後，
        # 才有資格把 candidate config 的 status 升成 approved；避免僅因 YAML 欄位齊全
        # 就把 generated component 發布成正式設定。
        formal_input_validation = validate_input_derivatives(
            input_root,
            config_override=ProjectConfig.model_validate(rewritten),
            formal=True,
        )
        if formal_input_validation.get("valid") is not True:
            blockers.extend(
                f"formal_input_invalid:{item}" for item in formal_input_validation.get("errors", [])
            )
    try:
        # formal gate 本身要求 config_status=approved；若直接拿 generated candidate
        # 驗證，狀態欄位會永遠製造一個假 blocker，導致「所有正式條件都已滿足」時仍
        # 無法產出 approved config。因此先用只存在於記憶體的 approved view 驗證，最後
        # 再依結果選擇寫入 approved 或 generated；不會把 template 或既有 config 覆寫。
        candidate_payload = json.loads(json.dumps(rewritten, ensure_ascii=False))
        candidate_payload["config_status"] = "approved"
        candidate = ProjectConfig.model_validate(candidate_payload)
        if formal:
            try:
                candidate.assert_formal_release_ready()
            except ValueError as exc:
                blockers.extend(str(exc).split("；"))
    except Exception as exc:
        blockers.append(f"config_schema_invalid:{type(exc).__name__}")
    artifact_bindings = []
    for kind, filename in sorted(ARTIFACT_FILENAMES.items()):
        _, fingerprint = read_canonical_json(input_root / filename)
        artifact_bindings.append(
            {
                "kind": kind,
                **fingerprint,
                "path": (Path(os.path.relpath(input_root, output.parent)) / filename).as_posix(),
            }
        )
    _, artifact_index_fp = read_canonical_json(input_root / "artifact_index.json")
    arrival_selection_binding = _arrival_selection_release_binding(candidate_config)
    rewritten["release_binding"] = {
        "schema_version": (
            BED_RESIDENCE_INPUT_SCHEMA_VERSION
            if source_bed is not None
            else DERIVED_INPUT_SCHEMA_VERSION
        ),
        "source_config_template_sha256": _sha256_file(template_path),
        # 這個 hash 取自 artifact 建置時的 source binding，代表母體 provenance；7 日
        # 與 30 日輸出可各自擁有不同 config hash，但不能把母體 hash 改寫成 target hash。
        "source_config_hash": support_evidence["source_config_hash"],
        "input_directory_artifact_index_sha256": artifact_index_fp["sha256"],
        "artifacts": artifact_bindings,
        "approved_only_after_exact_hash_validation": True,
        "backtrack_horizon_binding": support_evidence,
        # release config 與 common input 必須明示同一份 arrival population；不能只靠
        # artifact index hash 推測 2024 forcing 與 2025 observation 的分離仍然成立。
        "arrival_selection_binding": arrival_selection_binding,
    }
    if not blockers and formal:
        rewritten["config_status"] = "approved"
    elif not formal:
        rewritten["config_status"] = "generated"
    rewritten["release_approval"] = {
        "status": rewritten["config_status"],
        "blockers": blockers,
        "validated_input_summary": validation.get("summary", {}),
        "formal_input_validation_summary": (
            formal_input_validation.get("summary", {}) if formal_input_validation is not None else None
        ),
        "public_analysis_label_policy": {"A": "A 區分析域"},
    }
    rendered = yaml.safe_dump(rewritten, allow_unicode=True, sort_keys=False, default_flow_style=False)
    _atomic_write_text(output, rendered)
    return {
        "output": str(output),
        "config_status": rewritten["config_status"],
        "blockers": blockers,
        "input_artifact_index_sha256": artifact_index_fp["sha256"],
        "artifact_count": len(artifact_bindings),
    }


def validate_release_config(
    path: str | Path,
    *,
    input_directory: str | Path | None = None,
    formal: bool = True,
) -> dict[str, Any]:
    """唯讀驗證 release config 的 exact path/hash binding 與 config status。"""

    errors: list[str] = []
    config_path = _assert_regular_file(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"valid": False, "errors": [f"yaml_invalid:{type(exc).__name__}"], "warnings": []}
    if not isinstance(payload, dict):
        return {"valid": False, "errors": ["yaml_root_invalid"], "warnings": []}
    binding = payload.get("release_binding")
    if not isinstance(binding, Mapping):
        errors.append("release_binding_missing")
        return {"valid": False, "errors": errors, "warnings": []}
    raw_inputs = payload.get("inputs")
    support_declared = isinstance(raw_inputs, Mapping) and "backtrack_support_days" in raw_inputs
    release_config: ProjectConfig | None = None
    try:
        release_config = ProjectConfig.model_validate(payload)
    except Exception as exc:
        # 舊 release config 沒有新支援窗欄位時，保留原有 path/hash validator 的輸出；
        # 新欄位一旦出現則必須先通過完整 schema，否則不能藉 generated 狀態繞過 gate。
        if support_declared or "backtrack_horizon_binding" in binding:
            errors.append(f"release_config_schema_invalid:{type(exc).__name__}")
    # 新版 observation population 的四個身分欄位必須在 release binding 直接保存。
    # 舊 release config 可能沒有這個 nested binding，且仍需可唯讀載入；只有新版
    # policy 缺少 binding 時 fail closed，避免把不同 observation population 的 release
    # 誤當成同一份 common input。若 binding 存在，legacy 與新版都做 exact 比對。
    if release_config is not None:
        expected_arrival_binding = _arrival_selection_release_binding(release_config)
        selection = release_config.arrival_time_selection
        requires_arrival_binding = (
            selection is not None
            and selection.policy == ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1
        )
        raw_arrival_binding = binding.get("arrival_selection_binding")
        if raw_arrival_binding is None:
            if requires_arrival_binding:
                errors.append("release_arrival_selection_binding_missing")
        elif not isinstance(raw_arrival_binding, Mapping):
            errors.append("release_arrival_selection_binding_invalid")
        elif set(raw_arrival_binding) != set(expected_arrival_binding):
            errors.append("release_arrival_selection_binding_keys_invalid")
        elif any(
            raw_arrival_binding[field] != expected_arrival_binding[field]
            for field in expected_arrival_binding
        ):
            errors.append("release_arrival_selection_binding_mismatch")
    if binding.get("source_config_template_sha256") and not _SHA256_RE.fullmatch(
        str(binding["source_config_template_sha256"])
    ):
        errors.append("template_hash_invalid")
    input_root: Path | None = Path(input_directory) if input_directory is not None else None
    if input_root is None:
        # 從第一筆 artifact path 解析 sibling directory；仍拒絕絕對 path。
        raw_artifacts = binding.get("artifacts")
        if isinstance(raw_artifacts, list) and raw_artifacts and isinstance(raw_artifacts[0], Mapping):
            raw_path = raw_artifacts[0].get("path")
            if isinstance(raw_path, str) and not Path(raw_path).is_absolute():
                input_root = (config_path.parent / Path(raw_path)).parent
    if input_root is None:
        errors.append("input_directory_missing")
    else:
        try:
            input_root = _assert_regular_directory(input_root)
            # approved config 的 flow-domain resolver 可能已切換到 A 區 expanded domain，
            # 因此不能一律用 pilot loader 驗證；generated config 則保留非 formal 模式，
            # 讓尚待補齊正式數值閘門的衍生目錄仍可被當作開發 artifact 稽核。
            artifact_is_formal = payload.get("config_status") == "approved" and formal
            artifact_validation = validate_input_derivatives(
                input_root,
                config_path=config_path,
                formal=artifact_is_formal,
            )
            if artifact_validation.get("valid") is not True:
                errors.extend(f"input_invalid:{item}" for item in artifact_validation.get("errors", []))
            _, artifact_index_fp = read_canonical_json(input_root / "artifact_index.json")
            if binding.get("input_directory_artifact_index_sha256") != artifact_index_fp["sha256"]:
                errors.append("artifact_index_binding_mismatch")
            artifact_index_payload, _ = read_canonical_json(input_root / "artifact_index.json")
            artifact_source_bindings = artifact_index_payload.get("source_bindings")
            horizon_binding = binding.get("backtrack_horizon_binding")
            if horizon_binding is not None and not isinstance(horizon_binding, Mapping):
                errors.append("release_horizon_binding_invalid")
                horizon_binding = None
            if support_declared and horizon_binding is None:
                errors.append("release_horizon_binding_missing")
            gap_payload: Mapping[str, Any] | None = None
            if isinstance(horizon_binding, Mapping):
                try:
                    raw_gap_payload, _ = read_canonical_json(
                        input_root / ARTIFACT_FILENAMES["ocm_gap_safe_arrival_horizon"]
                    )
                    gap_payload = raw_gap_payload
                except Exception as exc:
                    errors.append(f"release_support_evidence_invalid:{type(exc).__name__}")
            if release_config is not None and support_declared:
                # 新支援窗配置必須同時綁定 source artifact 與 gap-safe 根節點；只看
                # release YAML 宣告的整日數，不足以證明母體真的建立過相同 horizon。
                if not isinstance(artifact_source_bindings, Mapping):
                    errors.append("release_source_binding_missing")
                else:
                    source_hash = artifact_source_bindings.get("config_hash")
                    bound_source_hash = binding.get("source_config_hash")
                    if type(source_hash) is not str or _SHA256_RE.fullmatch(source_hash) is None:
                        errors.append("release_artifact_source_config_hash_invalid")
                    if bound_source_hash != source_hash:
                        errors.append("release_source_config_hash_mismatch")
                try:
                    if gap_payload is None:
                        raise ValueError("gap-safe component unreadable")
                    artifact_support = gap_payload.get("max_backtrack_days")
                    if (
                        isinstance(artifact_support, bool)
                        or not isinstance(artifact_support, (int, float))
                        or not math.isfinite(float(artifact_support))
                    ):
                        raise ValueError("artifact support 非有限數")
                    support_days = release_config.inputs.backtrack_support_days
                    requested_days = release_config.boundaries.max_backtrack_days
                    bed_config = release_config.scenarios.bed_residence_time
                    if bed_config is None:
                        if support_days is not None and not math.isclose(
                            float(artifact_support),
                            float(support_days),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        ):
                            errors.append("release_support_evidence_mismatch")
                        if requested_days is not None and float(requested_days) > float(artifact_support):
                            errors.append("release_requested_horizon_exceeds_artifact_support")
                    else:
                        runtime_days = bed_config.runtime_horizon_support_days
                        if runtime_days is None:
                            runtime_days = (
                                int(requested_days)
                                if requested_days is not None and float(requested_days).is_integer()
                                else None
                            )
                        if (
                            support_days is None
                            or gap_payload.get("selection_support_days") != support_days
                            or gap_payload.get("runtime_support_days") != runtime_days
                            or runtime_days is None
                            or not math.isclose(
                                float(artifact_support), float(runtime_days), rel_tol=0.0, abs_tol=1e-12
                            )
                        ):
                            errors.append("release_bed_residence_support_evidence_mismatch")
                        if requested_days is not None and (
                            runtime_days is None or float(requested_days) > float(runtime_days)
                        ):
                            errors.append("release_requested_horizon_exceeds_runtime_support")
                except Exception as exc:
                    errors.append(f"release_support_evidence_invalid:{type(exc).__name__}")
            if isinstance(horizon_binding, Mapping) and release_config is not None:
                step_budget_error = _release_step_budget_error(release_config)
                if step_budget_error is not None:
                    errors.append("release_step_budget_insufficient")
                requested_bound = horizon_binding.get("requested_max_backtrack_days")
                requested_value = release_config.boundaries.max_backtrack_days
                if requested_value is None:
                    if requested_bound is not None:
                        errors.append("release_requested_horizon_binding_mismatch")
                elif (
                    isinstance(requested_bound, bool)
                    or not isinstance(requested_bound, (int, float))
                    or not math.isfinite(float(requested_bound))
                    or not math.isclose(
                        float(requested_bound), float(requested_value), rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    errors.append("release_requested_horizon_binding_mismatch")
                bound_source_hash = horizon_binding.get("source_config_hash")
                if isinstance(artifact_source_bindings, Mapping):
                    if bound_source_hash != artifact_source_bindings.get("config_hash"):
                        errors.append("release_horizon_source_hash_mismatch")
                elif bound_source_hash is not None:
                    errors.append("release_horizon_source_hash_mismatch")
                expected_support = release_config.inputs.backtrack_support_days if support_declared else None
                bound_support = horizon_binding.get("source_backtrack_support_days")
                if bound_support != expected_support:
                    errors.append("release_support_binding_mismatch")
                bed_config = release_config.scenarios.bed_residence_time
                expected_horizon_binding_keys = {
                    "source_config_hash",
                    "source_backtrack_support_days",
                    "artifact_backtrack_support_days",
                    "requested_max_backtrack_days",
                    "requested_maximum_step_count",
                }
                if bed_config is not None:
                    expected_horizon_binding_keys.update(
                        {
                            "selection_support_days",
                            "runtime_support_days",
                            "artifact_selection_support_days",
                            "artifact_runtime_support_days",
                            "backtrack_mode",
                        }
                    )
                if set(horizon_binding) != expected_horizon_binding_keys:
                    errors.append("release_horizon_binding_keys_invalid")
                expected_step_count = release_config.boundaries.maximum_step_count
                if horizon_binding.get("requested_maximum_step_count") != expected_step_count:
                    errors.append("release_step_count_binding_mismatch")
                if bed_config is not None:
                    runtime_days = bed_config.runtime_horizon_support_days
                    requested_value = release_config.boundaries.max_backtrack_days
                    if (
                        runtime_days is None
                        and requested_value is not None
                        and float(requested_value).is_integer()
                    ):
                        runtime_days = int(requested_value)
                    expected_bed_values = {
                        "selection_support_days": release_config.inputs.backtrack_support_days,
                        "runtime_support_days": runtime_days,
                        "artifact_selection_support_days": (
                            gap_payload.get("selection_support_days") if gap_payload is not None else None
                        ),
                        "artifact_runtime_support_days": (
                            gap_payload.get("runtime_support_days") if gap_payload is not None else None
                        ),
                        "backtrack_mode": bed_config.backtrack_mode,
                    }
                    for field, expected in expected_bed_values.items():
                        if horizon_binding.get(field) != expected:
                            errors.append(f"release_bed_residence_binding_mismatch:{field}")
                if "artifact_backtrack_support_days" not in horizon_binding:
                    errors.append("release_artifact_support_binding_missing")
                else:
                    try:
                        if gap_payload is None:
                            raise ValueError("gap-safe component unreadable")
                        gap_support = gap_payload.get("max_backtrack_days")
                        bound_artifact_support = horizon_binding.get("artifact_backtrack_support_days")
                        if (
                            isinstance(bound_artifact_support, bool)
                            or not isinstance(bound_artifact_support, (int, float))
                            or not math.isclose(
                                float(bound_artifact_support),
                                float(gap_support),
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            )
                        ):
                            errors.append("release_artifact_support_binding_mismatch")
                    except (TypeError, ValueError):
                        errors.append("release_artifact_support_binding_mismatch")
            raw_scenarios = payload.get("scenarios")
            raw_bed_residence = (
                raw_scenarios.get("bed_residence_time")
                if isinstance(raw_scenarios, Mapping)
                else None
            )
            expected_binding_schema_version = (
                BED_RESIDENCE_INPUT_SCHEMA_VERSION
                if raw_bed_residence is not None
                else DERIVED_INPUT_SCHEMA_VERSION
            )
            if binding.get("schema_version") != expected_binding_schema_version:
                errors.append("release_binding_schema_version_invalid")
            if binding.get("approved_only_after_exact_hash_validation") is not True:
                errors.append("release_binding_exact_hash_policy_invalid")
            raw_artifacts = binding.get("artifacts")
            if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(ARTIFACT_FILENAMES):
                errors.append("release_artifact_binding_count_invalid")
            else:
                relative_directory = Path(os.path.relpath(input_root, config_path.parent))
                expected_paths = {
                    kind: (relative_directory / filename).as_posix()
                    for kind, filename in ARTIFACT_FILENAMES.items()
                }
                expected_paths["artifact_index"] = (relative_directory / "artifact_index.json").as_posix()
                binding_by_kind: dict[str, Mapping[str, Any]] = {}
                for item in raw_artifacts:
                    if not isinstance(item, Mapping):
                        errors.append("release_artifact_binding_invalid")
                        continue
                    kind = item.get("kind")
                    if not isinstance(kind, str) or kind not in ARTIFACT_FILENAMES or kind in binding_by_kind:
                        errors.append(f"release_artifact_kind_invalid:{kind}")
                        continue
                    raw_path = item.get("path")
                    if not isinstance(raw_path, str) or Path(raw_path).is_absolute():
                        errors.append("release_artifact_absolute_path_forbidden")
                        continue
                    if raw_path != expected_paths[kind]:
                        errors.append(f"release_artifact_path_mismatch:{kind}")
                        continue
                    binding_by_kind[kind] = item
                    source = config_path.parent / Path(raw_path)
                    try:
                        _, fingerprint = read_canonical_json(source)
                        for field in ("sha256", "canonical_sha256", "size_bytes"):
                            if item.get(field) != fingerprint.get(field):
                                errors.append(f"release_artifact_hash_mismatch:{kind}:{field}")
                    except Exception as exc:
                        errors.append(f"release_artifact_unreadable:{kind}:{type(exc).__name__}")
                for kind in ARTIFACT_FILENAMES:
                    if kind not in binding_by_kind:
                        errors.append(f"release_artifact_missing:{kind}")
                # component path 必須同時出現在 config 的所有 runtime 參照與 release
                # binding 中；否則只綁定 hash 而沒有真正改變 runtime 讀取路徑，會形成
                # 難以發現的「驗證檔案與執行檔案不是同一份」問題。
                config_references = {
                    "inputs.ocm_gap_safe_arrival_manifest": (
                        payload.get("inputs", {}).get("ocm_gap_safe_arrival_manifest")
                        if isinstance(payload.get("inputs"), Mapping)
                        else None
                    ),
                    "inputs.nww_full_hourly_analysis_manifest": (
                        payload.get("inputs", {}).get("nww_full_hourly_analysis_manifest")
                        if isinstance(payload.get("inputs"), Mapping)
                        else None
                    ),
                    "inputs.derived_input_artifact_index": (
                        payload.get("inputs", {}).get("derived_input_artifact_index")
                        if isinstance(payload.get("inputs"), Mapping)
                        else None
                    ),
                    "scenarios.material_manifest": (
                        payload.get("scenarios", {}).get("material_manifest")
                        if isinstance(payload.get("scenarios"), Mapping)
                        else None
                    ),
                    "scenarios.receptor_manifest": (
                        payload.get("scenarios", {}).get("receptor_manifest")
                        if isinstance(payload.get("scenarios"), Mapping)
                        else None
                    ),
                    "scenarios.arrival_time_manifest": (
                        payload.get("scenarios", {}).get("arrival_time_manifest")
                        if isinstance(payload.get("scenarios"), Mapping)
                        else None
                    ),
                    "scenarios.receptor_arrival_initial_condition_manifest": (
                        payload.get("scenarios", {}).get("receptor_arrival_initial_condition_manifest")
                        if isinstance(payload.get("scenarios"), Mapping)
                        else None
                    ),
                    "physics.settling.material_manifest": (
                        payload.get("physics", {}).get("settling", {}).get("material_manifest")
                        if isinstance(payload.get("physics"), Mapping)
                        and isinstance(payload.get("physics", {}).get("settling"), Mapping)
                        else None
                    ),
                    "geometry.domain_manifest": (
                        payload.get("geometry", {}).get("domain_manifest")
                        if isinstance(payload.get("geometry"), Mapping)
                        else None
                    ),
                    "geometry.local_domain_manifest": (
                        payload.get("geometry", {}).get("local_domain_manifest")
                        if isinstance(payload.get("geometry"), Mapping)
                        else None
                    ),
                    "geometry.open_boundary_manifest": (
                        payload.get("geometry", {}).get("open_boundary_manifest")
                        if isinstance(payload.get("geometry"), Mapping)
                        else None
                    ),
                    "geometry.receptor_manifest": (
                        payload.get("geometry", {}).get("receptor_manifest")
                        if isinstance(payload.get("geometry"), Mapping)
                        else None
                    ),
                }
                reference_to_kind = {
                    "inputs.ocm_gap_safe_arrival_manifest": "ocm_gap_safe_arrival_horizon",
                    "inputs.nww_full_hourly_analysis_manifest": "nww_full_hourly",
                    "inputs.derived_input_artifact_index": "artifact_index",
                    "scenarios.material_manifest": "material",
                    "scenarios.receptor_manifest": "receptor",
                    "scenarios.arrival_time_manifest": "arrival",
                    "scenarios.receptor_arrival_initial_condition_manifest": "initial_condition",
                    "physics.settling.material_manifest": "material",
                    "geometry.domain_manifest": "domain_geometry",
                    "geometry.local_domain_manifest": "local_geometry",
                    "geometry.open_boundary_manifest": "open_boundary",
                    "geometry.receptor_manifest": "receptor",
                }
                for label, value in config_references.items():
                    kind = reference_to_kind[label]
                    if value != expected_paths.get(kind):
                        errors.append(f"config_reference_mismatch:{label}")
        except Exception as exc:
            errors.append(f"input_directory_invalid:{type(exc).__name__}")
    if payload.get("config_status") == "approved" and formal:
        try:
            config = load_config(config_path, formal_release=False)
            config.assert_formal_release_ready()
        except Exception as exc:
            errors.append(f"approved_config_gate_failed:{type(exc).__name__}")
    elif payload.get("config_status") not in {"approved", "generated"}:
        errors.append("config_status_invalid")
    summary = {
        "config_status": payload.get("config_status"),
        "artifact_index_bound": bool(binding.get("input_directory_artifact_index_sha256")),
    }
    return {"valid": not errors, "errors": errors, "warnings": [], "summary": summary}


# 短命名 alias 保留給 CLI、外部 runbook 與未來測試；所有 alias 都指向同一實作，避免
# 不同入口各自產生略有差異的 manifest schema。
build_derived_inputs = build_input_derivatives
validate_derived_inputs = validate_input_derivatives
create_approved_release_config = create_release_config
validate_approved_release_config = validate_release_config


__all__ = [
    "ARTIFACT_FILENAMES",
    "DERIVED_INPUT_SCHEMA_VERSION",
    "EXPECTED_NWW_HOURLY_STEPS",
    "InputDerivationError",
    "InputDerivationResult",
    "build_derived_inputs",
    "build_input_derivatives",
    "canonical_json_bytes",
    "create_approved_release_config",
    "create_release_config",
    "read_canonical_json",
    "validate_approved_release_config",
    "validate_derived_inputs",
    "validate_input_derivatives",
    "validate_release_config",
    "write_canonical_json",
]
