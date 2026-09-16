"""正式情境與巢狀邊界 manifest 的嚴格讀取器。

本模組把可由人或上游產製的 JSON 文件轉成既有的不可變科學資料類別。JSON 只負責
交換 WGS84 經緯度、識別碼、來源與版本；進入運算前，幾何一定會依設定中的 flow
domain 中心投影成公尺，避免把經緯度直接拿去做距離、邊界或粒子步進。所有 loader
都採 fail-fast 策略：未知欄位、缺值、非有限數值、重複識別碼、錯誤 CRS 與不一致的
跨參照會在執行前拒絕，不以預設值或最近值補齊。

material manifest 為相容既有 ``lbt behavior-manifest`` 的 schema 2.0.0；該格式沒有
共同 provenance object，因此保留原本的分類來源、速度來源及校準範圍三欄。receptor、
legacy arrival-time 與三種幾何文件採 schema 1.0.0；新增沉底時間 metadata 的 arrival
採獨立 schema 1.1.0，loader 依 config 模式拒絕跨版本載入。所有 root 都必須含完整
provenance。情境清單仍以 tuple 保存，50,000 個基礎情境只由既有 deterministic
cross-product builder 產生，不另建 NumPy object array。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from shapely.geometry import LineString, MultiLineString, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .bed_residence import (
    BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
    BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
    BED_RESIDENCE_POLICY_ID,
    BED_RESIDENCE_SAMPLING_METHOD_ID,
    BED_RESIDENCE_SAMPLING_POLICY_ID,
    sample_bed_residence_age_hours,
)
from .boundaries import BoundaryGeometry
from .config import ProjectConfig, resolve_flow_domain_id
from .geometry import DomainProjection
from .input_horizon import (
    BED_RESIDENCE_INPUT_SCHEMA_VERSION,
    LEGACY_INPUT_SCHEMA_VERSION,
)
from .scenarios import (
    ArrivalTime,
    Behavior,
    Receptor,
    ReceptorArrivalInitialCondition,
    Scenario,
    build_scenarios,
    stable_identifier,
    validate_baseline_coverage,
    validate_non_rising_behaviors,
)

MANIFEST_SCHEMA_VERSION = LEGACY_INPUT_SCHEMA_VERSION
"""本模組新增 manifest 的固定 schema 版本；material CLI 的 ``2.0.0`` 另行相容。"""

_MATERIAL_SCHEMA_VERSION = "2.0.0"
_WGS84 = "EPSG:4326"
_VERTICAL_REFERENCE = "z_m_positive_up"
_SEASONS = frozenset({"DJF", "MAM", "JJA", "SON"})
_ARRIVAL_TIDE_CLASSES = ("spring_proxy", "neap_proxy")
_ARRIVAL_TIDAL_PHASES = ("fastest_rising", "fastest_falling", "slack_proxy")
_ARRIVAL_EVENTS = ("high_wave_event", "strong_current_event")
_BED_RESIDENCE_ARRIVAL_SELECTION_METHOD_ID = (
    "server_v3_48_strata_plus_two_observation_anchors_then_random_deposition_v1"
)
_MANIFEST_STATUSES = frozenset({"approved", "pilot", "generated"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_MATERIAL_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "design_version",
        "classification_source",
        "velocity_unit",
        "velocity_source",
        "positive_or_zero_velocity_policy",
        "calibration_scope",
        "records",
    }
)
_MATERIAL_RECORD_KEYS = frozenset(
    {
        "material_id",
        "oca_category_zh",
        "material_family_zh",
        "representative_shape_zh",
        "settling_velocity_mps",
        "behavior_class",
        "applicability_condition_zh",
        "calibration_status",
        "evidence_grade",
    }
)
_RECEPTOR_ROOT_KEYS = frozenset(
    {
        "manifest_kind",
        "schema_version",
        "status",
        "design_version",
        "coordinate_reference",
        "vertical_reference",
        "generation_method_id",
        "provenance",
        "records",
    }
)
_RECEPTOR_RECORD_KEYS = frozenset(
    {
        "receptor_id",
        "study_site_id",
        "analysis_region_id",
        "lon",
        "lat",
        "z_m_positive_up",
        "vertical_id",
        "metadata",
    }
)
_ARRIVAL_ROOT_KEYS = frozenset(
    {
        "manifest_kind",
        "schema_version",
        "status",
        "design_version",
        "time_standard",
        "selection_method_id",
        "provenance",
        "records",
    }
)
_ARRIVAL_RECORD_KEYS = frozenset(
    {
        "arrival_time_id",
        "study_site_id",
        "time_utc_ns",
        "year",
        "season",
        "tide_class",
        "phase_or_event",
        "metadata",
    }
)
_GEOMETRY_ROOT_KEYS = frozenset(
    {
        "manifest_kind",
        "schema_version",
        "status",
        "design_version",
        "coordinate_reference",
        "provenance",
        "records",
    }
)
_DOMAIN_RECORD_KEYS = frozenset(
    {"analysis_region_id", "flow_domain_id", "geometry", "source_geometry_id"}
)
_LOCAL_RECORD_KEYS = frozenset(
    {
        "study_site_id",
        "analysis_region_id",
        "flow_domain_id",
        "local_equals_flow",
        "geometry",
        "source_geometry_id",
    }
)
_OPEN_RECORD_KEYS = frozenset(
    {
        "owner_kind",
        "owner_id",
        "analysis_region_id",
        "segment_id",
        "geometry",
        "source_geometry_id",
    }
)
_RECEPTOR_ARRIVAL_INITIAL_CONDITION_MANIFEST_KIND = (
    "receptor_arrival_initial_condition_manifest"
)
_RECEPTOR_ARRIVAL_INITIAL_CONDITION_ROOT_KEYS = frozenset(
    {
        "manifest_kind",
        "schema_version",
        "status",
        "design_version",
        "vertical_reference",
        "time_standard",
        "generation_method_id",
        "provenance",
        "records",
    }
)
_RECEPTOR_ARRIVAL_INITIAL_CONDITION_RECORD_KEYS = frozenset(
    {
        "receptor_id",
        "arrival_time_id",
        "study_site_id",
        "analysis_region_id",
        "flow_domain_id",
        "time_utc_ns",
        "vertical_id",
        "z_m_positive_up",
        "eta_m_positive_up",
        "bed_z_m_positive_up",
        "water_column_height_m",
        "height_above_bed_m",
        "zcor_lower_m_positive_up",
        "zcor_upper_m_positive_up",
        "vertical_bracket_alpha",
        "source_face_local_index",
        "source_face_global_index",
        "wetdry_elem_value",
        "wetdry_semantics_id",
        "ocm_month_yyyymm",
        "ocm_source_time_index",
        "ocm_time_origin",
    }
)
_WETDRY_SEMANTICS_ID = "schism_wetdry_elem_0_wet_1_dry"
_DYNAMIC_INITIAL_CONDITION_ABS_TOL = 1e-8


def _reject_json_constant(value: str) -> None:
    """拒絕 JSON 非標準的 NaN、Infinity 與 -Infinity 常數。"""

    raise ValueError(f"JSON 不允許非有限常數：{value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """保留 JSON object 的重複 key 錯誤，不讓解析器靜默採用最後一筆。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object 不可重複欄位：{key}")
        result[key] = value
    return result


def _read_json(path: str | Path) -> tuple[dict[str, Any], str, str, Path]:
    """讀取嚴格 JSON，回傳 payload、原始檔 hash、canonical hash 與實際路徑。

    原始 hash 用於確認磁碟上的確切檔案；canonical hash 則忽略 JSON 排版與 object key
    順序，供 run/checkpoint manifest 綁定語意內容。兩種 hash 都以 UTF-8 bytes 計算。
    """

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest 必須是存在的普通檔案：{manifest_path}")
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"無法解析嚴格 JSON manifest：{manifest_path}；{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"manifest root 必須是 JSON object：{manifest_path}")
    try:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest 含不可 canonicalize 的值：{manifest_path}") from exc
    return payload, sha256(raw).hexdigest(), sha256(canonical).hexdigest(), manifest_path


def resolve_manifest_path(
    config_path: str | Path | None, manifest_path: str | Path
) -> Path:
    """依 YAML 檔案所在目錄解析 manifest 路徑。

    相對路徑不能依賴目前工作目錄，因為 SERVER 排程器、互動 shell 與 container 的 cwd
    可能不同。只有呼叫端明示的 config YAML 路徑能作為相對基準；絕對路徑則原樣保留。
    空字串、首尾空白及沒有 config 基準的相對路徑一律拒絕。
    """

    raw = str(manifest_path)
    if not raw or not raw.strip() or raw != raw.strip():
        raise ValueError("manifest path 不可為空白或含首尾空白")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    if config_path is None:
        raise ValueError("相對 manifest path 必須同時提供 config YAML 路徑")
    config_file = Path(config_path)
    if not str(config_file).strip() or str(config_file) != str(config_file).strip():
        raise ValueError("config YAML path 不可為空白或含首尾空白")
    # config_path 本身是 caller 明示的定位依據；先把它固定成絕對 parent，讓後續開啟
    # manifest 不會因執行程序從另一個 cwd 啟動而改變。這不是替 manifest 猜測 cwd。
    if not config_file.is_absolute():
        config_file = config_file.resolve()
    return config_file.parent / candidate


def _expect_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    """要求 object 欄位集合完全符合契約，並列出缺少或未知欄位。"""

    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"缺少={missing}")
        if unknown:
            details.append(f"未知={unknown}")
        raise ValueError(f"{label} 欄位不符合 extra-forbid 契約：{'；'.join(details)}")


def _nonempty_string(value: Any, label: str) -> str:
    """驗證不可為空且不可全是空白的文字欄位，不進行靜默 trim。"""

    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} 必須是沒有首尾空白的非空字串")
    return value


def _finite_float(value: Any, label: str) -> float:
    """接受 JSON integer/float 但拒絕 bool、NaN 與無限值，並轉成 Python float。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必須是有限數值且不可為 bool")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} 不可為 NaN 或 Infinity")
    return converted


def _integer(value: Any, label: str) -> int:
    """驗證真正的 JSON integer，避免 bool 被 Python 視為 int。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} 必須是真正的 integer，不可為 bool 或浮點數")
    return value


def _boolean(value: Any, label: str) -> bool:
    """驗證 JSON boolean，不接受 0/1 或其他可轉換值。"""

    if not isinstance(value, bool):
        raise ValueError(f"{label} 必須是 boolean")
    return value


def _metadata(value: Any, label: str) -> dict[str, float | int | str]:
    """驗證 metadata 僅含非空 key 及有限 float、非 bool int、非空 string scalar。"""

    if not isinstance(value, dict):
        raise ValueError(f"{label} 必須是 object")
    result: dict[str, float | int | str] = {}
    for key, item in value.items():
        clean_key = _nonempty_string(key, f"{label} key")
        if isinstance(item, bool) or item is None or isinstance(item, (list, dict)):
            raise ValueError(f"{label}.{clean_key} 必須是有限 float、非 bool int 或非空 string")
        if isinstance(item, str):
            result[clean_key] = _nonempty_string(item, f"{label}.{clean_key}")
        elif isinstance(item, int):
            result[clean_key] = item
        elif isinstance(item, float):
            result[clean_key] = _finite_float(item, f"{label}.{clean_key}")
        else:
            raise ValueError(f"{label}.{clean_key} 含不支援的 metadata 型別")
    return result


def _ensure_json_safe(value: Any, label: str) -> None:
    """遞迴確認 provenance 沒有 NaN、非 JSON scalar 或非字串 object key。"""

    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{label} 不可含空白字串")
        return
    if isinstance(value, float):
        _finite_float(value, label)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_json_safe(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _nonempty_string(key, f"{label} key")
            _ensure_json_safe(item, f"{label}.{key}")
        return
    raise ValueError(f"{label} 不是 JSON-safe value")


def _provenance(value: Any, label: str) -> dict[str, Any]:
    """驗證 provenance 的最低追溯欄位與 64 位小寫 SHA-256 source hash。"""

    if not isinstance(value, dict):
        raise ValueError(f"{label} 必須是 JSON object")
    _ensure_json_safe(value, label)
    method_id = _nonempty_string(value.get("method_id"), f"{label}.method_id")
    created = _nonempty_string(value.get("created_at_utc"), f"{label}.created_at_utc")
    try:
        parsed = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label}.created_at_utc 必須是 ISO8601 UTC 時間") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label}.created_at_utc 必須明示 UTC offset")
    source_hashes = value.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError(f"{label}.source_hashes 必須是非空 object")
    normalized_hashes: dict[str, str] = {}
    for key, digest in source_hashes.items():
        clean_key = _nonempty_string(key, f"{label}.source_hashes key")
        clean_digest = _nonempty_string(digest, f"{label}.source_hashes.{clean_key}")
        if _SHA256_RE.fullmatch(clean_digest) is None:
            raise ValueError(f"{label}.source_hashes.{clean_key} 必須是 64 位小寫 SHA-256")
        normalized_hashes[clean_key] = clean_digest
    return {**value, "method_id": method_id, "created_at_utc": created, "source_hashes": normalized_hashes}


def _status(value: Any, label: str, *, formal: bool) -> str:
    """驗證 manifest lifecycle status；formal run 只接受 approved。"""

    status = _nonempty_string(value, label)
    if status not in _MANIFEST_STATUSES:
        raise ValueError(f"{label} 不在允許的 manifest status：{sorted(_MANIFEST_STATUSES)}")
    if formal and status != "approved":
        raise ValueError(f"正式 manifest 必須是 approved：{label}={status}")
    return status


def _design_version(payload: Mapping[str, Any], config: ProjectConfig | None, label: str) -> str:
    """驗證設計版本，並在有設定時阻擋跨設計版本混用。"""

    design = _nonempty_string(payload.get("design_version"), f"{label}.design_version")
    if config is not None and design != config.design_version:
        raise ValueError(f"{label}.design_version 與 config 不一致：{design} != {config.design_version}")
    return design


def _records(value: Any, label: str) -> list[dict[str, Any]]:
    """驗證 records 是非空 object list，讓各 component 再套用欄位契約。"""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} 必須是非空 array")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{index}] 必須是 object")
        result.append(item)
    return result


def _config_sites(config: ProjectConfig) -> dict[str, Any]:
    """建立設定中的站點索引，供所有 component 交叉驗證共用。"""

    return {site.study_site_id: site for site in config.study_sites}


def _validate_pilot_sites(site_ids: set[str], config: ProjectConfig, label: str) -> None:
    """允許 pilot 僅使用設定站點子集，但拒絕未知站點與空集合。"""

    known = set(_config_sites(config))
    if not site_ids or not site_ids <= known:
        raise ValueError(f"{label} pilot 站點必須是 config 的非空子集：{sorted(site_ids)}")


def _validate_formal_counts(
    site_ids: set[str], counts: Counter[str], config: ProjectConfig, *, per_site: int, total: int, label: str
) -> None:
    """確認正式 component 恰涵蓋設定五站及其固定逐站／全案數量。"""

    expected = set(_config_sites(config))
    if site_ids != expected:
        raise ValueError(f"{label} formal 站點 coverage 不完整：{sorted(site_ids)}")
    if sum(counts.values()) != total or any(counts[site] != per_site for site in expected):
        raise ValueError(f"{label} formal 筆數不符：{dict(counts)}")


@dataclass(frozen=True, slots=True)
class _ManifestDocument:
    """內部保存解析後 payload 與兩種 hash，避免 component loader 重複讀檔。"""

    payload: dict[str, Any]
    file_sha256: str
    canonical_sha256: str
    path: Path


def _document(path: str | Path) -> _ManifestDocument:
    """建立內部 manifest document。"""

    payload, file_hash, canonical_hash, actual_path = _read_json(path)
    return _ManifestDocument(payload, file_hash, canonical_hash, actual_path)


def _load_material_document(
    path: str | Path, config: ProjectConfig | None, *, formal: bool
) -> tuple[tuple[Behavior, ...], _ManifestDocument]:
    """讀取相容既有 CLI schema 2.0.0 的 material 文件。"""

    document = _document(path)
    payload = document.payload
    _expect_exact_keys(payload, _MATERIAL_ROOT_KEYS, "material manifest root")
    if payload["schema_version"] != _MATERIAL_SCHEMA_VERSION:
        raise ValueError(f"material schema_version 必須是 {_MATERIAL_SCHEMA_VERSION}")
    design = _design_version(payload, config, "material manifest")
    _nonempty_string(payload["classification_source"], "classification_source")
    _nonempty_string(payload["velocity_unit"], "velocity_unit")
    _nonempty_string(payload["velocity_source"], "velocity_source")
    _nonempty_string(payload["calibration_scope"], "calibration_scope")
    if payload["positive_or_zero_velocity_policy"] != "reject_config":
        raise ValueError("material manifest 必須拒絕零速與正值")
    rows = _records(payload["records"], "material.records")
    behaviors: list[Behavior] = []
    for index, row in enumerate(rows):
        label = f"material.records[{index}]"
        _expect_exact_keys(row, _MATERIAL_RECORD_KEYS, label)
        text_fields = (
            "material_id",
            "oca_category_zh",
            "material_family_zh",
            "representative_shape_zh",
            "behavior_class",
            "applicability_condition_zh",
            "calibration_status",
            "evidence_grade",
        )
        values = {field: _nonempty_string(row[field], f"{label}.{field}") for field in text_fields}
        velocity = _finite_float(row["settling_velocity_mps"], f"{label}.settling_velocity_mps")
        behaviors.append(
            Behavior(
                material_id=values["material_id"],
                oca_category_zh=values["oca_category_zh"],
                material_family_zh=values["material_family_zh"],
                representative_shape_zh=values["representative_shape_zh"],
                settling_velocity_mps=velocity,
                behavior_class=values["behavior_class"],
                applicability_condition_zh=values["applicability_condition_zh"],
                calibration_status=values["calibration_status"],
                evidence_grade=values["evidence_grade"],
            )
        )
    try:
        validate_non_rising_behaviors(behaviors)
    except ValueError as exc:
        raise ValueError(f"material manifest 非上浮契約失敗：{exc}") from exc
    if formal:
        if config is None:
            raise ValueError("material formal loader 必須提供 ProjectConfig")
        if design != config.design_version or len(behaviors) != 10:
            raise ValueError("material formal 必須使用 config design_version 且恰有 10 筆")
        expected_categories = {
            item["oca_category_zh"]
            for item in config.physics.get("settling", {}).get("material_classes", [])
            if isinstance(item, dict) and isinstance(item.get("oca_category_zh"), str)
        }
        actual_categories = {item.oca_category_zh for item in behaviors}
        if expected_categories and actual_categories != expected_categories:
            raise ValueError("material formal 分類集合與 config 不一致")
    if config is not None:
        configured = config.physics.get("settling", {}).get("material_classes", [])
        configured_by_id = {
            item.get("material_id"): item for item in configured if isinstance(item, dict)
        }
        for item in behaviors:
            expected = configured_by_id.get(item.material_id)
            if expected is None:
                raise ValueError(f"material {item.material_id} 不存在於 config")
            for field in _MATERIAL_RECORD_KEYS:
                expected_value = expected.get(field)
                actual_value = getattr(
                    item,
                    {
                        "material_id": "material_id",
                        "oca_category_zh": "oca_category_zh",
                        "material_family_zh": "material_family_zh",
                        "representative_shape_zh": "representative_shape_zh",
                        "settling_velocity_mps": "settling_velocity_mps",
                        "behavior_class": "behavior_class",
                        "applicability_condition_zh": "applicability_condition_zh",
                        "calibration_status": "calibration_status",
                        "evidence_grade": "evidence_grade",
                    }[field],
                )
                if actual_value != expected_value:
                    raise ValueError(f"material {item.material_id} 欄位 {field} 與 config 不一致")
    return tuple(behaviors), document


def load_material_manifest(
    path: str | Path, config: ProjectConfig | None = None, *, formal: bool = False
) -> tuple[Behavior, ...]:
    """載入 material schema 2.0.0 並回傳 tuple[Behavior, ...]。

    這個公開函式刻意保留既有 CLI manifest 的 exact top-level/record keys，不加入新的
    ``status`` 或 ``provenance`` 欄位，避免破壞既有產物；formal 模式會依固定 v2 設計、
    config 版本及十類負值分類 gate 判定其可用性。速度仍是 provisional sensitivity
    proxy，不代表 iOcean 清除統計推得的類別平均物性。
    """

    records, _ = _load_material_document(path, config, formal=formal)
    return records


def _validate_component_root(
    payload: dict[str, Any],
    expected_keys: frozenset[str],
    *,
    kind: str,
    config: ProjectConfig,
    formal: bool,
    require_coordinate_reference: bool = True,
    expected_schema_version: str = MANIFEST_SCHEMA_VERSION,
) -> tuple[str, str, dict[str, Any]]:
    """驗證 component root 版本、status、設計與 provenance。

    一般 component 維持既有 1.0.0；隨機沉底 arrival 由呼叫端要求 1.1.0，避免新舊
    row metadata 欄位集合交叉載入。其餘 receptor、geometry 與 initial-condition 文件
    不因 bed-residence 功能而改 schema。
    """

    _expect_exact_keys(payload, expected_keys, f"{kind} manifest root")
    if payload["manifest_kind"] != kind:
        raise ValueError(f"manifest_kind 必須是 {kind}")
    if payload["schema_version"] != expected_schema_version:
        raise ValueError(f"{kind} schema_version 必須是 {expected_schema_version}")
    status = _status(payload["status"], f"{kind}.status", formal=formal)
    design = _design_version(payload, config, kind)
    if require_coordinate_reference and payload["coordinate_reference"] != _WGS84:
        raise ValueError(f"{kind} 只接受 {_WGS84}")
    provenance = _provenance(payload["provenance"], f"{kind}.provenance")
    return status, design, provenance


def _site_reference(
    row: Mapping[str, Any],
    config: ProjectConfig,
    *,
    site_field: str = "study_site_id",
    label: str,
    formal: bool = False,
) -> Any:
    """確認 row 的 site 及其可選 region/flow domain 都和 ProjectConfig 對應。

    arrival-time 的既有 ``ArrivalTime`` 資料類別沒有 ``analysis_region_id`` 與
    ``flow_domain_id`` 欄位，因此該 component 只需以 study site 反查設定；receptor
    與 local geometry 則會提供並嚴格驗證這兩個跨參照欄位。含 flow-domain 的 row 會
    依 ``formal`` 使用共用 resolver；因此 formal A 可綁定 expanded ID，pilot 則仍綁定
    config 的 base ID。
    """

    site_id = _nonempty_string(row.get(site_field), f"{label}.{site_field}")
    site = _config_sites(config).get(site_id)
    if site is None:
        raise ValueError(f"{label} 的 study_site_id 不存在於 config：{site_id}")
    if "analysis_region_id" in row:
        region = _nonempty_string(row.get("analysis_region_id"), f"{label}.analysis_region_id")
        if region != site.analysis_region_id:
            raise ValueError(f"{label} 的 analysis_region_id 與 config 不一致")
    if "flow_domain_id" in row:
        expected_flow_id = resolve_flow_domain_id(
            config, site.analysis_region_id, formal=formal
        )
        if row["flow_domain_id"] != expected_flow_id:
            raise ValueError(f"{label} 的 flow_domain_id 與 config resolver 不一致")
    return site


def _load_receptor_document(
    path: str | Path, config: ProjectConfig, *, formal: bool
) -> tuple[tuple[Receptor, ...], _ManifestDocument]:
    """讀取 receptor schema 1.0.0 並檢查站點 coverage 與三維座標。"""

    document = _document(path)
    payload = document.payload
    _validate_component_root(
        payload, _RECEPTOR_ROOT_KEYS, kind="receptor_manifest", config=config, formal=formal
    )
    if payload["vertical_reference"] != _VERTICAL_REFERENCE:
        raise ValueError(f"receptor vertical_reference 必須是 {_VERTICAL_REFERENCE}")
    _nonempty_string(payload["generation_method_id"], "receptor_manifest.generation_method_id")
    rows = _records(payload["records"], "receptor.records")
    receptors: list[Receptor] = []
    ids: set[str] = set()
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        label = f"receptor.records[{index}]"
        _expect_exact_keys(row, _RECEPTOR_RECORD_KEYS, label)
        receptor_id = _nonempty_string(row["receptor_id"], f"{label}.receptor_id")
        if receptor_id in ids:
            raise ValueError(f"receptor_id 必須全案唯一：{receptor_id}")
        ids.add(receptor_id)
        site = _site_reference(row, config, label=label, formal=formal)
        lon = _finite_float(row["lon"], f"{label}.lon")
        lat = _finite_float(row["lat"], f"{label}.lat")
        if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
            raise ValueError(f"{label} lon/lat 超出 WGS84 bounds")
        z_m = _finite_float(row["z_m_positive_up"], f"{label}.z_m_positive_up")
        vertical_id = _nonempty_string(row["vertical_id"], f"{label}.vertical_id")
        receptors.append(
            Receptor(
                receptor_id=receptor_id,
                study_site_id=site.study_site_id,
                analysis_region_id=site.analysis_region_id,
                lon=lon,
                lat=lat,
                z_m_positive_up=z_m,
                vertical_id=vertical_id,
                metadata=_metadata(row["metadata"], f"{label}.metadata"),
            )
        )
        counts[site.study_site_id] += 1
    site_ids = set(counts)
    if formal:
        _validate_formal_counts(site_ids, counts, config, per_site=20, total=100, label="receptor")
    else:
        _validate_pilot_sites(site_ids, config, "receptor")
    return tuple(receptors), document


def load_receptor_manifest(
    path: str | Path, config: ProjectConfig, *, formal: bool = False
) -> tuple[Receptor, ...]:
    """載入 receptor manifest 並回傳 tuple，位置仍是 WGS84 交換座標。

    receptor 的 ``z_m_positive_up`` 只保存模板代表／候選深度，不能取代 arrival-specific
    OCM actual z；正式初始條件須另由 ``load_receptor_arrival_initial_condition_manifest``
    載入。此函式只交換 WGS84 經緯度與模板 vertical_id，不把經緯度當作運算座標，後續
    geometry/runtime 應依各 flow domain 的固定中心進行公尺制轉換。
    """

    records, _ = _load_receptor_document(path, config, formal=formal)
    return records


def _utc_datetime(time_utc_ns: int, label: str) -> datetime:
    """把 UTC 奈秒轉成可驗證年份的 datetime，超出系統支援範圍即拒絕。"""

    seconds, _ = divmod(time_utc_ns, 1_000_000_000)
    try:
        return datetime.fromtimestamp(seconds, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"{label} 無法轉成 UTC datetime") from exc


def _validate_formal_arrival_strata(
    arrivals: tuple[ArrivalTime, ...], config: ProjectConfig
) -> None:
    """逐站驗證正式 48 個潮汐分層與 2 個極端事件都恰有一筆。

    核心分層完全沿用 ``select_arrival_times`` 的資料語意：設定中的兩個年份，分別交叉
    北半球氣候季標籤 DJF／MAM／JJA／SON、spring/neap 潮差代理，以及最快漲、最快退與
    平潮代理，共 48 格。另兩筆必須以 ``tide_class=event`` 登錄在地高波與強流事件。
    此檢查不以總數取代分層內容，避免重複某一格後仍以 50 筆通過正式輸入閘門。
    """

    years = tuple(config.inputs.years)
    if (
        len(years) != 2
        or len(set(years)) != 2
        or any(isinstance(year, bool) or not isinstance(year, int) for year in years)
    ):
        raise ValueError("arrival formal 必須由 config.inputs.years 提供兩個不同整數年份")
    expected_core = {
        (year, season, tide_class, phase)
        for year in years
        for season in _SEASONS
        for tide_class in _ARRIVAL_TIDE_CLASSES
        for phase in _ARRIVAL_TIDAL_PHASES
    }
    expected_events = set(_ARRIVAL_EVENTS)
    by_site: dict[str, list[ArrivalTime]] = {
        site_id: [] for site_id in _config_sites(config)
    }
    for arrival in arrivals:
        by_site[arrival.study_site_id].append(arrival)
    for site_id, site_arrivals in by_site.items():
        core_counts: Counter[tuple[int, str, str, str]] = Counter()
        event_counts: Counter[str] = Counter()
        for arrival in site_arrivals:
            if arrival.year not in years:
                raise ValueError(
                    f"arrival formal {site_id} 含 config.inputs.years 以外年份：{arrival.year}"
                )
            if arrival.tide_class == "event":
                if arrival.phase_or_event not in expected_events:
                    raise ValueError(
                        f"arrival formal {site_id} 的 event 類別不合法：{arrival.phase_or_event}"
                    )
                event_counts[arrival.phase_or_event] += 1
                continue
            if (
                arrival.tide_class not in _ARRIVAL_TIDE_CLASSES
                or arrival.phase_or_event not in _ARRIVAL_TIDAL_PHASES
            ):
                raise ValueError(
                    "arrival formal 核心 strata 只接受 spring_proxy/neap_proxy 與三種既定相位"
                )
            core_counts[
                (
                    arrival.year,
                    arrival.season,
                    arrival.tide_class,
                    arrival.phase_or_event,
                )
            ] += 1
        if set(core_counts) != expected_core or any(count != 1 for count in core_counts.values()):
            missing = len(expected_core - set(core_counts))
            duplicate = sum(count - 1 for count in core_counts.values() if count > 1)
            raise ValueError(
                f"arrival formal {site_id} 的 48 個核心 strata 不完整或重複："
                f"missing={missing}, duplicate={duplicate}"
            )
        if set(event_counts) != expected_events or any(
            event_counts[event] != 1 for event in expected_events
        ):
            raise ValueError(
                f"arrival formal {site_id} 必須恰含 high_wave_event 與 strong_current_event 各一筆"
            )


def _season_for_observation_month(month: int) -> str:
    """依原 observation UTC 月份回傳既有北半球氣候季標籤。"""

    if month in {12, 1, 2}:
        return "DJF"
    if month in {3, 4, 5}:
        return "MAM"
    if month in {6, 7, 8}:
        return "JJA"
    return "SON"


def _validate_bed_residence_arrival_records(
    arrivals: tuple[ArrivalTime, ...],
    provenance: Mapping[str, Any],
    config: ProjectConfig,
) -> tuple[ArrivalTime, ...]:
    """重算每站共用的抽樣年齡，並回傳原觀測錨點供 48+2 分層驗證。

    arrival record 的 ``time_utc_ns`` 是粒子正式起算的沉底時刻；潮汐、季節與事件分層
    則必須可追溯回沉底前選出的 observation anchor。metadata 同時保存兩個 UTC 與
    age/stratum/seed/policy，這個 loader 依 config seed 重新抽出完整 age vector，逐站
    檢查相同 50 個分層、沉底時間差與 A 區 paired observation UTC。因而人工改一個 age、
    seed、observation 或 deposition 時刻，即使重簽 JSON hash 仍會在此被拒絕。
    """

    bed = config.scenarios.bed_residence_time
    if bed is None:
        raise ValueError("bed residence loader 缺少 config.scenarios.bed_residence_time")
    if bed.backtrack_mode not in {
        BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
        BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
    }:
        raise ValueError("bed residence backtrack_mode 未登錄")
    if bed.backtrack_mode not in bed.supported_backtrack_modes:
        raise ValueError("bed residence backtrack_mode 不在 supported_backtrack_modes")
    if bed.shared_age_offsets_across_sites is not True:
        raise ValueError("正式五站設計必須共用同一 50-age vector")
    offsets = sample_bed_residence_age_hours(
        maximum_age_days=bed.maximum_age_days,
        sample_count=bed.sample_count_per_site,
        seed=bed.sampling_seed,
    )
    if len(offsets) != bed.sample_count_per_site:
        raise ValueError("重算的 bed residence age vector 長度與設定不一致")
    age_vector_hash = sha256(
        json.dumps(list(offsets), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_provenance = {
        "policy_id": BED_RESIDENCE_POLICY_ID,
        "sampling_policy_id": BED_RESIDENCE_SAMPLING_POLICY_ID,
        "maximum_age_days": bed.maximum_age_days,
        "sample_count_per_site": bed.sample_count_per_site,
        "sampling_seed": bed.sampling_seed,
        "shared_age_offsets_across_sites": True,
        "age_offsets_hours_sha256": age_vector_hash,
    }
    sampling_provenance = provenance.get("bed_residence_sampling")
    if not isinstance(sampling_provenance, Mapping) or any(
        sampling_provenance.get(field) != expected for field, expected in expected_provenance.items()
    ):
        raise ValueError("arrival provenance 的 bed-residence sampling 不可由設定 seed 重現")
    if provenance.get("method_id") != _BED_RESIDENCE_ARRIVAL_SELECTION_METHOD_ID:
        raise ValueError("arrival provenance 未登錄 bed residence observation selector")
    if sampling_provenance.get("observation_anchor_selection_method_id") != (
        "server_v3_48_strata_plus_two_events_gap_safe_nww_metric_location_v2"
    ):
        raise ValueError("arrival provenance 缺少既有 48+2 observation selector 識別碼")
    selection_support = config.inputs.backtrack_support_days
    runtime_support = bed.runtime_horizon_support_days
    if runtime_support is None:
        requested = config.boundaries.max_backtrack_days
        runtime_support = int(requested) if requested is not None and float(requested).is_integer() else None
    if (
        selection_support is None
        or sampling_provenance.get("selection_support_days") != selection_support
        or sampling_provenance.get("runtime_support_days") != runtime_support
        or sampling_provenance.get("pre_window_policy") != bed.pre_window_policy
    ):
        raise ValueError("arrival provenance 的 selection/runtime support 與 config 不一致")

    expected_site_ids = set(_config_sites(config))
    # ``stratum_index`` 標記年齡落在哪個抽樣分層，不是觀測到達列在抽樣向量中的位置；
    # 因此各站的向量核對必須另按原 observation UTC 與原始 ID 排序。
    per_site: dict[str, list[tuple[int, str, int]]] = {
        site_id: [] for site_id in expected_site_ids
    }
    per_site_strata: dict[str, set[int]] = {
        site_id: set() for site_id in expected_site_ids
    }
    observation_ids: dict[str, set[str]] = {site_id: set() for site_id in expected_site_ids}
    observation_arrivals: list[ArrivalTime] = []
    hour_ns = 3_600_000_000_000
    design_hash = sha256(config.design_version.encode("utf-8")).hexdigest()
    for arrival in arrivals:
        metadata = arrival.metadata
        site_id = arrival.study_site_id
        if site_id not in per_site:
            raise ValueError(f"bed residence arrival 出現未設定站點：{site_id}")
        observation_ns = _integer(
            metadata.get("observation_time_utc_ns"),
            f"arrival[{arrival.arrival_time_id}].metadata.observation_time_utc_ns",
        )
        deposition_ns = _integer(
            metadata.get("deposition_time_utc_ns"),
            f"arrival[{arrival.arrival_time_id}].metadata.deposition_time_utc_ns",
        )
        age_hours = _integer(
            metadata.get("bed_residence_age_hours"),
            f"arrival[{arrival.arrival_time_id}].metadata.bed_residence_age_hours",
        )
        stratum_index = _integer(
            metadata.get("bed_residence_stratum_index"),
            f"arrival[{arrival.arrival_time_id}].metadata.bed_residence_stratum_index",
        )
        sampling_seed = _integer(
            metadata.get("bed_residence_sampling_seed"),
            f"arrival[{arrival.arrival_time_id}].metadata.bed_residence_sampling_seed",
        )
        if metadata.get("bed_residence_policy_id") != BED_RESIDENCE_POLICY_ID:
            raise ValueError(f"arrival[{arrival.arrival_time_id}] bed residence policy 不符")
        if metadata.get("bed_residence_sampling_policy_id") != BED_RESIDENCE_SAMPLING_POLICY_ID:
            raise ValueError(f"arrival[{arrival.arrival_time_id}] bed residence sampling policy 不符")
        if metadata.get("bed_residence_sampling_method_id") != BED_RESIDENCE_SAMPLING_METHOD_ID:
            raise ValueError(f"arrival[{arrival.arrival_time_id}] bed residence sampling method 不符")
        if sampling_seed != bed.sampling_seed:
            raise ValueError(f"arrival[{arrival.arrival_time_id}] bed residence seed 與 config 不一致")
        if metadata.get("bed_residence_maximum_age_days") != bed.maximum_age_days:
            raise ValueError(f"arrival[{arrival.arrival_time_id}] bed residence 最大年齡與 config 不一致")
        if metadata.get("bed_residence_design_version") != config.design_version:
            raise ValueError(f"arrival[{arrival.arrival_time_id}] bed residence design version 不一致")
        if metadata.get("bed_residence_design_hash") != design_hash:
            raise ValueError(f"arrival[{arrival.arrival_time_id}] bed residence design hash 不一致")
        if not 0 <= age_hours <= bed.maximum_age_days * 24:
            raise ValueError(f"arrival[{arrival.arrival_time_id}] bed residence age 超出設定上限")
        if not 0 <= stratum_index < bed.sample_count_per_site:
            raise ValueError(f"arrival[{arrival.arrival_time_id}] bed residence stratum index 無效")
        if stratum_index in per_site_strata[site_id]:
            raise ValueError(f"{site_id} bed residence stratum 重複：{stratum_index}")
        # 使用與 bed_residence 抽樣器相同的整數切分公式，檢查 stratum_index 確實是
        # 該 age_hours 所屬的分層；它不代表 age_offsets 向量的位置，兩者不可互換。
        total_hour_count = bed.maximum_age_days * 24 + 1
        stratum_start = (stratum_index * total_hour_count) // bed.sample_count_per_site
        stratum_stop = ((stratum_index + 1) * total_hour_count) // bed.sample_count_per_site
        if not stratum_start <= age_hours < stratum_stop:
            raise ValueError(f"arrival[{arrival.arrival_time_id}] age 不在標示的抽樣分層")
        if deposition_ns != arrival.time_utc_ns or deposition_ns != observation_ns - age_hours * hour_ns:
            raise ValueError(f"arrival[{arrival.arrival_time_id}] observation/deposition UTC 關係不一致")
        observation_utc_text = _nonempty_string(
            metadata.get("observation_time_utc"),
            f"arrival[{arrival.arrival_time_id}].metadata.observation_time_utc",
        )
        deposition_utc_text = _nonempty_string(
            metadata.get("deposition_time_utc"),
            f"arrival[{arrival.arrival_time_id}].metadata.deposition_time_utc",
        )
        expected_observation_text = (
            _utc_datetime(observation_ns, "observation_time_utc_ns")
            .isoformat()
            .replace("+00:00", "Z")
        )
        expected_deposition_text = _utc_datetime(deposition_ns, "deposition_time_utc_ns").isoformat().replace(
            "+00:00", "Z"
        )
        if (
            observation_utc_text != expected_observation_text
            or deposition_utc_text != expected_deposition_text
        ):
            raise ValueError(f"arrival[{arrival.arrival_time_id}] UTC 字串與 nanoseconds 不一致")
        observation_id = _nonempty_string(
            metadata.get("observation_arrival_time_id"),
            f"arrival[{arrival.arrival_time_id}].metadata.observation_arrival_time_id",
        )
        # 一般 event selector 的 ID 使用 event 名稱四欄；A 區龜山 paired clone 則刻意
        # 以潮汐類別與 phase 保留來源站位的 paired identity，包含兩筆 event 也使用五欄。
        # 必須依 clone provenance 判斷，不能只看 tide_class == "event"。
        is_paired_a_clone = (
            metadata.get("shared_A_forcing_reference_site") == "gongliao"
            and metadata.get("shared_A_forcing_policy") == "gongliao_paired_utc_reference_v1"
        )
        if arrival.tide_class == "event" and not is_paired_a_clone:
            observation_identity_fields = [
                site_id,
                str(observation_ns),
                arrival.phase_or_event,
                config.design_version,
            ]
        else:
            observation_identity_fields = [
                site_id,
                str(observation_ns),
                arrival.tide_class,
                arrival.phase_or_event,
                config.design_version,
            ]
        if observation_id != stable_identifier("arr", observation_identity_fields):
            raise ValueError(f"arrival[{arrival.arrival_time_id}] observation arrival identity 不可重算")
        if observation_id in observation_ids[site_id]:
            raise ValueError(f"{site_id} observation_arrival_time_id 重複：{observation_id}")
        observation_ids[site_id].add(observation_id)
        observation = _utc_datetime(observation_ns, "observation_time_utc_ns")
        observation_year = _integer(
            metadata.get("observation_year"),
            f"arrival[{arrival.arrival_time_id}].metadata.observation_year",
        )
        observation_season = _nonempty_string(
            metadata.get("observation_season"),
            f"arrival[{arrival.arrival_time_id}].metadata.observation_season",
        )
        deposition = _utc_datetime(deposition_ns, "deposition_time_utc_ns")
        if (
            observation_year != observation.year
            or observation_season != _season_for_observation_month(observation.month)
        ):
            raise ValueError(f"arrival[{arrival.arrival_time_id}] observation 年份／季節 metadata 不符")
        if (
            arrival.year != deposition.year
            or arrival.season != _season_for_observation_month(deposition.month)
        ):
            raise ValueError(f"arrival[{arrival.arrival_time_id}] 頂層年份／季節未依 deposition UTC 重算")
        expected_arrival_id = stable_identifier(
            "arrival_bed",
            [
                observation_id,
                str(deposition_ns),
                BED_RESIDENCE_POLICY_ID,
                BED_RESIDENCE_SAMPLING_POLICY_ID,
                str(bed.sampling_seed),
                design_hash,
            ],
        )
        if arrival.arrival_time_id != expected_arrival_id:
            raise ValueError(f"arrival[{arrival.arrival_time_id}] bed residence identity 不可重算")
        per_site[site_id].append((observation_ns, observation_id, age_hours))
        per_site_strata[site_id].add(stratum_index)
        observation_arrivals.append(
            ArrivalTime(
                arrival_time_id=observation_id,
                study_site_id=site_id,
                time_utc_ns=observation_ns,
                year=observation_year,
                season=observation_season,
                tide_class=arrival.tide_class,
                phase_or_event=arrival.phase_or_event,
                metadata={},
            )
        )

    expected_strata = set(range(bed.sample_count_per_site))
    for site_id, indexed in per_site.items():
        if per_site_strata[site_id] != expected_strata:
            raise ValueError(
                f"{site_id} bed residence strata 不完整："
                f"{len(per_site_strata[site_id])}/{len(expected_strata)}"
            )
        ordered = sorted(indexed, key=lambda item: (item[0], item[1]))
        actual_offsets = tuple(item[2] for item in ordered)
        if actual_offsets != offsets:
            raise ValueError(f"{site_id} age vector 與其他站點或 seed 抽樣結果不一致")
    if {"gongliao", "guishan"} <= expected_site_ids:
        paired_gongliao = sorted(per_site["gongliao"], key=lambda item: (item[0], item[1]))
        paired_guishan = sorted(per_site["guishan"], key=lambda item: (item[0], item[1]))
        if [item[0] for item in paired_gongliao] != [item[0] for item in paired_guishan]:
            raise ValueError("A 區 paired observation UTC 不一致")
    return tuple(observation_arrivals)


def _load_arrival_document(
    path: str | Path, config: ProjectConfig, *, formal: bool
) -> tuple[tuple[ArrivalTime, ...], _ManifestDocument]:
    """依設定讀取 legacy 1.0.0 或 bed-residence 1.1.0 arrival 文件。

    schema 版本與 config 模式必須一對一，避免把只含 observation UTC 的舊列誤當成
    deposition 起點，或讓 legacy runtime 意外載入含沉底轉換的 row metadata。
    """

    document = _document(path)
    payload = document.payload
    bed_residence_enabled = config.scenarios.bed_residence_time is not None
    _, _, provenance = _validate_component_root(
        payload,
        _ARRIVAL_ROOT_KEYS,
        kind="arrival_time_manifest",
        config=config,
        formal=formal,
        require_coordinate_reference=False,
        expected_schema_version=(
            BED_RESIDENCE_INPUT_SCHEMA_VERSION
            if bed_residence_enabled
            else MANIFEST_SCHEMA_VERSION
        ),
    )
    if payload["time_standard"] != "UTC":
        raise ValueError("arrival_time_manifest.time_standard 必須是 UTC")
    _nonempty_string(payload["selection_method_id"], "arrival_time_manifest.selection_method_id")
    rows = _records(payload["records"], "arrival.records")
    if (
        bed_residence_enabled
        and payload["selection_method_id"] != _BED_RESIDENCE_ARRIVAL_SELECTION_METHOD_ID
    ):
        raise ValueError("bed residence arrival_time_manifest.selection_method_id 不符")
    if (
        not bed_residence_enabled
        and payload["selection_method_id"] == _BED_RESIDENCE_ARRIVAL_SELECTION_METHOD_ID
    ):
        raise ValueError("legacy config 不得載入 bed residence arrival manifest")
    arrivals: list[ArrivalTime] = []
    ids: set[str] = set()
    site_times: set[tuple[str, int]] = set()
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        label = f"arrival.records[{index}]"
        _expect_exact_keys(row, _ARRIVAL_RECORD_KEYS, label)
        arrival_id = _nonempty_string(row["arrival_time_id"], f"{label}.arrival_time_id")
        if arrival_id in ids:
            raise ValueError(f"arrival_time_id 必須全案唯一：{arrival_id}")
        ids.add(arrival_id)
        site = _site_reference(row, config, label=label, formal=formal)
        time_ns = _integer(row["time_utc_ns"], f"{label}.time_utc_ns")
        utc = _utc_datetime(time_ns, f"{label}.time_utc_ns")
        if (site.study_site_id, time_ns) in site_times:
            raise ValueError(f"同一 study_site 的 UTC 必須唯一：{site.study_site_id}/{time_ns}")
        site_times.add((site.study_site_id, time_ns))
        year = _integer(row["year"], f"{label}.year")
        season = _nonempty_string(row["season"], f"{label}.season")
        if season not in _SEASONS:
            raise ValueError(f"{label}.season 不合法：{sorted(_SEASONS)}")
        metadata = _metadata(row["metadata"], f"{label}.metadata")
        if bed_residence_enabled:
            deposition_ns = _integer(
                metadata.get("deposition_time_utc_ns"),
                f"{label}.metadata.deposition_time_utc_ns",
            )
            deposition_utc = _utc_datetime(deposition_ns, f"{label}.metadata.deposition_time_utc_ns")
            if time_ns != deposition_ns:
                raise ValueError(f"{label}.time_utc_ns 必須等於 metadata deposition UTC")
            if year != deposition_utc.year or season != _season_for_observation_month(deposition_utc.month):
                raise ValueError(
                    f"{label}.year／season 與 deposition UTC 不一致：{year}/{season}"
                )
        elif year != utc.year:
            raise ValueError(f"{label}.year 與 UTC year 不一致：{year} != {utc.year}")
        arrivals.append(
            ArrivalTime(
                arrival_time_id=arrival_id,
                study_site_id=site.study_site_id,
                time_utc_ns=time_ns,
                year=year,
                season=season,
                tide_class=_nonempty_string(row["tide_class"], f"{label}.tide_class"),
                phase_or_event=_nonempty_string(row["phase_or_event"], f"{label}.phase_or_event"),
                metadata=metadata,
            )
        )
        counts[site.study_site_id] += 1
    site_ids = set(counts)
    if formal:
        _validate_formal_counts(site_ids, counts, config, per_site=50, total=250, label="arrival")
    if bed_residence_enabled:
        observation_arrivals = _validate_bed_residence_arrival_records(
            tuple(arrivals), provenance, config
        )
        if formal:
            # bed-residence arrival 的執行 UTC 已轉為沉底時刻；正式 48+2 統計仍以 metadata
            # 可重建的 observation anchor 分層，避免沉底年齡跨季或跨年改寫原始選時設計。
            _validate_formal_arrival_strata(observation_arrivals, config)
    elif formal:
        _validate_formal_arrival_strata(tuple(arrivals), config)
    else:
        _validate_pilot_sites(site_ids, config, "arrival")
    return tuple(arrivals), document


def load_arrival_time_manifest(
    path: str | Path, config: ProjectConfig, *, formal: bool = False
) -> tuple[ArrivalTime, ...]:
    """載入 UTC arrival-time manifest 並回傳 tuple[ArrivalTime, ...]。"""

    records, _ = _load_arrival_document(path, config, formal=formal)
    return records


def load_arrival_manifest(
    path: str | Path, config: ProjectConfig, *, formal: bool = False
) -> tuple[ArrivalTime, ...]:
    """``load_arrival_time_manifest`` 的簡短命名 alias，保持 API 意義清楚且不複製邏輯。"""

    return load_arrival_time_manifest(path, config, formal=formal)


def _dynamic_initial_condition_root(
    payload: dict[str, Any], config: ProjectConfig, *, formal: bool
) -> tuple[str, str, dict[str, Any]]:
    """驗證 receptor×arrival 初始條件 manifest 的 root 與正式來源契約。"""

    _expect_exact_keys(
        payload,
        _RECEPTOR_ARRIVAL_INITIAL_CONDITION_ROOT_KEYS,
        "receptor_arrival_initial_condition manifest root",
    )
    if payload["manifest_kind"] != _RECEPTOR_ARRIVAL_INITIAL_CONDITION_MANIFEST_KIND:
        raise ValueError(
            "manifest_kind 必須是 "
            f"{_RECEPTOR_ARRIVAL_INITIAL_CONDITION_MANIFEST_KIND}"
        )
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "receptor_arrival_initial_condition schema_version 必須是 "
            f"{MANIFEST_SCHEMA_VERSION}"
        )
    status = _status(
        payload["status"], "receptor_arrival_initial_condition.status", formal=formal
    )
    design = _design_version(payload, config, "receptor_arrival_initial_condition")
    if payload["vertical_reference"] != _VERTICAL_REFERENCE:
        raise ValueError(
            "receptor_arrival_initial_condition.vertical_reference 必須是 "
            f"{_VERTICAL_REFERENCE}"
        )
    if payload["time_standard"] != "UTC":
        raise ValueError("receptor_arrival_initial_condition.time_standard 必須是 UTC")
    _nonempty_string(
        payload["generation_method_id"],
        "receptor_arrival_initial_condition.generation_method_id",
    )
    provenance = _provenance(
        payload["provenance"], "receptor_arrival_initial_condition.provenance"
    )
    return status, design, provenance


def _dynamic_close(actual: float, expected: float, label: str) -> None:
    """以固定 1e-8 m／無因次小公差驗證推導數值等式。

    公差只吸收 JSON 浮點序列化造成的最後幾位差異；它不放寬 eta/bed、z 範圍或垂向
    bracket 的物理不等式。所有輸入仍先經有限值檢查，避免以 NaN 破壞 ``isclose``。
    """

    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=_DYNAMIC_INITIAL_CONDITION_ABS_TOL,
    ):
        raise ValueError(f"{label} 與 OCM 推導值不一致：{actual} != {expected}")


def _dynamic_template_indexes(
    config: ProjectConfig,
    receptors: Sequence[Receptor],
    arrivals: Sequence[ArrivalTime],
    *,
    formal: bool,
) -> tuple[dict[str, Receptor], dict[str, ArrivalTime], set[str]]:
    """建立模板索引並確認 formal 或 pilot 的 receptor／arrival 站點集合。"""

    receptors_by_id: dict[str, Receptor] = {}
    arrivals_by_id: dict[str, ArrivalTime] = {}
    receptor_counts: Counter[str] = Counter()
    arrival_counts: Counter[str] = Counter()
    for receptor in receptors:
        if receptor.receptor_id in receptors_by_id:
            raise ValueError(f"receptor_id 不可重複：{receptor.receptor_id}")
        receptors_by_id[receptor.receptor_id] = receptor
        receptor_counts[receptor.study_site_id] += 1
    for arrival in arrivals:
        if arrival.arrival_time_id in arrivals_by_id:
            raise ValueError(f"arrival_time_id 不可重複：{arrival.arrival_time_id}")
        arrivals_by_id[arrival.arrival_time_id] = arrival
        arrival_counts[arrival.study_site_id] += 1
    receptor_sites = set(receptor_counts)
    arrival_sites = set(arrival_counts)
    if receptor_sites != arrival_sites:
        raise ValueError("dynamic initial conditions 的 receptor/arrival 站點集合不一致")
    if formal:
        expected_sites = set(_config_sites(config))
        if receptor_sites != expected_sites:
            raise ValueError("dynamic initial conditions formal 必須涵蓋 config 五站")
        if any(receptor_counts[site] != 20 for site in expected_sites):
            raise ValueError("dynamic initial conditions formal receptor 必須每站 20 筆")
        if any(arrival_counts[site] != 50 for site in expected_sites):
            raise ValueError("dynamic initial conditions formal arrival 必須每站 50 筆")
    else:
        _validate_pilot_sites(receptor_sites, config, "dynamic initial conditions")
    return receptors_by_id, arrivals_by_id, receptor_sites


def _load_receptor_arrival_initial_condition_document(
    path: str | Path,
    config: ProjectConfig,
    receptors: Sequence[Receptor],
    arrivals: Sequence[ArrivalTime],
    *,
    formal: bool,
) -> tuple[tuple[ReceptorArrivalInitialCondition, ...], _ManifestDocument]:
    """讀取並驗證 OCM-derived receptor×arrival 初始深度資料列。"""

    document = _document(path)
    payload = document.payload
    _dynamic_initial_condition_root(payload, config, formal=formal)
    rows = _records(payload["records"], "receptor_arrival_initial_condition.records")
    receptors_by_id, arrivals_by_id, _ = _dynamic_template_indexes(
        config, receptors, arrivals, formal=formal
    )
    expected_flow_by_region = {
        region: resolve_flow_domain_id(config, region, formal=formal)
        for region in {item.analysis_region_id for item in config.domains}
    }
    result: list[ReceptorArrivalInitialCondition] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        label = f"receptor_arrival_initial_condition.records[{index}]"
        _expect_exact_keys(row, _RECEPTOR_ARRIVAL_INITIAL_CONDITION_RECORD_KEYS, label)
        receptor_id = _nonempty_string(row["receptor_id"], f"{label}.receptor_id")
        arrival_id = _nonempty_string(row["arrival_time_id"], f"{label}.arrival_time_id")
        pair = (receptor_id, arrival_id)
        if pair in seen_pairs:
            raise ValueError(f"receptor×arrival pair 不可重複：{pair}")
        seen_pairs.add(pair)
        receptor = receptors_by_id.get(receptor_id)
        arrival = arrivals_by_id.get(arrival_id)
        if receptor is None:
            raise ValueError(f"未知 receptor_id：{receptor_id}")
        if arrival is None:
            raise ValueError(f"未知 arrival_time_id：{arrival_id}")
        if receptor.study_site_id != arrival.study_site_id:
            raise ValueError(f"receptor/arrival 必須屬於同一 study_site：{pair}")
        study_site_id = _nonempty_string(row["study_site_id"], f"{label}.study_site_id")
        analysis_region_id = _nonempty_string(
            row["analysis_region_id"], f"{label}.analysis_region_id"
        )
        flow_domain_id = _nonempty_string(row["flow_domain_id"], f"{label}.flow_domain_id")
        if study_site_id != receptor.study_site_id:
            raise ValueError(f"{label}.study_site_id 與 receptor/arrival 不一致")
        if analysis_region_id != receptor.analysis_region_id:
            raise ValueError(f"{label}.analysis_region_id 與 receptor 不一致")
        expected_flow_id = expected_flow_by_region[receptor.analysis_region_id]
        if flow_domain_id != expected_flow_id:
            raise ValueError(
                f"{label}.flow_domain_id 與 {('formal' if formal else 'pilot')} resolver 不一致"
            )
        time_ns = _integer(row["time_utc_ns"], f"{label}.time_utc_ns")
        if time_ns != arrival.time_utc_ns:
            raise ValueError(f"{label}.time_utc_ns 與 arrival 不一致")
        vertical_id = _nonempty_string(row["vertical_id"], f"{label}.vertical_id")
        if vertical_id != receptor.vertical_id:
            raise ValueError(f"{label}.vertical_id 與 receptor 模板不一致")
        eta = _finite_float(row["eta_m_positive_up"], f"{label}.eta_m_positive_up")
        bed = _finite_float(row["bed_z_m_positive_up"], f"{label}.bed_z_m_positive_up")
        z = _finite_float(row["z_m_positive_up"], f"{label}.z_m_positive_up")
        water = _finite_float(row["water_column_height_m"], f"{label}.water_column_height_m")
        height = _finite_float(row["height_above_bed_m"], f"{label}.height_above_bed_m")
        lower = _finite_float(
            row["zcor_lower_m_positive_up"], f"{label}.zcor_lower_m_positive_up"
        )
        upper = _finite_float(
            row["zcor_upper_m_positive_up"], f"{label}.zcor_upper_m_positive_up"
        )
        alpha = _finite_float(row["vertical_bracket_alpha"], f"{label}.vertical_bracket_alpha")
        if eta <= bed:
            raise ValueError(f"{label} 必須滿足 eta > bed")
        _dynamic_close(water, eta - bed, f"{label}.water_column_height_m")
        if not bed <= z <= eta:
            raise ValueError(f"{label}.z_m_positive_up 必須位於 bed 與 eta 之間")
        if not 0.0 <= height <= water:
            raise ValueError(f"{label}.height_above_bed_m 必須位於 0 與 water column 之間")
        _dynamic_close(height, z - bed, f"{label}.height_above_bed_m")
        if upper <= lower:
            raise ValueError(f"{label} 必須滿足 zcor_upper > zcor_lower")
        # zcor bracket 是 OCM 垂向座標中包住粒子實際 z 的相鄰端點，兩端本身也必須
        # 屬於同一個有效水柱。只容許不超過固定序列化公差的端點尾差，因為 JSON
        # 浮點輸出可能在 bed 或 eta 的最後幾位產生誤差；若超過此公差，便是把
        # bracket 放到實際水體外，不得以「仍包住 z」為理由放行。這個 gate 不改變
        # eta > bed、bed <= z <= eta 等真正的物理不等式。
        if lower < bed - _DYNAMIC_INITIAL_CONDITION_ABS_TOL:
            raise ValueError(f"{label}.zcor_lower_m_positive_up 不可低於 bed")
        if upper > eta + _DYNAMIC_INITIAL_CONDITION_ABS_TOL:
            raise ValueError(f"{label}.zcor_upper_m_positive_up 不可高於 eta")
        if not lower <= z <= upper:
            raise ValueError(f"{label}.z_m_positive_up 必須位於 zcor bracket 之間")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"{label}.vertical_bracket_alpha 必須位於 [0, 1]")
        _dynamic_close(
            alpha,
            (z - lower) / (upper - lower),
            f"{label}.vertical_bracket_alpha",
        )
        source_face_local = _integer(
            row["source_face_local_index"], f"{label}.source_face_local_index"
        )
        source_face_global = _integer(
            row["source_face_global_index"], f"{label}.source_face_global_index"
        )
        source_time_index = _integer(
            row["ocm_source_time_index"], f"{label}.ocm_source_time_index"
        )
        if source_face_local < 0 or source_face_global < 0 or source_time_index < 0:
            raise ValueError(f"{label} 的 OCM source index 必須是非負整數")
        wetdry = _integer(row["wetdry_elem_value"], f"{label}.wetdry_elem_value")
        if wetdry != 0:
            raise ValueError(f"{label}.wetdry_elem_value=1 表示乾元素，不可作執行初始條件")
        semantics = _nonempty_string(
            row["wetdry_semantics_id"], f"{label}.wetdry_semantics_id"
        )
        if semantics != _WETDRY_SEMANTICS_ID:
            raise ValueError(f"{label}.wetdry_semantics_id 不符合固定 SCHISM 語意")
        month = _nonempty_string(row["ocm_month_yyyymm"], f"{label}.ocm_month_yyyymm")
        if re.fullmatch(r"[0-9]{6}", month) is None:
            raise ValueError(f"{label}.ocm_month_yyyymm 必須是 YYYYMM")
        utc = _utc_datetime(time_ns, f"{label}.time_utc_ns")
        if month != f"{utc.year:04d}{utc.month:02d}":
            raise ValueError(f"{label}.ocm_month_yyyymm 與 arrival UTC 月份不一致")
        origin = _nonempty_string(row["ocm_time_origin"], f"{label}.ocm_time_origin")
        if origin != "observed":
            raise ValueError(
                f"{label}.ocm_time_origin 目前只接受 observed；重建資料須另立核准 provenance"
            )
        result.append(
            ReceptorArrivalInitialCondition(
                receptor_id=receptor_id,
                arrival_time_id=arrival_id,
                study_site_id=receptor.study_site_id,
                analysis_region_id=receptor.analysis_region_id,
                flow_domain_id=expected_flow_id,
                time_utc_ns=time_ns,
                vertical_id=vertical_id,
                z_m_positive_up=z,
                eta_m_positive_up=eta,
                bed_z_m_positive_up=bed,
                water_column_height_m=water,
                height_above_bed_m=height,
                zcor_lower_m_positive_up=lower,
                zcor_upper_m_positive_up=upper,
                vertical_bracket_alpha=alpha,
                source_face_local_index=source_face_local,
                source_face_global_index=source_face_global,
                wetdry_elem_value=wetdry,
                wetdry_semantics_id=semantics,
                ocm_month_yyyymm=month,
                ocm_source_time_index=source_time_index,
                ocm_time_origin=origin,
            )
        )
    expected_pairs = {
        (receptor.receptor_id, arrival.arrival_time_id)
        for receptor in receptors
        for arrival in arrivals
        if receptor.study_site_id == arrival.study_site_id
    }
    if seen_pairs != expected_pairs:
        missing = len(expected_pairs - seen_pairs)
        extra = len(seen_pairs - expected_pairs)
        raise ValueError(
            f"dynamic initial conditions pair coverage 不完整：missing={missing}, extra={extra}"
        )
    expected_count = 5_000 if formal else len(expected_pairs)
    if len(result) != expected_count:
        raise ValueError(
            f"dynamic initial conditions {'formal' if formal else 'pilot'} row count 不符："
            f"{len(result)} != {expected_count}"
        )
    return tuple(result), document


def load_receptor_arrival_initial_condition_manifest(
    path: str | Path,
    config: ProjectConfig,
    receptors: Sequence[Receptor],
    arrivals: Sequence[ArrivalTime],
    *,
    formal: bool = False,
) -> tuple[ReceptorArrivalInitialCondition, ...]:
    """載入每個 receptor×arrival 的 OCM-derived actual initial depth。

    每列代表一個 pair 在 arrival UTC 的海面高程、bed、zcor bracket、wet/dry 與來源 face
    證據；深度與高度單位均為公尺且 z 軸向上為正。此 loader 只讀取已由 OCM 上游產出的
    JSON，不自行讀取 OCM、推算 eta/zcor 或產生 manifest。formal 必須是五站完整
    ``20×50=1,000`` pairs、共 5,000 rows；pilot 若被 caller 要求，也必須覆蓋所載模板
    站點的完整 Cartesian pairs。回傳 tuple immutable，10 個 material 不會重複展開。
    """

    records, _ = _load_receptor_arrival_initial_condition_document(
        path, config, receptors, arrivals, formal=formal
    )
    return records


def _configured_manifest_reference(
    config: ProjectConfig, explicit: str | Path | None, *, name: str, fallbacks: tuple[Any, ...]
) -> str | Path:
    """選擇明示路徑或設定路徑，並拒絕 component 路徑互相矛盾的設定。"""

    values = [value for value in fallbacks if value is not None]
    if explicit is not None:
        explicit_text = str(explicit)
        if values and any(explicit_text != str(value) for value in values):
            raise ValueError(f"{name} explicit path 與設定 fallback 不一致：{explicit_text} != {values}")
        return explicit
    if not values:
        raise ValueError(f"缺少 {name} manifest path")
    text_values = [str(value) for value in values]
    if len(set(text_values)) != 1:
        raise ValueError(f"{name} manifest 在設定中有多個不同路徑：{text_values}")
    return values[0]


@dataclass(frozen=True, slots=True)
class ScenarioInputs:
    """已驗證的 component records、基礎情境與輸入 hash。

    ``scenarios`` 是 deterministic cross-product 的 tuple；不保存 object dtype NumPy array，
    因為那會在 parquet／checkpoint 邊界失去清楚的欄位契約。``initial_conditions`` 是
    receptor×arrival 的 tuple，不重複展開十種 material；``initial_conditions_by_pair``
    以 ``(receptor_id, arrival_time_id)`` 查找正式 actual z。``file_sha256`` 綁定磁碟 bytes，
    ``canonical_component_hashes`` 綁定 JSON 語意內容，Phase 3B 可直接納入 run manifest
    與 checkpoint binding。mapping 以 read-only proxy 暴露，避免呼叫端改寫 hash。
    """

    materials: tuple[Behavior, ...]
    receptors: tuple[Receptor, ...]
    arrival_times: tuple[ArrivalTime, ...]
    scenarios: tuple[Scenario, ...]
    file_sha256: Mapping[str, str]
    canonical_component_hashes: Mapping[str, str]
    design_version: str
    initial_conditions: tuple[ReceptorArrivalInitialCondition, ...] = ()
    initial_conditions_by_pair: Mapping[
        tuple[str, str], ReceptorArrivalInitialCondition
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """將所有 component container 與 hash mapping 固定為不可重新賦值的 tuple/proxy。"""

        object.__setattr__(self, "materials", tuple(self.materials))
        object.__setattr__(self, "receptors", tuple(self.receptors))
        object.__setattr__(self, "arrival_times", tuple(self.arrival_times))
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(self, "initial_conditions", tuple(self.initial_conditions))
        object.__setattr__(self, "file_sha256", MappingProxyType(dict(self.file_sha256)))
        object.__setattr__(
            self,
            "canonical_component_hashes",
            MappingProxyType(dict(self.canonical_component_hashes)),
        )
        expected_pairs = {
            (item.receptor_id, item.arrival_time_id): item for item in self.initial_conditions
        }
        provided_pairs = dict(self.initial_conditions_by_pair)
        if provided_pairs and provided_pairs != expected_pairs:
            raise ValueError("initial_conditions_by_pair 必須與 initial_conditions 完全一致")
        object.__setattr__(
            self,
            "initial_conditions_by_pair",
            MappingProxyType(provided_pairs or expected_pairs),
        )

    @property
    def behaviors(self) -> tuple[Behavior, ...]:
        """相容既有命名的 materials alias。"""

        return self.materials

    @property
    def records(self) -> Mapping[str, tuple[Any, ...]]:
        """以 read-only mapping 暴露三個 component records，方便 run controller 統一取用。"""

        return MappingProxyType(
            {
                "materials": self.materials,
                "receptors": self.receptors,
                "arrival_times": self.arrival_times,
                "initial_conditions": self.initial_conditions,
            }
        )

    @property
    def component_sha256(self) -> Mapping[str, str]:
        """file_sha256 的語意別名，便於輸出 manifest 欄位命名。"""

        return self.file_sha256


def load_scenario_inputs(
    config: ProjectConfig,
    *,
    config_path: str | Path | None = None,
    material_path: str | Path | None = None,
    receptor_path: str | Path | None = None,
    arrival_path: str | Path | None = None,
    initial_condition_path: str | Path | None = None,
    require_dynamic_initial_conditions: bool | None = None,
    formal: bool = False,
) -> ScenarioInputs:
    """讀取 component manifest、建 scenarios 並回傳 hash-bound bundle。

    未提供 explicit path 時，material 依序使用 ``scenarios.material_manifest`` 或
    ``physics.settling.material_manifest``；receptor 依序使用 scenarios 與 geometry 設定。
    每個相對 path 都以 ``config_path`` 的 YAML 目錄解析。formal 模式最後套用既有
    ``validate_baseline_coverage``，要求五站各 10,000、A 區 20,000、全案 50,000，並自動
    要求完整 receptor×arrival initial-condition manifest。非 formal 預設保留 legacy
    synthetic API，可沒有 dynamic manifest；tracked pilot 可傳
    ``require_dynamic_initial_conditions=True`` 強制載入並驗證所選站點的完整 Cartesian
    pairs。explicit path 與 config fallback 若不一致會 fail-fast；dynamic loader 只讀
    OCM 上游已產出的 JSON，不讀取 OCM 或接 runtime。
    """

    if require_dynamic_initial_conditions is not None and not isinstance(
        require_dynamic_initial_conditions, bool
    ):
        raise ValueError("require_dynamic_initial_conditions 必須是 bool 或 None")
    dynamic_required = (
        formal if require_dynamic_initial_conditions is None else require_dynamic_initial_conditions
    )
    material_ref = _configured_manifest_reference(
        config,
        material_path,
        name="material",
        fallbacks=(
            config.scenarios.material_manifest,
            config.physics.get("settling", {}).get("material_manifest"),
        ),
    )
    receptor_ref = _configured_manifest_reference(
        config,
        receptor_path,
        name="receptor",
        fallbacks=(config.scenarios.receptor_manifest, config.geometry.get("receptor_manifest")),
    )
    arrival_ref = _configured_manifest_reference(
        config, arrival_path, name="arrival", fallbacks=(config.scenarios.arrival_time_manifest,)
    )
    dynamic_fallbacks = (
        config.scenarios.receptor_arrival_initial_condition_manifest,
    )
    if initial_condition_path is not None or any(value is not None for value in dynamic_fallbacks):
        dynamic_ref = _configured_manifest_reference(
            config,
            initial_condition_path,
            name="receptor_arrival_initial_condition",
            fallbacks=dynamic_fallbacks,
        )
    elif dynamic_required:
        raise ValueError("缺少 receptor_arrival_initial_condition manifest path")
    else:
        dynamic_ref = None
    paths = {
        "material": resolve_manifest_path(config_path, material_ref),
        "receptor": resolve_manifest_path(config_path, receptor_ref),
        "arrival": resolve_manifest_path(config_path, arrival_ref),
    }
    if dynamic_ref is not None:
        paths["initial_condition"] = resolve_manifest_path(config_path, dynamic_ref)
    materials, material_document = _load_material_document(paths["material"], config, formal=formal)
    receptors, receptor_document = _load_receptor_document(paths["receptor"], config, formal=formal)
    arrivals, arrival_document = _load_arrival_document(paths["arrival"], config, formal=formal)
    initial_conditions: tuple[ReceptorArrivalInitialCondition, ...] = ()
    initial_condition_document: _ManifestDocument | None = None
    if dynamic_ref is not None:
        initial_conditions, initial_condition_document = (
            _load_receptor_arrival_initial_condition_document(
                paths["initial_condition"],
                config,
                receptors,
                arrivals,
                formal=formal,
            )
        )
    if {item.study_site_id for item in receptors} != {item.study_site_id for item in arrivals}:
        raise ValueError("receptor 與 arrival 的 study_site_id 集合必須一致")
    scenario_list = build_scenarios(
        behaviors=materials,
        receptors=receptors,
        arrival_times=arrivals,
        design_version=config.design_version,
    )
    for item in scenario_list:
        if item.design_version != config.design_version:
            raise ValueError("scenario design_version 與 config 不一致")
        expected_id = stable_identifier(
            "scn",
            [
                item.study_site_id,
                item.material_id,
                item.receptor_id,
                item.arrival_time_id,
                item.design_version,
            ],
        )
        if item.scenario_id != expected_id:
            raise ValueError(f"scenario stable ID 驗證失敗：{item.scenario_id}")
    if formal:
        validate_baseline_coverage(scenario_list)
    file_sha256 = {
        "material": material_document.file_sha256,
        "receptor": receptor_document.file_sha256,
        "arrival": arrival_document.file_sha256,
    }
    canonical_component_hashes = {
        "material": material_document.canonical_sha256,
        "receptor": receptor_document.canonical_sha256,
        "arrival": arrival_document.canonical_sha256,
    }
    if initial_condition_document is not None:
        file_sha256["receptor_arrival_initial_condition"] = initial_condition_document.file_sha256
        canonical_component_hashes[
            "receptor_arrival_initial_condition"
        ] = initial_condition_document.canonical_sha256
    if formal and "receptor_arrival_initial_condition" not in canonical_component_hashes:
        raise ValueError("formal ScenarioInputs 必須保存 dynamic initial-condition canonical hash")
    return ScenarioInputs(
        materials=materials,
        receptors=receptors,
        arrival_times=arrivals,
        scenarios=tuple(scenario_list),
        file_sha256=file_sha256,
        canonical_component_hashes=canonical_component_hashes,
        design_version=config.design_version,
        initial_conditions=initial_conditions,
    )


def _geojson_position(value: Any, label: str) -> tuple[float, float]:
    """驗證單一世界大地測量系統 1984（WGS84）位置恰為 ``[lon, lat]``。

    GeoJSON 經 JSON 解析後的位置必須是兩元素 array；布林、文字、第三維、NaN 與無限值
    均在交給 Shapely 前拒絕。經度與緯度也分別限制在全球合法範圍，避免投影器收到可轉
    成數字但沒有地理意義的座標。
    """

    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} 必須是二維 [lon, lat] array")
    lon = _finite_float(value[0], f"{label}[0]")
    lat = _finite_float(value[1], f"{label}[1]")
    if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
        raise ValueError(f"{label} 超出 WGS84 lon/lat bounds")
    return lon, lat


def _geojson_line_coordinates(value: Any, label: str) -> list[tuple[float, float]]:
    """驗證 LineString 的位置層級固定且至少有兩個二維點。"""

    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{label} 必須是至少兩個位置的 array")
    return [
        _geojson_position(position, f"{label}[{index}]")
        for index, position in enumerate(value)
    ]


def _validate_geojson_coordinates(value: Any, geometry_type: str, label: str) -> None:
    """依 geometry type 驗證 GeoJSON 巢狀層級，不接受混合或模糊 nesting。

    本專案只交換 Polygon、LineString 與 MultiLineString。Polygon 的每個環至少四點且首尾
    相同；MultiLineString 的每條線均套用同一二維位置規則。完成此純 JSON gate 後才建立
    Shapely geometry，因此解析器不會替錯誤 nesting、布林值或第三維座標做隱式容錯。
    """

    if geometry_type == "LineString":
        _geojson_line_coordinates(value, label)
        return
    if geometry_type == "MultiLineString":
        if not isinstance(value, list) or not value:
            raise ValueError(f"{label} 必須是非空 LineString array")
        for index, line in enumerate(value):
            _geojson_line_coordinates(line, f"{label}[{index}]")
        return
    if geometry_type == "Polygon":
        if not isinstance(value, list) or not value:
            raise ValueError(f"{label} 必須是非空 linear-ring array")
        for ring_index, ring in enumerate(value):
            ring_label = f"{label}[{ring_index}]"
            if not isinstance(ring, list) or len(ring) < 4:
                raise ValueError(f"{ring_label} 必須至少有四個二維位置")
            positions = [
                _geojson_position(position, f"{ring_label}[{index}]")
                for index, position in enumerate(ring)
            ]
            if positions[0] != positions[-1]:
                raise ValueError(f"{ring_label} 首尾位置必須相同")
        return
    raise ValueError(f"{label} 未支援 geometry type：{geometry_type}")


def _geometry_object(value: Any, expected_types: set[str], label: str) -> BaseGeometry:
    """先驗證嚴格二維 GeoJSON，再解析並拒絕空或無效幾何。"""

    if not isinstance(value, dict):
        raise ValueError(f"{label} 必須是 GeoJSON geometry object")
    _expect_exact_keys(value, frozenset({"type", "coordinates"}), label)
    geometry_type = _nonempty_string(value["type"], f"{label}.type")
    if geometry_type not in expected_types:
        raise ValueError(f"{label}.type 必須是 {sorted(expected_types)}")
    _validate_geojson_coordinates(value["coordinates"], geometry_type, f"{label}.coordinates")
    try:
        geometry = shape(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 無法解析為 Shapely geometry") from exc
    if geometry.is_empty or not geometry.is_valid or geometry.has_z:
        raise ValueError(f"{label} 不可為空、invalid 或 3-D geometry")
    if isinstance(geometry, Polygon) and geometry.area <= 0:
        raise ValueError(f"{label} polygon area 必須為正")
    if isinstance(geometry, (LineString, MultiLineString)) and geometry.length <= 0:
        raise ValueError(f"{label} line length 必須為正")
    return geometry


def _validate_metric_geometry(geometry: BaseGeometry, label: str) -> None:
    """確認投影後幾何仍有效，且每個公尺座標都是有限二維數值。

    WGS84 輸入通過並不代表座標轉換一定成功；投影中心或外部函式異常仍可能產生非有限
    公尺值。本檢查在任何 containment、boundary crossing 或物理距離計算前執行，確保回傳
    的 ``BoundaryGeometry`` 不會夾帶經緯度或 NaN／Infinity。
    """

    if geometry.is_empty or not geometry.is_valid or geometry.has_z:
        raise ValueError(f"{label} 投影後不可為空、invalid 或 3-D geometry")
    coordinate_sequences: list[Any] = []
    if isinstance(geometry, Polygon):
        coordinate_sequences.append(geometry.exterior.coords)
        coordinate_sequences.extend(interior.coords for interior in geometry.interiors)
    elif isinstance(geometry, LineString):
        coordinate_sequences.append(geometry.coords)
    elif isinstance(geometry, MultiLineString):
        coordinate_sequences.extend(part.coords for part in geometry.geoms)
    else:
        raise ValueError(f"{label} 投影後 geometry type 不受支援")
    for sequence in coordinate_sequences:
        for coordinate in sequence:
            if len(coordinate) != 2 or not all(
                math.isfinite(float(item)) for item in coordinate
            ):
                raise ValueError(f"{label} 投影後含非有限或非二維公尺座標")


def _boundary_line_within(
    line: BaseGeometry, polygon: Polygon, *, tolerance_m: float, label: str
) -> None:
    """確認 open-water line 全部落在指定 polygon boundary 的公尺容許帶內。"""

    boundary_support = (
        polygon.boundary
        if tolerance_m == 0.0
        else polygon.boundary.buffer(tolerance_m)
    )
    if not boundary_support.covers(line):
        raise ValueError(f"{label} 必須落在對應 polygon boundary tolerance 內")


@dataclass(frozen=True, slots=True)
class BoundaryGeometryBundle(Mapping[str, BoundaryGeometry]):
    """每站點的公尺制 BoundaryGeometry 及幾何 manifest hash。

    bundle 直接實作 Mapping，因此既可用 ``bundle[study_site_id]`` 當作 dict，也可保存
    projection 與 hash 供後續 run manifest 使用。``geometries`` 內不保留經緯度 geometry；
    WGS84 僅存在上游 JSON 與外部交換層。
    """

    geometries: Mapping[str, BoundaryGeometry]
    projections: Mapping[str, DomainProjection]
    file_sha256: Mapping[str, str]
    canonical_component_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        """封裝 bundle mapping，防止呼叫端改動已驗證的站點與 hash 對應。"""

        object.__setattr__(self, "geometries", MappingProxyType(dict(self.geometries)))
        object.__setattr__(self, "projections", MappingProxyType(dict(self.projections)))
        object.__setattr__(self, "file_sha256", MappingProxyType(dict(self.file_sha256)))
        object.__setattr__(
            self,
            "canonical_component_hashes",
            MappingProxyType(dict(self.canonical_component_hashes)),
        )

    def __getitem__(self, key: str) -> BoundaryGeometry:
        """以 study_site_id 取回公尺制邊界。"""

        return self.geometries[key]

    def __iter__(self):
        """依固定 mapping 順序迭代站點識別碼。"""

        return iter(self.geometries)

    def __len__(self) -> int:
        """回傳已載入站點數。"""

        return len(self.geometries)

    @property
    def by_study_site(self) -> Mapping[str, BoundaryGeometry]:
        """提供語意化的 geometry mapping alias。"""

        return self.geometries


def _load_geometry_document(
    path: str | Path,
    config: ProjectConfig,
    *,
    kind: Literal["domain_geometry_manifest", "local_geometry_manifest", "open_boundary_manifest"],
    formal: bool,
) -> _ManifestDocument:
    """讀取三種 geometry manifest 的共用 root 契約。"""

    document = _document(path)
    _validate_component_root(
        document.payload, _GEOMETRY_ROOT_KEYS, kind=kind, config=config, formal=formal
    )
    return document


def _geometry_manifest_reference(
    config: ProjectConfig, explicit: str | Path | None, key: str
) -> str | Path:
    """從 geometry 設定取出必需文件 path。"""

    if explicit is not None:
        return explicit
    value = config.geometry.get(key)
    if value is None:
        raise ValueError(f"geometry.{key} 尚未指定 manifest path")
    return value


def load_boundary_geometries(
    config: ProjectConfig,
    domain_manifest_path: str | Path | None = None,
    local_domain_manifest_path: str | Path | None = None,
    open_boundary_manifest_path: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    formal: bool = False,
) -> BoundaryGeometryBundle:
    """載入 domain/local/open-boundary v1 文件並建構每站點公尺制邊界 bundle。

    三種 JSON 均只接受 EPSG:4326。domain 與 local polygon 先經 Shapely validity、
    containment、``local_equals_flow`` topology gate，再依 config domain center 建立固定
    AEQD 公尺座標；open-water line 也在同一公尺座標檢查 boundary tolerance。foreign local
    domains 只從相同 flow_domain 的其他已選站點加入，不能跨 forcing domain。B-D 的
    ``local_equals_flow=true`` 若沒有重複 local line，會安全共用該 flow line；A 區 local
    boundary 則必須有自己的有效 line。formal 必須完整涵蓋 config 的四個 resolved
    flow-domain 與五個 study site；pilot 則允許 config study site 的非空子集，但 domain
    與 flow open-boundary 只能涵蓋該子集實際使用的 flow-domain，不能夾帶未使用的域。
    回傳值內不保存 lon/lat geometry。
    """

    domain_ref = _geometry_manifest_reference(config, domain_manifest_path, "domain_manifest")
    local_ref = _geometry_manifest_reference(config, local_domain_manifest_path, "local_domain_manifest")
    open_ref = _geometry_manifest_reference(config, open_boundary_manifest_path, "open_boundary_manifest")
    paths = {
        "domain": resolve_manifest_path(config_path, domain_ref),
        "local": resolve_manifest_path(config_path, local_ref),
        "open_boundary": resolve_manifest_path(config_path, open_ref),
    }
    domain_document = _load_geometry_document(
        paths["domain"], config, kind="domain_geometry_manifest", formal=formal
    )
    local_document = _load_geometry_document(
        paths["local"], config, kind="local_geometry_manifest", formal=formal
    )
    open_document = _load_geometry_document(
        paths["open_boundary"], config, kind="open_boundary_manifest", formal=formal
    )
    # 幾何 manifest 的 flow-domain ID 必須依執行模式解析，而不是直接拿
    # StudySiteConfig.flow_domain_id。A 區在 formal 可能已換成南向擴充的 v4；若此處
    # 仍以 site 設定中的 pilot v3 查找，domain、local 與 open-boundary 會被錯誤綁回
    # 舊網格，後續即使幾何形狀看似合法也不能代表同一份正式 forcing。解析結果只保存
    # in-memory lookup，不改寫輸入 JSON 或 config。
    config_domains_by_region = {domain.analysis_region_id: domain for domain in config.domains}
    resolved_flow_by_region = {
        region: resolve_flow_domain_id(config, region, formal=formal)
        for region in config_domains_by_region
    }
    if len(set(resolved_flow_by_region.values())) != len(resolved_flow_by_region):
        raise ValueError("不同 analysis_region_id 不可解析到重複 flow_domain_id")
    # ``expected_flow_ids`` 是完整 config 的 resolved 集合。formal 需要整集合；pilot
    # 先以它限制可接受的 base ID，待 local records 解析出 selected sites 後，再收斂為
    # selected_flows。這個兩階段檢查同時允許 pilot 子集，又避免「domain 多給一個但本次
    # 沒有任何 selected site 使用」的外來資料混入。
    expected_flow_ids = set(resolved_flow_by_region.values())
    domain_records = _records(domain_document.payload["records"], "domain.records")
    domain_metric: dict[str, Polygon] = {}
    domain_projection: dict[str, DomainProjection] = {}
    for index, row in enumerate(domain_records):
        label = f"domain.records[{index}]"
        _expect_exact_keys(row, _DOMAIN_RECORD_KEYS, label)
        region = _nonempty_string(row["analysis_region_id"], f"{label}.analysis_region_id")
        flow_id = _nonempty_string(row["flow_domain_id"], f"{label}.flow_domain_id")
        if flow_id in domain_metric:
            raise ValueError(f"flow_domain_id 不可重複：{flow_id}")
        domain_config = config_domains_by_region.get(region)
        if domain_config is None or resolved_flow_by_region[region] != flow_id:
            raise ValueError(f"{label} region/flow domain 與 config resolver 不一致")
        _nonempty_string(row["source_geometry_id"], f"{label}.source_geometry_id")
        polygon = _geometry_object(row["geometry"], {"Polygon"}, f"{label}.geometry")
        assert isinstance(polygon, Polygon)
        projection = DomainProjection(*domain_config.center_lonlat)
        metric_polygon = projection.project_geometry(polygon)
        _validate_metric_geometry(metric_polygon, f"{label}.geometry")
        domain_metric[flow_id] = metric_polygon
        domain_projection[flow_id] = projection
    if formal:
        if set(domain_metric) != expected_flow_ids:
            raise ValueError("formal domain geometry 必須恰涵蓋 config 四個 resolved flow domains")
    elif not domain_metric or not set(domain_metric) <= expected_flow_ids:
        raise ValueError("pilot domain geometry 必須是 config resolved base flow domains 的非空子集")
    local_records = _records(local_document.payload["records"], "local.records")
    local_metric: dict[str, Polygon] = {}
    local_equal: dict[str, bool] = {}
    site_flow: dict[str, str] = {}
    for index, row in enumerate(local_records):
        label = f"local.records[{index}]"
        _expect_exact_keys(row, _LOCAL_RECORD_KEYS, label)
        site = _site_reference(row, config, label=label, formal=formal)
        site_id = site.study_site_id
        if site_id in local_metric:
            raise ValueError(f"study_site_id 不可重複：{site_id}")
        flow_id = _nonempty_string(row["flow_domain_id"], f"{label}.flow_domain_id")
        expected_flow_id = resolve_flow_domain_id(
            config, site.analysis_region_id, formal=formal
        )
        if flow_id != expected_flow_id or flow_id not in domain_metric:
            raise ValueError(f"{label}.flow_domain_id 與 domain manifest/config resolver 不一致")
        _nonempty_string(row["source_geometry_id"], f"{label}.source_geometry_id")
        polygon = _geometry_object(row["geometry"], {"Polygon"}, f"{label}.geometry")
        assert isinstance(polygon, Polygon)
        metric = domain_projection[flow_id].project_geometry(polygon)
        _validate_metric_geometry(metric, f"{label}.geometry")
        local_metric[site_id] = metric
        local_equal[site_id] = _boolean(row["local_equals_flow"], f"{label}.local_equals_flow")
        site_flow[site_id] = flow_id
    selected_sites = set(local_metric)
    if formal:
        if selected_sites != set(_config_sites(config)):
            raise ValueError("local formal 必須恰涵蓋 config 五個 study sites")
    else:
        _validate_pilot_sites(selected_sites, config, "local geometry")
    selected_flows = {site_flow[site_id] for site_id in selected_sites}
    if not formal and set(domain_metric) != selected_flows:
        raise ValueError(
            "pilot domain geometry 必須恰涵蓋 selected sites 實際使用的 flow domains"
        )
    tolerance = config.geometry.get("coordinate_round_trip_tolerance_m", 0.05)
    tolerance_m = _finite_float(tolerance, "geometry.coordinate_round_trip_tolerance_m")
    if tolerance_m < 0:
        raise ValueError("geometry.coordinate_round_trip_tolerance_m 不可為負")
    for site_id, local_polygon in local_metric.items():
        flow_polygon = domain_metric[site_flow[site_id]]
        if local_equal[site_id]:
            if not local_polygon.equals(flow_polygon):
                raise ValueError(f"{site_id} local_equals_flow=true 但 topology 不等於 flow polygon")
        elif not flow_polygon.buffer(tolerance_m).covers(local_polygon):
            raise ValueError(f"{site_id} local polygon 超出對應 flow polygon tolerance")
    open_records = _records(open_document.payload["records"], "open_boundary.records")
    flow_lines: dict[str, list[BaseGeometry]] = {}
    local_lines: dict[str, list[BaseGeometry]] = {}
    seen_segments: set[tuple[str, str, str]] = set()
    for index, row in enumerate(open_records):
        label = f"open_boundary.records[{index}]"
        _expect_exact_keys(row, _OPEN_RECORD_KEYS, label)
        owner_kind = _nonempty_string(row["owner_kind"], f"{label}.owner_kind")
        owner_id = _nonempty_string(row["owner_id"], f"{label}.owner_id")
        region = _nonempty_string(row["analysis_region_id"], f"{label}.analysis_region_id")
        segment_id = _nonempty_string(row["segment_id"], f"{label}.segment_id")
        key = (owner_kind, owner_id, segment_id)
        if key in seen_segments:
            raise ValueError(f"open boundary segment 不可重複：{key}")
        seen_segments.add(key)
        _nonempty_string(row["source_geometry_id"], f"{label}.source_geometry_id")
        if owner_kind == "flow_domain":
            expected_flow_id = resolve_flow_domain_id(config, region, formal=formal)
            if owner_id != expected_flow_id or owner_id not in domain_metric:
                raise ValueError(f"{label} flow owner 與 config/domain 不一致")
            owner_polygon = domain_metric[owner_id]
            owner_line = domain_projection[owner_id].project_geometry(
                _geometry_object(row["geometry"], {"LineString", "MultiLineString"}, f"{label}.geometry")
            )
            _validate_metric_geometry(owner_line, f"{label}.geometry")
            _boundary_line_within(owner_line, owner_polygon, tolerance_m=tolerance_m, label=label)
            flow_lines.setdefault(owner_id, []).append(owner_line)
        elif owner_kind == "local_domain":
            if owner_id not in local_metric or site_flow[owner_id] not in domain_metric:
                raise ValueError(f"{label} local owner 必須是已選站點")
            if _config_sites(config)[owner_id].analysis_region_id != region:
                raise ValueError(f"{label} local owner region 與 config 不一致")
            flow_id = site_flow[owner_id]
            owner_line = domain_projection[flow_id].project_geometry(
                _geometry_object(row["geometry"], {"LineString", "MultiLineString"}, f"{label}.geometry")
            )
            _validate_metric_geometry(owner_line, f"{label}.geometry")
            _boundary_line_within(owner_line, local_metric[owner_id], tolerance_m=tolerance_m, label=label)
            local_lines.setdefault(owner_id, []).append(owner_line)
        else:
            raise ValueError(f"{label}.owner_kind 必須是 flow_domain 或 local_domain")
    if any(flow_id not in flow_lines for flow_id in selected_flows):
        missing = sorted(selected_flows - set(flow_lines))
        raise ValueError(f"缺少 selected flow open boundary：{missing}")
    if formal:
        if set(flow_lines) != expected_flow_ids:
            raise ValueError(
                "formal open boundary 必須恰涵蓋 config 四個 resolved flow domains"
            )
    elif set(flow_lines) != selected_flows:
        raise ValueError(
            "pilot open boundary 必須恰涵蓋 selected sites 實際使用的 flow domains"
        )
    geometries: dict[str, BoundaryGeometry] = {}
    projections: dict[str, DomainProjection] = {}
    for site_id in sorted(selected_sites):
        flow_id = site_flow[site_id]
        flow_line = unary_union(flow_lines[flow_id])
        if not isinstance(flow_line, (LineString, MultiLineString)):
            raise ValueError(f"flow {flow_id} open boundary union 後不是 line geometry")
        local_values = local_lines.get(site_id, [])
        if local_values:
            local_line = unary_union(local_values)
            if not isinstance(local_line, (LineString, MultiLineString)):
                raise ValueError(f"local {site_id} open boundary union 後不是 line geometry")
        elif local_equal[site_id]:
            local_line = flow_line
        else:
            raise ValueError(f"local {site_id} 不是 local_equals_flow，必須提供自己的 open line")
        foreign = {
            other_site: local_metric[other_site]
            for other_site in sorted(selected_sites)
            if other_site != site_id and site_flow[other_site] == flow_id
        }
        geometries[site_id] = BoundaryGeometry(
            own_local_domain=local_metric[site_id],
            flow_domain=domain_metric[flow_id],
            foreign_local_domains=foreign,
            own_local_open_boundary=local_line,
            flow_open_boundary=flow_line,
            local_equals_flow=local_equal[site_id],
            own_local_boundary_segment_id=f"{site_id}_local_open_boundary",
            flow_boundary_segment_id=f"{flow_id}_flow_open_boundary",
            boundary_match_tolerance_m=tolerance_m,
        )
        projections[site_id] = domain_projection[flow_id]
    return BoundaryGeometryBundle(
        geometries=geometries,
        projections=projections,
        file_sha256={
            "domain": domain_document.file_sha256,
            "local": local_document.file_sha256,
            "open_boundary": open_document.file_sha256,
        },
        canonical_component_hashes={
            "domain": domain_document.canonical_sha256,
            "local": local_document.canonical_sha256,
            "open_boundary": open_document.canonical_sha256,
        },
    )


__all__ = [
    "BED_RESIDENCE_INPUT_SCHEMA_VERSION",
    "BoundaryGeometryBundle",
    "MANIFEST_SCHEMA_VERSION",
    "ScenarioInputs",
    "load_arrival_manifest",
    "load_arrival_time_manifest",
    "load_boundary_geometries",
    "load_material_manifest",
    "load_receptor_arrival_initial_condition_manifest",
    "load_receptor_manifest",
    "load_scenario_inputs",
    "resolve_manifest_path",
]
