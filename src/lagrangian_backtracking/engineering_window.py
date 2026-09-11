"""建立並執行單站工程性回溯時窗。

本模組是正式五站、五萬情境輸入建置之外的獨立 adapter。它讀取已驗證的 source
config／derived input 與指定 OCM native、NWW3 analysis 產品，只抽取一個站點、一個
arrival UTC、指定材質及可選的垂向層位，建立 ``engineering_only=true`` 的新 artifact。
artifact 仍交給正式 ``RuntimeRequestFactory``、``RunController`` 及 checkpoint binding
執行，因此工程測速會走真實 forcing 與物理 gate；但它不會把 5 或 20 個小範圍情境冒充
正式五站成果，也不會把 H30 的時間軸證明解讀成整條粒子路徑已通過空間支援。

輸入的 OCM／NWW 大型 NPY 只以既有 memory-map helper 讀取必要月份。prepare 階段只
對五個受體的 arrival 端點執行 NWW 四角 static/dynamic mask、有限值與物理條件檢查，
並對 OCM/NWW canonical time 軸逐時檢查 H30 的 721 個節點；沿途每個 runtime stage 的
空間與物理驗證仍由真正 run 負責。這個取捨避免先做一輪等同 trajectory 的前置運算，
同時把限制寫進 artifact provenance，讓量測結果能和正式成本估算分開解讀。
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import ProjectConfig, load_config, resolve_flow_domain_id
from .input_derivation import (
    ARTIFACT_FILENAMES,
    DERIVED_INPUT_SCHEMA_VERSION,
    InputDerivationError,
    _array_record_hash,
    _assert_no_symlink_components,
    _assert_regular_directory,
    _assert_regular_file,
    _build_face_vertical_support,
    _dynamic_initial_payload,
    _env_root,
    _load_native_mesh,
    _load_nww_runtime_cache,
    _load_ocm_pair_cache,
    _load_product,
    _nww_exact_hour_samples,
    _provenance,
    canonical_json_bytes,
    read_canonical_json,
    validate_input_derivatives,
    write_canonical_json,
)
from .input_horizon import HorizonContractError, build_horizon_window, compute_horizon_coverage, utc_string
from .manifests import (
    load_boundary_geometries,
    load_scenario_inputs,
    resolve_manifest_path,
)
from .provenance import collect_code_provenance
from .run_control import (
    RunController,
    RunExecutionSummary,
    initialize_run_workspace,
    load_run_plan,
    load_run_progress,
)
from .run_validation import validate_run
from .runtime import RuntimeRequestFactory, _read_strict_json_object
from .scenarios import ArrivalTime, Receptor, stable_identifier

ENGINEERING_ARTIFACT_SCHEMA_VERSION = "1.0.0"
"""工程性 artifact 的版本；變更來源／狀態欄位時必須升版。"""

ENGINEERING_GENERATION_METHOD_ID = "server_v3_engineering_window_adapter_v1"
"""本 adapter 的生成方法識別碼，與正式五萬輸入 builder 分開。"""

ENGINEERING_RECEPTOR_METHOD_ID = "server_v3_engineering_window_receptor_subset_v1"
ENGINEERING_ARRIVAL_METHOD_ID = "server_v3_engineering_window_arrival_v1"
ENGINEERING_COMPONENT_DIR = "manifests"
ENGINEERING_MANIFEST_NAME = "engineering_manifest.json"
ENGINEERING_CONFIG_NAME = "config.yaml"
ENGINEERING_INVENTORY_NAME = "input_inventory.json"
ENGINEERING_DEFAULT_EXPERIMENT_CASE_ID = "finite_depth_stokes"
ENGINEERING_DEFAULT_SHARD_SCENARIO_COUNT = 5

_UTC_HOUR_NS = 3_600_000_000_000
_SUPPORTED_VERTICAL_IDS = frozenset(
    {"upper_water_column", "mid_upper_water_column", "mid_lower_water_column", "near_bed"}
)
_REQUIRED_ARTIFACT_KEYS = frozenset(
    {
        "artifact_schema_version",
        "artifact_kind",
        "status",
        "engineering_only",
        "run_kind",
        "config_path",
        "input_inventory_path",
        "input_inventory_sha256",
        "study_site_id",
        "analysis_region_id",
        "flow_domain_id",
        "material_id",
        "arrival_time_id",
        "arrival_utc",
        "backtrack_days",
        "horizon",
        "vertical_scope",
        "receptor_count",
        "pair_count",
        "scenario_count",
        "source_lineage",
        "component_files",
        "geometry_files",
        "config_hash",
        "component_canonical_hashes",
        "geometry_canonical_hashes",
        "shard_scenario_count",
        "code_provenance",
    }
)


class EngineeringWindowError(ValueError):
    """工程時窗、來源 lineage 或真實 forcing gate 不符合契約。"""


@dataclass(frozen=True, slots=True)
class EngineeringHorizon:
    """一筆工程 arrival 的 inclusive exact-hour 回溯窗口。"""

    arrival_time_ns: int
    arrival_utc: str
    start_time_ns: int
    start_utc: str
    support_days: int
    expected_step_count: int
    months: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EngineeringWindowArtifact:
    """prepare 完成後可交給 run CLI 的 immutable artifact 摘要。"""

    directory: Path
    manifest_path: Path
    config_path: Path
    input_inventory_path: Path
    study_site_id: str
    arrival_utc: str
    backtrack_days: int
    vertical_id: str | None
    receptor_count: int
    pair_count: int
    scenario_count: int

    def to_dict(self) -> dict[str, Any]:
        """輸出不含絕對 source root 的 JSON-safe 摘要，供 CLI 回報。"""

        return {
            "artifact_directory": str(self.directory),
            "manifest": str(self.manifest_path),
            "config": str(self.config_path),
            "input_inventory": str(self.input_inventory_path),
            "study_site_id": self.study_site_id,
            "arrival_utc": self.arrival_utc,
            "backtrack_days": self.backtrack_days,
            "vertical_id": self.vertical_id,
            "receptor_count": self.receptor_count,
            "pair_count": self.pair_count,
            "scenario_count": self.scenario_count,
            "engineering_only": True,
        }


def _json_read(path: str | Path) -> tuple[dict[str, Any], str, str]:
    """以既有 strict JSON 與 sidecar binding 讀取文件，回傳 raw/canonical SHA-256。"""

    source = _assert_regular_file(path)
    try:
        payload, fingerprint = read_canonical_json(source)
    except (OSError, TypeError, ValueError) as exc:
        raise EngineeringWindowError(f"無法讀取 JSON：{source}") from exc
    if not isinstance(payload, dict):
        raise EngineeringWindowError(f"JSON root 必須是 object：{source}")
    try:
        canonical = canonical_json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise EngineeringWindowError(f"JSON 無法 canonicalize：{source}") from exc
    # 重新 canonicalize 只作 defensive check；hash 值採 sidecar 驗證後的實際 bytes
    # fingerprint，避免 caller 以解析後 mapping 取代檔案 lineage。
    if fingerprint.get("canonical_sha256") != sha256(canonical).hexdigest():
        raise EngineeringWindowError(f"JSON canonical hash 重算不一致：{source}")
    return payload, str(fingerprint["sha256"]), str(fingerprint["canonical_sha256"])


def parse_arrival_utc(value: str) -> tuple[int, str]:
    """解析並固定化 exact-hour UTC ``Z`` 字串。

    arrival 不是 local time，也不接受 offset 形式或分鐘／秒；整數 epoch nanoseconds
    會直接交給 OCM/NWW canonical time 軸比對，避免浮點 timestamp 在跨月邊界改變節點。
    """

    if not isinstance(value, str) or not value or value != value.strip() or not value.endswith("Z"):
        raise EngineeringWindowError("arrival_utc 必須是沒有首尾空白的 UTC Z 字串")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EngineeringWindowError(f"arrival_utc 格式無效：{value!r}") from exc
    if parsed.utcoffset() != timedelta(0) or parsed.minute or parsed.second or parsed.microsecond:
        raise EngineeringWindowError("arrival_utc 必須落在 exact-hour UTC")
    canonical = parsed.isoformat().replace("+00:00", "Z")
    if canonical != value:
        raise EngineeringWindowError("arrival_utc 必須使用固定 ISO8601 UTC Z 格式")
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = parsed - epoch
    time_ns = (delta.days * 86_400 + delta.seconds) * 1_000_000_000
    return int(time_ns), canonical


def _months_between(start_ns: int, end_ns: int) -> tuple[str, ...]:
    """回傳涵蓋 inclusive 時窗的最小曆月集合，不讀取任何 forcing 陣列。"""

    start = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=UTC)
    end = datetime.fromtimestamp(end_ns / 1_000_000_000, tz=UTC)
    labels: list[str] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        labels.append(f"{year:04d}{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    if not labels:
        raise EngineeringWindowError("arrival 時窗未涵蓋任何月份")
    return tuple(labels)


def build_engineering_horizon(arrival_utc: str, backtrack_days: int) -> EngineeringHorizon:
    """以通用正整日參數建立 inclusive H 時窗；不寫死 7、30 或 60。"""

    if isinstance(backtrack_days, bool) or not isinstance(backtrack_days, int) or backtrack_days < 1:
        raise EngineeringWindowError("backtrack_days 必須是正整數")
    arrival_ns, canonical = parse_arrival_utc(arrival_utc)
    try:
        window = build_horizon_window(arrival_ns, backtrack_days)
    except HorizonContractError as exc:
        raise EngineeringWindowError(f"無法建立工程回溯窗口：{exc}") from exc
    start_utc = utc_string(window.start_time_ns)
    return EngineeringHorizon(
        arrival_time_ns=window.arrival_time_ns,
        arrival_utc=canonical,
        start_time_ns=window.start_time_ns,
        start_utc=start_utc,
        support_days=window.support_days,
        expected_step_count=window.expected_step_count,
        months=_months_between(window.start_time_ns, window.end_time_ns),
    )


def _source_component_payloads(
    source_input_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    """讀取 source 的固定 component 文件與 raw/canonical hash。"""

    payloads: dict[str, dict[str, Any]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for kind, filename in ARTIFACT_FILENAMES.items():
        payload, raw_hash, canonical_hash = _json_read(source_input_directory / filename)
        payloads[kind] = payload
        hashes[kind] = {"sha256": raw_hash, "canonical_sha256": canonical_hash}
    return payloads, hashes


def _copy_component_provenance(
    *,
    source_hashes: Mapping[str, str],
    method_id: str,
    study_site_id: str,
    analysis_region_id: str,
    flow_domain_id: str,
    vertical_id: str | None = None,
) -> dict[str, Any]:
    """建立 generated component 的 provenance，不把 source accepted 狀態複製成新狀態。"""

    extra: dict[str, Any] = {
        "engineering_only": True,
        "selection_scope": "single_site_single_arrival_engineering_window",
        "study_site_id": study_site_id,
        "analysis_region_id": analysis_region_id,
        "flow_domain_id": flow_domain_id,
    }
    if vertical_id is not None:
        extra["vertical_id"] = vertical_id
    return _provenance(method_id=method_id, source_hashes=source_hashes, **extra)


def _subset_geometry_payload(
    payload: Mapping[str, Any],
    *,
    kind: str,
    study_site_id: str,
    analysis_region_id: str,
    flow_domain_id: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """從 source geometry 保留選定站點所需的 domain/local/open records。"""

    result = copy.deepcopy(dict(payload))
    records = payload.get("records")
    if not isinstance(records, list):
        raise EngineeringWindowError(f"{kind} source records 必須是 list")
    if kind == "domain":
        selected = [
            row
            for row in records
            if isinstance(row, Mapping) and row.get("analysis_region_id") == analysis_region_id
        ]
    elif kind == "local":
        selected = [
            row for row in records if isinstance(row, Mapping) and row.get("study_site_id") == study_site_id
        ]
    elif kind == "open_boundary":
        selected = [
            row
            for row in records
            if isinstance(row, Mapping)
            and (
                (row.get("owner_kind") == "flow_domain" and row.get("owner_id") == flow_domain_id)
                or (row.get("owner_kind") == "local_domain" and row.get("owner_id") == study_site_id)
            )
        ]
    else:
        raise EngineeringWindowError(f"未知 geometry kind：{kind}")
    if not selected:
        raise EngineeringWindowError(f"source {kind} 沒有選定站點所需 geometry")
    result["records"] = copy.deepcopy(selected)
    result["status"] = "generated"
    result["provenance"] = _copy_component_provenance(
        source_hashes=source_hashes,
        method_id=ENGINEERING_GENERATION_METHOD_ID,
        study_site_id=study_site_id,
        analysis_region_id=analysis_region_id,
        flow_domain_id=flow_domain_id,
    )
    return result


def _subset_receptor_payload(
    payload: Mapping[str, Any],
    *,
    receptors: Sequence[Receptor],
    source_hashes: Mapping[str, str],
    study_site_id: str,
    analysis_region_id: str,
    flow_domain_id: str,
    vertical_id: str | None,
) -> dict[str, Any]:
    """依 selected ``vertical_id`` 選出既有 receptor，不建立或複製新的 XY。"""

    selected_ids = {item.receptor_id for item in receptors}
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise EngineeringWindowError("source receptor records 必須是 list")
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("receptor_id") in selected_ids]
    if len(selected) != len(receptors):
        raise EngineeringWindowError("selected receptor 與 source record 數量不一致")
    result = copy.deepcopy(dict(payload))
    result["records"] = copy.deepcopy(selected)
    result["status"] = "generated"
    result["provenance"] = _copy_component_provenance(
        source_hashes=source_hashes,
        method_id=ENGINEERING_RECEPTOR_METHOD_ID,
        study_site_id=study_site_id,
        analysis_region_id=analysis_region_id,
        flow_domain_id=flow_domain_id,
        vertical_id=vertical_id,
    )
    return result


def _subset_material_payload(payload: Mapping[str, Any], material_id: str) -> dict[str, Any]:
    """保留單一材質 row；schema 2.0 root 不擴充 status/provenance 以維持 loader 相容。"""

    rows = payload.get("records")
    if not isinstance(rows, list):
        raise EngineeringWindowError("source material records 必須是 list")
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("material_id") == material_id]
    if len(selected) != 1:
        raise EngineeringWindowError(f"source material_id 不唯一或不存在：{material_id}")
    result = copy.deepcopy(dict(payload))
    result["records"] = copy.deepcopy(selected)
    return result


def _source_grid_signature(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """把 grid record 路徑正規化後比較 structural metadata，忽略 root 前綴。"""

    normalized: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, Mapping) or not str(item.get("file_kind", "")).startswith("grid"):
            continue
        value = copy.deepcopy(dict(item))
        path = value.get("path")
        if not isinstance(path, str) or "/grid/" not in path:
            raise EngineeringWindowError("grid fingerprint path 缺少 /grid/ 分隔")
        value["path"] = "grid/" + path.split("/grid/", 1)[1]
        normalized.append(value)
    return tuple(sorted(normalized, key=lambda item: str(item["path"])))


def _assert_grid_lineage(
    *,
    forcing_inventory: Mapping[str, Any],
    current_product: Any,
    product: str,
    analysis_region_id: str,
) -> dict[str, Any]:
    """比較 source accepted inventory 與目前產品的 grid metadata/header/size。"""

    products = forcing_inventory.get("products")
    if not isinstance(products, list):
        raise EngineeringWindowError("source forcing_inventory.products 缺少")
    candidates = [
        row
        for row in products
        if isinstance(row, Mapping)
        and row.get("product") == product
        and row.get("analysis_region_id") == analysis_region_id
    ]
    if len(candidates) != 1:
        raise EngineeringWindowError(f"source forcing inventory 缺少唯一 {product}/{analysis_region_id} grid")
    source = candidates[0]
    source_records = source.get("files")
    if not isinstance(source_records, list):
        raise EngineeringWindowError("source forcing product files 缺少")
    current_signature = _source_grid_signature(current_product.source_file_records)
    source_signature = _source_grid_signature(source_records)
    if current_signature != source_signature:
        raise EngineeringWindowError(f"{product}/{analysis_region_id} grid lineage 不一致")
    source_fp = source.get("product_fingerprint_sha256")
    if not isinstance(source_fp, str) or len(source_fp) != 64:
        raise EngineeringWindowError("source forcing product fingerprint 缺少")
    return {
        "product": product,
        "flow_domain_id": current_product.flow_domain_id,
        "months": [item.label for item in current_product.months],
        "source_product_fingerprint_sha256": source_fp,
        "current_product_fingerprint_sha256": _array_record_hash(current_product),
        "grid_records_sha256": sha256(canonical_json_bytes({"files": source_signature})).hexdigest(),
        "grid_match": True,
        # 這些是兩個選定月份的 metadata/header/size structural records；保存它們可在
        # run/resume 時重讀實際 root 並拒絕換到另一份同名但結構已漂移的產品，不把大型
        # NPY 內容誤宣稱已逐 byte checksum。
        "source_grid_file_records": list(source_signature),
        "current_source_file_records": copy.deepcopy(list(current_product.source_file_records)),
    }


def _assert_receptor_mesh_lineage(
    *,
    mesh: Any,
    projection: Any,
    receptors: Sequence[Receptor],
) -> dict[str, Any]:
    """以真正 NativeMesh.locate 比對 source local/global face，不信任 metadata fallback。"""

    checked: list[dict[str, Any]] = []
    for receptor in receptors:
        local = receptor.metadata.get("source_face_local_index")
        global_index = receptor.metadata.get("source_face_global_index")
        if isinstance(local, bool) or not isinstance(local, int):
            raise EngineeringWindowError(f"receptor 缺少 source_face_local_index：{receptor.receptor_id}")
        if isinstance(global_index, bool) or not isinstance(global_index, int):
            raise EngineeringWindowError(f"receptor 缺少 source_face_global_index：{receptor.receptor_id}")
        if local < 0 or local >= mesh.face_nodes_local.shape[0]:
            raise EngineeringWindowError(f"receptor face local 超出目前 mesh：{receptor.receptor_id}")
        actual_global = int(mesh.source_face_global_index[local])
        if actual_global != global_index:
            raise EngineeringWindowError(
                f"receptor source face global 不一致：{receptor.receptor_id} {global_index}!={actual_global}"
            )
        x_m, y_m = projection.project(float(receptor.lon), float(receptor.lat))
        located = mesh.locate(float(x_m), float(y_m))
        if (
            located is None
            or located.source_face_local_index != local
            or located.source_face_global_index != global_index
        ):
            raise EngineeringWindowError(f"NativeMesh.locate 未命中 source face：{receptor.receptor_id}")
        checked.append(
            {
                "receptor_id": receptor.receptor_id,
                "source_face_local_index": local,
                "source_face_global_index": global_index,
                "located_triangle_id": int(located.triangle_id),
            }
        )
    return {"checked_count": len(checked), "records": checked}


def _assert_horizon_coverage(
    product: Any, horizon: EngineeringHorizon, *, product_label: str
) -> dict[str, Any]:
    """由 canonical bounds/gaps 重算 H 時窗，要求 exact 721 等資料節點且不跨 gap。"""

    try:
        window = build_horizon_window(horizon.arrival_time_ns, horizon.support_days)
        coverage = compute_horizon_coverage(
            window,
            expected_start_ns=horizon.start_time_ns,
            expected_end_ns=horizon.arrival_time_ns,
            canonical_start_ns=int(product.canonical.time_utc_ns[0]),
            canonical_end_ns=int(product.canonical.time_utc_ns[-1]),
            canonical_gaps=product.canonical.gaps,
        )
    except (HorizonContractError, IndexError, ValueError) as exc:
        raise EngineeringWindowError(f"{product_label} H 時窗驗證失敗：{exc}") from exc
    if coverage.crossed_gap or coverage.supported_step_count != horizon.expected_step_count:
        missing = [utc_string(value) for value in coverage.missing_time_ns[:5]]
        raise EngineeringWindowError(
            f"{product_label} H 時窗缺節點：missing={len(coverage.missing_time_ns)} {missing}"
        )
    return {
        "product": product_label,
        "start_utc": horizon.start_utc,
        "end_utc": horizon.arrival_utc,
        "expected_step_count": horizon.expected_step_count,
        "supported_step_count": coverage.supported_step_count,
        "missing_utc": [],
        "crossed_gap": False,
        "spatial_scope": "time_axis_only;沿途空間支援由runtime逐stage驗證",
    }


def _assert_nww_arrival_support(
    *,
    product: Any,
    receptors: Sequence[Receptor],
    arrival_ns: int,
) -> dict[str, Any]:
    """對每個獨立 XY 做真正 NWW 四角 endpoint gate；不預算整段 trajectory。"""

    cache = _load_nww_runtime_cache(product)
    unique: dict[tuple[float, float], list[str]] = {}
    for receptor in receptors:
        unique.setdefault((float(receptor.lon), float(receptor.lat)), []).append(receptor.receptor_id)
    records: list[dict[str, Any]] = []
    for (lon, lat), receptor_ids in sorted(unique.items()):
        values, valid = _nww_exact_hour_samples(cache, lon=lon, lat=lat, requested_times=(arrival_ns,))
        available = bool(valid.get(arrival_ns, False))
        value = float(values.get(arrival_ns, float("nan")))
        if not available or not math.isfinite(value):
            raise EngineeringWindowError(
                f"NWW endpoint 四角 mask/finite/physical gate 失敗：{lon},{lat},{arrival_ns}"
            )
        records.append(
            {
                "lon": lon,
                "lat": lat,
                "receptor_ids": sorted(receptor_ids),
                "arrival_utc": utc_string(arrival_ns),
                "valid": True,
                "significant_wave_height_m": value,
                "gate": "runtime_equivalent_four_corner_static_dynamic_mask_finite_physical",
            }
        )
    return {
        "unique_xy_count": len(records),
        "receptor_count": len(receptors),
        "arrival_endpoint_only": True,
        "records": records,
        "limitation": "未宣稱整條H30粒子路徑的NWW空間支援；由runtime每stage驗證",
    }


def _arrival_record(
    *, study_site_id: str, arrival_ns: int, design_version: str
) -> tuple[ArrivalTime, dict[str, Any]]:
    """建立一筆明示工程 arrival 與其 schema-compatible JSON row。"""

    dt = datetime.fromtimestamp(arrival_ns / 1_000_000_000, tz=UTC)
    season = {
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
    }[dt.month]
    arrival_id = stable_identifier("engarr", [study_site_id, str(arrival_ns), design_version])
    metadata: dict[str, float | int | str] = {
        "engineering_only": "true",
        "selection_policy": ENGINEERING_GENERATION_METHOD_ID,
        "arrival_utc": utc_string(arrival_ns),
    }
    value = ArrivalTime(
        arrival_time_id=arrival_id,
        study_site_id=study_site_id,
        time_utc_ns=arrival_ns,
        year=dt.year,
        season=season,
        tide_class="engineering_window",
        phase_or_event="explicit_arrival",
        metadata=metadata,
    )
    return value, {
        "arrival_time_id": arrival_id,
        "study_site_id": study_site_id,
        "time_utc_ns": arrival_ns,
        "year": dt.year,
        "season": season,
        "tide_class": "engineering_window",
        "phase_or_event": "explicit_arrival",
        "metadata": metadata,
    }


def _generated_config_payload(
    source_payload: Mapping[str, Any],
    *,
    study_site_id: str,
    arrival_utc: str,
    backtrack_days: int,
    material_id: str,
    vertical_id: str | None,
    shard_scenario_count: int,
) -> dict[str, Any]:
    """建立 config-relative 的工程設定，移除會觸發舊 release binding 的 stale refs。

    shard 大小與 ``maximum_step_count`` 是這個工程時窗的執行設定，不沿用完整母體的
    隱含值。步數上限依「回溯秒數／目前設定的最小步長」向上取整，再加一個 inclusive
    endpoint 節點，讓 H30／H60 等不同參數都能使用同一介面；它不是為了加速而改變
    ``dt_min_seconds`` 或物理模式。
    """

    result = copy.deepcopy(dict(source_payload))
    result["config_status"] = "generated"
    inputs = result.get("inputs")
    if not isinstance(inputs, dict):
        raise EngineeringWindowError("source config.inputs 必須是 mapping")
    # 工程 H30 的真實證據另存 engineering manifest；不能把舊 7 日 artifact index
    # 重新掛到新 config，否則 runtime 會誤認 source accepted release 已涵蓋新設定。
    for key in (
        "backtrack_support_days",
        "derived_input_artifact_index",
        "ocm_gap_safe_arrival_manifest",
        "nww_full_hourly_analysis_manifest",
    ):
        inputs.pop(key, None)
    boundaries = result.get("boundaries")
    if not isinstance(boundaries, dict):
        raise EngineeringWindowError("source config.boundaries 必須是 mapping")
    boundaries["max_backtrack_days"] = backtrack_days
    integration = result.get("integration")
    if not isinstance(integration, dict):
        raise EngineeringWindowError("source config.integration 必須是 mapping")
    dt_min_seconds = integration.get("dt_min_seconds")
    if isinstance(dt_min_seconds, bool) or not isinstance(dt_min_seconds, (int, float)):
        raise EngineeringWindowError("source config.integration.dt_min_seconds 必須是數值")
    if not math.isfinite(float(dt_min_seconds)) or float(dt_min_seconds) <= 0.0:
        raise EngineeringWindowError("source config.integration.dt_min_seconds 必須是有限正數")
    boundaries["maximum_step_count"] = int(math.ceil(backtrack_days * 86_400.0 / float(dt_min_seconds)) + 1)
    geometry = result.get("geometry")
    scenarios = result.get("scenarios")
    physics = result.get("physics")
    if not isinstance(geometry, dict) or not isinstance(scenarios, dict) or not isinstance(physics, dict):
        raise EngineeringWindowError("source config 缺少 geometry/scenarios/physics mapping")
    execution = result.get("execution")
    if not isinstance(execution, dict):
        raise EngineeringWindowError("source config.execution 必須是 mapping")
    execution["shard_scenario_count"] = shard_scenario_count
    # 這三項是 source 的正式 release binding；engineering artifact 只以自己的
    # manifest／inventory／run plan 保存 hash，不能讓舊 release 或 pilot binding 看似仍
    # 適用於新的單站時窗。物性中的 calibration_status 則仍由既有 material schema 契約
    # 使用，因此只移除 binding/ref 容器，不刪除物性紀錄欄位。
    for key in ("release_binding", "release_approval", "pilot_execution_binding"):
        result.pop(key, None)
    for key in ("calibration", "calibration_manifest", "calibration_manifest_path", "calibration_refs"):
        result.pop(key, None)
    relative = {
        "domain_manifest": f"{ENGINEERING_COMPONENT_DIR}/domain.json",
        "local_domain_manifest": f"{ENGINEERING_COMPONENT_DIR}/local.json",
        "open_boundary_manifest": f"{ENGINEERING_COMPONENT_DIR}/open.json",
        "receptor_manifest": f"{ENGINEERING_COMPONENT_DIR}/receptor.json",
    }
    geometry.update(relative)
    scenarios.update(
        {
            "receptor_manifest": relative["receptor_manifest"],
            "arrival_time_manifest": f"{ENGINEERING_COMPONENT_DIR}/arrival.json",
            "material_manifest": f"{ENGINEERING_COMPONENT_DIR}/material.json",
            "receptor_arrival_initial_condition_manifest": (
                f"{ENGINEERING_COMPONENT_DIR}/initial_conditions.json"
            ),
        }
    )
    settling = physics.get("settling")
    if not isinstance(settling, dict):
        raise EngineeringWindowError("source config.physics.settling 必須是 mapping")
    settling["material_manifest"] = f"{ENGINEERING_COMPONENT_DIR}/material.json"
    result["engineering_window"] = {
        "engineering_only": True,
        "artifact_schema_version": ENGINEERING_ARTIFACT_SCHEMA_VERSION,
        "generation_method_id": ENGINEERING_GENERATION_METHOD_ID,
        "study_site_id": study_site_id,
        "arrival_utc": arrival_utc,
        "backtrack_days": backtrack_days,
        "material_id": material_id,
        "vertical_id": vertical_id,
        "source_horizon_evidence": "engineering_manifest.json",
        "formal_policy": "formal runtime must reject engineering artifact",
    }
    return result


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    """以原子 UTF-8 YAML 寫入新 config，避免半成品被 run CLI 看到。"""

    rendered = yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False)
    _assert_no_symlink_components(path.parent, allow_missing_leaf=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"不可覆寫既有 config：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"config 暫存檔已存在：{temporary}")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def _component_file_records(directory: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """重新讀取生成後 JSON bytes，建立 artifact component hash closure。"""

    result: dict[str, dict[str, Any]] = {}
    for name in names:
        payload, raw_hash, canonical_hash = _json_read(directory / name)
        del payload
        result[name] = {
            "path": name,
            "sha256": raw_hash,
            "canonical_sha256": canonical_hash,
            "size_bytes": (directory / name).stat().st_size,
        }
    return result


def _build_engineering_inventory(
    *,
    config_hash: str,
    horizon: EngineeringHorizon,
    coverage: Sequence[Mapping[str, Any]],
    nww_support: Mapping[str, Any],
    product_fingerprints: Mapping[str, Any],
) -> dict[str, Any]:
    """建立 runtime pilot inventory；根欄位固定符合既有 strict inventory loader。"""

    return {
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "config_hash": config_hash,
        "mode": "engineering_pilot",
        "formal_ready": False,
        "inventories": [
            {
                "kind": "engineering_window",
                "status": "generated",
                "engineering_only": True,
                "horizon_start_utc": horizon.start_utc,
                "horizon_end_utc": horizon.arrival_utc,
                "expected_step_count": horizon.expected_step_count,
            },
            {"kind": "product_fingerprint", "products": copy.deepcopy(dict(product_fingerprints))},
        ],
        "time_axes": [copy.deepcopy(dict(row)) for row in coverage],
        "findings": [
            {
                "code": "ENGINEERING_NWW_ENDPOINT_GATE",
                "severity": "info",
                "unique_xy_count": nww_support["unique_xy_count"],
                "arrival_endpoint_only": True,
            },
            {
                "code": "ENGINEERING_PATH_SPATIAL_GATE_DEFERRED_TO_RUNTIME",
                "severity": "info",
                "message": "Horizon time coverage does not establish trajectory spatial support.",
            },
        ],
    }


def _assert_source_component_lineage(
    *,
    config_path: Path,
    source_directory: Path,
    config: ProjectConfig,
) -> tuple[Any, Any]:
    """以同一組 explicit source paths 載入 scenario／geometry 並核對 config lineage。

    ``validate_input_derivatives`` 只證明指定目錄本身的十個檔案互相一致；若再用 config
    的相對路徑讀到另一份同名 artifact，便可能出現「驗證 A、mesh 卻使用 B」的錯綁。
    這裡要求 config 的七個 manifest reference 正好解析到指定 source directory，並把
    loader 的 canonical component hashes 與已讀 raw JSON hash 逐一比對，才能把後續
    receptor face／dynamic pair 證據繫結到同一組 source bytes。
    """

    expected_paths = {
        "material": source_directory / ARTIFACT_FILENAMES["material"],
        "receptor": source_directory / ARTIFACT_FILENAMES["receptor"],
        "arrival": source_directory / ARTIFACT_FILENAMES["arrival"],
        "initial_condition": source_directory / ARTIFACT_FILENAMES["initial_condition"],
        "domain": source_directory / ARTIFACT_FILENAMES["domain_geometry"],
        "local": source_directory / ARTIFACT_FILENAMES["local_geometry"],
        "open_boundary": source_directory / ARTIFACT_FILENAMES["open_boundary"],
    }
    configured_refs = {
        "material": config.scenarios.material_manifest
        or config.physics.get("settling", {}).get("material_manifest"),
        "receptor": config.scenarios.receptor_manifest or config.geometry.get("receptor_manifest"),
        "arrival": config.scenarios.arrival_time_manifest,
        "initial_condition": config.scenarios.receptor_arrival_initial_condition_manifest,
        "domain": config.geometry.get("domain_manifest"),
        "local": config.geometry.get("local_domain_manifest"),
        "open_boundary": config.geometry.get("open_boundary_manifest"),
    }
    for kind, expected in expected_paths.items():
        reference = configured_refs[kind]
        if not isinstance(reference, str) or not reference.strip():
            raise EngineeringWindowError(f"source config 缺少 {kind} manifest reference")
        resolved = Path(resolve_manifest_path(config_path, reference))
        if Path(os.path.abspath(resolved)) != Path(os.path.abspath(expected)):
            raise EngineeringWindowError(
                f"source config {kind} reference 未指向指定 source directory：{resolved}"
            )

    scenario_inputs = load_scenario_inputs(
        config,
        config_path=config_path,
        require_dynamic_initial_conditions=True,
        formal=False,
    )
    geometries = load_boundary_geometries(
        config,
        config_path=config_path,
        formal=False,
    )
    return scenario_inputs, geometries


def prepare_engineering_window(
    *,
    source_config: str | Path,
    source_input_directory: str | Path,
    study_site_id: str,
    arrival_utc: str,
    backtrack_days: int,
    material_id: str,
    destination: str | Path,
    ocm_native_root: str | Path | None = None,
    nww_analysis_root: str | Path | None = None,
    project_root: str | Path | None = None,
    vertical_id: str | None = None,
    shard_scenario_count: int = ENGINEERING_DEFAULT_SHARD_SCENARIO_COUNT,
) -> EngineeringWindowArtifact:
    """建立單站工程 artifact，先驗證 source 再寫入 generated 文件。

    ``vertical_id`` 為 ``None`` 時保留所選站點全部 source receptors；本輪龜山島 near-bed
    首測應傳 ``near_bed``，得到五個既有 XY 的五筆 dynamic pair。任何垂向過濾只會從
    source receptor manifest 選 row，不會新建 20 XY 或把舊四層資料複製成不同 receptor。
    """

    if (
        not isinstance(study_site_id, str)
        or not study_site_id.strip()
        or study_site_id != study_site_id.strip()
    ):
        raise EngineeringWindowError("study_site_id 必須是沒有空白的非空字串")
    if vertical_id is not None and vertical_id not in _SUPPORTED_VERTICAL_IDS:
        raise EngineeringWindowError(f"vertical_id 不支援：{vertical_id}")
    if (
        isinstance(shard_scenario_count, bool)
        or not isinstance(shard_scenario_count, int)
        or shard_scenario_count < 1
    ):
        raise EngineeringWindowError("shard_scenario_count 必須是正整數")
    horizon = build_engineering_horizon(arrival_utc, backtrack_days)
    config_path = _assert_regular_file(source_config)
    source_directory = _assert_regular_directory(source_input_directory)
    source_config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(source_config_payload, dict):
        raise EngineeringWindowError("source config root 必須是 mapping")
    source_config_model = load_config(config_path, formal_release=False)
    source_config_model.assert_research_domain_policy()
    source_validation = validate_input_derivatives(source_directory, config_path=config_path, formal=False)
    if source_validation.get("valid") is not True:
        errors = source_validation.get("errors", [])
        raise EngineeringWindowError(f"source derived input 未通過驗證：{errors}")
    source_payloads, source_hashes = _source_component_payloads(source_directory)
    source_index_payload, source_index_raw_hash, _ = _json_read(source_directory / "artifact_index.json")
    del source_index_payload
    source_config_raw_hash = sha256(config_path.read_bytes()).hexdigest()
    source_config_hash = source_config_model.config_hash()
    source_scenario_inputs, source_geometries = _assert_source_component_lineage(
        config_path=config_path,
        source_directory=source_directory,
        config=source_config_model,
    )
    expected_scenario_hashes = {
        "material": source_hashes["material"],
        "receptor": source_hashes["receptor"],
        "arrival": source_hashes["arrival"],
        "receptor_arrival_initial_condition": source_hashes["initial_condition"],
    }
    for kind, record in expected_scenario_hashes.items():
        if (
            source_scenario_inputs.file_sha256.get(kind) != record["sha256"]
            or source_scenario_inputs.canonical_component_hashes.get(kind) != record["canonical_sha256"]
        ):
            raise EngineeringWindowError(f"source {kind} loader hash 與指定 artifact 不一致")
    expected_geometry_hashes = {
        "domain": source_hashes["domain_geometry"],
        "local": source_hashes["local_geometry"],
        "open_boundary": source_hashes["open_boundary"],
    }
    for kind, record in expected_geometry_hashes.items():
        if (
            source_geometries.file_sha256.get(kind) != record["sha256"]
            or source_geometries.canonical_component_hashes.get(kind) != record["canonical_sha256"]
        ):
            raise EngineeringWindowError(f"source {kind} geometry hash 與指定 artifact 不一致")

    sites = {site.study_site_id: site for site in source_config_model.study_sites}
    site = sites.get(study_site_id)
    if site is None:
        raise EngineeringWindowError(f"study_site_id 不存在於 source config：{study_site_id}")
    analysis_region_id = site.analysis_region_id
    flow_domain_id = resolve_flow_domain_id(source_config_model, analysis_region_id, formal=False)
    material_rows = source_payloads["material"].get("records")
    if not isinstance(material_rows, list) or not any(
        isinstance(row, Mapping) and row.get("material_id") == material_id for row in material_rows
    ):
        raise EngineeringWindowError(f"material_id 不存在於 source material：{material_id}")

    source_receptor_rows = source_payloads["receptor"].get("records")
    if not isinstance(source_receptor_rows, list):
        raise EngineeringWindowError("source receptor records 缺少")
    selected_rows = [
        row
        for row in source_receptor_rows
        if isinstance(row, Mapping)
        and row.get("study_site_id") == study_site_id
        and (vertical_id is None or row.get("vertical_id") == vertical_id)
    ]
    if not selected_rows:
        raise EngineeringWindowError("selected site/vertical 沒有 source receptors")
    # 先由 loader 重建 dataclass，確保 receptor row 的 schema、座標與站點 cross-reference
    # 已通過既有 gate；不自行從 JSON 猜測 face 或 vertical 欄位。
    source_receptors = source_scenario_inputs.receptors
    selected_receptors = tuple(
        item
        for item in source_receptors
        if item.study_site_id == study_site_id and (vertical_id is None or item.vertical_id == vertical_id)
    )
    if len(selected_receptors) != len(selected_rows):
        raise EngineeringWindowError("source receptor row 與 loader 選定數量不一致")

    native_root = _env_root(ocm_native_root, source_config_model.inputs.ocm_native_root_env)
    nww_root = _env_root(nww_analysis_root, source_config_model.inputs.nww_analysis_root_env)
    assert native_root is not None and nww_root is not None
    native_product = _load_product(
        product="ocm_native",
        root=native_root,
        root_token=source_config_model.inputs.ocm_native_root_env,
        flow_domain_id=flow_domain_id,
        months=horizon.months,
        config=source_config_model,
    )
    nww_product = _load_product(
        product="nww3_analysis",
        root=nww_root,
        root_token=source_config_model.inputs.nww_analysis_root_env,
        flow_domain_id=flow_domain_id,
        months=horizon.months,
        config=source_config_model,
    )
    product_fingerprints = {
        "ocm_native": _assert_grid_lineage(
            forcing_inventory=source_payloads["forcing_inventory"],
            current_product=native_product,
            product="ocm_native",
            analysis_region_id=analysis_region_id,
        ),
        "nww3_analysis": _assert_grid_lineage(
            forcing_inventory=source_payloads["forcing_inventory"],
            current_product=nww_product,
            product="nww3_analysis",
            analysis_region_id=analysis_region_id,
        ),
    }
    domain = next(
        item for item in source_config_model.domains if item.analysis_region_id == analysis_region_id
    )
    projection = __import__(
        "lagrangian_backtracking.geometry", fromlist=["DomainProjection"]
    ).DomainProjection(*domain.center_lonlat)
    mesh = _load_native_mesh(native_product, projection)
    receptor_mesh_binding = _assert_receptor_mesh_lineage(
        mesh=mesh, projection=projection, receptors=selected_receptors
    )
    # OCM native 的 arrival 端點從實際 eta/bed/zcor/wetdry 生成新 dynamic pair；不讀 source
    # 5000-row dynamic 值作捷徑，也不把舊 hash 寫成新 hash。
    _by_time, depth, month_arrays = _load_ocm_pair_cache(native_product)
    source = _by_time.get(horizon.arrival_time_ns)
    if source is None:
        raise EngineeringWindowError("arrival UTC 不在 OCM native canonical source")
    month, local = source
    expected_arrival_month = datetime.fromtimestamp(horizon.arrival_time_ns / 1_000_000_000, tz=UTC).strftime(
        "%Y%m"
    )
    if month.label != expected_arrival_month:
        # dynamic manifest schema 將月份視為 arrival 的來源月份；若跨月 halo 只在前一份
        # 檔案出現，不能把 July row 改標成 August 來繞過契約。caller 應改選一個在其
        # canonical 月份檔案有 exact observed row 的 arrival，再重新建置 artifact。
        raise EngineeringWindowError(
            "arrival exact UTC 只在非同月 OCM chunk 找到；禁止偽改 ocm_month_yyyymm："
            f"arrival={expected_arrival_month}, source={month.label}"
        )
    zcor, elev, wetdry = month_arrays[month.label]
    for receptor in selected_receptors:
        face = receptor.metadata.get("source_face_local_index")
        if not isinstance(face, int) or isinstance(face, bool):
            raise EngineeringWindowError(f"receptor face 缺少：{receptor.receptor_id}")
        node_count = int(mesh.face_node_count[face])
        nodes = mesh.face_nodes_local[face, :node_count]
        wet_value = int(round(float(np.asarray(wetdry[local], dtype=np.float64)[face])))
        if wet_value != 0:
            raise EngineeringWindowError(f"arrival endpoint OCM face 非 wet：{receptor.receptor_id}")
        try:
            _build_face_vertical_support(
                zcor_node_layer=np.asarray(zcor[local, nodes], dtype=np.float64),
                node_elev_m=np.asarray(elev[local, nodes], dtype=np.float64),
                node_depth_m=np.asarray(depth[nodes], dtype=np.float64),
                vertical_ids=(receptor.vertical_id,),
            )
        except InputDerivationError as exc:
            raise EngineeringWindowError(
                f"arrival endpoint OCM vertical gate 失敗：{receptor.receptor_id}"
            ) from exc
    coverage = (
        _assert_horizon_coverage(native_product, horizon, product_label="ocm_native"),
        _assert_horizon_coverage(nww_product, horizon, product_label="nww3_analysis"),
    )
    nww_support = _assert_nww_arrival_support(
        product=nww_product,
        receptors=selected_receptors,
        arrival_ns=horizon.arrival_time_ns,
    )

    arrival, arrival_row = _arrival_record(
        study_site_id=study_site_id,
        arrival_ns=horizon.arrival_time_ns,
        design_version=source_config_model.design_version,
    )
    product_source_hashes = {
        f"ocm_native:{flow_domain_id}": product_fingerprints["ocm_native"][
            "current_product_fingerprint_sha256"
        ],
        f"nww3_analysis:{flow_domain_id}": product_fingerprints["nww3_analysis"][
            "current_product_fingerprint_sha256"
        ],
    }
    dynamic_payload = _dynamic_initial_payload(
        config=source_config_model,
        products_by_region={analysis_region_id: native_product},
        receptors=selected_receptors,
        arrivals=(arrival,),
        source_hashes=product_source_hashes,
        strict=False,
        expected_pair_count=len(selected_receptors),
        generation_method_id=ENGINEERING_GENERATION_METHOD_ID,
        provenance_extra={
            "engineering_only": True,
            "source_component_canonical_sha256": {
                "receptor": source_hashes["receptor"]["canonical_sha256"],
                "arrival": source_hashes["arrival"]["canonical_sha256"],
            },
            "horizon": {
                "start_utc": horizon.start_utc,
                "end_utc": horizon.arrival_utc,
                "expected_step_count": horizon.expected_step_count,
                "spatial_scope": "endpoint only; runtime validates trajectory",
            },
            "vertical_scope": vertical_id or "all_source_verticals",
        },
    )
    arrival_payload = {
        "manifest_kind": "arrival_time_manifest",
        "schema_version": DERIVED_INPUT_SCHEMA_VERSION,
        "status": "generated",
        "design_version": source_config_model.design_version,
        "time_standard": "UTC",
        "selection_method_id": ENGINEERING_ARRIVAL_METHOD_ID,
        "provenance": _copy_component_provenance(
            source_hashes={
                "source_arrival": source_hashes["arrival"]["canonical_sha256"],
                **product_source_hashes,
            },
            method_id=ENGINEERING_ARRIVAL_METHOD_ID,
            study_site_id=study_site_id,
            analysis_region_id=analysis_region_id,
            flow_domain_id=flow_domain_id,
            vertical_id=vertical_id,
        ),
        "records": [arrival_row],
    }
    geometry_payloads = {
        "domain": _subset_geometry_payload(
            source_payloads["domain_geometry"],
            kind="domain",
            study_site_id=study_site_id,
            analysis_region_id=analysis_region_id,
            flow_domain_id=flow_domain_id,
            source_hashes={
                "source_domain": source_hashes["domain_geometry"]["canonical_sha256"],
                **product_source_hashes,
            },
        ),
        "local": _subset_geometry_payload(
            source_payloads["local_geometry"],
            kind="local",
            study_site_id=study_site_id,
            analysis_region_id=analysis_region_id,
            flow_domain_id=flow_domain_id,
            source_hashes={
                "source_local": source_hashes["local_geometry"]["canonical_sha256"],
                **product_source_hashes,
            },
        ),
        "open_boundary": _subset_geometry_payload(
            source_payloads["open_boundary"],
            kind="open_boundary",
            study_site_id=study_site_id,
            analysis_region_id=analysis_region_id,
            flow_domain_id=flow_domain_id,
            source_hashes={
                "source_open": source_hashes["open_boundary"]["canonical_sha256"],
                **product_source_hashes,
            },
        ),
    }
    receptor_payload = _subset_receptor_payload(
        source_payloads["receptor"],
        receptors=selected_receptors,
        source_hashes={
            "source_receptor": source_hashes["receptor"]["canonical_sha256"],
            **product_source_hashes,
        },
        study_site_id=study_site_id,
        analysis_region_id=analysis_region_id,
        flow_domain_id=flow_domain_id,
        vertical_id=vertical_id,
    )
    material_payload = _subset_material_payload(source_payloads["material"], material_id)
    generated_config_payload = _generated_config_payload(
        source_config_payload,
        study_site_id=study_site_id,
        arrival_utc=horizon.arrival_utc,
        backtrack_days=backtrack_days,
        material_id=material_id,
        vertical_id=vertical_id,
        shard_scenario_count=shard_scenario_count,
    )
    artifact_directory = Path(destination)
    _assert_no_symlink_components(artifact_directory, allow_missing_leaf=True)
    if artifact_directory.exists() or artifact_directory.is_symlink():
        raise FileExistsError(f"工程 artifact 目錄已存在：{artifact_directory}")
    artifact_directory.mkdir(parents=True)
    component_directory = artifact_directory / ENGINEERING_COMPONENT_DIR
    component_directory.mkdir()
    config_output = artifact_directory / ENGINEERING_CONFIG_NAME
    _write_yaml(config_output, generated_config_payload)
    component_files = {
        "domain.json": geometry_payloads["domain"],
        "local.json": geometry_payloads["local"],
        "open.json": geometry_payloads["open_boundary"],
        "receptor.json": receptor_payload,
        "arrival.json": arrival_payload,
        "material.json": material_payload,
        "initial_conditions.json": dynamic_payload,
    }
    for filename, payload in component_files.items():
        write_canonical_json(component_directory / filename, payload)
    generated_config_model = load_config(config_output, formal_release=False)
    generated_config_model.assert_research_domain_policy()
    # 重新由磁碟上的 generated component 載入，而不是相信剛才記憶體中的 payload；這一步
    # 會實際驗證單站 geometry、material、arrival、receptor 與 dynamic pair 的 cross-reference，
    # 讓 prepare 在發布 artifact 前就攔下路徑或欄位錯綁。
    generated_scenario_inputs = load_scenario_inputs(
        generated_config_model,
        config_path=config_output,
        require_dynamic_initial_conditions=True,
        formal=False,
    )
    generated_geometries = load_boundary_geometries(
        generated_config_model,
        config_path=config_output,
        formal=False,
    )
    if len(generated_scenario_inputs.receptors) != len(selected_receptors):
        raise EngineeringWindowError("generated receptor loader 數量與 selected source 不一致")
    if len(generated_scenario_inputs.initial_conditions) != len(selected_receptors):
        raise EngineeringWindowError("generated dynamic pair loader 數量與 selected source 不一致")
    if len(generated_scenario_inputs.scenarios) != len(selected_receptors):
        raise EngineeringWindowError("generated scenario 數量與 selected source 不一致")
    if set(generated_geometries.geometries) != {study_site_id}:
        raise EngineeringWindowError("generated geometry 必須只涵蓋指定 study_site_id")
    generated_config_hash = generated_config_model.config_hash()
    inventory_payload = _build_engineering_inventory(
        config_hash=generated_config_hash,
        horizon=horizon,
        coverage=coverage,
        nww_support=nww_support,
        product_fingerprints=product_fingerprints,
    )
    inventory_output = artifact_directory / ENGINEERING_INVENTORY_NAME
    write_canonical_json(inventory_output, inventory_payload)
    inventory_raw_hash = sha256(inventory_output.read_bytes()).hexdigest()
    component_records = _component_file_records(component_directory, tuple(component_files.keys()))
    component_canonical_hashes = {
        name.removesuffix(".json"): record["canonical_sha256"] for name, record in component_records.items()
    }
    # geometry hash 命名必須與 runtime static loader 的 fixed keys 完全一致。
    geometry_canonical_hashes = {
        "domain": component_records["domain.json"]["canonical_sha256"],
        "local": component_records["local.json"]["canonical_sha256"],
        "open_boundary": component_records["open.json"]["canonical_sha256"],
    }
    component_canonical_hashes["receptor_arrival_initial_condition"] = component_records[
        "initial_conditions.json"
    ]["canonical_sha256"]
    code_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    code_provenance = collect_code_provenance(code_root, formal=False).to_dict()
    artifact_manifest = {
        "artifact_schema_version": ENGINEERING_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "engineering_window_artifact",
        "status": "generated",
        "engineering_only": True,
        "run_kind": "pilot",
        "config_path": ENGINEERING_CONFIG_NAME,
        "input_inventory_path": ENGINEERING_INVENTORY_NAME,
        "input_inventory_sha256": inventory_raw_hash,
        "study_site_id": study_site_id,
        "analysis_region_id": analysis_region_id,
        "flow_domain_id": flow_domain_id,
        "material_id": material_id,
        "arrival_time_id": arrival.arrival_time_id,
        "arrival_utc": horizon.arrival_utc,
        "backtrack_days": backtrack_days,
        "horizon": {
            "start_utc": horizon.start_utc,
            "end_utc": horizon.arrival_utc,
            "expected_step_count": horizon.expected_step_count,
            "months": list(horizon.months),
            "coverage": list(coverage),
            "nww_endpoint_support": nww_support,
            "trajectory_spatial_support": "deferred_to_runtime_stage_gate",
        },
        "vertical_scope": {
            "vertical_id": vertical_id,
            "selected_receptor_count": len(selected_receptors),
            "policy": "filter_existing_source_receptors_only",
        },
        "receptor_count": len(selected_receptors),
        "pair_count": len(selected_receptors),
        "scenario_count": len(selected_receptors),
        "source_lineage": {
            "source_config_sha256": source_config_raw_hash,
            "source_config_hash": source_config_hash,
            "source_input_artifact_index_sha256": source_index_raw_hash,
            "source_component_hashes": source_hashes,
            "product_fingerprints": product_fingerprints,
            "receptor_mesh_binding": receptor_mesh_binding,
            "physics_candidate": "inherited finite_depth_stokes; no new H30 calibration performed",
            "old_input_release_reused": False,
            "old_pilot_execution_binding_reused": False,
        },
        "component_files": component_records,
        "geometry_files": {
            "domain": component_records["domain.json"],
            "local": component_records["local.json"],
            "open_boundary": component_records["open.json"],
        },
        "config_hash": generated_config_hash,
        "component_canonical_hashes": component_canonical_hashes,
        "geometry_canonical_hashes": geometry_canonical_hashes,
        "shard_scenario_count": shard_scenario_count,
        "code_provenance": code_provenance,
    }
    manifest_output = artifact_directory / ENGINEERING_MANIFEST_NAME
    write_canonical_json(manifest_output, artifact_manifest)
    return EngineeringWindowArtifact(
        directory=artifact_directory,
        manifest_path=manifest_output,
        config_path=config_output,
        input_inventory_path=inventory_output,
        study_site_id=study_site_id,
        arrival_utc=horizon.arrival_utc,
        backtrack_days=backtrack_days,
        vertical_id=vertical_id,
        receptor_count=len(selected_receptors),
        pair_count=len(selected_receptors),
        scenario_count=len(selected_receptors),
    )


def _assert_runtime_product_lineage(
    *,
    manifest: Mapping[str, Any],
    config: ProjectConfig,
    ocm_native_root: Path,
    nww_analysis_root: Path,
) -> dict[str, Any]:
    """在 run／resume 前重讀 prepare 所用月份，核對產品結構與時間軸指紋。

    prepare 的產品檢查只代表當時看到的 OCM/NWW root；現場可能在兩次 CLI 之間換了
    mount、metadata 或 time array。這裡依 artifact 保存的 flow、月份與
    ``source_file_records`` 重新呼叫既有低記憶體 loader，再比對產品級 fingerprint 與完整
    structural records。任一漂移都在建立 request factory 前拒絕，避免以不同 forcing 繼續
    checkpoint resume；大型 NPY 仍只核對既有 header／size／metadata／time hash 契約。
    """

    source_lineage = manifest.get("source_lineage")
    horizon = manifest.get("horizon")
    if not isinstance(source_lineage, Mapping) or not isinstance(horizon, Mapping):
        raise EngineeringWindowError("artifact 缺少產品來源追溯或 horizon")
    product_fingerprints = source_lineage.get("product_fingerprints")
    months = horizon.get("months")
    if not isinstance(product_fingerprints, Mapping) or not isinstance(months, list) or not months:
        raise EngineeringWindowError("artifact 缺少選定月份產品指紋")
    if any(not isinstance(month, str) or len(month) != 6 for month in months):
        raise EngineeringWindowError("artifact horizon months 格式不合法")
    analysis_region_id = manifest.get("analysis_region_id")
    flow_domain_id = manifest.get("flow_domain_id")
    if not isinstance(analysis_region_id, str) or not isinstance(flow_domain_id, str):
        raise EngineeringWindowError("artifact 缺少產品 region/flow identity")
    if resolve_flow_domain_id(config, analysis_region_id, formal=False) != flow_domain_id:
        raise EngineeringWindowError("artifact flow_domain_id 與 generated config resolver 不一致")
    roots = {"ocm_native": ocm_native_root, "nww3_analysis": nww_analysis_root}
    current: dict[str, Any] = {}
    for product, root in roots.items():
        expected = product_fingerprints.get(product)
        if not isinstance(expected, Mapping):
            raise EngineeringWindowError(f"artifact 缺少 {product} product fingerprint")
        expected_months = expected.get("months")
        if expected_months != months:
            raise EngineeringWindowError(f"artifact {product} months binding 不一致")
        loaded = _load_product(
            product=product,
            root=root,
            root_token=(
                config.inputs.ocm_native_root_env
                if product == "ocm_native"
                else config.inputs.nww_analysis_root_env
            ),
            flow_domain_id=flow_domain_id,
            months=tuple(months),
            config=config,
        )
        actual_hash = _array_record_hash(loaded)
        if actual_hash != expected.get("current_product_fingerprint_sha256"):
            raise EngineeringWindowError(f"run {product} product fingerprint 漂移")
        expected_records = expected.get("current_source_file_records")
        if not isinstance(expected_records, list) or list(loaded.source_file_records) != expected_records:
            raise EngineeringWindowError(f"run {product} source_file_records 漂移")
        current[product] = loaded
    return current


def _validate_engineering_artifact(artifact: str | Path) -> tuple[Path, dict[str, Any], Path, Path]:
    """run 前重新讀取並驗證 artifact hash、engineering flag 與 config/inventory binding。"""

    directory = _assert_regular_directory(artifact)
    manifest_path = directory / ENGINEERING_MANIFEST_NAME
    payload, _raw_hash, _canonical_hash = _json_read(manifest_path)
    missing = sorted(_REQUIRED_ARTIFACT_KEYS - set(payload))
    if missing:
        raise EngineeringWindowError(f"engineering manifest 缺少欄位：{missing}")
    if (
        payload.get("artifact_kind") != "engineering_window_artifact"
        or payload.get("artifact_schema_version") != ENGINEERING_ARTIFACT_SCHEMA_VERSION
    ):
        raise EngineeringWindowError("artifact kind/schema 不符")
    if payload.get("status") != "generated" or payload.get("engineering_only") is not True:
        raise EngineeringWindowError("artifact 必須保持 generated 且 engineering_only=true")
    if payload.get("run_kind") != "pilot":
        raise EngineeringWindowError("engineering artifact 只允許 run_kind=pilot")
    config_ref = payload.get("config_path")
    inventory_ref = payload.get("input_inventory_path")
    if config_ref != ENGINEERING_CONFIG_NAME or inventory_ref != ENGINEERING_INVENTORY_NAME:
        raise EngineeringWindowError("artifact config/inventory path 必須是固定相對檔名")
    config_path = directory / ENGINEERING_CONFIG_NAME
    inventory_path = directory / ENGINEERING_INVENTORY_NAME
    config = load_config(config_path, formal_release=False)
    config.assert_research_domain_policy()
    if config.config_hash() != payload.get("config_hash"):
        raise EngineeringWindowError("artifact config_hash 與 config 不一致")
    config_extra = getattr(config, "model_extra", {})
    engineering_config = config_extra.get("engineering_window") if isinstance(config_extra, Mapping) else None
    if not isinstance(engineering_config, Mapping) or engineering_config.get("engineering_only") is not True:
        raise EngineeringWindowError("generated config 缺少 engineering_only=true binding")
    if engineering_config.get("formal_policy") != "formal runtime must reject engineering artifact":
        raise EngineeringWindowError("generated config formal rejection binding 不一致")
    inventory, _inventory_raw_hash, _inventory_canonical_hash = _json_read(inventory_path)
    if (
        inventory.get("config_hash") != config.config_hash()
        or inventory.get("formal_ready") is not False
        or payload.get("input_inventory_sha256") != _inventory_raw_hash
    ):
        raise EngineeringWindowError("engineering inventory config/formal binding 不一致")
    component_records = payload.get("component_files")
    if not isinstance(component_records, Mapping):
        raise EngineeringWindowError("artifact component_files 必須是 mapping")
    expected_component_names = {
        "domain.json",
        "local.json",
        "open.json",
        "receptor.json",
        "arrival.json",
        "material.json",
        "initial_conditions.json",
    }
    if set(component_records) != expected_component_names:
        raise EngineeringWindowError("engineering component 文件集合不完整或含未知檔案")
    for name, record in component_records.items():
        if not isinstance(name, str) or not isinstance(record, Mapping) or record.get("path") != name:
            raise EngineeringWindowError("artifact component path record 不合法")
        path = directory / ENGINEERING_COMPONENT_DIR / name
        _assert_regular_file(path)
        _payload, raw_hash, canonical_hash = _json_read(path)
        if record.get("sha256") != raw_hash or record.get("canonical_sha256") != canonical_hash:
            raise EngineeringWindowError(f"artifact component hash 不一致：{name}")
    scenario_hashes = payload.get("component_canonical_hashes")
    geometry_hashes = payload.get("geometry_canonical_hashes")
    if not isinstance(scenario_hashes, Mapping) or not isinstance(geometry_hashes, Mapping):
        raise EngineeringWindowError("artifact component/geometry canonical hashes 缺少")
    if scenario_hashes.get("material") != component_records["material.json"].get("canonical_sha256"):
        raise EngineeringWindowError("artifact material canonical hash binding 不一致")
    if scenario_hashes.get("receptor") != component_records["receptor.json"].get("canonical_sha256"):
        raise EngineeringWindowError("artifact receptor canonical hash binding 不一致")
    if scenario_hashes.get("arrival") != component_records["arrival.json"].get("canonical_sha256"):
        raise EngineeringWindowError("artifact arrival canonical hash binding 不一致")
    if scenario_hashes.get("receptor_arrival_initial_condition") != component_records[
        "initial_conditions.json"
    ].get("canonical_sha256"):
        raise EngineeringWindowError("artifact dynamic canonical hash binding 不一致")
    for kind, filename in (
        ("domain", "domain.json"),
        ("local", "local.json"),
        ("open_boundary", "open.json"),
    ):
        if geometry_hashes.get(kind) != component_records[filename].get("canonical_sha256"):
            raise EngineeringWindowError(f"artifact {kind} geometry canonical hash binding 不一致")
    return directory, payload, config_path, inventory_path


def run_engineering_window(
    *,
    artifact: str | Path,
    destination: str | Path,
    run_id: str,
    checkpoint_root: str | Path | None = None,
    ocm_native_root: str | Path | None = None,
    nww_analysis_root: str | Path | None = None,
    project_root: str | Path | None = None,
    experiment_case_id: str = ENGINEERING_DEFAULT_EXPERIMENT_CASE_ID,
    sweep_budget: int | None = None,
    shard_ids: Sequence[str] | None = None,
    resume: bool = False,
) -> tuple[RunExecutionSummary, ...]:
    """從 engineering artifact 建立／恢復 pilot workspace 並執行指定 shard。

    ``shard_ids`` 可使用 immutable plan 的 hash ID；為方便現場 shell，也接受非負整數
    index 並在讀取 plan 後轉成 exact ID。``resume=True`` 只恢復相同 run binding，不會在
    artifact 或 checkpoint 改變後自動接續；``sweep_budget`` 讓首測可先取得真 checkpoint
    與 step cost。函式不接受 formal 模式，也不會把工程 artifact 交給 formal initializer。
    """

    if type(resume) is not bool:
        raise EngineeringWindowError("resume 必須是 bool")
    if sweep_budget is not None and (
        isinstance(sweep_budget, bool) or not isinstance(sweep_budget, int) or sweep_budget < 1
    ):
        raise EngineeringWindowError("sweep_budget 必須是正整數或 None")
    artifact_directory, manifest, config_path, inventory_path = _validate_engineering_artifact(artifact)
    config = load_config(config_path, formal_release=False)
    scenario_inputs = load_scenario_inputs(
        config, config_path=config_path, require_dynamic_initial_conditions=True, formal=False
    )
    geometries = load_boundary_geometries(config, config_path=config_path, formal=False)
    expected_scenarios = int(manifest["scenario_count"])
    if len(scenario_inputs.scenarios) != expected_scenarios:
        raise EngineeringWindowError("artifact scenario_count 與 current manifests 不一致")
    if len(scenario_inputs.initial_conditions) != int(manifest["pair_count"]):
        raise EngineeringWindowError("artifact pair_count 與 current dynamic manifest 不一致")
    if len(scenario_inputs.receptors) != int(manifest["receptor_count"]):
        raise EngineeringWindowError("artifact receptor_count 與 current manifests 不一致")
    expected_component_hashes = manifest["component_canonical_hashes"]
    expected_scenario_keys = (
        "material",
        "receptor",
        "arrival",
        "receptor_arrival_initial_condition",
    )
    current_component_hashes = dict(scenario_inputs.canonical_component_hashes)
    if {key: current_component_hashes.get(key) for key in expected_scenario_keys} != {
        key: expected_component_hashes.get(key) for key in expected_scenario_keys
    }:
        raise EngineeringWindowError("run scenario component canonical hash 與 artifact 不一致")
    expected_geometry_keys = ("domain", "local", "open_boundary")
    current_geometry_hashes = dict(geometries.canonical_component_hashes)
    if {key: current_geometry_hashes.get(key) for key in expected_geometry_keys} != {
        key: manifest["geometry_canonical_hashes"].get(key) for key in expected_geometry_keys
    }:
        raise EngineeringWindowError("run geometry canonical hash 與 artifact 不一致")
    native_root = _env_root(ocm_native_root, config.inputs.ocm_native_root_env)
    nww_root = _env_root(nww_analysis_root, config.inputs.nww_analysis_root_env)
    assert native_root is not None and nww_root is not None
    code_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    provenance = collect_code_provenance(code_root, formal=False)
    if provenance.to_dict() != manifest["code_provenance"]:
        raise EngineeringWindowError("run code provenance 與 prepare artifact 不一致")
    # 只重新檢查選定 H 月份的產品結構與 canonical UTC 軸；不把 H30 的 endpoint gate
    # 誤當成沿途空間證明，也不在這裡預載大型 velocity array。
    _assert_runtime_product_lineage(
        manifest=manifest,
        config=config,
        ocm_native_root=native_root,
        nww_analysis_root=nww_root,
    )
    component_hashes = {key: str(value) for key, value in current_component_hashes.items()}
    geometry_hashes = {key: str(value) for key, value in current_geometry_hashes.items()}
    required_execution_values = {
        "master_seed": config.scenarios.master_seed,
        "seed_policy": config.scenarios.seed_policy,
        "members_per_scenario": config.scenarios.members_per_scenario,
        "shard_scenario_count": config.execution.shard_scenario_count,
        "checkpoint_interval_sweeps": config.execution.checkpoint_interval_sweeps,
    }
    if any(value is None for value in required_execution_values.values()):
        raise EngineeringWindowError("generated config 缺少 run execution scalar")
    destination_root = Path(destination)
    workspace_path = destination_root / run_id
    if resume:
        if not workspace_path.exists() or workspace_path.is_symlink():
            raise EngineeringWindowError("resume 要求既有且非 symlink run workspace")
    else:
        initialize_run_workspace(
            destination_root,
            run_id=run_id,
            scenarios=scenario_inputs.scenarios,
            normalized_config=config.normalized_payload(),
            config_hash=config.config_hash(),
            input_inventory_file=inventory_path,
            component_canonical_hashes=component_hashes,
            geometry_canonical_hashes=geometry_hashes,
            provenance=provenance,
            experiment_case_id=experiment_case_id,
            master_seed=int(config.scenarios.master_seed),
            seed_policy=str(config.scenarios.seed_policy),
            members_per_scenario=int(config.scenarios.members_per_scenario),
            shard_scenario_count=int(config.execution.shard_scenario_count),
            checkpoint_interval_sweeps=int(config.execution.checkpoint_interval_sweeps),
            active_chunk_size=(
                None
                if config.execution.active_chunk_size is None
                else int(config.execution.active_chunk_size)
            ),
            run_kind="pilot",
        )
    plan = load_run_plan(workspace_path)
    run_validation = validate_run(
        workspace_path,
        require_complete=False,
        checkpoint_root=checkpoint_root,
    )
    if not isinstance(run_validation, Mapping) or run_validation.get("valid") is not True:
        errors = run_validation.get("errors", []) if isinstance(run_validation, Mapping) else []
        raise EngineeringWindowError(f"engineering run workspace 驗證失敗：{errors}")
    load_run_progress(workspace_path)
    if plan.get("run_kind") != "pilot" or plan.get("experiment_case_id") != experiment_case_id:
        raise EngineeringWindowError("run plan 只允許相同 pilot experiment_case_id")
    if plan.get("config_hash") != config.config_hash():
        raise EngineeringWindowError("run plan config_hash 與 generated config 不一致")
    if plan.get("component_canonical_hashes") != component_hashes:
        raise EngineeringWindowError("run plan component canonical hash binding 不一致")
    if plan.get("geometry_canonical_hashes") != geometry_hashes:
        raise EngineeringWindowError("run plan geometry canonical hash binding 不一致")
    workspace_inventory = workspace_path / ENGINEERING_INVENTORY_NAME
    if not workspace_inventory.is_file() or workspace_inventory.is_symlink():
        raise EngineeringWindowError("run workspace input_inventory.json 缺失或不是普通檔案")
    workspace_inventory_hash = sha256(workspace_inventory.read_bytes()).hexdigest()
    if workspace_inventory_hash != plan.get(
        "raw_input_inventory_sha256"
    ) or workspace_inventory_hash != manifest.get("input_inventory_sha256"):
        raise EngineeringWindowError("run workspace input inventory hash binding 不一致")
    if plan.get("code_provenance") != provenance.to_dict():
        raise EngineeringWindowError("run plan code provenance 與目前程式不一致")
    if plan.get("scenario_count") != expected_scenarios:
        raise EngineeringWindowError("run plan scenario_count 與 engineering artifact 不一致")
    plan_scalar_fields = (
        "master_seed",
        "seed_policy",
        "members_per_scenario",
        "shard_scenario_count",
        "checkpoint_interval_sweeps",
        "active_chunk_size",
    )
    expected_plan_scalars = {
        **required_execution_values,
        "active_chunk_size": config.execution.active_chunk_size,
    }
    for field in plan_scalar_fields:
        if plan.get(field) != expected_plan_scalars[field]:
            raise EngineeringWindowError(f"run plan {field} 與 generated config 不一致")
    normalized_payload = _read_strict_json_object(
        workspace_path / "normalized_config.json", label="normalized_config"
    )
    if normalized_payload != config.normalized_payload():
        raise EngineeringWindowError("run plan normalized_config 與 generated config 不一致")
    declared_shards = [str(row["shard_id"]) for row in plan["shards"]]
    if shard_ids:
        selected_shards: list[str] = []
        for raw in shard_ids:
            if not isinstance(raw, str) or not raw.strip():
                raise EngineeringWindowError("shard_ids 必須是非空字串")
            if raw in declared_shards:
                selected = raw
            elif raw.isdigit():
                index = int(raw)
                if index < 0 or index >= len(declared_shards):
                    raise EngineeringWindowError(f"shard index 超出範圍：{raw}")
                selected = declared_shards[index]
            else:
                raise EngineeringWindowError(f"未知 shard_id：{raw}")
            if selected not in selected_shards:
                selected_shards.append(selected)
    else:
        selected_shards = declared_shards
    # 不呼叫正式 ``open_pilot_run_controller``：它會再套用完整母體的 static selection／
    # inventory gate，與本 artifact 明示的單站 generated bundle 不相容。這裡沿用同一個
    # validated manifest loader 與現有 run-control lock/checkpoint，直接建立小 bundle 的
    # RuntimeRequestFactory；factory 仍會在每一筆 request 走真實 OCM/NWW、mesh、wet/dry
    # 與物理條件檢查，沒有以 synthetic 或 source 舊 hash 代替 runtime 驗證。
    factory = RuntimeRequestFactory(
        config=config,
        scenario_inputs=scenario_inputs,
        geometries=geometries,
        ocm_native_root=native_root,
        nww_analysis_root=nww_root,
        experiment_case_id=experiment_case_id,
        run_kind="pilot",
    )
    controller = RunController(
        workspace_path,
        request_factory=factory,
        resume=resume,
        checkpoint_root=checkpoint_root,
        resource_reporter=factory.resource_stats,
    )
    return tuple(controller.run_shard(shard_id, sweep_budget=sweep_budget) for shard_id in selected_shards)


__all__ = [
    "ENGINEERING_ARTIFACT_SCHEMA_VERSION",
    "ENGINEERING_DEFAULT_EXPERIMENT_CASE_ID",
    "EngineeringHorizon",
    "EngineeringWindowArtifact",
    "EngineeringWindowError",
    "build_engineering_horizon",
    "parse_arrival_utc",
    "prepare_engineering_window",
    "run_engineering_window",
]
