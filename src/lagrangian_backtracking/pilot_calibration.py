"""建立、讀取與驗證 OCM 真資料 pilot 校準證據。

本模組只接受已由 Slice 1 release config 綁定的 OCM schema 3 衍生輸入，並從
``ForcingWindowManager`` 的 OCM-only route 取樣 receptor×arrival pair。它不讀取 raw
NetCDF、transfer archive 或 NWW3，也不把無效取樣改寫成零速度。輸出是一個不可變的
三檔目錄：固定 Arrow schema 的 ``pair_samples.parquet``、可重算統計與候選值的
``calibration_report.json``，以及列出檔案大小／SHA-256 的 ``manifest.json``。

校準結果僅是條件式來源足跡流程的工程證據與候選係數，尚未是正式擴散基線；在
well-mixed、PDE 障壁、時間／網格收斂與實際軌跡檢查完成前，報告不得解讀成絕對來源
機率、因果歸因或已核定物理參數。
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import ProjectConfig, load_config
from .diffusion import SmagorinskySettings
from .forcing_window import ForcingWindowManager
from .input_derivation import read_canonical_json, validate_release_config
from .manifests import BoundaryGeometryBundle, ScenarioInputs, load_boundary_geometries, load_scenario_inputs
from .models import SampleQC, VelocitySample
from .outputs import sha256_file
from .provenance import collect_code_provenance

PILOT_CALIBRATION_SCHEMA_VERSION: Final[str] = "1.1.0"
# 1.0.0 的 time-limit report 使用跨軸 combined diffusion 公式；它只保留給唯讀
# validator／reader 相容解析，builder 永遠輸出目前的 1.1.0 雙軸契約，避免新程式把
# 舊 evidence 靜默當成新公式。
_PILOT_CALIBRATION_LEGACY_SCHEMA_VERSION: Final[str] = "1.0.0"
_PILOT_CALIBRATION_SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset(
    {_PILOT_CALIBRATION_LEGACY_SCHEMA_VERSION, PILOT_CALIBRATION_SCHEMA_VERSION}
)
# 校準目錄的固定識別與 root topology；validator 會拒絕額外檔案，避免未驗證 payload
# 被下游誤讀。這些名稱不是 OCM source path，因此可安全寫入 portable manifest。
PILOT_CALIBRATION_ARTIFACT_KIND: Final[str] = "ocm_pilot_calibration_evidence"
PILOT_CALIBRATION_FILES: Final[tuple[str, ...]] = (
    "pair_samples.parquet",
    "calibration_report.json",
    "manifest.json",
)
_PAYLOAD_FILES: Final[tuple[str, ...]] = ("pair_samples.parquet", "calibration_report.json")
# quantile level 固定包含規劃要求的 q0、q0.5%、q1%、q5%、q25%、q50%、q75%、q95%、
# q99%、q99.5% 與 q100%；線性方法固定下來後，不同 NumPy 預設不會改變報告語意。
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_YYYYMM_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]{6}$")
_QUANTILE_LEVELS: Final[tuple[float, ...]] = (
    0.0,
    0.005,
    0.01,
    0.05,
    0.25,
    0.5,
    0.75,
    0.95,
    0.99,
    0.995,
    1.0,
)
# 三個 Cs 僅作敏感度取樣；它們不是已核定的正式 runtime 參數。pilot 取樣 cap 使用
# 極大的有限值，目的是保存 raw current-triangle Kh，而不是先把候選 cap 截斷。
_CS_VALUES: Final[tuple[tuple[str, float], ...]] = (
    ("010", 0.10),
    ("015", 0.15),
    ("020", 0.20),
)
# pilot 取樣必須使用可表示的最大有限值，讓 Cs=.20 raw candidate 不先被人工 cap 截斷；
# 真正的 cap candidate 另由有效 raw current-triangle Kh 的 q99.5 產生。
_PILOT_SAMPLE_CAP_M2PS: Final[float] = sys.float_info.max
_REPORT_ROOT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "evidence_class",
        "source_status",
        "engineering_measurement_not_scientific_result",
        "ocm_only",
        "nww_accessed",
        "recommendation_status",
        "completion_status",
        "selection_policy",
        "input_binding",
        "code_provenance",
        "counts",
        "statistics",
        "candidates",
        "time_limit_candidates",
        "resource",
    }
)


def _field(name: str, data_type: pa.DataType, *, nullable: bool = True) -> pa.Field:
    """建立一個具名 Arrow 欄位，集中保存 nullable 政策與 schema 可讀性。"""

    return pa.field(name, data_type, nullable=nullable)


def _build_pair_schema() -> pa.Schema:
    """建立 pair 取樣表的固定欄位契約。

    identity、時間、來源 face、wet/dry 與各路由 QC 一律不可為空；物理量只有在對應
    取樣成功時才填入有限數值，其餘情況維持 Arrow null，讓「無資料」不會與有效的零值
    混淆。所有位置與尺度單位是公尺，速度是 m/s，擴散係數是 m²/s，時間是 UTC 奈秒。
    """

    fields: list[pa.Field] = [
        _field("receptor_id", pa.string(), nullable=False),
        _field("arrival_time_id", pa.string(), nullable=False),
        _field("study_site_id", pa.string(), nullable=False),
        _field("analysis_region_id", pa.string(), nullable=False),
        _field("flow_domain_id", pa.string(), nullable=False),
        _field("time_utc_ns", pa.int64(), nullable=False),
        _field("ocm_month_yyyymm", pa.string(), nullable=False),
        _field("ocm_source_time_index", pa.int64(), nullable=False),
        _field("ocm_time_origin", pa.string(), nullable=False),
        _field("vertical_id", pa.string(), nullable=False),
        _field("z_m_positive_up", pa.float64(), nullable=False),
        _field("eta_m_positive_up", pa.float64(), nullable=False),
        _field("bed_z_m_positive_up", pa.float64(), nullable=False),
        _field("water_column_height_m", pa.float64(), nullable=False),
        _field("height_above_bed_m", pa.float64(), nullable=False),
        _field("zcor_lower_m_positive_up", pa.float64(), nullable=False),
        _field("zcor_upper_m_positive_up", pa.float64(), nullable=False),
        _field("vertical_bracket_alpha", pa.float64(), nullable=False),
        _field("source_face_local_index", pa.int64(), nullable=False),
        _field("source_face_global_index", pa.int64(), nullable=False),
        _field("wetdry_elem_value", pa.int64(), nullable=False),
        _field("wetdry_semantics_id", pa.string(), nullable=False),
        _field("ocm_qc", pa.int64(), nullable=False),
        _field("u_mps", pa.float64()),
        _field("v_mps", pa.float64()),
        _field("w_mps", pa.float64()),
        # 垂向 advection limit 還需考慮 material manifest 的最大沉降速率；將同一
        # immutable design scalar 帶入每列，validator 才能只依 parquet rows 重算時間候選。
        _field("max_abs_settling_velocity_mps", pa.float64(), nullable=False),
        _field("speed_mps", pa.float64()),
        _field("horizontal_scale_m", pa.float64()),
        _field("vertical_scale_m", pa.float64()),
        _field("ocm_sampled_kz_m2ps", pa.float64()),
        _field("velocity_source_face_id", pa.int64()),
        _field("velocity_triangle_id", pa.int64()),
        _field("velocity_forcing_month_id", pa.string()),
    ]
    for token, _coefficient in _CS_VALUES:
        prefix = f"smag_cs_{token}"
        fields.extend(
            (
                _field(f"{prefix}_qc", pa.int64(), nullable=False),
                _field(f"{prefix}_kh_m2ps", pa.float64()),
                _field(f"{prefix}_raw_current_triangle_kh_m2ps", pa.float64()),
                _field(f"{prefix}_d_kh_dx_mps", pa.float64()),
                _field(f"{prefix}_d_kh_dy_mps", pa.float64()),
                _field(f"{prefix}_floor_hit", pa.bool_()),
                _field(f"{prefix}_cap_hit", pa.bool_()),
                _field(f"{prefix}_triangle_id", pa.int64()),
                _field(f"{prefix}_forcing_month_id", pa.string()),
            )
        )
    return pa.schema(fields)


PAIR_SAMPLE_SCHEMA: Final[pa.Schema] = _build_pair_schema()


@dataclass(frozen=True, slots=True)
class PilotCalibration:
    """通過 immutable validator 後的 pilot 校準目錄內容。

    ``pair_samples`` 是唯讀流程所讀出的 Arrow table；``report`` 與 ``manifest`` 以
    mapping proxy 暴露，避免呼叫端把已驗證的摘要欄位改成與 parquet 不一致。此資料類別
    仍不把 OCM root、raw path 或 server 絕對路徑帶入回傳值。
    """

    path: Path
    pair_samples: pa.Table
    report: Mapping[str, Any]
    manifest: Mapping[str, Any]


def _native(value: Any) -> Any:
    """將 NumPy scalar 轉成 Python scalar，避免 JSON／Arrow 邊界留下不可序列化型別。"""

    return value.item() if isinstance(value, np.generic) else value


def _finite(value: Any) -> float | None:
    """只接受有限數值；無法表示科學值時回傳 null 而不補零。"""

    value = _native(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _json_safe(value: Any) -> Any:
    """遞迴轉換 JSON scalar 並拒絕 NaN、Infinity、Path 與未登錄物件。"""

    value = _native(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON 不可含 NaN 或 Infinity")
        return value
    raise TypeError(f"不支援的 JSON 型別：{type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """以固定排序與 UTF-8 pretty JSON 寫入 partial artifact。"""

    safe = _json_safe(payload)
    text = json.dumps(safe, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    """嚴格讀取單一 JSON，拒絕常數 NaN、重複 key、symlink 與非 mapping root。"""

    if path.is_symlink() or not path.is_file():
        raise ValueError("artifact JSON 檔案節點無效")

    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON 常數無效：{value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON 不可含重複 key")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate,
    )
    if not isinstance(payload, dict):
        raise ValueError("JSON root 必須是 mapping")
    return payload


def _sha256_schema(schema: pa.Schema) -> str:
    """對欄名、Arrow 型別與 nullable 旗標做 canonical schema fingerprint。"""

    descriptor = [
        {"name": field.name, "type": str(field.type), "nullable": bool(field.nullable)}
        for field in schema
    ]
    encoded = json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _regular_directory(path: str | Path) -> Path:
    """驗證既有 artifact 目錄不是 symlink；不解析或掃描其外部父層。"""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("pilot calibration path 必須是普通目錄")
    return root


def _prepare_destination(path: str | Path) -> Path:
    """建立新的 partial 發布目標並拒絕覆寫既有節點。"""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("pilot calibration destination 已存在")
    parent = destination.parent
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise ValueError("pilot calibration destination parent 無效")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise ValueError("pilot calibration destination parent 不可為 symlink")
    partial = parent / f".{destination.name}.partial-{uuid4().hex}"
    partial.mkdir()
    return partial


def _validate_pair_limit(value: int | None) -> int | None:
    """驗證每站抽樣上限；bool 不得利用 Python 整數子類別混入設定。"""

    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("pair_limit_per_site 必須是正整數或 None")
    return value


def _pair_key(pair: Any) -> tuple[str, str]:
    """回傳 pair identity；此 identity 不含 material，避免重複展開十種材質。"""

    return str(pair.receptor_id), str(pair.arrival_time_id)


def _stable_pair_digest(pair: Any) -> str:
    """以 canonical identity 產生跨 Python process 穩定的 SHA-256 抽樣排序鍵。"""

    identity = "|".join(
        (
            str(pair.study_site_id),
            str(pair.receptor_id),
            str(pair.arrival_time_id),
            str(int(pair.time_utc_ns)),
        )
    )
    return sha256(identity.encode("utf-8")).hexdigest()


def _pilot_pair_execution_sort_key(pair: Any) -> tuple[str, str, int, str, str]:
    """建立 pilot pair 的 deterministic OCM month-locality execution sort key。

    ``_select_pairs`` 的 stable SHA-256 排序是「哪些 pair 被選入 sample」的資料契約；
    本函式只處理「已選 pair 以何種順序讀取 OCM」的 I/O locality 契約，兩者不能互相
    取代。排序欄位固定為 ``(flow_domain_id, ocm_month_yyyymm, time_utc_ns,
    receptor_id, arrival_time_id)``，先聚合同一 flow domain／月份，再以 UTC、receptor
    與 arrival identity 固定同月內順序；它不重新計算 stable digest，也不改變 selected
    identity 集合。

    flow domain、study site 與 receptor／arrival identity 必須是非空字串；月份必須是
    明確提供的六位 ``YYYYMM`` 字串，不能由 UTC 或檔案路徑猜測。``time_utc_ns`` 必須
    是 Python／NumPy 整數，不能以浮點或字串假冒。任一 pair 欄位缺失、空白、月份格式
    不符或時間型別非法都立即 raise ``ValueError``，避免 locality sorting 把壞 pair
    隱藏到 OCM cache 或後續統計中。
    """

    missing = object()

    def nonempty_text(field: str) -> str:
        """讀取並正規化 pair 的識別字串；缺值不以 fallback 補入。"""

        value = getattr(pair, field, missing)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"pilot pair {field} 必須是非空字串")
        return value.strip()

    flow_domain_id = nonempty_text("flow_domain_id")
    nonempty_text("study_site_id")
    receptor_id = nonempty_text("receptor_id")
    arrival_time_id = nonempty_text("arrival_time_id")
    month = nonempty_text("ocm_month_yyyymm")
    if _YYYYMM_RE.fullmatch(month) is None:
        raise ValueError("pilot pair ocm_month_yyyymm 必須是明確的六位 YYYYMM")
    # 六位數格式仍可能包含不存在的月份；在排序前拒絕這類輸入，避免把
    # 無法對應到 OCM 月份產品的值當成可用的 cache locality 分組依據。
    year = int(month[:4])
    month_number = int(month[4:])
    if year < 1 or not 1 <= month_number <= 12:
        raise ValueError("pilot pair ocm_month_yyyymm 必須使用有效的 YYYYMM 月份")
    time_utc_ns = getattr(pair, "time_utc_ns", missing)
    if isinstance(time_utc_ns, (bool, np.bool_)) or not isinstance(time_utc_ns, (int, np.integer)):
        raise ValueError("pilot pair time_utc_ns 必須是整數")
    return flow_domain_id, month, int(time_utc_ns), receptor_id, arrival_time_id


def _order_pairs_for_execution(pairs: Sequence[Any]) -> list[Any]:
    """依固定 OCM locality key 排序已選 pair，並拒絕無法唯一決定的 identity。

    這個 helper 只在 ``_select_pairs`` 完成 stable-hash selection 後執行，因此不會改變
    pair_limit、可用數、selected count 或 selected set。它先為每筆 pair 建立並驗證
    `_pilot_pair_execution_sort_key`，再以完整 key 排序；若兩筆 pair 具有相同完整 key，
    代表 identity／時間／月份契約重複，不能依輸入列次序任意決定 Parquet row order，
    故直接 fail closed。輸出順序可讓同一 flow domain／OCM 月份連續讀取，降低 NFS
    month cache eviction 與重載，但不影響任何物理取樣或統計公式。
    """

    keyed = [(_pilot_pair_execution_sort_key(pair), pair) for pair in pairs]
    keys = [key for key, _pair in keyed]
    if len(keys) != len(set(keys)):
        raise ValueError("selected pilot pairs 的 locality execution key 不唯一")
    keyed.sort(key=lambda item: item[0])
    return [pair for _key, pair in keyed]


def _select_pairs(
    pairs: Sequence[Any], *, pair_limit_per_site: int | None
) -> tuple[list[Any], dict[str, int], dict[str, int]]:
    """依站點以 stable hash 選取 pair，並回傳可用／選取計數。

    不使用輸入 manifest 的列次序或 Python ``hash``；因此相同 identity 集合在不同
    worker、不同 process 與不同 JSON 排序下都會得到同一批有限 pilot sample。
    """

    by_site: dict[str, list[Any]] = defaultdict(list)
    for pair in pairs:
        by_site[str(pair.study_site_id)].append(pair)
    available: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    selected: list[Any] = []
    for site_id in sorted(by_site):
        ordered = sorted(
            by_site[site_id],
            key=lambda item: (_stable_pair_digest(item), str(item.receptor_id), str(item.arrival_time_id)),
        )
        available[site_id] = len(ordered)
        limit = len(ordered) if pair_limit_per_site is None else min(pair_limit_per_site, len(ordered))
        selected.extend(ordered[:limit])
        selected_counts[site_id] = limit
    return selected, available, selected_counts


def _sample_kz(velocity: VelocitySample) -> float | None:
    """從 OCM velocity diagnostics 取有效 Kz；診斷缺值不以零假裝成實測係數。"""

    raw = velocity.diagnostics.get("kz_m2ps")
    value = _finite(raw)
    return value if value is not None and value >= 0.0 else None


def _diagnostic(sample: Any, key: str) -> Any:
    """從 provider 診斷 mapping 取值，兼容 fake／實際 provider 的 Mapping 實作。"""

    diagnostics = getattr(sample, "diagnostics", {})
    return diagnostics.get(key) if isinstance(diagnostics, Mapping) else None


def _optional_int(value: Any) -> int | None:
    """轉換 nullable integer provenance；拒絕 bool 與非整數浮點。"""

    value = _native(value)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    return int(value)


def _provenance_check(pair: Any, velocity: VelocitySample, diffusion: Mapping[str, Any]) -> None:
    """確認 velocity 與 Smagorinsky 使用相同 OCM month／triangle provenance。

    只有 provider 實際提供的 provenance 才比較；缺少 provenance 不自行推導。若兩條
    路由都提供值卻不一致，代表座標、月份或 mesh 定位不是同一證據，建置立即失敗，避免
    產生看似完整但不可比較的校準表。
    """

    expected_month = str(pair.ocm_month_yyyymm)
    velocity_month = velocity.forcing_month_id
    if velocity_month is not None and velocity_month != expected_month:
        raise ValueError("velocity forcing month provenance mismatch")
    smag_month = diffusion.get("forcing_month_id")
    if smag_month is not None and str(smag_month) != expected_month:
        raise ValueError("diffusion forcing month provenance mismatch")
    velocity_triangle = velocity.triangle_id
    smag_triangle = _optional_int(diffusion.get("triangle_id"))
    if (
        velocity_triangle is not None
        and smag_triangle is not None
        and int(velocity_triangle) != smag_triangle
    ):
        raise ValueError("velocity/diffusion triangle provenance mismatch")
    velocity_face = velocity.source_face_id
    if velocity_face is not None and int(velocity_face) != int(pair.source_face_global_index):
        raise ValueError("velocity source face provenance mismatch")


def _base_pair_row(
    pair: Any,
    velocity: VelocitySample,
    sampled_kz: float | None,
    max_abs_settling_velocity_mps: float,
) -> dict[str, Any]:
    """建立 pair identity、實際 z 與 OCM velocity 欄位，無效數值保留 null。"""

    row: dict[str, Any] = {
        "receptor_id": str(pair.receptor_id),
        "arrival_time_id": str(pair.arrival_time_id),
        "study_site_id": str(pair.study_site_id),
        "analysis_region_id": str(pair.analysis_region_id),
        "flow_domain_id": str(pair.flow_domain_id),
        "time_utc_ns": int(pair.time_utc_ns),
        "ocm_month_yyyymm": str(pair.ocm_month_yyyymm),
        "ocm_source_time_index": int(pair.ocm_source_time_index),
        "ocm_time_origin": str(pair.ocm_time_origin),
        "vertical_id": str(pair.vertical_id),
        "z_m_positive_up": float(pair.z_m_positive_up),
        "eta_m_positive_up": float(pair.eta_m_positive_up),
        "bed_z_m_positive_up": float(pair.bed_z_positive_up)
        if hasattr(pair, "bed_z_positive_up")
        else float(pair.bed_z_m_positive_up),
        "water_column_height_m": float(pair.water_column_height_m),
        "height_above_bed_m": float(pair.height_above_bed_m),
        "zcor_lower_m_positive_up": float(pair.zcor_lower_m_positive_up),
        "zcor_upper_m_positive_up": float(pair.zcor_upper_m_positive_up),
        "vertical_bracket_alpha": float(pair.vertical_bracket_alpha),
        "source_face_local_index": int(pair.source_face_local_index),
        "source_face_global_index": int(pair.source_face_global_index),
        "wetdry_elem_value": int(pair.wetdry_elem_value),
        "wetdry_semantics_id": str(pair.wetdry_semantics_id),
        "ocm_qc": int(velocity.qc),
        "u_mps": None,
        "v_mps": None,
        "w_mps": None,
        "max_abs_settling_velocity_mps": max_abs_settling_velocity_mps,
        "speed_mps": None,
        "horizontal_scale_m": None,
        "vertical_scale_m": None,
        "ocm_sampled_kz_m2ps": sampled_kz,
        "velocity_source_face_id": _optional_int(velocity.source_face_id),
        "velocity_triangle_id": _optional_int(velocity.triangle_id),
        "velocity_forcing_month_id": velocity.forcing_month_id,
    }
    if velocity.valid:
        for field in ("u_mps", "v_mps", "w_mps", "horizontal_scale_m", "vertical_scale_m"):
            row[field] = _finite(getattr(velocity, field))
        if row["u_mps"] is not None and row["v_mps"] is not None:
            row["speed_mps"] = math.hypot(row["u_mps"], row["v_mps"])
    return row


def _diffusion_row_fields(row: dict[str, Any], token: str, sample: Any) -> None:
    """將單一 Cs 的 QC、Kh、梯度、floor/cap 與 triangle provenance 寫入 row。"""

    prefix = f"smag_cs_{token}"
    diagnostics = sample.diagnostics
    row[f"{prefix}_qc"] = int(sample.qc)
    if sample.valid:
        row[f"{prefix}_kh_m2ps"] = _finite(diagnostics.get("kh_m2ps"))
        row[f"{prefix}_raw_current_triangle_kh_m2ps"] = _finite(
            diagnostics.get("raw_current_triangle_kh_m2ps")
        )
        row[f"{prefix}_d_kh_dx_mps"] = _finite(diagnostics.get("d_kh_dx_mps"))
        row[f"{prefix}_d_kh_dy_mps"] = _finite(diagnostics.get("d_kh_dy_mps"))
        row[f"{prefix}_floor_hit"] = bool(diagnostics.get("floor_hit"))
        row[f"{prefix}_cap_hit"] = bool(diagnostics.get("cap_hit"))
        row[f"{prefix}_triangle_id"] = _optional_int(diagnostics.get("triangle_id"))
        row[f"{prefix}_forcing_month_id"] = (
            str(diagnostics["forcing_month_id"])
            if diagnostics.get("forcing_month_id") is not None
            else None
        )
    else:
        for suffix in (
            "kh_m2ps",
            "raw_current_triangle_kh_m2ps",
            "d_kh_dx_mps",
            "d_kh_dy_mps",
            "floor_hit",
            "cap_hit",
            "triangle_id",
            "forcing_month_id",
        ):
            row[f"{prefix}_{suffix}"] = None


def _statistic(values: Sequence[float]) -> dict[str, Any]:
    """產生可重算的有限值統計；空集合的 min/max/quantile 明確為 null。"""

    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        quantiles: list[float | None] = [None] * len(_QUANTILE_LEVELS)
        return {
            "n": 0,
            "min": None,
            "max": None,
            "quantile_levels": list(_QUANTILE_LEVELS),
            "quantile_values": quantiles,
            "q50": None,
            "q995": None,
        }
    array = np.asarray(finite_values, dtype=np.float64)
    quantile_values = [float(value) for value in np.quantile(array, _QUANTILE_LEVELS, method="linear")]
    q50_index = _QUANTILE_LEVELS.index(0.5)
    q995_index = _QUANTILE_LEVELS.index(0.995)
    return {
        "n": len(finite_values),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "quantile_levels": list(_QUANTILE_LEVELS),
        "quantile_values": quantile_values,
        "q50": quantile_values[q50_index],
        "q995": quantile_values[q995_index],
    }


def _valid_metric(rows: Sequence[Mapping[str, Any]], field: str, qc_field: str | None = None) -> list[float]:
    """依 QC=0 與有限 nullable 值擷取一個報告統計欄位。"""

    result: list[float] = []
    for row in rows:
        if qc_field is not None and int(row[qc_field]) != int(SampleQC.OK):
            continue
        value = _finite(row.get(field))
        if value is not None:
            result.append(value)
    return result


def _time_statistic(values: Sequence[float], unlimited_count: int) -> dict[str, Any]:
    """保存有限時間步候選統計及零分母造成的無限制筆數。"""

    result = _statistic(values)
    result["unlimited_count"] = int(unlimited_count)
    return result


def _candidate(
    *,
    values: Sequence[float],
    rule: str,
    metric: str,
    quantile_level: float,
    reason_if_empty: str,
    floor: float | None = None,
) -> dict[str, Any]:
    """由固定 pooled quantile 建立候選係數，空集合或物理界線錯誤時不可用。"""

    statistic = _statistic(values)
    value: float | None = None
    reason = reason_if_empty
    if values:
        index = _QUANTILE_LEVELS.index(quantile_level)
        value = statistic["quantile_values"][index]
        reason = "valid sampled values available"
        if floor is not None and value <= floor:
            value = None
            reason = "candidate cap must be greater than floor"
    return {
        "available": value is not None,
        "value": value,
        "unit": "m2/s",
        "rule": rule,
        "source_metric": metric,
        "quantile_level": quantile_level,
        "reason": reason,
    }


def _max_settling_speed(scenario_inputs: ScenarioInputs) -> float:
    """回傳材質清單中的最大沉降速率絕對值，供垂向保守步長分母使用。"""

    values = [abs(float(item.settling_velocity_mps)) for item in scenario_inputs.materials]
    finite_values = [value for value in values if math.isfinite(value)]
    return max(finite_values, default=0.0)


def _build_time_limit_candidates(
    rows: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    max_settling: float,
    *,
    schema_version: str = PILOT_CALIBRATION_SCHEMA_VERSION,
) -> dict[str, Any]:
    """依指定 report schema 推導保守的 time-limit quantiles。

    advection 維持既有水平／垂向分軸公式 ``0.25*scale/speed``；分母為零或尺度無效
    時，該筆只增加 ``unlimited_count``，不以零值製造有限步長。新 schema 1.1.0 依
    Brownian 三軸方差 ``2*K_axis*dt`` 分開計算水平與垂向 diffusion：水平尺度只配對
    ``constant_kh_m2ps``，垂向尺度只配對 ``constant_kz_m2ps``，兩軸的 K 必須個別大於
    零。這裡禁止 ``min(horizontal_scale_m, vertical_scale_m)`` 與
    ``max(Kh, Kz)`` 的跨軸組合，因為它會把一個軸的細尺度與另一個軸的大擴散係數
    任意配對。1.0.0 則明示保留舊 combined diffusion contract，僅供 legacy artifact
    驗證，不能被當成 1.1.0 公式。

    ``schema_version`` 是資料契約 dispatch，不是可由 rows 或 candidate 猜出的值；只
    接受已登錄的 1.0.0／1.1.0，未知版本直接 ``ValueError``。輸入 rows 的尺度單位是
    m、速度是 m/s、擴散係數是 m²/s，輸出候選步長單位是 s。這個報告候選只供後續
    pilot gate 比較，不會直接修改正式 runtime 的 dt 設定。
    """

    if (
        not isinstance(schema_version, str)
        or schema_version not in _PILOT_CALIBRATION_SUPPORTED_SCHEMA_VERSIONS
    ):
        raise ValueError(f"unsupported pilot calibration schema version: {schema_version!r}")

    horizontal: list[float] = []
    vertical: list[float] = []
    horizontal_unlimited = 0
    vertical_unlimited = 0
    settling_speed = _finite(max_settling)
    for row in rows:
        speed = _finite(row.get("speed_mps"))
        h_scale = _finite(row.get("horizontal_scale_m"))
        w_speed = _finite(row.get("w_mps"))
        v_scale = _finite(row.get("vertical_scale_m"))
        if speed is not None and h_scale is not None and h_scale > 0.0 and speed > 0.0:
            value = 0.25 * h_scale / speed
            if math.isfinite(value):
                horizontal.append(value)
            else:
                horizontal_unlimited += 1
        else:
            horizontal_unlimited += 1
        denominator = (
            abs(w_speed) + settling_speed
            if w_speed is not None and settling_speed is not None and settling_speed >= 0.0
            else None
        )
        if (
            v_scale is not None
            and v_scale > 0.0
            and denominator is not None
            and denominator > 0.0
        ):
            value = 0.25 * v_scale / denominator
            if math.isfinite(value):
                vertical.append(value)
            else:
                vertical_unlimited += 1
        else:
            vertical_unlimited += 1

    kh = candidates["constant_kh_m2ps"]
    kz = candidates["constant_kz_m2ps"]
    kh_value = _finite(kh.get("value")) if kh.get("available") is True else None
    kz_value = _finite(kz.get("value")) if kz.get("available") is True else None
    common = {
        "max_abs_settling_velocity_mps": max_settling,
        "horizontal_advection": {
            "formula": "0.25*horizontal_scale_m/speed_mps",
            "statistics": _time_statistic(horizontal, horizontal_unlimited),
        },
        "vertical_advection": {
            "formula": "0.25*vertical_scale_m/(abs(w_mps)+max_abs_settling_velocity_mps)",
            "statistics": _time_statistic(vertical, vertical_unlimited),
        },
    }
    if schema_version == _PILOT_CALIBRATION_LEGACY_SCHEMA_VERSION:
        # 這段只服務 1.0.0 的唯讀驗證。即使兩個 candidate 中只有一個存在，也必須
        # 重現當年的 combined unavailable 語意，不能把舊資料改寫成新雙軸報告。
        diffusion_values: list[float] = []
        diffusion_unlimited = 0
        if kh_value is not None and kz_value is not None:
            denominator = 2.0 * max(kh_value, kz_value)
            for row in rows:
                h_scale = _finite(row.get("horizontal_scale_m"))
                v_scale = _finite(row.get("vertical_scale_m"))
                if (
                    h_scale is None
                    or v_scale is None
                    or h_scale <= 0.0
                    or v_scale <= 0.0
                    or denominator <= 0.0
                ):
                    diffusion_unlimited += 1
                    continue
                value = (0.25 * min(h_scale, v_scale)) ** 2 / denominator
                if math.isfinite(value):
                    diffusion_values.append(value)
                else:
                    diffusion_unlimited += 1
            diffusion_reason = None
        else:
            diffusion_unlimited = len(rows)
            diffusion_reason = "constant Kh or Kz candidate unavailable"
        common["diffusion"] = {
            "formula": "(0.25*min(horizontal_scale_m,vertical_scale_m))^2/(2*max(Kh,Kz))",
            "statistics": _time_statistic(diffusion_values, diffusion_unlimited),
            "unavailable_reason": diffusion_reason,
        }
        return common

    # 1.1.0 對兩軸各自建立候選；某一軸 candidate 不可用時，只將該軸的有限值集合
    # 留空並保存原因，另一軸仍依自己的 K 與尺度重算，避免錯誤的全域失效。
    def axis_diffusion_candidate(
        *,
        scale_field: str,
        coefficient: float | None,
        coefficient_label: str,
        formula: str,
    ) -> dict[str, Any]:
        """依單一軸尺度與 K 建立 diffusion statistics，禁止跨軸補值。"""

        if coefficient is None or coefficient <= 0.0:
            reason = (
                f"{coefficient_label} candidate unavailable"
                if coefficient is None
                else f"{coefficient_label} candidate must be greater than zero"
            )
            return {
                "formula": formula,
                "statistics": _time_statistic([], len(rows)),
                "unavailable_reason": reason,
            }
        values: list[float] = []
        unlimited = 0
        for row in rows:
            scale = _finite(row.get(scale_field))
            if scale is None or scale <= 0.0:
                unlimited += 1
                continue
            value = (0.25 * scale) ** 2 / (2.0 * coefficient)
            if math.isfinite(value):
                values.append(value)
            else:
                unlimited += 1
        return {
            "formula": formula,
            "statistics": _time_statistic(values, unlimited),
            "unavailable_reason": None,
        }

    common["horizontal_diffusion"] = axis_diffusion_candidate(
        scale_field="horizontal_scale_m",
        coefficient=kh_value,
        coefficient_label="constant Kh",
        formula="(0.25*horizontal_scale_m)^2/(2*constant_kh_m2ps)",
    )
    common["vertical_diffusion"] = axis_diffusion_candidate(
        scale_field="vertical_scale_m",
        coefficient=kz_value,
        coefficient_label="constant Kz",
        formula="(0.25*vertical_scale_m)^2/(2*constant_kz_m2ps)",
    )
    return common


def _qc_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    """將 QC 整數以字串 key 排序輸出，保留每個失敗旗標的原始計數。"""

    counts = Counter(int(row[field]) for row in rows)
    return {str(key): int(counts[key]) for key in sorted(counts)}


def _build_counts(
    rows: Sequence[Mapping[str, Any]],
    available_by_site: Mapping[str, int],
    selected_by_site: Mapping[str, int],
    receptor_count: int,
    arrival_count: int,
) -> dict[str, Any]:
    """建立全域、站點、flow-domain 與月份 QC/valid count 摘要。"""

    nested: dict[str, dict[str, Any]] = {}
    for dimension, field in (
        ("per_flow_domain", "flow_domain_id"),
        ("per_month", "ocm_month_yyyymm"),
    ):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        nested[dimension] = {}
        for key in sorted(groups):
            group = groups[key]
            nested[dimension][key] = {
                "selected_pair_count": len(group),
                "valid_velocity_count": sum(int(row["ocm_qc"]) == 0 for row in group),
                "velocity_qc_counts": _qc_counts(group, "ocm_qc"),
                "smagorinsky_qc_counts": {
                    token: _qc_counts(group, f"smag_cs_{token}_qc") for token, _ in _CS_VALUES
                },
            }
    per_site: dict[str, Any] = {}
    site_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        site_groups[str(row["study_site_id"])].append(row)
    for site_id in sorted(available_by_site):
        group = site_groups.get(site_id, [])
        per_site[site_id] = {
            "available_pair_count": int(available_by_site[site_id]),
            "selected_pair_count": int(selected_by_site.get(site_id, 0)),
            "valid_velocity_count": sum(int(row["ocm_qc"]) == 0 for row in group),
            "velocity_qc_counts": _qc_counts(group, "ocm_qc"),
            "smagorinsky_qc_counts": {
                token: _qc_counts(group, f"smag_cs_{token}_qc") for token, _ in _CS_VALUES
            },
        }
    return {
        "available_pair_count": int(sum(available_by_site.values())),
        "selected_pair_count": len(rows),
        "unique_pair_count": len({(row["receptor_id"], row["arrival_time_id"]) for row in rows}),
        "receptor_count": int(receptor_count),
        "arrival_count": int(arrival_count),
        "valid_velocity_count": sum(int(row["ocm_qc"]) == 0 for row in rows),
        "valid_smagorinsky_count": {
            token: sum(int(row[f"smag_cs_{token}_qc"]) == 0 for row in rows)
            for token, _ in _CS_VALUES
        },
        "per_site": per_site,
        **nested,
    }


def _selected_group_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    """從 parquet rows 重算指定分組的 pair、valid velocity 與各 Cs QC 計數。"""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        key: {
            "selected_pair_count": len(group),
            "valid_velocity_count": sum(int(row["ocm_qc"]) == 0 for row in group),
            "velocity_qc_counts": _qc_counts(group, "ocm_qc"),
            "smagorinsky_qc_counts": {
                token: _qc_counts(group, f"smag_cs_{token}_qc") for token, _ in _CS_VALUES
            },
        }
        for key, group in sorted(groups.items())
    }


def _input_binding(
    config: ProjectConfig,
    input_directory: Path,
    scenario_inputs: ScenarioInputs,
    geometries: BoundaryGeometryBundle,
    release_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """保存 config、release、scenario 與 geometry 的 hash binding，不保存 root path。"""

    _, artifact_index_fingerprint = read_canonical_json(input_directory / "artifact_index.json")
    return {
        "config_hash": config.config_hash(),
        "input_artifact_index_sha256": artifact_index_fingerprint["sha256"],
        "release_config_status": release_validation.get("summary", {}).get("config_status"),
        "scenario_file_sha256": dict(sorted(scenario_inputs.file_sha256.items())),
        "scenario_canonical_component_hashes": dict(
            sorted(scenario_inputs.canonical_component_hashes.items())
        ),
        "geometry_file_sha256": dict(sorted(geometries.file_sha256.items())),
        "geometry_canonical_component_hashes": dict(
            sorted(geometries.canonical_component_hashes.items())
        ),
    }


def _manager_settings(config: ProjectConfig) -> int:
    """取得 forcing cache 容量；缺值只使用 config model 的既有預設，不猜測資料路徑。"""

    value = config.execution.max_resident_forcing_months
    if type(value) is not int or value < 1:
        raise ValueError("execution.max_resident_forcing_months 必須是正整數")
    return value


def _build_report(
    *,
    config: ProjectConfig,
    project_root: Path,
    input_binding: Mapping[str, Any],
    release_validation: Mapping[str, Any],
    scenario_inputs: ScenarioInputs,
    rows: Sequence[Mapping[str, Any]],
    available_by_site: Mapping[str, int],
    selected_by_site: Mapping[str, int],
    pair_limit_per_site: int | None,
    managers: Mapping[str, ForcingWindowManager],
    started: float,
) -> dict[str, Any]:
    """把取樣資料組合成 fixed report；所有摘要均可由 parquet 重新計算。"""

    metrics = {
        "speed_mps": _valid_metric(rows, "speed_mps", "ocm_qc"),
        "horizontal_scale_m": _valid_metric(rows, "horizontal_scale_m", "ocm_qc"),
        "vertical_scale_m": _valid_metric(rows, "vertical_scale_m", "ocm_qc"),
        "ocm_sampled_kz_m2ps": _valid_metric(rows, "ocm_sampled_kz_m2ps", "ocm_qc"),
    }
    for token, _coefficient in _CS_VALUES:
        metrics[f"smag_cs_{token}_kh_m2ps"] = _valid_metric(
            rows, f"smag_cs_{token}_kh_m2ps", f"smag_cs_{token}_qc"
        )
        metrics[f"smag_cs_{token}_raw_current_triangle_kh_m2ps"] = _valid_metric(
            rows,
            f"smag_cs_{token}_raw_current_triangle_kh_m2ps",
            f"smag_cs_{token}_qc",
        )
    statistics = {key: _statistic(value) for key, value in metrics.items()}
    candidates: dict[str, Any] = {
        "constant_kz_m2ps": _candidate(
            values=metrics["ocm_sampled_kz_m2ps"],
            rule="pooled q50 of valid OCM sampled Kz",
            metric="ocm_sampled_kz_m2ps",
            quantile_level=0.5,
            reason_if_empty="no valid OCM sampled Kz",
        ),
        # 常數 Kh 只有一個正式候選，且規劃已固定以 Cs=0.15 的有效 particle Kh q50
        # 作為來源；Cs=.10／.20 仍保留在 statistics 供 sensitivity 稽核，不另產生
        # constant candidate，避免下游誤把敏感度參數當成 baseline。
        "constant_kh_m2ps": _candidate(
            values=metrics["smag_cs_015_kh_m2ps"],
            rule="pooled q50 of valid Smagorinsky particle Kh at Cs=0.15",
            metric="smag_cs_015_kh_m2ps",
            quantile_level=0.5,
            reason_if_empty="no valid Smagorinsky Kh at Cs=0.15",
        ),
        "floor_m2ps": {
            "available": bool(metrics["smag_cs_015_kh_m2ps"]),
            "value": 0.0 if metrics["smag_cs_015_kh_m2ps"] else None,
            "unit": "m2/s",
            "rule": "fixed floor candidate 0.0 after at least one valid Cs=0.15 sample",
            "source_metric": "smag_cs_015_kh_m2ps",
            "quantile_level": None,
            "reason": "valid Cs=0.15 samples available"
            if metrics["smag_cs_015_kh_m2ps"]
            else "no valid Cs=0.15 Kh",
        },
    }
    floor_value = candidates["floor_m2ps"]["value"]
    candidates["cap_m2ps"] = _candidate(
        values=metrics["smag_cs_020_raw_current_triangle_kh_m2ps"],
        rule="pooled q99.5 of valid raw current-triangle Kh at Cs=0.20",
        metric="smag_cs_020_raw_current_triangle_kh_m2ps",
        quantile_level=0.995,
        reason_if_empty="no valid raw current-triangle Kh at Cs=0.20",
        floor=floor_value,
    )
    time_limits = _build_time_limit_candidates(
        rows,
        {
            "constant_kz_m2ps": candidates["constant_kz_m2ps"],
            "constant_kh_m2ps": candidates["constant_kh_m2ps"],
        },
        _max_settling_speed(scenario_inputs),
    )
    cache = [manager.cache_stats for manager in managers.values()]
    resource = {
        "elapsed_seconds": max(0.0, float(time.monotonic() - started)),
        "manager_count": len(managers),
        "sample_call_count": len(rows) * (1 + len(_CS_VALUES)),
        "ocm_load_count": sum(item.ocm_load_count for item in cache),
        "ocm_cache_hit_count": sum(item.ocm_cache_hit_count for item in cache),
        "ocm_cache_miss_count": sum(item.ocm_cache_miss_count for item in cache),
        "nww_load_count": sum(item.nww_load_count for item in cache),
        "nww_cache_hit_count": sum(item.nww_cache_hit_count for item in cache),
        "nww_cache_miss_count": sum(item.nww_cache_miss_count for item in cache),
        "eviction_count": sum(item.eviction_count for item in cache),
        "resident_month_count": sum(len(item.resident_month_ids) for item in cache),
        "resident_ndarray_bytes": sum(item.resident_ndarray_bytes for item in cache),
    }
    counts = _build_counts(
        rows,
        available_by_site,
        selected_by_site,
        len(scenario_inputs.receptors),
        len(scenario_inputs.arrival_times),
    )
    complete = (
        counts["available_pair_count"] == 5_000
        and counts["selected_pair_count"] == 5_000
        and counts["unique_pair_count"] == 5_000
        and counts["receptor_count"] == 100
        and counts["arrival_count"] == 250
        and len(available_by_site) == 5
        and len(selected_by_site) == 5
        and all(value == 1_000 for value in selected_by_site.values())
        and (pair_limit_per_site is None or pair_limit_per_site >= 1_000)
    )
    source_status = str(release_validation.get("summary", {}).get("config_status", config.config_status))
    # builder 的輸入 gate 已經要求 accepted OCM release；本專案不將 synthetic accepted
    # product 送入此流程，因此 evidence class 固定表達 SERVER 真資料 pilot candidate。
    # config status 另存為 source_status，不能用 generated/approved 推導出 synthetic。
    evidence_class = "server_real_data_pilot_candidate"
    return {
        "schema_version": PILOT_CALIBRATION_SCHEMA_VERSION,
        "artifact_kind": PILOT_CALIBRATION_ARTIFACT_KIND,
        "evidence_class": evidence_class,
        "source_status": source_status,
        "engineering_measurement_not_scientific_result": True,
        "ocm_only": True,
        "nww_accessed": resource["nww_load_count"] > 0 or resource["nww_cache_miss_count"] > 0,
        "recommendation_status": (
            "candidate_pending_trajectory_convergence_and_scientific_validation"
        ),
        "completion_status": "complete" if complete else "partial_engineering_sample",
        "selection_policy": {
            "algorithm_id": "per_site_sha256_pair_identity_sort_v1",
            "pair_limit_per_site": pair_limit_per_site,
            "available_pair_count": counts["available_pair_count"],
            "selected_pair_count": counts["selected_pair_count"],
            "status": "full_design_pair_calibration" if complete else "partial_engineering_sample",
            "full_design": {
                "receptor_count": 100,
                "arrival_count": 250,
                "pair_count": 5_000,
                "pairs_per_site": 1_000,
            },
        },
        "input_binding": dict(input_binding),
        "code_provenance": collect_code_provenance(project_root, formal=False).to_dict(),
        "counts": counts,
        "statistics": statistics,
        "candidates": candidates,
        "time_limit_candidates": time_limits,
        "resource": resource,
    }


def build_pilot_calibration(
    *,
    config_path: str | Path,
    input_directory: str | Path,
    ocm_native_root: str | Path,
    destination: str | Path,
    project_root: str | Path,
    pair_limit_per_site: int | None = None,
) -> Path:
    """從 accepted OCM products 建立 immutable pilot calibration evidence。

    建置順序先驗證 release config 與 input artifact closure，再載入 actual-z pair 與
    公尺制 geometry；只有所有輸入 gate 通過後才建立 hidden partial 目錄。每個 pair 使用
    ``ReceptorArrivalInitialCondition.z_m_positive_up`` 與 arrival UTC，velocity 與三個
    Cs sensitivity 皆由同一 OCM-only manager 取樣，不建立 NWW loader。成功後才 rename
    成 destination；任何例外都刪除本次 partial，既有 destination 永不覆寫。

    ``pair_limit_per_site`` 只控制 deterministic engineering sample；省略或每站至少
    1,000 且輸入本身具備 5 sites、100 receptors、250 arrivals、5,000 unique pairs 時，
    報告才標記 ``completion_status=complete``，否則為 ``partial_engineering_sample``。
    stable-hash selection 完成後，已選 pair 會另依 flow domain／OCM 月份／UTC／identity
    排序，以降低月份 cache reload；這個 execution order 不改 pair set、seed、取樣係數、
    統計公式或 evidence status。即使完整，候選係數仍需後續科學 validation 才能升級。
    """

    limit = _validate_pair_limit(pair_limit_per_site)
    config_file = Path(config_path)
    input_root = Path(input_directory)
    config = load_config(config_file, formal_release=False)
    release_validation = validate_release_config(
        config_file,
        input_directory=input_root,
        formal=False,
    )
    if release_validation.get("valid") is not True:
        raise ValueError("pilot calibration input release gate failed")
    input_root = _regular_directory(input_root)
    scenario_inputs = load_scenario_inputs(
        config,
        config_path=config_file,
        require_dynamic_initial_conditions=True,
        formal=False,
    )
    geometries = load_boundary_geometries(config, config_path=config_file, formal=False)
    selected_pairs, available_by_site, selected_by_site = _select_pairs(
        scenario_inputs.initial_conditions,
        pair_limit_per_site=limit,
    )
    # selection 與 execution order 是兩個獨立契約：前者維持 stable-hash sample set，後者
    # 只為讓同一 flow domain／月份的 OCM memory-map 連續使用，避免完整 5,000 pair 在
    # NFS 上因 hash 順序反覆淘汰月份。這裡先驗證所有 selected pair 的欄位與 key 唯一性，
    # 再進入任何 forcing manager 或物理取樣，非法 pair 會 fail closed。
    execution_pairs = _order_pairs_for_execution(selected_pairs)
    receptor_by_id = {item.receptor_id: item for item in scenario_inputs.receptors}
    if {pair.study_site_id for pair in selected_pairs} - set(geometries.projections):
        raise ValueError("selected pair site 缺少已驗證 projection")
    manager_by_domain: dict[str, ForcingWindowManager] = {}
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    max_abs_settling_velocity_mps = _max_settling_speed(scenario_inputs)
    partial: Path | None = None
    try:
        for pair in execution_pairs:
            receptor = receptor_by_id.get(pair.receptor_id)
            if receptor is None:
                raise ValueError("initial condition receptor 不存在")
            projection = geometries.projections[pair.study_site_id]
            manager = manager_by_domain.get(pair.flow_domain_id)
            if manager is None:
                manager = ForcingWindowManager.from_roots(
                    flow_domain_id=pair.flow_domain_id,
                    projection=projection,
                    ocm_root=ocm_native_root,
                    nww_root=None,
                    max_resident_months=_manager_settings(config),
                )
                manager_by_domain[pair.flow_domain_id] = manager
            x_values, y_values = projection.project(receptor.lon, receptor.lat)
            x_m = float(np.asarray(x_values).reshape(()))
            y_m = float(np.asarray(y_values).reshape(()))
            velocity = manager.sample(
                x_m,
                y_m,
                float(pair.z_m_positive_up),
                int(pair.time_utc_ns),
                settling_velocity_mps=0.0,
                include_stokes=False,
                triangle_hint=None,
            )
            sampled_kz = _sample_kz(velocity)
            settings_kz = sampled_kz if sampled_kz is not None else 0.0
            row = _base_pair_row(
                pair,
                velocity,
                sampled_kz,
                max_abs_settling_velocity_mps,
            )
            for token, coefficient in _CS_VALUES:
                diffusion = manager.sample_smagorinsky_diffusion(
                    x_m,
                    y_m,
                    float(pair.z_m_positive_up),
                    int(pair.time_utc_ns),
                    settings=SmagorinskySettings(
                        coefficient_cs=coefficient,
                        floor_m2ps=0.0,
                        cap_m2ps=_PILOT_SAMPLE_CAP_M2PS,
                        constant_kz_m2ps=settings_kz,
                    ),
                    triangle_hint=None,
                )
                _provenance_check(pair, velocity, diffusion.diagnostics)
                _diffusion_row_fields(row, token, diffusion)
            rows.append(row)
        report = _build_report(
            config=config,
            project_root=Path(project_root),
            input_binding=_input_binding(
                config,
                input_root,
                scenario_inputs,
                geometries,
                release_validation,
            ),
            release_validation=release_validation,
            scenario_inputs=scenario_inputs,
            rows=rows,
            available_by_site=available_by_site,
            selected_by_site=selected_by_site,
            pair_limit_per_site=limit,
            managers=manager_by_domain,
            started=started,
        )
        partial = _prepare_destination(destination)
        table = pa.Table.from_pylist(rows, schema=PAIR_SAMPLE_SCHEMA)
        pq.write_table(table, partial / "pair_samples.parquet", compression="zstd", use_dictionary=False)
        _write_json(partial / "calibration_report.json", report)
        manifest = {
            "schema_version": PILOT_CALIBRATION_SCHEMA_VERSION,
            "artifact_kind": PILOT_CALIBRATION_ARTIFACT_KIND,
            "closure_policy": "exact_root_files_v1",
            "root_files": list(PILOT_CALIBRATION_FILES),
            "files": {
                "pair_samples.parquet": {
                    "size_bytes": int((partial / "pair_samples.parquet").stat().st_size),
                    "sha256": sha256_file(partial / "pair_samples.parquet"),
                    "row_count": table.num_rows,
                    "schema_sha256": _sha256_schema(PAIR_SAMPLE_SCHEMA),
                },
                "calibration_report.json": {
                    "size_bytes": int((partial / "calibration_report.json").stat().st_size),
                    "sha256": sha256_file(partial / "calibration_report.json"),
                },
            },
        }
        _write_json(partial / "manifest.json", manifest)
        final = Path(destination)
        if final.exists() or final.is_symlink():
            raise FileExistsError("pilot calibration destination appeared during build")
        os.replace(partial, final)
        partial = None
        return final
    except Exception:
        if partial is not None:
            shutil.rmtree(partial, ignore_errors=True)
        raise


def _compare_number(left: Any, right: Any, *, tolerance: float = 1.0e-12) -> bool:
    """以固定相對／絕對容許度比較報告重算的有限 scalar。"""

    if left is None or right is None:
        return left is None and right is None
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    )


def _compare_json_structure(actual: Any, expected: Any) -> bool:
    """遞迴比較 validator 需要重算的 mapping/list/scalar 結構與有限浮點數。"""

    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            return False
        return set(actual) == set(expected) and all(
            _compare_json_structure(actual[key], expected[key]) for key in expected
        )
    if isinstance(actual, list) or isinstance(expected, list):
        if not isinstance(actual, list) or not isinstance(expected, list):
            return False
        return len(actual) == len(expected) and all(
            _compare_json_structure(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(actual, float) or isinstance(expected, float):
        return _compare_number(actual, expected)
    return actual == expected


def _validate_statistic(
    reported: Any, values: Sequence[float], errors: list[str], label: str
) -> None:
    """驗證一個報告統計的 n、界線、quantile levels 與值。"""

    expected = _statistic(values)
    if not isinstance(reported, Mapping):
        errors.append(f"{label}_invalid")
        return
    for field in ("n", "min", "max", "q50", "q995"):
        if field == "n":
            if reported.get(field) != expected[field]:
                errors.append(f"{label}_{field}_mismatch")
        elif not _compare_number(reported.get(field), expected[field]):
            errors.append(f"{label}_{field}_mismatch")
    if reported.get("quantile_levels") != expected["quantile_levels"]:
        errors.append(f"{label}_levels_mismatch")
    actual_quantiles = reported.get("quantile_values")
    expected_quantiles = expected["quantile_values"]
    if not isinstance(actual_quantiles, list) or len(actual_quantiles) != len(expected_quantiles):
        errors.append(f"{label}_quantiles_invalid")
    else:
        for index, (actual, expected_value) in enumerate(
            zip(actual_quantiles, expected_quantiles, strict=True)
        ):
            if not _compare_number(actual, expected_value):
                errors.append(f"{label}_quantile_{index}_mismatch")
    if expected["n"] > 0 and isinstance(actual_quantiles, list):
        for actual, next_value in zip(
            actual_quantiles[:-1], actual_quantiles[1:], strict=False
        ):
            if (
                actual is not None
                and next_value is not None
                and float(actual) > float(next_value)
            ):
                errors.append(f"{label}_quantiles_not_monotonic")
                break


def _report_has_forbidden_root(report: Mapping[str, Any]) -> bool:
    """拒絕報告把 OCM root 或任意 root-like absolute path 寫入 evidence。"""

    def visit(value: Any, key: str = "") -> bool:
        if isinstance(value, Mapping):
            return any(visit(item, str(name)) for name, item in value.items())
        if isinstance(value, (list, tuple)):
            return any(visit(item, key) for item in value)
        if isinstance(value, str) and ("root" in key.lower() or "path" in key.lower()):
            return Path(value).is_absolute()
        return False

    return visit(report)


def _is_supported_pilot_calibration_schema_version(value: object) -> bool:
    """判定 calibration artifact 版本是否為已明示登錄的可讀格式。

    1.1.0 是 builder 目前寫出的雙軸 diffusion 契約；1.0.0 僅代表舊的 combined
    diffusion report，仍可由唯讀 validator／reader 驗證。版本必須直接存在於 JSON，
    不從 report topology、candidate 欄位或資料內容推測，避免把未知格式靜默套用到
    錯誤的公式。輸入不是字串或未登錄版本時回傳 False，由呼叫端加入固定錯誤碼。
    """

    return (
        isinstance(value, str)
        and value in _PILOT_CALIBRATION_SUPPORTED_SCHEMA_VERSIONS
    )


def _validate_manifest(root: Path, errors: list[str]) -> dict[str, Any] | None:
    """驗證固定 root topology、版本 dispatch 與 manifest payload checksum。

    manifest 的 `schema_version` 只能是目前 builder 輸出的 1.1.0 或唯讀相容的 1.0.0；
    兩版本共用相同三檔 topology 與 pair-table schema，但 report 內 diffusion time-limit
    語意不同。此函式只驗證 manifest 宣告與檔案 checksum，實際 report 公式由 semantic
    validator 依同一版本再次重算。
    """

    try:
        root_files = {item.name for item in root.iterdir()}
    except Exception:
        errors.append("root_files_unreadable")
        root_files = set()
    if root_files != set(PILOT_CALIBRATION_FILES):
        errors.append("root_files_invalid")
    try:
        manifest = _read_json(root / "manifest.json")
    except Exception:
        errors.append("manifest_invalid")
        return None
    if set(manifest) != {"schema_version", "artifact_kind", "closure_policy", "root_files", "files"}:
        errors.append("manifest_keys_invalid")
    if not _is_supported_pilot_calibration_schema_version(manifest.get("schema_version")):
        errors.append("manifest_schema_version_invalid")
    if manifest.get("artifact_kind") != PILOT_CALIBRATION_ARTIFACT_KIND:
        errors.append("manifest_kind_invalid")
    if manifest.get("closure_policy") != "exact_root_files_v1":
        errors.append("manifest_closure_policy_invalid")
    if manifest.get("root_files") != list(PILOT_CALIBRATION_FILES):
        errors.append("manifest_root_files_invalid")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(_PAYLOAD_FILES):
        errors.append("manifest_payload_files_invalid")
        return manifest
    for filename in _PAYLOAD_FILES:
        record = files.get(filename)
        if not isinstance(record, Mapping):
            errors.append(f"manifest_record_invalid:{filename}")
            continue
        required = {"size_bytes", "sha256"}
        if filename.endswith("parquet"):
            required |= {"row_count", "schema_sha256"}
        if set(record) != required:
            errors.append(f"manifest_record_keys_invalid:{filename}")
            continue
        path = root / filename
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError
            if type(record["size_bytes"]) is not int or record["size_bytes"] != path.stat().st_size:
                errors.append(f"manifest_size_mismatch:{filename}")
            if not isinstance(record["sha256"], str) or _SHA256_RE.fullmatch(record["sha256"]) is None:
                errors.append(f"manifest_sha256_invalid:{filename}")
            elif record["sha256"] != sha256_file(path):
                errors.append(f"manifest_sha256_mismatch:{filename}")
        except Exception:
            errors.append(f"manifest_payload_unreadable:{filename}")
    return manifest


def _validate_rows(table: pa.Table, report: Mapping[str, Any], errors: list[str]) -> list[dict[str, Any]]:
    """驗證 Arrow schema、nullable numeric policy、identity uniqueness 與 QC fields。"""

    if not table.schema.equals(PAIR_SAMPLE_SCHEMA, check_metadata=True):
        errors.append("pair_schema_mismatch")
    rows = table.to_pylist()
    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        identity = (row.get("receptor_id"), row.get("arrival_time_id"))
        if identity in identities:
            errors.append("pair_identity_duplicate")
        identities.add(identity)
        if not isinstance(row.get("ocm_month_yyyymm"), str) or len(row["ocm_month_yyyymm"]) != 6:
            errors.append("pair_month_invalid")
        for field in PAIR_SAMPLE_SCHEMA:
            value = row.get(field.name)
            if value is not None and isinstance(value, float) and not math.isfinite(value):
                errors.append(f"pair_nonfinite:{index}:{field.name}")
        for token, _coefficient in _CS_VALUES:
            qc = row.get(f"smag_cs_{token}_qc")
            if type(qc) is not int or qc < 0:
                errors.append(f"pair_smag_qc_invalid:{token}")
        if type(row.get("ocm_qc")) is not int or row["ocm_qc"] < 0:
            errors.append("pair_ocm_qc_invalid")
    if report.get("counts", {}).get("unique_pair_count") != len(identities):
        errors.append("pair_unique_count_mismatch")
    return rows


def _validate_report_against_rows(
    report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], errors: list[str]
) -> None:
    """依 report schema 重算 counts、statistics、候選值與 completion marker。

    report 的 schema version 是 time-limit 語意的唯一 dispatch；1.1.0 必須含獨立的
    horizontal／vertical diffusion products，1.0.0 則必須精確重算舊 combined diffusion
    product。兩者不做欄位猜測或自動轉換，讓舊 artifact 可以被驗證但不會被誤讀為新公式。
    """

    if set(report) != _REPORT_ROOT_KEYS:
        errors.append("report_keys_invalid")
        return
    report_schema_version = report.get("schema_version")
    if not _is_supported_pilot_calibration_schema_version(report_schema_version):
        errors.append("report_schema_version_invalid")
        return
    if report.get("artifact_kind") != PILOT_CALIBRATION_ARTIFACT_KIND:
        errors.append("report_kind_invalid")
    if report.get("engineering_measurement_not_scientific_result") is not True:
        errors.append("report_scientific_status_invalid")
    if report.get("ocm_only") is not True or report.get("nww_accessed") is not False:
        errors.append("report_ocm_only_invalid")
    if report.get("evidence_class") != "server_real_data_pilot_candidate":
        errors.append("report_evidence_class_invalid")
    if report.get("recommendation_status") != (
        "candidate_pending_trajectory_convergence_and_scientific_validation"
    ):
        errors.append("report_recommendation_status_invalid")
    if _report_has_forbidden_root(report):
        errors.append("report_absolute_root_forbidden")
    counts = report.get("counts")
    if not isinstance(counts, Mapping):
        errors.append("report_counts_invalid")
        return
    selected = len(rows)
    expected_count_keys = {
        "available_pair_count",
        "selected_pair_count",
        "unique_pair_count",
        "receptor_count",
        "arrival_count",
        "valid_velocity_count",
        "valid_smagorinsky_count",
        "per_site",
        "per_flow_domain",
        "per_month",
    }
    if set(counts) != expected_count_keys:
        errors.append("report_count_keys_invalid")
    for field in (
        "available_pair_count",
        "selected_pair_count",
        "unique_pair_count",
        "receptor_count",
        "arrival_count",
        "valid_velocity_count",
    ):
        value = counts.get(field)
        if type(value) is not int or value < 0:
            errors.append(f"report_count_invalid:{field}")
    if counts.get("selected_pair_count") != selected or counts.get("unique_pair_count") != selected:
        errors.append("report_pair_count_mismatch")
    available_pair_count = counts.get("available_pair_count")
    if (
        type(available_pair_count) is int
        and type(counts.get("selected_pair_count")) is int
        and available_pair_count < counts["selected_pair_count"]
    ):
        errors.append("report_available_pair_count_less_than_selected")
    per_site = counts.get("per_site")
    if not isinstance(per_site, Mapping):
        errors.append("report_per_site_invalid")
    else:
        per_site_selected_total = 0
        per_site_available_total = 0
        per_site_counts_valid = True
        for item in per_site.values():
            if not isinstance(item, Mapping):
                per_site_counts_valid = False
                continue
            selected_count = item.get("selected_pair_count")
            available_count = item.get("available_pair_count")
            if type(selected_count) is not int or selected_count < 0:
                per_site_counts_valid = False
            else:
                per_site_selected_total += selected_count
            if type(available_count) is not int or available_count < 0:
                per_site_counts_valid = False
            else:
                per_site_available_total += available_count
                if type(selected_count) is int and available_count < selected_count:
                    per_site_counts_valid = False
        if not per_site_counts_valid:
            errors.append("report_per_site_count_invalid")
        if per_site_selected_total != selected:
            errors.append("report_per_site_count_mismatch")
        if (
            type(available_pair_count) is int
            and per_site_available_total != available_pair_count
        ):
            errors.append("report_per_site_available_count_mismatch")
        expected_sites = _selected_group_counts(rows, "study_site_id")
        if set(per_site) != set(expected_sites) and not (
            not expected_sites and all(
                item.get("selected_pair_count") == 0 for item in per_site.values()
            )
        ):
            errors.append("report_per_site_keys_mismatch")
        for site_id, expected in expected_sites.items():
            reported = per_site.get(site_id)
            if not isinstance(reported, Mapping):
                errors.append(f"report_per_site_missing:{site_id}")
                continue
            for field in (
                "selected_pair_count",
                "valid_velocity_count",
                "velocity_qc_counts",
                "smagorinsky_qc_counts",
            ):
                if not _compare_json_structure(reported.get(field), expected[field]):
                    errors.append(f"report_per_site_field_mismatch:{site_id}:{field}")
    for dimension in ("per_flow_domain", "per_month"):
        reported_groups = counts.get(dimension)
        expected_groups = _selected_group_counts(
            rows,
            "flow_domain_id" if dimension == "per_flow_domain" else "ocm_month_yyyymm",
        )
        if not isinstance(reported_groups, Mapping) or not _compare_json_structure(
            reported_groups, expected_groups
        ):
            errors.append(f"report_{dimension}_mismatch")
    if counts.get("valid_velocity_count") != sum(int(row["ocm_qc"]) == 0 for row in rows):
        errors.append("report_velocity_count_mismatch")
    for token, _coefficient in _CS_VALUES:
        expected = sum(int(row[f"smag_cs_{token}_qc"]) == 0 for row in rows)
        if counts.get("valid_smagorinsky_count", {}).get(token) != expected:
            errors.append(f"report_smag_valid_count_mismatch:{token}")
    statistics = report.get("statistics")
    if not isinstance(statistics, Mapping):
        errors.append("report_statistics_invalid")
    else:
        metric_fields = {
            "speed_mps": ("speed_mps", "ocm_qc"),
            "horizontal_scale_m": ("horizontal_scale_m", "ocm_qc"),
            "vertical_scale_m": ("vertical_scale_m", "ocm_qc"),
            "ocm_sampled_kz_m2ps": ("ocm_sampled_kz_m2ps", "ocm_qc"),
        }
        for token, _coefficient in _CS_VALUES:
            metric_fields[f"smag_cs_{token}_kh_m2ps"] = (
                f"smag_cs_{token}_kh_m2ps",
                f"smag_cs_{token}_qc",
            )
            metric_fields[f"smag_cs_{token}_raw_current_triangle_kh_m2ps"] = (
                f"smag_cs_{token}_raw_current_triangle_kh_m2ps",
                f"smag_cs_{token}_qc",
            )
        if set(statistics) != set(metric_fields):
            errors.append("report_statistics_keys_invalid")
        for metric, (field, qc_field) in metric_fields.items():
            _validate_statistic(
                statistics.get(metric),
                _valid_metric(rows, field, qc_field),
                errors,
                f"statistics:{metric}",
            )
    candidates = report.get("candidates")
    if not isinstance(candidates, Mapping):
        errors.append("report_candidates_invalid")
        return
    expected_kz = _candidate(
        values=_valid_metric(rows, "ocm_sampled_kz_m2ps", "ocm_qc"),
        rule="pooled q50 of valid OCM sampled Kz",
        metric="ocm_sampled_kz_m2ps",
        quantile_level=0.5,
        reason_if_empty="no valid OCM sampled Kz",
    )
    expected_kh = _candidate(
        values=_valid_metric(rows, "smag_cs_015_kh_m2ps", "smag_cs_015_qc"),
        rule="pooled q50 of valid Smagorinsky particle Kh at Cs=0.15",
        metric="smag_cs_015_kh_m2ps",
        quantile_level=0.5,
        reason_if_empty="no valid Smagorinsky Kh at Cs=0.15",
    )
    valid_015 = _valid_metric(rows, "smag_cs_015_kh_m2ps", "smag_cs_015_qc")
    expected_floor = {
        "available": bool(valid_015),
        "value": 0.0 if valid_015 else None,
        "unit": "m2/s",
        "rule": "fixed floor candidate 0.0 after at least one valid Cs=0.15 sample",
        "source_metric": "smag_cs_015_kh_m2ps",
        "quantile_level": None,
        "reason": "valid Cs=0.15 samples available" if valid_015 else "no valid Cs=0.15 Kh",
    }
    raw_020 = _valid_metric(
        rows,
        "smag_cs_020_raw_current_triangle_kh_m2ps",
        "smag_cs_020_qc",
    )
    expected_cap_value = _statistic(raw_020)["q995"] if raw_020 else None
    if expected_cap_value is not None and expected_cap_value <= 0.0:
        expected_cap_value = None
    expected_cap = _candidate(
        values=raw_020,
        rule="pooled q99.5 of valid raw current-triangle Kh at Cs=0.20",
        metric="smag_cs_020_raw_current_triangle_kh_m2ps",
        quantile_level=0.995,
        reason_if_empty="no valid raw current-triangle Kh at Cs=0.20",
        floor=expected_floor["value"],
    )
    expected_candidates = {
        "constant_kz_m2ps": expected_kz,
        "constant_kh_m2ps": expected_kh,
        "floor_m2ps": expected_floor,
        "cap_m2ps": expected_cap,
    }
    if set(candidates) != set(expected_candidates):
        errors.append("candidate_keys_invalid")
    for name, expected in expected_candidates.items():
        candidate = candidates.get(name)
        if not isinstance(candidate, Mapping) or not _compare_json_structure(candidate, expected):
            errors.append(f"candidate_contract_mismatch:{name}")
    time_limits = report.get("time_limit_candidates")
    reported_max_settling = (
        time_limits.get("max_abs_settling_velocity_mps")
        if isinstance(time_limits, Mapping)
        else None
    )
    row_settling = [
        _finite(row.get("max_abs_settling_velocity_mps"))
        for row in rows
    ]
    finite_row_settling = [value for value in row_settling if value is not None]
    max_settling = max(finite_row_settling, default=0.0)
    if not rows and isinstance(reported_max_settling, (int, float)):
        max_settling = float(reported_max_settling)
    if (
        any(value is None or value < 0.0 for value in row_settling)
        or not isinstance(reported_max_settling, (int, float))
        or not math.isfinite(float(reported_max_settling))
        or reported_max_settling < 0.0
        or (rows and not _compare_number(reported_max_settling, max_settling))
    ):
        errors.append("time_limit_settling_speed_invalid")
    else:
        expected_time_limits = _build_time_limit_candidates(
            rows,
            {"constant_kh_m2ps": expected_kh, "constant_kz_m2ps": expected_kz},
            float(max_settling),
            schema_version=report_schema_version,
        )
        if not _compare_json_structure(time_limits, expected_time_limits):
            errors.append("time_limit_candidates_mismatch")
    selection = report.get("selection_policy")
    if not isinstance(selection, Mapping):
        errors.append("selection_policy_invalid")
    else:
        expected_selection_keys = {
            "algorithm_id",
            "pair_limit_per_site",
            "available_pair_count",
            "selected_pair_count",
            "status",
            "full_design",
        }
        if set(selection) != expected_selection_keys:
            errors.append("selection_policy_keys_invalid")
        if selection.get("algorithm_id") != "per_site_sha256_pair_identity_sort_v1":
            errors.append("selection_algorithm_invalid")
        if selection.get("available_pair_count") != counts.get("available_pair_count"):
            errors.append("selection_available_count_mismatch")
        if selection.get("selected_pair_count") != counts.get("selected_pair_count"):
            errors.append("selection_selected_count_mismatch")
        selection_limit = selection.get("pair_limit_per_site")
        if selection_limit is not None and (type(selection_limit) is not int or selection_limit < 1):
            errors.append("selection_pair_limit_invalid")
        if selection.get("full_design") != {
            "receptor_count": 100,
            "arrival_count": 250,
            "pair_count": 5_000,
            "pairs_per_site": 1_000,
        }:
            errors.append("selection_full_design_invalid")
        full = (
            counts.get("available_pair_count") == 5_000
            and selected == 5_000
            and counts.get("receptor_count") == 100
            and counts.get("arrival_count") == 250
            and len(counts.get("per_site", {})) == 5
            and all(
                item.get("selected_pair_count") == 1_000
                for item in counts.get("per_site", {}).values()
            )
            and (
                selection_limit is None
                or (type(selection_limit) is int and selection_limit >= 1_000)
            )
        )
        expected_status = "complete" if full else "partial_engineering_sample"
        if report.get("completion_status") != expected_status:
            errors.append("completion_status_mismatch")
        expected_selection_status = (
            "full_design_pair_calibration" if full else "partial_engineering_sample"
        )
        if selection.get("status") != expected_selection_status:
            errors.append("selection_status_mismatch")
    resource = report.get("resource")
    if not isinstance(resource, Mapping):
        errors.append("report_resource_invalid")
    else:
        for field in (
            "manager_count",
            "sample_call_count",
            "ocm_load_count",
            "ocm_cache_hit_count",
            "ocm_cache_miss_count",
            "nww_load_count",
            "nww_cache_hit_count",
            "nww_cache_miss_count",
            "eviction_count",
            "resident_month_count",
            "resident_ndarray_bytes",
        ):
            if type(resource.get(field)) is not int or resource[field] < 0:
                errors.append(f"resource_{field}_invalid")
        if resource.get("sample_call_count") != selected * (1 + len(_CS_VALUES)):
            errors.append("resource_sample_call_count_mismatch")
        nww_fields = ("nww_load_count", "nww_cache_hit_count", "nww_cache_miss_count")
        if any(resource.get(field) != 0 for field in nww_fields):
            errors.append("resource_nww_accessed")
        elapsed = resource.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or elapsed < 0.0:
            errors.append("resource_elapsed_invalid")


def validate_pilot_calibration(
    path: str | Path,
    *,
    config_path: str | Path | None = None,
    input_directory: str | Path | None = None,
) -> dict[str, Any]:
    """唯讀驗證 pilot calibration topology、checksums、schema、QC 與候選統計。

    若提供 config／input directory，validator 會再次呼叫 release config gate 並比對
    input artifact index 與 config hash；未提供時仍完整驗證 artifact 內部 closure，但不會
    猜測外部 root。1.1.0 report 使用雙軸 diffusion time-limit；既有 1.0.0 report 以
    明示的 combined diffusion contract 驗證，兩者不做靜默轉換。回傳固定 JSON-safe
    ``valid/errors/summary``，不把第三方例外或絕對路徑洩漏給 CLI。
    """

    errors: list[str] = []
    try:
        root = _regular_directory(path)
    except Exception:
        return {"valid": False, "errors": ["root_invalid"], "summary": {}}
    manifest = _validate_manifest(root, errors)
    report: dict[str, Any] | None = None
    table: pa.Table | None = None
    try:
        report = _read_json(root / "calibration_report.json")
    except Exception:
        errors.append("report_unreadable")
    # manifest 與 report 必須共同宣告同一資料契約；1.0.0／1.1.0 雖都可讀，不能
    # 讓一份 legacy manifest 搭配新 report（或反向）而繞過 diffusion schema dispatch。
    if (
        manifest is not None
        and report is not None
        and manifest.get("schema_version") != report.get("schema_version")
    ):
        errors.append("manifest_report_schema_version_mismatch")
    try:
        parquet_path = root / "pair_samples.parquet"
        if parquet_path.is_symlink() or not parquet_path.is_file():
            raise ValueError
        table = pq.read_table(parquet_path)
        if manifest is not None:
            record = manifest.get("files", {}).get("pair_samples.parquet", {})
            if isinstance(record, Mapping) and record.get("row_count") != table.num_rows:
                errors.append("manifest_row_count_mismatch")
            if (
                isinstance(record, Mapping)
                and record.get("schema_sha256") != _sha256_schema(PAIR_SAMPLE_SCHEMA)
            ):
                errors.append("manifest_schema_sha256_mismatch")
    except Exception:
        errors.append("pair_table_unreadable")
    if report is not None and table is not None:
        try:
            rows = _validate_rows(table, report, errors)
            _validate_report_against_rows(report, rows, errors)
        except Exception:
            # Parquet／JSON 被竄改到不符合欄位型別時，validator 仍須回傳固定 invalid
            # report，而不能把第三方例外或檔案路徑洩漏給 CLI caller。
            errors.append("semantic_validation_failed")
        binding = report.get("input_binding")
        if not isinstance(binding, Mapping):
            errors.append("input_binding_invalid")
        else:
            for field in ("config_hash", "input_artifact_index_sha256"):
                value = binding.get(field)
                if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                    errors.append(f"input_binding_hash_invalid:{field}")
            if config_path is not None and input_directory is None:
                # 有 config 卻省略 input directory 時，release validator 可能自行從 YAML 推導
                # sibling path；pilot CLI 必須拒絕這種看似有驗證、實際未明示 exact binding 的
                # 呼叫，避免 operator 以錯誤 input 目錄誤跑校準。
                errors.append("input_directory_required_with_config")
            elif input_directory is not None:
                try:
                    input_root = _regular_directory(input_directory)
                    _, fingerprint = read_canonical_json(input_root / "artifact_index.json")
                    if binding.get("input_artifact_index_sha256") != fingerprint.get("sha256"):
                        errors.append("input_artifact_index_hash_mismatch")
                    if config_path is not None:
                        config = load_config(config_path, formal_release=False)
                        release = validate_release_config(
                            config_path,
                            input_directory=input_root,
                            formal=False,
                        )
                        if release.get("valid") is not True:
                            errors.append("input_release_gate_failed")
                        if binding.get("config_hash") != config.config_hash():
                            errors.append("input_config_hash_mismatch")
                except Exception:
                    errors.append("input_binding_external_invalid")
    summary: dict[str, Any] = {}
    if report is not None:
        summary = {
            "schema_version": report.get("schema_version"),
            "evidence_class": report.get("evidence_class"),
            "completion_status": report.get("completion_status"),
            "selected_pair_count": report.get("counts", {}).get("selected_pair_count")
            if isinstance(report.get("counts"), Mapping)
            else None,
            "valid_velocity_count": report.get("counts", {}).get("valid_velocity_count")
            if isinstance(report.get("counts"), Mapping)
            else None,
        }
    return {"valid": not errors, "errors": errors, "summary": summary}


def read_pilot_calibration(path: str | Path) -> PilotCalibration:
    """完整驗證並讀取 immutable pilot calibration artifact。"""

    validation = validate_pilot_calibration(path)
    if validation.get("valid") is not True:
        raise ValueError("pilot calibration 驗證失敗")
    root = _regular_directory(path)
    report = _read_json(root / "calibration_report.json")
    manifest = _read_json(root / "manifest.json")
    table = pq.read_table(root / "pair_samples.parquet")
    return PilotCalibration(
        path=root,
        pair_samples=table,
        report=MappingProxyType(report),
        manifest=MappingProxyType(manifest),
    )


__all__ = [
    "PAIR_SAMPLE_SCHEMA",
    "PILOT_CALIBRATION_ARTIFACT_KIND",
    "PILOT_CALIBRATION_FILES",
    "PILOT_CALIBRATION_SCHEMA_VERSION",
    "PilotCalibration",
    "build_pilot_calibration",
    "read_pilot_calibration",
    "validate_pilot_calibration",
]
