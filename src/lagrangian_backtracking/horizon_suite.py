"""建立與驗證共用一次 input-build 的多 horizon／沉底時間模式設定集合。

本模組提供一個可由 CLI 重建的原子流程：先將未綁定的設定 template 深拷貝成
``common-config.yaml``，把最大指定日數寫入共同支援窗，接著只呼叫一次
``build_input_derivatives`` 建立 ``common-input``，最後以同一批到達時刻、受體、
材質與 dynamic initial-condition 產生每個 release YAML。bed-residence 設計以
``最大 horizon + 最大沉底年齡`` 建立 observation 選時包絡，再將同一母體註冊為
``fixed_calendar_window`` 與 ``full_horizon_from_deposition`` 兩種 release；因此
30／60／90 日、雙模式共六份 release 不必重跑昂貴的 input-build。

本模組不讀取 raw NetCDF、不以零值或最近值補真正缺口；缺時仍由既有 gap-safe
validator 與版本化重建政策決定是否可用。共同母體通過只代表輸入與設定契約已通過，
不保證每一粒子都能在物理邊界與資料支援下走滿所要求的 horizon。

suite 對外成功的最後提交點是固定 publication marker；native exclusive rename 不可用時，
只在 backend 明確回報 unsupported 的 NFS 兩階段複製才會建立該 marker。未含 marker 的
partial／reserved destination 僅供本模組 private gate 稽核，不能由公開 validator 當成成功。
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from .bed_residence import (
    BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
    BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
)
from .config import (
    ARRIVAL_SELECTION_POLICY_LEGACY_TWO_YEAR_V1,
    CURRENT_DESIGN_VERSION,
    ProjectConfig,
)
from .input_derivation import (
    ARTIFACT_FILENAMES,
    DENOMINATOR_POLICY_ID,
    DEPOSITION_AVAILABILITY_POLICY_ID,
    DERIVED_INPUT_SCHEMA_VERSION,
    OBSERVED_GAP_CENSORED_STOP_POLICY_ID,
    _approved_reconstruction_release_binding_enabled,
    _assert_no_symlink_components,
    _assert_regular_directory,
    _assert_regular_file,
    _atomic_write_bytes,
    _validate_artifact_directory,
    build_input_derivatives,
    canonical_json_bytes,
    create_release_config,
    read_canonical_json,
    validate_input_derivatives,
    validate_release_config,
    write_canonical_json,
)
from .input_horizon import (
    BED_RESIDENCE_INPUT_SCHEMA_VERSION,
    GAP_CENSORED_BED_RESIDENCE_INPUT_SCHEMA_VERSION,
)
from .report_release import (
    _call_exclusive_rename,
    _load_exclusive_rename_backend,
    _ReportReleaseAtomicRenameUnsupported,
)

HORIZON_SUITE_SCHEMA_VERSION = "1.1.0"
"""多 horizon suite 的固定 schema 版本。"""

HORIZON_SUITE_MANIFEST_FILENAME = "horizon-suite-manifest.json"
"""suite 根目錄的 manifest 檔名。"""

HORIZON_SUITE_COMMON_CONFIG_FILENAME = "common-config.yaml"
"""共同 input 建置設定的檔名。"""

HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME = "source-template.yaml"
"""suite 內保存的原始 template 副本檔名。"""

HORIZON_SUITE_COMMON_INPUT_DIRECTORY = "common-input"
"""共同 input artifact 的固定子目錄。"""

HORIZON_SUITE_RELEASE_DIRECTORY = "release-configs"
"""release YAML 的固定子目錄。"""

HORIZON_SUITE_VALIDATION_DIRECTORY = "validations"
"""input／release validation JSON 的固定子目錄。"""

HORIZON_SUITE_PUBLICATION_MARKER_FILENAME = "horizon-suite-publication.json"
"""正式 suite 根目錄的不可變發布完成標記檔名。"""

HORIZON_SUITE_PUBLICATION_POLICY_ID = "exclusive_native_or_nfs_two_phase_copy"
"""發布策略識別碼；只允許 native exclusive rename 或受控 NFS fallback。"""

HORIZON_SUITE_PUBLICATION_POLICY_VERSION = "1.0.0"
"""發布策略版本；marker 以此版本綁定寫入與驗證語意。"""

HORIZON_SUITE_NATIVE_PUBLICATION_METHOD = "native_exclusive_rename_v1"
"""同一 parent 目錄內以 native exclusive rename 完成的發布方法。"""

HORIZON_SUITE_NFS_PUBLICATION_METHOD = "nfs_two_phase_copy_v1"
"""NFS 不支援 exclusive rename 時使用的保留目錄、複製、marker 兩階段方法。"""

_ALLOWED_PUBLICATION_METHODS = frozenset(
    {
        HORIZON_SUITE_NATIVE_PUBLICATION_METHOD,
        HORIZON_SUITE_NFS_PUBLICATION_METHOD,
    }
)

HORIZON_SUITE_POLICY_ID = "shared_horizon_and_bed_age_input_one_build_v1"
"""共同母體只建置一次的政策識別碼。"""

HORIZON_SUITE_METHOD_ID = "lbt_horizon_suite_build_v2"
"""本建置方法識別碼。"""

_IDENTITY_KINDS = ("arrival", "receptor", "material", "initial_condition")
_DERIVED_FIELDS = (
    ("inputs", "ocm_gap_reconstruction_manifest"),
    ("inputs", "ocm_gap_safe_arrival_manifest"),
    ("inputs", "nww_full_hourly_analysis_manifest"),
    ("inputs", "derived_input_artifact_index"),
    ("scenarios", "material_manifest"),
    ("scenarios", "receptor_manifest"),
    ("scenarios", "arrival_time_manifest"),
    ("scenarios", "receptor_arrival_initial_condition_manifest"),
    ("geometry", "domain_manifest"),
    ("geometry", "local_domain_manifest"),
    ("geometry", "open_boundary_manifest"),
    ("geometry", "receptor_manifest"),
    ("geometry", "reporting_region_manifest"),
)
_ALLOWED_INITIAL_CONDITION_PLACEHOLDER = (
    "manifests/receptor_arrival_initial_condition.json"
)

_SUITE_PATHS = {
    "source_template": HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME,
    "common_config": HORIZON_SUITE_COMMON_CONFIG_FILENAME,
    "common_input": HORIZON_SUITE_COMMON_INPUT_DIRECTORY,
    "release_configs": HORIZON_SUITE_RELEASE_DIRECTORY,
    "validations": HORIZON_SUITE_VALIDATION_DIRECTORY,
}
"""manifest.paths 的固定 suite-relative topology；不能由 manifest 任意注入路徑。"""

_COMMON_INPUT_EXTRA_FILENAMES = {"artifact_index.json", "artifact_bindings.json"}
"""既有 input builder 產出的兩個目錄層級 closure 檔案。"""

_SUITE_MANIFEST_KEYS = frozenset(
    {
        "manifest_kind",
        "schema_version",
        "source_schema_version",
        "status",
        "gate_mode",
        "policy_id",
        "method_id",
        "horizons_days",
        "forcing_years",
        "observation_years",
        "arrival_selection_policy_id",
        "replicates_per_stratum",
        "arrival_core_count",
        "arrival_event_count",
        "selection_support_days",
        "runtime_support_days",
        "backtrack_modes",
        "step_count_formula",
        "dt_min_seconds",
        "maximum_step_count_by_horizon",
        "source_template_fingerprint",
        "common_config_fingerprint",
        "common_artifact_index_fingerprint",
        "common_artifact_fingerprints",
        "common_identity_fingerprints",
        "input_validation_fingerprint",
        "input_validation_context",
        "releases",
        "paths",
        "input_build_count",
        "recovery_method",
        "recovery_source_fingerprint",
    }
)
"""根 manifest 的 exact 欄位集合；未登錄欄位不能靠重簽 sidecar 加入。"""

_RELEASE_RECORD_KEYS = frozenset(
    {
        "backtrack_days",
        "backtrack_mode",
        "maximum_step_count",
        "config_status",
        "config_path",
        "validation_path",
        "config_fingerprint",
        "validation_fingerprint",
        "identity_fingerprints",
    }
)
"""manifest.releases 每筆紀錄的固定欄位集合。"""

_RELEASE_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "source_config_template_sha256",
        "source_config_hash",
        "input_directory_artifact_index_sha256",
        "artifacts",
        "approved_only_after_exact_hash_validation",
        "backtrack_horizon_binding",
        "arrival_selection_binding",
    }
)
"""create_release_config 登錄的 release binding 欄位集合。"""

_RELEASE_ARRIVAL_SELECTION_BINDING_KEYS = frozenset(
    {"forcing_years", "observation_years", "policy", "replicates"}
)
"""release 直接保存的 forcing／observation 母體身分欄位。"""

_RELEASE_ARTIFACT_RECORD_KEYS = frozenset(
    {"kind", "sha256", "canonical_sha256", "size_bytes", "path"}
)
"""每個 release artifact binding 的 exact 欄位集合。"""

_RELEASE_HORIZON_BINDING_KEYS = frozenset(
    {
        "source_config_hash",
        "source_backtrack_support_days",
        "artifact_backtrack_support_days",
        "requested_max_backtrack_days",
        "requested_maximum_step_count",
    }
)
"""共同支援窗與單一 release horizon 的固定 evidence 欄位。"""

_BED_RELEASE_HORIZON_BINDING_KEYS = _RELEASE_HORIZON_BINDING_KEYS | frozenset(
    {
        "selection_support_days",
        "runtime_support_days",
        "artifact_selection_support_days",
        "artifact_runtime_support_days",
        "backtrack_mode",
    }
)

_GAP_CENSOR_RELEASE_BINDING_KEYS = frozenset(
    {
        "gap_policy_id",
        "deposition_availability_policy_id",
        "denominator_policy_id",
    }
)
_GAP_CENSOR_SUITE_MANIFEST_KEYS = frozenset(
    {
        "gap_policy_id",
        "deposition_availability_policy_id",
        "denominator_policy_id",
    }
)
"""隨機沉底設計將選時包絡與沉底後 runtime 支援分開綁定。"""

_BED_RESIDENCE_MODES = (
    BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
    BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
)


def _payload_gap_censoring_enabled(payload: Mapping[str, Any]) -> bool:
    """判定 suite source/common/release YAML 是否完整明示缺時截尾契約。

    suite 必須把政策寫入每一份 release binding；這裡只接受 bed-residence block、
    三個 exact policy ID 與 ``stop_at_data_gap=true``，legacy template 不會因 suite
    validator 的預設值而被重新解讀。
    """

    inputs = payload.get("inputs")
    boundaries = payload.get("boundaries")
    scenarios = payload.get("scenarios")
    bed = scenarios.get("bed_residence_time") if isinstance(scenarios, Mapping) else None
    time_contract = (
        inputs.get("time_axis_contract")
        if isinstance(inputs, Mapping)
        else None
    )
    # canonical gap policy 位於 inputs.time_axis_contract；為了讓 validator 能讀取
    # 尚未正規化的 YAML 與已產出的 release，仍接受過渡期把同一欄位放在 root／inputs
    # 的 alias。這些 mapping 只作 exact policy 比對，不會由 validator 自行推導缺時策略。
    sections = [payload, inputs, time_contract, boundaries, scenarios, bed]

    def read(name: str) -> Any:
        values = [
            section.get(name)
            for section in sections
            if isinstance(section, Mapping) and name in section
        ]
        if not values:
            return None
        if len({json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values}) != 1:
            raise HorizonSuiteError(f"suite 缺時 policy 欄位 {name} 宣告衝突")
        return values[0]

    gap_value = next(
        (
            read(name)
            for name in ("gap_policy", "ocm_gap_policy", "time_gap_policy", "data_gap_policy")
            if read(name) is not None
        ),
        None,
    )
    deposition_value = next(
        (
            read(name)
            for name in (
                "deposition_availability_policy",
                "deposition_policy",
                "availability_conditioning_policy",
            )
            if read(name) is not None
        ),
        None,
    )
    denominator_value = read("denominator_policy")
    return (
        isinstance(bed, Mapping)
        and gap_value == OBSERVED_GAP_CENSORED_STOP_POLICY_ID
        and deposition_value == DEPOSITION_AVAILABILITY_POLICY_ID
        and denominator_value == DENOMINATOR_POLICY_ID
        and isinstance(boundaries, Mapping)
        and boundaries.get("stop_at_data_gap") is True
    )


def _payload_gap_censoring_contract(payload: Mapping[str, Any]) -> dict[str, str] | None:
    """回傳 suite manifest 要保存的 exact policy snapshot，未知設定回傳 None。"""

    if not _payload_gap_censoring_enabled(payload):
        return None
    return {
        "gap_policy_id": OBSERVED_GAP_CENSORED_STOP_POLICY_ID,
        "deposition_availability_policy_id": DEPOSITION_AVAILABILITY_POLICY_ID,
        "denominator_policy_id": DENOMINATOR_POLICY_ID,
    }


def _arrival_population_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """從 source/common/release YAML 重建 forcing 與 observation 母體契約。

    ``inputs.years`` 是 forcing 產品聯集，正式新版則透過根層
    ``arrival_time_selection`` 把可作 arrival anchor 的 observation 年份縮成
    ``[2025]``。此 helper 同時相容過渡期的 ``scenarios`` 巢狀 block 與舊兩年份
    policy，讓 suite validator 能用同一份 canonical payload 比對 source、common、
    六份 release，而不依賴 YAML 中任意自述的摘要欄位。
    """

    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise HorizonSuiteError("config.inputs 必須是 mapping")
    raw_forcing = inputs.get("years")
    if not isinstance(raw_forcing, list) or not raw_forcing:
        raise HorizonSuiteError("config.inputs.years 必須是非空 list")
    forcing_years: list[int] = []
    for value in raw_forcing:
        if isinstance(value, bool) or type(value) is not int:
            raise HorizonSuiteError("config.inputs.years 必須是非 bool 整數")
        forcing_years.append(int(value))
    if len(set(forcing_years)) != len(forcing_years):
        raise HorizonSuiteError("config.inputs.years 不得重複")

    selection: Any = payload.get("arrival_time_selection")
    if selection is None:
        scenarios = payload.get("scenarios")
        if isinstance(scenarios, Mapping):
            selection = scenarios.get("arrival_time_selection")
    if selection is None:
        selection = {}
    if not isinstance(selection, Mapping):
        raise HorizonSuiteError("arrival_time_selection 必須是 mapping")
    observation_raw = selection.get("observation_years")
    observation_years = (
        list(forcing_years)
        if observation_raw is None
        else [int(value) for value in observation_raw]
        if isinstance(observation_raw, list)
        else None
    )
    if not observation_years or len(set(observation_years)) != len(observation_years):
        raise HorizonSuiteError("arrival_time_selection.observation_years 必須是非空 list")
    if not set(observation_years).issubset(set(forcing_years)):
        raise HorizonSuiteError("observation_years 必須是 forcing_years 子集")
    replicates = selection.get("replicates_per_stratum", selection.get("replicates", 1))
    if isinstance(replicates, bool) or type(replicates) is not int or replicates < 1:
        raise HorizonSuiteError("arrival selection replicates 必須是正整數")
    event_count = selection.get("event_supplement_count", selection.get("event_count", 2))
    if isinstance(event_count, bool) or type(event_count) is not int or event_count < 0:
        raise HorizonSuiteError("arrival selection event count 必須是非負整數")
    expected_core = len(observation_years) * 4 * 2 * 3 * replicates
    core_count = selection.get("core_count", expected_core)
    if isinstance(core_count, bool) or type(core_count) is not int or core_count != expected_core:
        raise HorizonSuiteError("arrival selection core_count 與 observation strata 不一致")
    policy = selection.get(
        "policy_id", selection.get("policy", selection.get("selection_policy_id"))
    )
    if policy is not None and (not isinstance(policy, str) or not policy.strip()):
        raise HorizonSuiteError("arrival selection policy 必須是非空字串")
    enabled = bool(
        policy is not None
        or observation_years != forcing_years
        or replicates != 1
        or event_count != 2
    )
    if enabled and policy is None:
        raise HorizonSuiteError("新版 observation 母體必須明示 arrival selection policy")
    effective_policy = policy or ARRIVAL_SELECTION_POLICY_LEGACY_TWO_YEAR_V1
    return {
        "forcing_years": forcing_years,
        "observation_years": observation_years,
        "arrival_selection_policy_id": effective_policy,
        "replicates_per_stratum": replicates,
        "arrival_core_count": core_count,
        "arrival_event_count": event_count,
    }

_RELEASE_APPROVAL_KEYS = frozenset(
    {
        "status",
        "blockers",
        "validated_input_summary",
        "formal_input_validation_summary",
        "public_analysis_label_policy",
    }
)
"""create_release_config 登錄的 release approval 欄位集合。"""

_WETDRY_RELEASE_APPROVAL_KEY = "wetdry_semantics_approval"
"""正式 current-design release 的 wet/dry 衍生核准 evidence 欄位名稱。"""


class HorizonSuiteError(ValueError):
    """horizon suite 的輸入、完整性或 gate 不符合時使用的例外。"""


def normalize_horizons(values: Sequence[Any]) -> tuple[int, ...]:
    """將回溯日數嚴格正規化為升冪且唯一的整數 tuple。

    ``values`` 是研究者希望比較的日數，例如亂序的 ``[90, 30, 60]``。核心 API
    刻意拒絕字串、浮點數、布林、零、負數、空集合與重複值，避免 Python／YAML 的
    隱式轉型改變研究設計；CLI 解析後的原生 ``int`` 也會再次經過這個 gate。
    """

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise HorizonSuiteError("backtrack_days 必須是非空正整數序列")
    result: list[int] = []
    for value in values:
        if type(value) is not int or value <= 0:
            raise HorizonSuiteError("backtrack_days 必須全部是唯一正整數日")
        result.append(value)
    if not result:
        raise HorizonSuiteError("backtrack_days 不得為空")
    if len(set(result)) != len(result):
        raise HorizonSuiteError("backtrack_days 不得重複")
    return tuple(sorted(result))


def _bed_residence_settings(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """取得 template/common 中的 bed-residence mapping；缺欄位代表 legacy suite。"""

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, Mapping):
        return None
    bed = scenarios.get("bed_residence_time")
    if bed is None:
        return None
    if not isinstance(bed, Mapping):
        raise HorizonSuiteError("scenarios.bed_residence_time 必須是 mapping")
    return bed


def _suite_support_contract(
    payload: Mapping[str, Any], maximum_horizon: int
) -> tuple[int, int, tuple[str, ...]]:
    """計算共同選時包絡、沉底後支援與本 suite 必須發布的模式集合。

    legacy template 的 inputs support 等於最大 horizon，且每個 horizon 只有一份 release。
    bed-residence template 則把最大隨機沉底年齡加在最長回溯期之前，因此一次 build
    同時覆蓋兩個模式；runtime support 仍等於最長 release horizon，並固定發布兩種具名模式。
    """

    bed = _bed_residence_settings(payload)
    if bed is None:
        return maximum_horizon, maximum_horizon, ()
    maximum_age = bed.get("maximum_age_days")
    if type(maximum_age) is not int or maximum_age <= 0:
        raise HorizonSuiteError("bed residence maximum_age_days 必須是正整數")
    modes = bed.get("supported_backtrack_modes")
    if not isinstance(modes, list) or tuple(modes) != _BED_RESIDENCE_MODES:
        raise HorizonSuiteError("bed residence supported_backtrack_modes 必須登錄兩種固定模式且順序一致")
    if bed.get("backtrack_mode") not in _BED_RESIDENCE_MODES:
        raise HorizonSuiteError("bed residence template backtrack_mode 未登錄")
    return maximum_horizon + maximum_age, maximum_horizon, _BED_RESIDENCE_MODES


def maximum_step_count_for_horizon(days: Any, dt_min_seconds: Any) -> int:
    """依最小時間步長計算 ``ceil(days*86400/dt_min_seconds)+1``。

    ``days`` 是正整數日，``dt_min_seconds`` 是 integration contract 的最小秒數。
    多出的節點涵蓋回溯邊界的部分步；函式不讀 forcing，也不替代 runtime 的資料
    缺口與粒子停止狀態判定。
    """

    if type(days) is not int or days <= 0:
        raise HorizonSuiteError("horizon days 必須是正整數")
    if isinstance(dt_min_seconds, bool) or not isinstance(dt_min_seconds, (int, float)):
        raise HorizonSuiteError("integration.dt_min_seconds 必須是有限正數")
    dt_min = float(dt_min_seconds)
    if not math.isfinite(dt_min) or dt_min <= 0.0:
        raise HorizonSuiteError("integration.dt_min_seconds 必須是有限正數")
    required = math.ceil(days * 86_400.0 / dt_min) + 1
    if required > (1 << 63) - 1:
        raise HorizonSuiteError("horizon step count 超出可安全表示範圍")
    return int(required)


def _canonical_payload(payload: Mapping[str, Any]) -> bytes:
    """以固定 JSON 排序規則計算 YAML 語意 fingerprint。"""

    try:
        return json.dumps(
            dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HorizonSuiteError(f"YAML mapping 無法 canonicalize：{type(exc).__name__}") from exc


def _yaml_fingerprint_from_snapshot(
    raw: bytes, payload: Mapping[str, Any], relative_path: str
) -> dict[str, Any]:
    """由同一次讀取的 bytes 與 mapping 計算 YAML fingerprint。

    source template、common config 與 release YAML 都可能位於共享檔案系統；若先讀
    bytes 再重新開檔解析，兩次讀取之間可能被其他程序替換，造成保存的 raw hash、
    canonical payload 與實際送進 builder 的設定彼此不一致。因此 fingerprint 必須直接
    由同一個 snapshot 產生。
    """

    return {
        "path": relative_path,
        "size_bytes": len(raw),
        "sha256": sha256(raw).hexdigest(),
        "canonical_sha256": sha256(_canonical_payload(payload)).hexdigest(),
    }


def _read_yaml_snapshot(
    path: Path, label: str, relative_path: str
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    """一次讀取 YAML bytes，同時解析 mapping 與產生 raw/canonical fingerprints。

    回傳順序為 ``(payload, fingerprint, raw_bytes)``。呼叫端應重用這三項，不要再
    對同一路徑呼叫 ``read_text`` 或 ``_yaml_fingerprint``；此約束是 suite provenance
    的一部分，也讓 source snapshot 能證明 builder 使用的確切 bytes。
    """

    source = Path(path)
    # 先以 parent dirfd 固定檔案所在的目錄，再用 O_NOFOLLOW 開啟 leaf；這比
    # ``Path.read_bytes`` 安全，因為檔案開啟後即使名稱被替換，後續 bytes 仍來自同一
    # 個 descriptor。fstat 也會確認 descriptor 真的是普通檔案，避免把 directory、
    # FIFO 或 symbolic link 當成設定內容。
    try:
        _assert_no_symlink_components(source.parent, allow_missing_leaf=False)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(source.parent, directory_flags)
        descriptor: int | None = None
        try:
            descriptor = os.open(
                source.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            file_status = os.fstat(descriptor)
            if stat.S_ISLNK(file_status.st_mode) or not stat.S_ISREG(file_status.st_mode):
                raise HorizonSuiteError(f"{label} 必須是普通檔案")
            entry_status = os.stat(
                source.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (entry_status.st_dev, entry_status.st_ino) != (
                file_status.st_dev,
                file_status.st_ino,
            ):
                raise HorizonSuiteError(f"{label} 檔案 identity 已改變")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            raw = b"".join(chunks)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)
    except HorizonSuiteError:
        raise
    except Exception as exc:
        raise HorizonSuiteError(f"{label} YAML 無法讀取：{type(exc).__name__}") from exc
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except Exception as exc:
        raise HorizonSuiteError(f"{label} YAML 無法讀取：{type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise HorizonSuiteError(f"{label} YAML root 必須是 mapping")
    return payload, _yaml_fingerprint_from_snapshot(raw, payload, relative_path), raw


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    """讀取普通 YAML mapping，錯誤訊息不回傳檔案內容。

    這是給不需要 fingerprint 的內部小工具保留的簡化入口；需要 provenance 的
    builder／validator 會直接使用 ``_read_yaml_snapshot``，避免同一檔案讀兩次。
    """

    payload, _, _ = _read_yaml_snapshot(path, label, path.name)
    return payload


def _yaml_fingerprint(path: Path, relative_path: str) -> dict[str, Any]:
    """讀取一次 YAML 並回傳實際 bytes 與語意 fingerprint。"""

    _, fingerprint, _ = _read_yaml_snapshot(path, relative_path, relative_path)
    return fingerprint


def _fingerprint_matches(
    recorded: object, actual: Mapping[str, Any]
) -> bool:
    """要求 provenance fingerprint 的欄位集合與實際值完全相同。"""

    return isinstance(recorded, Mapping) and dict(recorded) == dict(actual)


def _config_hash_from_payload(payload: Mapping[str, Any]) -> str:
    """由已解析的 common YAML snapshot 計算與 ``load_config`` 相同的 config hash。

    ``load_config(path)`` 的唯一語意是 YAML 解析後呼叫 ``ProjectConfig.model_validate``
    與 ``config_hash``；validator 已經持有同一次 read 的 mapping，因此直接重現這段
    純函式流程可避免為了核對 artifact index 而再次讀共享檔案。deep copy 也避免
    Pydantic 的 normalized payload 或後續 caller 修改 snapshot。
    """

    try:
        return ProjectConfig.model_validate(copy.deepcopy(dict(payload))).config_hash()
    except Exception as exc:
        raise HorizonSuiteError(f"common config hash 無法重建：{type(exc).__name__}") from exc


def _assert_unbound_template(payload: Mapping[str, Any]) -> None:
    """拒絕舊 release binding 與非預期衍生路徑。

    example config 為了讓下游 loader 看見欄位，保留一個固定的初始條件 placeholder；
    它不是已發布 artifact，且既有 ``create_release_config`` 會在寫 release 時精確
    改成 common-input 路徑，因此只允許這一個字串。其他非空衍生 reference 一律拒絕。
    """

    for field in ("release_binding", "release_approval", "pilot_execution_binding"):
        if payload.get(field) is not None:
            raise HorizonSuiteError(f"template 已有 {field}，必須使用未綁定 template")
    for section, field in _DERIVED_FIELDS:
        section_payload = payload.get(section)
        if not isinstance(section_payload, Mapping):
            continue
        value = section_payload.get(field)
        if (
            section == "scenarios"
            and field == "receptor_arrival_initial_condition_manifest"
            and value == _ALLOWED_INITIAL_CONDITION_PLACEHOLDER
        ):
            continue
        if value is not None:
            raise HorizonSuiteError(f"template 已有非預期衍生 manifest：{section}.{field}")
    physics = payload.get("physics")
    settling = physics.get("settling") if isinstance(physics, Mapping) else None
    if isinstance(settling, Mapping) and settling.get("material_manifest") is not None:
        raise HorizonSuiteError("template 已有非預期衍生 manifest：physics.settling.material_manifest")


def _read_dt_min_and_validate_common(payload: Mapping[str, Any], maximum_horizon: int) -> float:
    """驗證 selection/runtime support contract 與最大步數，並回傳有效最小步長。"""

    inputs = payload.get("inputs")
    support = inputs.get("backtrack_support_days") if isinstance(inputs, Mapping) else None
    expected_selection, expected_runtime, _ = _suite_support_contract(payload, maximum_horizon)
    if type(support) is not int or support != expected_selection:
        raise HorizonSuiteError(
            "inputs.backtrack_support_days 必須等於最大 runtime horizon 加最大 bed age"
        )
    integration = payload.get("integration")
    dt_min = integration.get("dt_min_seconds") if isinstance(integration, Mapping) else None
    if isinstance(dt_min, bool) or not isinstance(dt_min, (int, float)):
        raise HorizonSuiteError("integration.dt_min_seconds 必須是有限正數")
    dt_min_float = float(dt_min)
    if not math.isfinite(dt_min_float) or dt_min_float <= 0.0:
        raise HorizonSuiteError("integration.dt_min_seconds 必須是有限正數")
    boundaries = payload.get("boundaries")
    if not isinstance(boundaries, Mapping):
        raise HorizonSuiteError("config.boundaries 必須是 mapping")
    if boundaries.get("max_backtrack_days") != float(expected_runtime):
        raise HorizonSuiteError("common boundaries.max_backtrack_days 必須等於 runtime support")
    bed = _bed_residence_settings(payload)
    if bed is not None and bed.get("runtime_horizon_support_days") != expected_runtime:
        raise HorizonSuiteError("bed runtime_horizon_support_days 必須等於最大 horizon")
    return dt_min_float


def _set_common_manifest_references(payload: dict[str, Any]) -> None:
    """將所有 builder／runtime manifest reference 指向 suite-relative common-input。

    這些欄位在 ``inputs-build`` 前就必須存在，否則 builder 會把 null 當成未建置而
    產生無法由 release loader 重建的設定。accepted forcing root 仍只以參數傳入，
    不寫入 config copy 或 suite manifest。
    """

    inputs = payload.get("inputs")
    scenarios = payload.get("scenarios")
    geometry = payload.get("geometry")
    physics = payload.get("physics")
    settling = physics.get("settling") if isinstance(physics, Mapping) else None
    if not all(isinstance(value, dict) for value in (inputs, scenarios, geometry, physics, settling)):
        raise HorizonSuiteError("template 的 manifest 所在區塊必須是 mapping")
    base = HORIZON_SUITE_COMMON_INPUT_DIRECTORY
    inputs["ocm_gap_safe_arrival_manifest"] = f"{base}/ocm_gap_safe_arrival.json"
    inputs["nww_full_hourly_analysis_manifest"] = f"{base}/nww_full_hourly.json"
    inputs["derived_input_artifact_index"] = f"{base}/artifact_index.json"
    scenarios["material_manifest"] = f"{base}/material.json"
    scenarios["receptor_manifest"] = f"{base}/receptor.json"
    scenarios["arrival_time_manifest"] = f"{base}/arrival.json"
    scenarios["receptor_arrival_initial_condition_manifest"] = f"{base}/initial_conditions.json"
    geometry["domain_manifest"] = f"{base}/domain.json"
    geometry["local_domain_manifest"] = f"{base}/local.json"
    geometry["open_boundary_manifest"] = f"{base}/open.json"
    geometry["receptor_manifest"] = f"{base}/receptor.json"
    settling["material_manifest"] = f"{base}/material.json"


def _derive_common_payload(
    source: Mapping[str, Any], maximum_horizon: int
) -> dict[str, Any]:
    """從同一個未綁定來源重建唯一允許的 common config。

    ``source`` 是 ``source-template.yaml`` 的 YAML mapping；函式先完整 deep copy，再
    只改四類已登錄欄位：共同支援日數、共同 requested horizon、最大 horizon 的安全
    step budget，以及所有固定的 suite-relative manifest reference。這個函式同時被
    builder 與 validator 使用，讓 validator 能以來源重新推導 expected mapping，拒絕
    任何額外的研究設定漂移。它不會讀取 forcing、raw NetCDF 或衍生 artifact，也不會
    以零值／最近值填補缺時；輸入來源必須仍是未綁定 template。
    """

    if type(maximum_horizon) is not int or maximum_horizon <= 0:
        raise HorizonSuiteError("common maximum horizon 必須是正整數")
    if not isinstance(source, Mapping):
        raise HorizonSuiteError("source template 必須是 mapping")
    _assert_unbound_template(source)
    payload = copy.deepcopy(dict(source))
    inputs = payload.get("inputs")
    boundaries = payload.get("boundaries")
    integration = payload.get("integration")
    if not isinstance(inputs, dict) or not isinstance(boundaries, dict):
        raise HorizonSuiteError("template.inputs／boundaries 必須是 mapping")
    dt_min = integration.get("dt_min_seconds") if isinstance(integration, Mapping) else None
    step_count = maximum_step_count_for_horizon(maximum_horizon, dt_min)
    selection_support, runtime_support, _ = _suite_support_contract(payload, maximum_horizon)
    inputs["backtrack_support_days"] = selection_support
    boundaries["max_backtrack_days"] = float(maximum_horizon)
    boundaries["maximum_step_count"] = step_count
    bed = _bed_residence_settings(payload)
    if bed is not None:
        scenarios = payload["scenarios"]
        bed_payload = scenarios["bed_residence_time"]
        bed_payload["runtime_horizon_support_days"] = runtime_support
    _set_common_manifest_references(payload)
    return payload


def _expected_suite_paths() -> dict[str, str]:
    """回傳不受 caller/manifest 影響的固定 suite path mapping。"""

    return dict(_SUITE_PATHS)


def _component_fingerprints(input_root: Path) -> dict[str, dict[str, Any]]:
    """重新讀取固定十個 component 與 artifact index 的 sidecar fingerprints。"""

    result: dict[str, dict[str, Any]] = {}
    for kind, filename in sorted(ARTIFACT_FILENAMES.items()):
        _, fingerprint = read_canonical_json(input_root / filename)
        result[kind] = {
            field: fingerprint[field] for field in ("sha256", "canonical_sha256", "size_bytes")
        }
    _, fingerprint = read_canonical_json(input_root / "artifact_index.json")
    result["artifact_index"] = {
        field: fingerprint[field] for field in ("sha256", "canonical_sha256", "size_bytes")
    }
    return result


def _read_artifact_index(input_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """讀取共同 input 的 artifact index，並保留其已驗證 fingerprint。"""

    payload, fingerprint = read_canonical_json(input_root / "artifact_index.json")
    if payload.get("manifest_kind") != "derived_input_artifact_index":
        raise HorizonSuiteError("artifact_index manifest_kind 不符")
    if payload.get("schema_version") != DERIVED_INPUT_SCHEMA_VERSION:
        raise HorizonSuiteError("artifact_index schema_version 不符")
    return payload, fingerprint


def _validate_artifact_closure(
    input_root: Path,
    component_fingerprints: Mapping[str, Mapping[str, Any]],
) -> None:
    """驗證 artifact_bindings closure 逐項指向固定 component 與 artifact index。"""

    payload, _ = read_canonical_json(input_root / "artifact_bindings.json")
    if payload.get("manifest_kind") != "derived_input_artifact_closure":
        raise HorizonSuiteError("artifact_bindings manifest_kind 不符")
    if payload.get("schema_version") != DERIVED_INPUT_SCHEMA_VERSION:
        raise HorizonSuiteError("artifact_bindings schema_version 不符")
    records = payload.get("artifacts")
    if not isinstance(records, list) or len(records) != len(ARTIFACT_FILENAMES) + 1:
        raise HorizonSuiteError("artifact_bindings artifact count 不符")
    seen: set[str] = set()
    expected = set(ARTIFACT_FILENAMES) | {"artifact_index"}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("kind"), str):
            raise HorizonSuiteError("artifact_bindings record 不符")
        kind = record["kind"]
        if kind not in expected or kind in seen:
            raise HorizonSuiteError("artifact_bindings kind 不符")
        seen.add(kind)
        actual = component_fingerprints[kind]
        if any(record.get(field) != actual.get(field) for field in actual):
            raise HorizonSuiteError(f"artifact_bindings fingerprint 不符：{kind}")
        expected_path = ARTIFACT_FILENAMES.get(kind, "artifact_index.json")
        if record.get("path") != expected_path:
            raise HorizonSuiteError(f"artifact_bindings path 不符：{kind}")
    if seen != expected:
        raise HorizonSuiteError("artifact_bindings kind set 不完整")


def _assert_exact_entries(
    directory: Path,
    *,
    expected_files: set[str],
    expected_directories: set[str],
    label: str,
) -> None:
    """驗證目錄只含契約列出的普通檔案與普通子目錄。

    manifest 內的路徑只是自述，不能用來放寬檔案系統拓撲；因此 validator 會列舉每
    一層目錄，拒絕額外的暫存檔、缺少的 sidecar、symbolic link 或意外巢狀目錄。這個
    gate 對 common-input 也適用，避免產物目錄中藏入未被 hash 覆蓋的資料。
    """

    directory = _assert_regular_directory(directory)
    expected = expected_files | expected_directories
    try:
        actual = {entry.name for entry in directory.iterdir()}
    except OSError as exc:
        raise HorizonSuiteError(f"{label} 目錄無法列舉：{type(exc).__name__}") from exc
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise HorizonSuiteError(f"{label} 拓撲不符：missing={missing},extra={extra}")
    for name in expected_files:
        _assert_regular_file(directory / name)
    for name in expected_directories:
        _assert_regular_directory(directory / name)


def _release_stem(days: int, mode: str | None) -> str:
    """建立含天數及可選模式的 release/validation 唯一名稱。"""

    return f"release-{days}d" if mode is None else f"release-{days}d-{mode}"


def _assert_suite_topology(
    suite_root: Path,
    horizons: Sequence[int],
    backtrack_modes: Sequence[str],
    *,
    publication_marker_required: bool = True,
) -> None:
    """驗證 suite 根、release、validation 與 common-input 的固定 closure。

    固定檔名是為了讓下游 CLI 能以 suite-relative 路徑重建，不允許 caller 透過
    manifest 新增任意 input 或 release。十個 component 以及 artifact index／closure
    的 sidecar 都必須存在，檔案內容的 hash 仍由後續 canonical reader 再驗證。
    公開 validator 要求發布完成 marker；只有 builder 在 marker 尚未寫入的 private
    partial gate 才能明示 ``publication_marker_required=False``，避免未完成目錄被
    外部 CLI 當成正式 suite。
    """

    root_files = {
        HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME,
        HORIZON_SUITE_COMMON_CONFIG_FILENAME,
        HORIZON_SUITE_MANIFEST_FILENAME,
        f"{HORIZON_SUITE_MANIFEST_FILENAME}.sha256",
    }
    if publication_marker_required:
        root_files.update(
            {
                HORIZON_SUITE_PUBLICATION_MARKER_FILENAME,
                f"{HORIZON_SUITE_PUBLICATION_MARKER_FILENAME}.sha256",
            }
        )
    root_directories = {
        HORIZON_SUITE_COMMON_INPUT_DIRECTORY,
        HORIZON_SUITE_RELEASE_DIRECTORY,
        HORIZON_SUITE_VALIDATION_DIRECTORY,
    }
    _assert_exact_entries(
        suite_root,
        expected_files=root_files,
        expected_directories=root_directories,
        label="suite root",
    )
    common_files = {
        filename
        for filename in ARTIFACT_FILENAMES.values()
    }
    common_files.update(_COMMON_INPUT_EXTRA_FILENAMES)
    common_files.update(f"{filename}.sha256" for filename in tuple(common_files))
    _assert_exact_entries(
        suite_root / HORIZON_SUITE_COMMON_INPUT_DIRECTORY,
        expected_files=common_files,
        expected_directories=set(),
        label="common-input",
    )
    mode_slots: tuple[str | None, ...] = tuple(backtrack_modes) if backtrack_modes else (None,)
    expected_release_files = {
        f"{_release_stem(days, mode)}.yaml" for days in horizons for mode in mode_slots
    }
    _assert_exact_entries(
        suite_root / HORIZON_SUITE_RELEASE_DIRECTORY,
        expected_files=expected_release_files,
        expected_directories=set(),
        label="release-configs",
    )
    expected_validation_files = {
        "input.json",
        "input.json.sha256",
    }
    expected_validation_files.update(
        f"{_release_stem(days, mode)}{suffix}"
        for days in horizons
        for mode in mode_slots
        for suffix in (".json", ".json.sha256")
    )
    _assert_exact_entries(
        suite_root / HORIZON_SUITE_VALIDATION_DIRECTORY,
        expected_files=expected_validation_files,
        expected_directories=set(),
        label="validations",
    )


def _identity_fingerprints(fingerprints: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """保留比較研究母體所需的 arrival／receptor／material／initial identity。"""

    return {
        kind: {
            field: fingerprints[kind][field]
            for field in ("sha256", "canonical_sha256", "size_bytes")
        }
        for kind in _IDENTITY_KINDS
    }


def _release_records(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """解析 release binding，並核對 component 與 observation 母體身分。

    artifact hash 能證明檔案未變，但不能單獨說明 2024–2025 forcing 中哪些年份可作
    observation anchor；因此 release 必須另存四欄母體快照，且與 release YAML 的
    arrival policy 精確一致。這項檢查同時適用新版與 legacy suite，避免產生只有部分
    release 帶有母體身分的混合拓撲。
    """

    binding = payload.get("release_binding")
    gap_contract = _payload_gap_censoring_contract(payload)
    expected_binding_keys = _RELEASE_BINDING_KEYS | (
        _GAP_CENSOR_RELEASE_BINDING_KEYS if gap_contract is not None else frozenset()
    )
    if not isinstance(binding, Mapping) or set(binding) != expected_binding_keys:
        raise HorizonSuiteError("release_binding 欄位集合不符")
    if gap_contract is not None and any(
        binding.get(field) != expected for field, expected in gap_contract.items()
    ):
        raise HorizonSuiteError("release_binding 缺時截尾 policy 不一致")
    arrival_binding = binding.get("arrival_selection_binding")
    if (
        not isinstance(arrival_binding, Mapping)
        or set(arrival_binding) != _RELEASE_ARRIVAL_SELECTION_BINDING_KEYS
    ):
        raise HorizonSuiteError("release_binding.arrival_selection_binding 欄位集合不符")
    population = _arrival_population_contract(payload)
    expected_arrival_binding = {
        "forcing_years": population["forcing_years"],
        "observation_years": population["observation_years"],
        "policy": population["arrival_selection_policy_id"],
        "replicates": population["replicates_per_stratum"],
    }
    if dict(arrival_binding) != expected_arrival_binding:
        raise HorizonSuiteError("release_binding arrival population 與 release config 不一致")
    records = binding.get("artifacts") if isinstance(binding, Mapping) else None
    if not isinstance(records, list):
        raise HorizonSuiteError("release_binding.artifacts 必須是 list")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise HorizonSuiteError("release artifact record 必須是 mapping")
        if set(record) != _RELEASE_ARTIFACT_RECORD_KEYS:
            raise HorizonSuiteError("release artifact record 欄位集合不符")
        kind = record.get("kind")
        if not isinstance(kind, str) or kind not in ARTIFACT_FILENAMES or kind in result:
            raise HorizonSuiteError(f"release artifact kind 不合法：{kind}")
        result[kind] = record
    if set(result) != set(ARTIFACT_FILENAMES):
        raise HorizonSuiteError("release binding artifact set 不完整")
    if [record.get("kind") for record in records] != sorted(ARTIFACT_FILENAMES):
        raise HorizonSuiteError("release binding artifact 順序不符")
    return result


def _set_release_manifest_references(payload: dict[str, Any]) -> None:
    """把 common config 的固定 reference 改成 release config 的相對路徑。

    ``create_release_config`` 以 release YAML 所在的 ``release-configs`` 目錄為基準，
    因此同一批 immutable artifact 在 release 中必須寫成 ``../common-input``。這個
    純記憶體轉換和 builder 使用的固定 artifact filename 對齊，供 validator 從
    ``common-config.yaml`` 重建 expected release payload；目前 design 且明示核准
    reconstruction policy 時，runtime 所需的 reconstruction manifest 也與
    schema／closure 使用的 gap-safe manifest exact 指向同一檔案。這裡只綁定
    common-input 證據，不把外部 reconstruction root index 寫入 release；不接受
    manifest 自述的任意路徑，也不會重讀 raw forcing。
    """

    inputs = payload.get("inputs")
    scenarios = payload.get("scenarios")
    geometry = payload.get("geometry")
    physics = payload.get("physics")
    settling = physics.get("settling") if isinstance(physics, Mapping) else None
    if not all(
        isinstance(value, dict)
        for value in (inputs, scenarios, geometry, physics, settling)
    ):
        raise HorizonSuiteError("release manifest 所在 config 區塊必須是 mapping")
    base = f"../{HORIZON_SUITE_COMMON_INPUT_DIRECTORY}"
    inputs["ocm_gap_safe_arrival_manifest"] = (
        f"{base}/{ARTIFACT_FILENAMES['ocm_gap_safe_arrival_horizon']}"
    )
    if _approved_reconstruction_release_binding_enabled(payload):
        # reconstruction 與 gap-safe 欄位故意共用 immutable common-input 支援證據；
        # 外部 OCM_RECONSTRUCTION_ROOT 只在 runtime／preflight 解析，不進 release YAML。
        inputs["ocm_gap_reconstruction_manifest"] = (
            f"{base}/{ARTIFACT_FILENAMES['ocm_gap_safe_arrival_horizon']}"
        )
    inputs["nww_full_hourly_analysis_manifest"] = (
        f"{base}/{ARTIFACT_FILENAMES['nww_full_hourly']}"
    )
    inputs["derived_input_artifact_index"] = f"{base}/artifact_index.json"
    scenarios["material_manifest"] = f"{base}/{ARTIFACT_FILENAMES['material']}"
    scenarios["receptor_manifest"] = f"{base}/{ARTIFACT_FILENAMES['receptor']}"
    scenarios["arrival_time_manifest"] = f"{base}/{ARTIFACT_FILENAMES['arrival']}"
    scenarios["receptor_arrival_initial_condition_manifest"] = (
        f"{base}/{ARTIFACT_FILENAMES['initial_condition']}"
    )
    geometry["domain_manifest"] = f"{base}/{ARTIFACT_FILENAMES['domain_geometry']}"
    geometry["local_domain_manifest"] = f"{base}/{ARTIFACT_FILENAMES['local_geometry']}"
    geometry["open_boundary_manifest"] = f"{base}/{ARTIFACT_FILENAMES['open_boundary']}"
    geometry["receptor_manifest"] = f"{base}/{ARTIFACT_FILENAMES['receptor']}"
    settling["material_manifest"] = f"{base}/{ARTIFACT_FILENAMES['material']}"


def _release_payload_without_registered_fields(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """移除 create_release_config 登錄的 release metadata，保留其餘 config 供 exact 比較。"""

    result = copy.deepcopy(dict(payload))
    for field in (
        "config_status",
        "release_binding",
        "release_approval",
        "pilot_execution_binding",
    ):
        result.pop(field, None)
    return result


def _derive_expected_release_payload(
    common_payload: Mapping[str, Any],
    *,
    days: int,
    maximum_step_count: int,
    inventory_payload: Mapping[str, Any] | None = None,
    backtrack_mode: str | None = None,
    formal: bool = False,
) -> dict[str, Any]:
    """從 common payload 建立單一 release 應有的完整設定 mapping。

    release 只允許四類差異：requested horizon、其 safe step budget、已登錄的 bed-residence
    mode，以及
    ``create_release_config`` 登錄的 status／binding／approval metadata；artifact path
    因 YAML 位置改變而由固定 helper 轉成 ``../common-input``。若 forcing inventory
    含有四區三產品的完整 flow-domain ID，會重用既有 binding 純邏輯預先重建正式欄位；
    無法解析的 synthetic inventory 不會在這裡猜測 ID，而會由 input closure／下游
    validator 報錯。這樣 execution、physics、scenario 與 geometry 等未登錄欄位的
    任意漂移都會被 exact mapping 比較攔下。
    """

    if type(days) is not int or days <= 0:
        raise HorizonSuiteError("release horizon 必須是正整數")
    if type(maximum_step_count) is not int or maximum_step_count <= 0:
        raise HorizonSuiteError("release maximum_step_count 必須是正整數")
    expected = copy.deepcopy(dict(common_payload))
    boundaries = expected.get("boundaries")
    if not isinstance(boundaries, dict):
        raise HorizonSuiteError("common config boundaries 必須是 mapping")
    boundaries["max_backtrack_days"] = float(days)
    boundaries["maximum_step_count"] = maximum_step_count
    bed = _bed_residence_settings(expected)
    if bed is None:
        if backtrack_mode is not None:
            raise HorizonSuiteError("legacy suite 不允許 backtrack_mode override")
    else:
        if backtrack_mode not in _BED_RESIDENCE_MODES:
            raise HorizonSuiteError("bed-residence release 必須明示合法 backtrack_mode")
        bed_payload = expected["scenarios"]["bed_residence_time"]
        if not isinstance(bed_payload, dict):
            raise HorizonSuiteError("bed residence release config 必須是 mapping")
        bed_payload["backtrack_mode"] = backtrack_mode
    _set_release_manifest_references(expected)
    if formal and expected.get("design_version") == CURRENT_DESIGN_VERSION:
        forcing = expected.get("forcing")
        ocm = forcing.get("ocm") if isinstance(forcing, Mapping) else None
        if isinstance(ocm, dict) and ocm.get(
            "wetdry_semantics_decision_status"
        ) == "derived_pending_server_preflight":
            # formal create_release_config 只有在既有 validator 已確認 dynamic
            # initial-condition 全部為 wet 時才會寫入 approved；expected payload 也要
            # 反映同一個 deterministic promotion，否則 suite validator 會把合法
            # release 誤判為 common-config 漂移。pilot／legacy 不套用此轉換。
            ocm["wetdry_semantics_decision_status"] = "approved"

    # 真實 inventory 的 flow-domain 綁定可能需要同步 release 的 formal 欄位。採用
    # 既有 input_derivation 純函式會增加對 synthetic double 的依賴，因此以局部 import
    # 並只在 inventory 結構完整時套用；不完整 inventory 本身已在 closure gate 失敗。
    if isinstance(inventory_payload, Mapping) and isinstance(
        inventory_payload.get("products"), list
    ):
        try:
            from .input_derivation import _bind_inventory_flow_domains, _flow_domain_ids_from_inventory

            flow_ids = _flow_domain_ids_from_inventory(inventory_payload)
            _bind_inventory_flow_domains(expected, inventory_flow_ids=flow_ids)
        except Exception:
            # 不把缺少 synthetic inventory 欄位的測試 double 轉成錯誤的「猜測」；
            # 真實流程仍會由 artifact/index 與下游 validator 對完整 inventory fail closed。
            pass
    return expected


def _directory_identity(path: Path) -> tuple[int, int]:
    """取得普通目錄的 device／inode identity，供競態檢查使用。"""

    status = os.lstat(path)
    if stat.S_ISLNK(status.st_mode):
        raise HorizonSuiteError("path 不得是 symbolic link")
    if not stat.S_ISDIR(status.st_mode):
        raise HorizonSuiteError("path 必須是普通目錄")
    return status.st_dev, status.st_ino


def _prepare_partial(destination: Path) -> tuple[Path, tuple[int, int], tuple[int, int]]:
    """建立自有 hidden partial，並保存 parent／partial 的 identity。

    parent 與 partial 的 device／inode 會在發布前再次核對。若共享檔案系統上的其他
    程序替換了任一目錄，失敗路徑會保留 partial 供人工稽核；不自動追蹤或刪除競態
    中出現的對方資料。
    """

    _assert_no_symlink_components(destination.parent, allow_missing_leaf=True)
    if destination.exists() or destination.is_symlink():
        # 不把可能包含 SERVER 絕對路徑的 destination 放進例外文字；操作者可依
        # runbook 以自身指定的輸出根目錄定位既有成果，錯誤本身只需表達拒絕覆寫。
        raise FileExistsError("不可覆寫既有 horizon suite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(destination.parent, allow_missing_leaf=False)
    parent_identity = _directory_identity(destination.parent)
    partial = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    return partial, parent_identity, _directory_identity(partial)


def _assert_owned_partial(partial: Path, partial_identity: tuple[int, int]) -> None:
    """確認 partial 仍是本次建立的非 symlink 普通目錄。"""

    actual_identity = _directory_identity(partial)
    if actual_identity != partial_identity:
        raise HorizonSuiteError("partial identity 已改變")


def _directory_open_flags() -> int:
    """回傳 no-follow directory descriptor 所需旗標；不支援時 fail closed。"""

    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise HorizonSuiteError("平台缺少安全 directory descriptor 旗標")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _lock_directory(descriptor: int) -> None:
    """以 advisory lock 序列化本流程在同一 parent 下的 publish／cleanup。"""

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except (OSError, AttributeError) as exc:
        raise HorizonSuiteError("無法取得 suite parent lock") from exc


def _unlock_directory(descriptor: int) -> None:
    """釋放 publish／cleanup 使用的 parent advisory lock。"""

    with contextlib.suppress(OSError, AttributeError):
        fcntl.flock(descriptor, fcntl.LOCK_UN)


def _cleanup_partial(
    partial: Path,
    partial_identity: tuple[int, int],
    parent_identity: tuple[int, int] | None = None,
) -> None:
    """保留失敗 partial，完全不自動遞迴刪除任何檔案或目錄。

    partial 內含設定、input artifact、release 與 validator evidence；在共享 SERVER
    上，即使先核對 inode，後續 ``stat`` 到 ``unlink`` 之間仍可能被同帳號程序交換
    basename。若自動清理，無法證明每一個刪除的檔案仍屬於本次建置，因此本流程採
    fail-safe 政策：失敗時保留 ``.partial-*`` 供人工稽核與明確清除，且不追蹤或刪除
    任一 replacement／foreign sentinel。參數保留是為維持 builder 的錯誤處理介面；
    此函式刻意不讀取 path、不開啟 descriptor，也不回傳敏感絕對路徑。
    """

    del partial, partial_identity, parent_identity


def _fsync_directory(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    """以不追 symlink 的 directory descriptor 同步目錄項目。

    partial 的固定子目錄在 exclusive rename 前同步，確保 YAML、JSON 與 sidecar 的
    directory entry 已送至檔案系統；final rename 成功後 caller 會再同步 parent。若
    平台或檔案系統不支援 directory fsync，直接回傳例外而不宣稱 durability 已確認。
    """

    directory = Path(path)
    _assert_no_symlink_components(directory, allow_missing_leaf=False)
    descriptor = os.open(directory, _directory_open_flags())
    try:
        opened = os.fstat(descriptor)
        opened_identity = (opened.st_dev, opened.st_ino)
        if not stat.S_ISDIR(opened.st_mode) or (
            expected_identity is not None and opened_identity != expected_identity
        ):
            raise HorizonSuiteError("fsync target descriptor 非普通目錄")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if expected_identity is not None and _directory_identity(directory) != expected_identity:
        raise HorizonSuiteError("fsync 後 parent identity 已改變")


def _publication_fingerprint(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """擷取 marker 可公開保存的 manifest fingerprint，不帶入絕對路徑。"""

    fields = ("sha256", "canonical_sha256", "size_bytes")
    if any(field not in fingerprint for field in fields):
        raise HorizonSuiteError("manifest fingerprint 欄位不完整")
    return {field: fingerprint[field] for field in fields}


def _publication_marker_payload(
    manifest_fingerprint: Mapping[str, Any], publication_method: str
) -> dict[str, Any]:
    """組裝固定欄位的 suite publication marker。"""

    if publication_method not in _ALLOWED_PUBLICATION_METHODS:
        raise HorizonSuiteError("未知的 suite publication method")
    return {
        "marker_kind": "horizon_suite_publication_commit",
        "schema_version": HORIZON_SUITE_SCHEMA_VERSION,
        "publication_policy_id": HORIZON_SUITE_PUBLICATION_POLICY_ID,
        "publication_policy_version": HORIZON_SUITE_PUBLICATION_POLICY_VERSION,
        "publication_method": publication_method,
        "manifest_fingerprint": _publication_fingerprint(manifest_fingerprint),
    }


def _write_no_replace_at(directory_descriptor: int, name: str, data: bytes) -> None:
    """以 directory fd 寫入不可覆寫的普通檔案並同步檔案 descriptor。

    NFS fallback 的 destination 是先保留、後複製的空目錄；因此 marker 與 sidecar
    不能沿用 path-based ``os.replace``。這裡以 ``O_EXCL``／``O_NOFOLLOW`` 把「檔名
    尚不存在」與建立檔案合併，任何既有 node、symbolic link 或中途寫入失敗都保留
    現場並 fail closed。
    """

    if not hasattr(os, "O_NOFOLLOW"):
        raise HorizonSuiteError("平台缺少安全 no-follow file flag")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, 0o644, dir_fd=directory_descriptor)
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("publication marker 寫入沒有前進")
            offset += written
        os.fsync(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_publication_marker(
    destination: Path,
    publication_method: str,
    parent_identity: tuple[int, int],
    destination_identity: tuple[int, int],
) -> None:
    """在已固定的 destination inode 上原子建立 publication marker 與 sidecar。

    marker 是正式 validator 的完成提交點：manifest 已先寫入且通過 private partial
    gate，這裡再以 parent／destination dirfd 核對 inode，建立不可覆寫的 marker 與
    SHA-256 sidecar，並同步 destination／parent 目錄。payload 僅保存 bytes fingerprint
    與發布策略，不保存任何 destination、partial 或 forcing root 絕對路徑。
    """

    if publication_method not in _ALLOWED_PUBLICATION_METHODS:
        raise HorizonSuiteError("未知的 suite publication method")
    _assert_no_symlink_components(destination, allow_missing_leaf=False)
    manifest_payload, manifest_fingerprint = read_canonical_json(
        destination / HORIZON_SUITE_MANIFEST_FILENAME
    )
    if manifest_payload.get("schema_version") != HORIZON_SUITE_SCHEMA_VERSION:
        raise HorizonSuiteError("publication marker 的 manifest schema 不符")
    marker_payload = _publication_marker_payload(manifest_fingerprint, publication_method)
    marker_bytes = (
        json.dumps(
            marker_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    marker_binding = {
        "algorithm": "sha256",
        "canonical_sha256": sha256(canonical_json_bytes(marker_payload)).hexdigest(),
        "sha256": sha256(marker_bytes).hexdigest(),
        "size_bytes": len(marker_bytes),
    }
    binding_bytes = (
        json.dumps(
            marker_binding,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    parent_descriptor: int | None = None
    destination_descriptor: int | None = None
    lock_acquired = False
    try:
        parent_descriptor = os.open(destination.parent, _directory_open_flags())
        parent_status = os.fstat(parent_descriptor)
        actual_parent = (parent_status.st_dev, parent_status.st_ino)
        if not stat.S_ISDIR(parent_status.st_mode) or actual_parent != parent_identity:
            raise HorizonSuiteError("publication marker parent identity 已改變")
        _lock_directory(parent_descriptor)
        lock_acquired = True
        destination_descriptor = os.open(
            destination.name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        destination_status = os.fstat(destination_descriptor)
        actual_destination = (destination_status.st_dev, destination_status.st_ino)
        if (
            not stat.S_ISDIR(destination_status.st_mode)
            or actual_destination != destination_identity
        ):
            raise HorizonSuiteError("publication marker destination identity 已改變")
        _write_no_replace_at(
            destination_descriptor,
            HORIZON_SUITE_PUBLICATION_MARKER_FILENAME,
            marker_bytes,
        )
        _write_no_replace_at(
            destination_descriptor,
            f"{HORIZON_SUITE_PUBLICATION_MARKER_FILENAME}.sha256",
            binding_bytes,
        )
        os.fsync(destination_descriptor)
        final_destination = os.fstat(destination_descriptor)
        if (final_destination.st_dev, final_destination.st_ino) != destination_identity:
            raise HorizonSuiteError("publication marker 後 destination identity 已改變")
        os.fsync(parent_descriptor)
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if lock_acquired and parent_descriptor is not None:
            _unlock_directory(parent_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _validate_publication_marker(
    suite_root: Path, manifest_fingerprint: Mapping[str, Any]
) -> list[str]:
    """驗證正式 suite 的 immutable publication marker 與 manifest fingerprint。"""

    errors: list[str] = []
    marker_path = suite_root / HORIZON_SUITE_PUBLICATION_MARKER_FILENAME
    try:
        marker, _ = read_canonical_json(marker_path)
    except Exception as exc:
        return [f"publication_marker_invalid:{type(exc).__name__}"]
    expected_keys = {
        "marker_kind",
        "schema_version",
        "publication_policy_id",
        "publication_policy_version",
        "publication_method",
        "manifest_fingerprint",
    }
    if set(marker) != expected_keys:
        errors.append("publication_marker_key_set_invalid")
    if marker.get("marker_kind") != "horizon_suite_publication_commit":
        errors.append("publication_marker_kind_invalid")
    if marker.get("schema_version") != HORIZON_SUITE_SCHEMA_VERSION:
        errors.append("publication_marker_schema_version_invalid")
    if marker.get("publication_policy_id") != HORIZON_SUITE_PUBLICATION_POLICY_ID:
        errors.append("publication_marker_policy_invalid")
    if marker.get("publication_policy_version") != HORIZON_SUITE_PUBLICATION_POLICY_VERSION:
        errors.append("publication_marker_policy_version_invalid")
    if marker.get("publication_method") not in _ALLOWED_PUBLICATION_METHODS:
        errors.append("publication_marker_method_invalid")
    try:
        if marker.get("manifest_fingerprint") != _publication_fingerprint(manifest_fingerprint):
            errors.append("publication_marker_manifest_fingerprint_mismatch")
    except Exception:
        errors.append("publication_marker_manifest_fingerprint_invalid")
    return errors


def _suite_copy_layout(partial: Path) -> dict[str, tuple[set[str], set[str]]]:
    """從 partial manifest 推導 NFS fallback 唯一允許複製的檔案拓撲。"""

    manifest, _ = read_canonical_json(partial / HORIZON_SUITE_MANIFEST_FILENAME)
    horizons = normalize_horizons(manifest.get("horizons_days"))
    raw_modes = manifest.get("backtrack_modes")
    if not isinstance(raw_modes, list) or any(not isinstance(mode, str) for mode in raw_modes):
        raise HorizonSuiteError("partial manifest backtrack_modes 不合法")
    backtrack_modes = tuple(raw_modes)
    _assert_suite_topology(
        partial,
        horizons,
        backtrack_modes,
        publication_marker_required=False,
    )
    common_files = set(ARTIFACT_FILENAMES.values()) | _COMMON_INPUT_EXTRA_FILENAMES
    common_files.update(f"{filename}.sha256" for filename in tuple(common_files))
    mode_slots: tuple[str | None, ...] = backtrack_modes if backtrack_modes else (None,)
    release_files = {
        f"{_release_stem(days, mode)}.yaml" for days in horizons for mode in mode_slots
    }
    validation_files = {"input.json", "input.json.sha256"}
    validation_files.update(
        f"{_release_stem(days, mode)}{suffix}"
        for days in horizons
        for mode in mode_slots
        for suffix in (".json", ".json.sha256")
    )
    return {
        "": (
            {
                HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME,
                HORIZON_SUITE_COMMON_CONFIG_FILENAME,
                HORIZON_SUITE_MANIFEST_FILENAME,
                f"{HORIZON_SUITE_MANIFEST_FILENAME}.sha256",
            },
            {
                HORIZON_SUITE_COMMON_INPUT_DIRECTORY,
                HORIZON_SUITE_RELEASE_DIRECTORY,
                HORIZON_SUITE_VALIDATION_DIRECTORY,
            },
        ),
        HORIZON_SUITE_COMMON_INPUT_DIRECTORY: (common_files, set()),
        HORIZON_SUITE_RELEASE_DIRECTORY: (release_files, set()),
        HORIZON_SUITE_VALIDATION_DIRECTORY: (validation_files, set()),
    }


def _assert_nfs_source_immutable(partial: Path) -> None:
    """在 NFS 保留 destination 前重驗 partial 的 manifest／sidecar closure。

    native rename 只搬移已通過 private validator 的目錄；NFS fallback 會逐檔複製，
    因此必須再確認 source 沒有在兩個 gate 之間被竄改。此檢查不讀 forcing root，只
    比對 suite 已保存的 YAML、canonical JSON、component 與 validation fingerprints。
    """

    manifest, _ = read_canonical_json(partial / HORIZON_SUITE_MANIFEST_FILENAME)
    source_payload, source_fingerprint, _ = _read_yaml_snapshot(
        partial / HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME,
        "NFS fallback source template",
        HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME,
    )
    _assert_unbound_template(source_payload)
    if not _fingerprint_matches(manifest.get("source_template_fingerprint"), source_fingerprint):
        raise HorizonSuiteError("NFS fallback source template fingerprint 已改變")
    common_payload, common_fingerprint, _ = _read_yaml_snapshot(
        partial / HORIZON_SUITE_COMMON_CONFIG_FILENAME,
        "NFS fallback common config",
        HORIZON_SUITE_COMMON_CONFIG_FILENAME,
    )
    if not _fingerprint_matches(manifest.get("common_config_fingerprint"), common_fingerprint):
        raise HorizonSuiteError("NFS fallback common config fingerprint 已改變")
    input_root = partial / HORIZON_SUITE_COMMON_INPUT_DIRECTORY
    artifact_errors, _, _ = _validate_artifact_directory(input_root)
    if artifact_errors:
        raise HorizonSuiteError("NFS fallback common-input closure 已改變")
    component_fingerprints = _component_fingerprints(input_root)
    _validate_artifact_closure(input_root, component_fingerprints)
    if manifest.get("common_artifact_index_fingerprint") != component_fingerprints["artifact_index"]:
        raise HorizonSuiteError("NFS fallback artifact index fingerprint 已改變")
    recorded_components = manifest.get("common_artifact_fingerprints")
    if recorded_components != {
        kind: component_fingerprints[kind] for kind in sorted(ARTIFACT_FILENAMES)
    }:
        raise HorizonSuiteError("NFS fallback component fingerprint 已改變")
    validation_root = partial / HORIZON_SUITE_VALIDATION_DIRECTORY
    input_validation, input_validation_fingerprint = read_canonical_json(
        validation_root / "input.json"
    )
    if input_validation.get("valid") is not True or not _fingerprint_matches(
        manifest.get("input_validation_fingerprint"), input_validation_fingerprint
    ):
        raise HorizonSuiteError("NFS fallback input validation evidence 已改變")
    releases = manifest.get("releases")
    if not isinstance(releases, list):
        raise HorizonSuiteError("NFS fallback release manifest 不合法")
    horizons = normalize_horizons(manifest.get("horizons_days"))
    raw_modes = manifest.get("backtrack_modes")
    if not isinstance(raw_modes, list) or any(not isinstance(mode, str) for mode in raw_modes):
        raise HorizonSuiteError("NFS fallback manifest modes 不合法")
    mode_slots: tuple[str | None, ...] = tuple(raw_modes) if raw_modes else (None,)
    expected_release_paths = {
        (
            f"{HORIZON_SUITE_RELEASE_DIRECTORY}/{_release_stem(days, mode)}.yaml",
            f"{HORIZON_SUITE_VALIDATION_DIRECTORY}/{_release_stem(days, mode)}.json",
        )
        for days in horizons
        for mode in mode_slots
    }
    for record in releases:
        if not isinstance(record, Mapping):
            raise HorizonSuiteError("NFS fallback release record 不合法")
        release_relative = record.get("config_path")
        validation_relative = record.get("validation_path")
        if not isinstance(release_relative, str) or not isinstance(validation_relative, str):
            raise HorizonSuiteError("NFS fallback release path 不合法")
        if (release_relative, validation_relative) not in expected_release_paths:
            raise HorizonSuiteError("NFS fallback release path 不在固定白名單")
        _, release_fingerprint, _ = _read_yaml_snapshot(
            partial / release_relative,
            "NFS fallback release",
            release_relative,
        )
        if not _fingerprint_matches(record.get("config_fingerprint"), release_fingerprint):
            raise HorizonSuiteError("NFS fallback release fingerprint 已改變")
        _, validation_fingerprint = read_canonical_json(partial / validation_relative)
        if not _fingerprint_matches(
            record.get("validation_fingerprint"), validation_fingerprint
        ):
            raise HorizonSuiteError("NFS fallback release validation fingerprint 已改變")
    del common_payload


def _copy_file_no_follow(
    source_directory: int, destination_directory: int, name: str
) -> None:
    """以兩個 dirfd 複製單一普通檔案，拒絕 symlink、覆寫與內容競態。"""

    if not hasattr(os, "O_NOFOLLOW"):
        raise HorizonSuiteError("平台缺少安全 no-follow file flag")
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    try:
        source_descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=source_directory,
        )
        source_initial = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_initial.st_mode):
            raise HorizonSuiteError("NFS fallback 只允許複製普通檔案")
        destination_descriptor = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            stat.S_IMODE(source_initial.st_mode) & 0o777,
            dir_fd=destination_directory,
        )
        digest = sha256()
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            offset = 0
            while offset < len(chunk):
                written = os.write(destination_descriptor, chunk[offset:])
                if written <= 0:
                    raise OSError("NFS fallback file copy 沒有前進")
                offset += written
        os.fsync(destination_descriptor)
        source_after = os.fstat(source_descriptor)
        if (
            (source_after.st_dev, source_after.st_ino) != (source_initial.st_dev, source_initial.st_ino)
            or source_after.st_size != source_initial.st_size
            or source_after.st_mtime_ns != source_initial.st_mtime_ns
            or source_after.st_ctime_ns != source_initial.st_ctime_ns
        ):
            raise HorizonSuiteError("NFS fallback source file 在複製期間改變")
        # 再讀一次 source，確認即使 metadata 沒有變化也沒有內容競態。
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        second_digest = sha256()
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            second_digest.update(chunk)
        if second_digest.digest() != digest.digest():
            raise HorizonSuiteError("NFS fallback source file checksum 在複製期間改變")
        os.lseek(destination_descriptor, 0, os.SEEK_SET)
        destination_digest = sha256()
        while True:
            chunk = os.read(destination_descriptor, 1024 * 1024)
            if not chunk:
                break
            destination_digest.update(chunk)
        if destination_digest.digest() != digest.digest():
            raise HorizonSuiteError("NFS fallback destination file checksum 不符")
        destination_status = os.fstat(destination_descriptor)
        if destination_status.st_size != source_after.st_size:
            raise HorizonSuiteError("NFS fallback destination file size 不符")
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if source_descriptor is not None:
            os.close(source_descriptor)


def _copy_directory_fd_tree(
    source_directory: int,
    destination_directory: int,
    relative_name: str,
    layout: Mapping[str, tuple[set[str], set[str]]],
) -> None:
    """依固定白名單遞迴複製 partial，所有檔案與目錄均以 no-follow dirfd 操作。"""

    expected_files, expected_directories = layout[relative_name]
    try:
        actual = set(os.listdir(source_directory))
    except OSError as exc:
        raise HorizonSuiteError("NFS fallback source directory 無法列舉") from exc
    expected = expected_files | expected_directories
    if actual != expected:
        raise HorizonSuiteError(
            f"NFS fallback source topology 不符：missing={sorted(expected - actual)},"
            f"extra={sorted(actual - expected)}"
        )
    for name in sorted(expected_files):
        status = os.stat(name, dir_fd=source_directory, follow_symlinks=False)
        if not stat.S_ISREG(status.st_mode):
            raise HorizonSuiteError("NFS fallback 禁止 symlink 或特殊檔案")
        _copy_file_no_follow(source_directory, destination_directory, name)
    for name in sorted(expected_directories):
        source_status = os.stat(name, dir_fd=source_directory, follow_symlinks=False)
        if not stat.S_ISDIR(source_status.st_mode):
            raise HorizonSuiteError("NFS fallback 只允許普通目錄")
        os.mkdir(name, 0o755, dir_fd=destination_directory)
        source_child = os.open(name, _directory_open_flags(), dir_fd=source_directory)
        destination_child = os.open(name, _directory_open_flags(), dir_fd=destination_directory)
        try:
            child_name = name if not relative_name else f"{relative_name}/{name}"
            _copy_directory_fd_tree(source_child, destination_child, child_name, layout)
            os.fsync(destination_child)
        finally:
            os.close(destination_child)
            os.close(source_child)


def _publish_partial_nfs_two_phase(
    partial: Path,
    destination: Path,
    parent_descriptor: int,
    parent_identity: tuple[int, int],
    partial_descriptor: int,
    partial_identity: tuple[int, int],
) -> tuple[str, tuple[int, int]]:
    """NFS fallback：保留 destination basename 後，以白名單複製完整 partial。

    fallback 只由明確的 ``_ReportReleaseAtomicRenameUnsupported`` 觸發。先在已鎖定
    parent dirfd 以 ``mkdir`` 不可覆寫地保留 basename，再以 no-follow、普通檔／目錄
    白名單複製；任何中斷都保留無 marker 的 reserved destination，原 partial 不修改。
    """

    layout = _suite_copy_layout(partial)
    _assert_tree_has_no_symlink_or_special_file(partial)
    _assert_nfs_source_immutable(partial)
    source_status = os.fstat(partial_descriptor)
    if (
        not stat.S_ISDIR(source_status.st_mode)
        or (source_status.st_dev, source_status.st_ino) != partial_identity
    ):
        raise HorizonSuiteError("NFS fallback source partial identity 已改變")
    current_parent = os.fstat(parent_descriptor)
    if (
        not stat.S_ISDIR(current_parent.st_mode)
        or (current_parent.st_dev, current_parent.st_ino) != parent_identity
    ):
        raise HorizonSuiteError("NFS fallback parent identity 已改變")
    try:
        os.mkdir(destination.name, 0o755, dir_fd=parent_descriptor)
    except FileExistsError as exc:
        raise FileExistsError("horizon suite destination 已存在") from exc
    os.fsync(parent_descriptor)
    destination_descriptor: int | None = None
    try:
        destination_descriptor = os.open(
            destination.name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        destination_status = os.fstat(destination_descriptor)
        destination_identity = (destination_status.st_dev, destination_status.st_ino)
        if not stat.S_ISDIR(destination_status.st_mode):
            raise HorizonSuiteError("NFS fallback destination 不是普通目錄")
        _copy_directory_fd_tree(partial_descriptor, destination_descriptor, "", layout)
        os.fsync(destination_descriptor)
        after_source = os.fstat(partial_descriptor)
        if (after_source.st_dev, after_source.st_ino) != partial_identity:
            raise HorizonSuiteError("NFS fallback source partial identity 在複製後改變")
        after_destination = os.fstat(destination_descriptor)
        if (after_destination.st_dev, after_destination.st_ino) != destination_identity:
            raise HorizonSuiteError("NFS fallback destination identity 在複製後改變")
        copied_manifest, _ = read_canonical_json(
            destination / HORIZON_SUITE_MANIFEST_FILENAME
        )
        _assert_suite_topology(
            destination,
            normalize_horizons(copied_manifest["horizons_days"]),
            tuple(copied_manifest["backtrack_modes"]),
            publication_marker_required=False,
        )
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
    os.fsync(parent_descriptor)
    return HORIZON_SUITE_NFS_PUBLICATION_METHOD, destination_identity


def _publish_partial(
    partial: Path,
    destination: Path,
    parent_identity: tuple[int, int],
    partial_identity: tuple[int, int],
) -> tuple[str, tuple[int, int]]:
    """以同一組 parent／partial dirfd 做 exclusive rename，封閉 basename race。

    ``report_release`` 的 native backend 仍負責 ``renameat2(RENAME_NOREPLACE)``／
    ``renameatx_np(RENAME_EXCL)``；本 wrapper 額外把 parent descriptor、partial descriptor
    與 source inode 一起固定，避免「先查 partial path、後由另一個 path lookup rename」
    把 replacement sentinel 發布出去。rename 後再由同一 parent dirfd 開啟 destination
    比對 inode；若任何邊界無法證明是本次 partial，直接失敗，caller 不會收到成功結果。
    """

    if partial.parent != destination.parent:
        raise HorizonSuiteError("partial 與 final 必須位於同一 parent")
    parent_descriptor: int | None = None
    partial_descriptor: int | None = None
    lock_acquired = False
    try:
        _assert_no_symlink_components(destination.parent, allow_missing_leaf=False)
        parent_descriptor = os.open(destination.parent, _directory_open_flags())
        parent_status = os.fstat(parent_descriptor)
        actual_parent = (parent_status.st_dev, parent_status.st_ino)
        if not stat.S_ISDIR(parent_status.st_mode) or actual_parent != parent_identity:
            raise HorizonSuiteError("final parent identity 已改變")
        _lock_directory(parent_descriptor)
        lock_acquired = True
        # 保留公開的 ownership hook，讓 caller／測試可在最後一個 path-level gate
        # 發現 race；後面的 dirfd/fstat 仍是實際安全邊界，會再拒絕被替換的 basename。
        _assert_owned_partial(partial, partial_identity)
        partial_descriptor = os.open(
            partial.name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        partial_status = os.fstat(partial_descriptor)
        actual_partial = (partial_status.st_dev, partial_status.st_ino)
        if not stat.S_ISDIR(partial_status.st_mode) or actual_partial != partial_identity:
            raise HorizonSuiteError("partial identity 已改變")
        source_entry = os.stat(partial.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (source_entry.st_dev, source_entry.st_ino) != partial_identity:
            raise HorizonSuiteError("partial basename identity 已改變")

        # 這裡直接使用 report_release 相同的 native exclusive backend，但所有五個
        # renameat 參數中的 directory 皆來自已驗證的同一個 dirfd；不重新解析 parent
        # 或 partial 的絕對路徑。只有 backend 明確回報「不支援」時才進入 NFS fallback；
        # collision、ABI 失敗及其他 OSError 絕不以一般 rename 或 unchecked mv 取代。
        try:
            rename_function, rename_flags = _load_exclusive_rename_backend()
            _call_exclusive_rename(
                rename_function,
                parent_descriptor=parent_descriptor,
                source_name=partial.name,
                destination_name=destination.name,
                rename_flags=rename_flags,
            )
        except _ReportReleaseAtomicRenameUnsupported:
            return _publish_partial_nfs_two_phase(
                partial,
                destination,
                parent_descriptor,
                actual_parent,
                partial_descriptor,
                actual_partial,
            )
        destination_descriptor = os.open(
            destination.name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        try:
            destination_status = os.fstat(destination_descriptor)
            destination_identity = (destination_status.st_dev, destination_status.st_ino)
            if not stat.S_ISDIR(destination_status.st_mode) or destination_identity != partial_identity:
                raise HorizonSuiteError("rename 後 destination identity 不符")
        finally:
            os.close(destination_descriptor)
        return HORIZON_SUITE_NATIVE_PUBLICATION_METHOD, destination_identity
    except FileExistsError as exc:
        # native exclusive backend 的 collision 保留 FileExistsError，讓 caller 知道
        # final 是既有成果；不對該 node 做任何 cleanup。
        raise FileExistsError("horizon suite destination 已存在") from exc
    finally:
        if partial_descriptor is not None:
            os.close(partial_descriptor)
        if lock_acquired and parent_descriptor is not None:
            _unlock_directory(parent_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _normalise_publication_result(
    result: object, partial_identity: tuple[int, int]
) -> tuple[str, tuple[int, int]]:
    """將內部 publish hook 統一成 method／destination inode 結果。

    舊有測試或受控部署 hook 可能只回傳 ``None``；這只代表 native hook 已完成，並
    不放寬正式 marker gate。真正的 production ``_publish_partial`` 一律回傳兩項
    tuple，fallback 的新 destination inode 也因此會被後續 marker writer 固定核對。
    """

    if isinstance(result, tuple) and len(result) == 2:
        method, identity = result
        if (
            isinstance(method, str)
            and method in _ALLOWED_PUBLICATION_METHODS
            and isinstance(identity, tuple)
            and len(identity) == 2
            and all(type(item) is int for item in identity)
        ):
            return method, identity
        raise HorizonSuiteError("publish hook 回傳的 publication identity 不合法")
    # 相容既有 native-only test hook；marker writer 仍會重新開啟並驗證 destination。
    return HORIZON_SUITE_NATIVE_PUBLICATION_METHOD, partial_identity


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """以固定 UTF-8 YAML 格式原子寫入 mapping 並回傳 fingerprint。"""

    rendered = yaml.safe_dump(
        dict(payload), allow_unicode=True, sort_keys=False, default_flow_style=False
    ).encode("utf-8")
    _atomic_write_bytes(path, rendered, suffix=".yaml")
    return _yaml_fingerprint(path, path.name)


def _build_releases(
    *,
    common_config: Path,
    common_input: Path,
    partial: Path,
    horizons: tuple[int, ...],
    step_counts: Mapping[int, int],
    backtrack_modes: tuple[str, ...],
    selection_support_days: int,
    runtime_support_days: int,
    formal: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """以同一 common input 建立各 horizon×mode release，並核對共同 binding identity。"""

    release_dir = partial / HORIZON_SUITE_RELEASE_DIRECTORY
    validation_dir = partial / HORIZON_SUITE_VALIDATION_DIRECTORY
    release_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    reference_bindings: dict[str, Mapping[str, Any]] | None = None
    reference_index_hash: str | None = None
    reference_identity: dict[str, Any] | None = None
    mode_slots: tuple[str | None, ...] = backtrack_modes if backtrack_modes else (None,)
    for days in horizons:
        for mode in mode_slots:
            stem = _release_stem(days, mode)
            release_path = release_dir / f"{stem}.yaml"
            created = create_release_config(
                config_template_path=common_config,
                input_directory=common_input,
                output_path=release_path,
                formal=formal,
                max_backtrack_days=float(days),
                maximum_step_count=step_counts[days],
                backtrack_mode_override=mode,
            )
            label = f"release {days} 日/{mode}" if mode is not None else f"release {days} 日"
            relative_config = f"{HORIZON_SUITE_RELEASE_DIRECTORY}/{stem}.yaml"
            release_payload, release_fp, _ = _read_yaml_snapshot(
                release_path, label, relative_config
            )
            validation = validate_release_config(
                release_path, input_directory=common_input, formal=formal
            )
            if not isinstance(validation, Mapping) or validation.get("valid") is not True:
                raise HorizonSuiteError(f"{label} validator 未通過")
            expected_status = "approved" if formal else "generated"
            created_status = created.get("config_status") if isinstance(created, Mapping) else None
            if created_status != expected_status:
                raise HorizonSuiteError(f"{label} config_status 不符：{expected_status}")
            if release_payload.get("config_status") != expected_status:
                raise HorizonSuiteError(f"{label} YAML config_status 不符：{expected_status}")
            approval = release_payload.get("release_approval")
            if not isinstance(approval, Mapping) or approval.get("status") != expected_status:
                raise HorizonSuiteError(f"{label} release_approval.status 不符")
            boundaries = release_payload.get("boundaries")
            inputs = release_payload.get("inputs")
            scenarios = release_payload.get("scenarios")
            bed = scenarios.get("bed_residence_time") if isinstance(scenarios, Mapping) else None
            if not isinstance(boundaries, Mapping) or boundaries.get("max_backtrack_days") != float(days):
                raise HorizonSuiteError(f"{label} requested horizon 不符")
            if boundaries.get("maximum_step_count") != step_counts[days]:
                raise HorizonSuiteError(f"{label} maximum_step_count 不符")
            if (
                not isinstance(inputs, Mapping)
                or inputs.get("backtrack_support_days") != selection_support_days
            ):
                raise HorizonSuiteError(f"{label} selection support 不符")
            if mode is None:
                if bed is not None:
                    raise HorizonSuiteError("legacy suite release 不得意外加入 bed residence")
            elif (
                not isinstance(bed, Mapping)
                or bed.get("backtrack_mode") != mode
                or bed.get("runtime_horizon_support_days") != runtime_support_days
            ):
                raise HorizonSuiteError(f"{label} bed-residence mode/runtime support 不符")
            bindings = _release_records(release_payload)
            release_binding = release_payload.get("release_binding")
            index_hash = (
                release_binding.get("input_directory_artifact_index_sha256")
                if isinstance(release_binding, Mapping)
                else None
            )
            identity = _identity_fingerprints(
                {
                    kind: {
                        field: bindings[kind].get(field)
                        for field in ("sha256", "canonical_sha256", "size_bytes")
                    }
                    for kind in _IDENTITY_KINDS
                }
            )
            for kind, filename in ARTIFACT_FILENAMES.items():
                expected_path = f"../{HORIZON_SUITE_COMMON_INPUT_DIRECTORY}/{filename}"
                if bindings[kind].get("path") != expected_path:
                    raise HorizonSuiteError(f"{label} artifact path 不符：{kind}")
            if reference_bindings is None:
                reference_bindings = dict(bindings)
                reference_index_hash = index_hash
                reference_identity = identity
            else:
                for kind in ARTIFACT_FILENAMES:
                    for field in ("path", "sha256", "canonical_sha256", "size_bytes"):
                        if bindings[kind].get(field) != reference_bindings[kind].get(field):
                            raise HorizonSuiteError(f"{label} artifact binding 不一致：{kind}")
                if index_hash != reference_index_hash or identity != reference_identity:
                    raise HorizonSuiteError(f"{label} 共同 identity 不一致")
            validation_path = validation_dir / f"{stem}.json"
            validation_fp = write_canonical_json(validation_path, dict(validation))
            records.append(
                {
                    "backtrack_days": days,
                    "backtrack_mode": mode,
                    "maximum_step_count": step_counts[days],
                    "config_status": release_payload.get("config_status"),
                    "config_path": relative_config,
                    "validation_path": f"{HORIZON_SUITE_VALIDATION_DIRECTORY}/{stem}.json",
                    "config_fingerprint": release_fp,
                    "validation_fingerprint": validation_fp,
                    "identity_fingerprints": identity,
                }
            )
    if reference_bindings is None or reference_identity is None:
        raise HorizonSuiteError("未建立任何 release")
    return records, {"artifact_index_sha256": reference_index_hash, "identity": reference_identity}


def _validate_release_registered_metadata(
    payload: Mapping[str, Any],
    *,
    expected_status: str,
    formal: bool,
    days: int,
    maximum_step_count: int | None,
    common_config_fingerprint: Mapping[str, Any] | None,
    artifact_index_sha256: str | None,
    artifact_source_config_hash: str | None,
    artifact_support_days: float | None,
    selection_support_days: int,
    runtime_support_days: int,
    backtrack_mode: str | None,
    nonformal_input_summary: Mapping[str, Any] | None,
    formal_input_summary: Mapping[str, Any] | None,
) -> list[str]:
    """逐欄驗證 create_release_config 登錄的 binding／approval evidence。

    release config 的大部分內容必須由 common config 精確重建；只有本 helper 列出的
    metadata 是 release 建置時允許新增的欄位。這裡仍核對 source config hash、共同
    artifact index hash、支援窗與 requested step，避免攻擊者同時重簽 YAML、manifest
    與 sidecar 後偽造另一個輸入母體。approval 的摘要直接比對 validator 目前重放的
    nonformal／formal 結果；回傳錯誤清單而不拋例外，讓公開 validator 能以 JSON-safe
    ``valid:false`` 回應損壞資料。
    """

    errors: list[str] = []
    binding = payload.get("release_binding")
    if not isinstance(binding, Mapping):
        return ["release_binding_missing"]
    gap_contract = _payload_gap_censoring_contract(payload)
    expected_release_binding_keys = _RELEASE_BINDING_KEYS | (
        _GAP_CENSOR_RELEASE_BINDING_KEYS if gap_contract is not None else frozenset()
    )
    if set(binding) != expected_release_binding_keys:
        errors.append(f"release_binding_keys_invalid:{days}")
    if gap_contract is not None:
        for field, expected in gap_contract.items():
            if binding.get(field) != expected:
                errors.append(f"release_gap_policy_binding_invalid:{days}:{field}")
    expected_binding_schema_version = (
        GAP_CENSORED_BED_RESIDENCE_INPUT_SCHEMA_VERSION
        if backtrack_mode is not None and gap_contract is not None
        else BED_RESIDENCE_INPUT_SCHEMA_VERSION
        if backtrack_mode is not None
        else DERIVED_INPUT_SCHEMA_VERSION
    )
    if binding.get("schema_version") != expected_binding_schema_version:
        errors.append(f"release_binding_schema_invalid:{days}")
    if common_config_fingerprint is None or binding.get("source_config_template_sha256") != (
        common_config_fingerprint.get("sha256")
        if isinstance(common_config_fingerprint, Mapping)
        else None
    ):
        errors.append(f"release_source_template_hash_invalid:{days}")
    if binding.get("source_config_hash") != artifact_source_config_hash:
        errors.append(f"release_source_config_hash_invalid:{days}")
    if binding.get("input_directory_artifact_index_sha256") != artifact_index_sha256:
        errors.append(f"release_artifact_index_hash_invalid:{days}")
    if binding.get("approved_only_after_exact_hash_validation") is not True:
        errors.append(f"release_exact_hash_policy_invalid:{days}")

    arrival_binding = binding.get("arrival_selection_binding")
    if (
        not isinstance(arrival_binding, Mapping)
        or set(arrival_binding) != _RELEASE_ARRIVAL_SELECTION_BINDING_KEYS
    ):
        errors.append(f"release_arrival_selection_binding_invalid:{days}")
    else:
        try:
            population = _arrival_population_contract(payload)
        except Exception:
            errors.append(f"release_arrival_population_invalid:{days}")
        else:
            expected_arrival_binding = {
                "forcing_years": population["forcing_years"],
                "observation_years": population["observation_years"],
                "policy": population["arrival_selection_policy_id"],
                "replicates": population["replicates_per_stratum"],
            }
            if dict(arrival_binding) != expected_arrival_binding:
                errors.append(f"release_arrival_selection_binding_mismatch:{days}")

    horizon_binding = binding.get("backtrack_horizon_binding")
    if not isinstance(horizon_binding, Mapping):
        errors.append(f"release_horizon_binding_missing:{days}")
    else:
        expected_horizon_values: dict[str, Any] = {
            "source_config_hash": artifact_source_config_hash,
            "source_backtrack_support_days": selection_support_days,
            "artifact_backtrack_support_days": artifact_support_days,
            "requested_max_backtrack_days": float(days),
            "requested_maximum_step_count": maximum_step_count,
        }
        expected_binding_keys = _RELEASE_HORIZON_BINDING_KEYS
        if backtrack_mode is not None:
            expected_binding_keys = _BED_RELEASE_HORIZON_BINDING_KEYS
            expected_horizon_values.update(
                {
                    "selection_support_days": selection_support_days,
                    "runtime_support_days": runtime_support_days,
                    "artifact_selection_support_days": selection_support_days,
                    "artifact_runtime_support_days": runtime_support_days,
                    "backtrack_mode": backtrack_mode,
                }
            )
        if set(horizon_binding) != expected_binding_keys:
            errors.append(f"release_horizon_binding_keys_invalid:{days}:{backtrack_mode}")
        for field, expected in expected_horizon_values.items():
            if horizon_binding.get(field) != expected:
                errors.append(f"release_horizon_binding_invalid:{days}:{field}")

    approval = payload.get("release_approval")
    if not isinstance(approval, Mapping):
        errors.append(f"release_approval_missing:{days}")
    else:
        expected_approval_keys = _RELEASE_APPROVAL_KEYS
        # wet/dry evidence 只在正式 current-design promotion 時出現；pilot、legacy
        # 與已經具有 confirmed/approved source status 的 release 維持原欄位集合。
        # 證據內容與 dynamic artifact hash 由 input_derivation validator 另行重放。
        if _WETDRY_RELEASE_APPROVAL_KEY in approval:
            expected_approval_keys = _RELEASE_APPROVAL_KEYS | {
                _WETDRY_RELEASE_APPROVAL_KEY
            }
        if set(approval) != expected_approval_keys:
            errors.append(f"release_approval_keys_invalid:{days}")
        if approval.get("status") != expected_status:
            errors.append(f"release_approval_status_invalid:{days}")
        if approval.get("blockers") != []:
            errors.append(f"release_approval_blockers_invalid:{days}")
        if (
            nonformal_input_summary is None
            or approval.get("validated_input_summary") != dict(nonformal_input_summary)
        ):
            errors.append(f"release_approval_input_summary_mismatch:{days}")
        formal_summary = approval.get("formal_input_validation_summary")
        if formal:
            if (
                formal_input_summary is None
                or formal_summary != dict(formal_input_summary)
            ):
                errors.append(f"release_approval_formal_summary_mismatch:{days}")
        elif formal_summary is not None:
            errors.append(f"release_approval_pilot_summary_invalid:{days}")
        if approval.get("public_analysis_label_policy") != {"A": "A 區分析域"}:
            errors.append(f"release_approval_label_policy_invalid:{days}")
    if payload.get("config_status") != expected_status:
        errors.append(f"release_status_invalid:{days}")
    return errors


def _validate_suite_contents(
    suite_root: Path,
    manifest: Mapping[str, Any],
    *,
    formal: bool,
    publication_marker_required: bool,
    ocm_native_root: str | Path | None,
    ocm_surface_root: str | Path | None,
    nww_analysis_root: str | Path | None,
) -> dict[str, Any]:
    """重新計算所有 suite fingerprint 與 downstream gate，任何例外均轉成 errors。"""

    errors: list[str] = []
    manifest_gap_keys_present = _GAP_CENSOR_SUITE_MANIFEST_KEYS.issubset(set(manifest))
    if set(manifest) not in (
        _SUITE_MANIFEST_KEYS,
        _SUITE_MANIFEST_KEYS | _GAP_CENSOR_SUITE_MANIFEST_KEYS,
    ):
        errors.append("manifest_key_set_invalid")
    try:
        horizons = normalize_horizons(manifest.get("horizons_days"))
    except Exception:
        horizons = ()
        errors.append("manifest_horizons_invalid")
    maximum_horizon = max(horizons) if horizons else None
    if manifest.get("manifest_kind") != "horizon_suite_manifest":
        errors.append("manifest_kind_invalid")
    if manifest.get("schema_version") != HORIZON_SUITE_SCHEMA_VERSION:
        errors.append("manifest_schema_version_invalid")
    expected_gate_mode = "formal" if formal else "pilot"
    expected_status = "approved" if formal else "generated"
    if manifest.get("gate_mode") != expected_gate_mode:
        errors.append("manifest_gate_mode_invalid")
    if manifest.get("status") != expected_status:
        errors.append("manifest_status_invalid")
    if manifest.get("policy_id") != HORIZON_SUITE_POLICY_ID:
        errors.append("manifest_policy_invalid")
    if manifest.get("method_id") != HORIZON_SUITE_METHOD_ID:
        errors.append("manifest_method_invalid")
    if maximum_horizon is not None and manifest.get("horizons_days") != list(horizons):
        errors.append("manifest_horizons_not_canonical")
    raw_modes = manifest.get("backtrack_modes")
    if not isinstance(raw_modes, list) or any(not isinstance(mode, str) for mode in raw_modes):
        manifest_modes: tuple[str, ...] = ()
        errors.append("manifest_backtrack_modes_invalid")
    else:
        manifest_modes = tuple(raw_modes)
        if len(set(manifest_modes)) != len(manifest_modes) or any(
            mode not in _BED_RESIDENCE_MODES for mode in manifest_modes
        ):
            errors.append("manifest_backtrack_modes_invalid")
    if manifest.get("step_count_formula") != "ceil(days*86400/dt_min_seconds)+1":
        errors.append("manifest_step_formula_invalid")
    if type(manifest.get("input_build_count")) is not int or manifest.get("input_build_count") != 1:
        errors.append("manifest_input_build_count_invalid")
    if manifest.get("paths") != _expected_suite_paths():
        errors.append("manifest_paths_invalid")
    input_context = manifest.get("input_validation_context")
    if (
        not isinstance(input_context, Mapping)
        or type(input_context.get("formal")) is not bool
        or type(input_context.get("roots_supplied")) is not bool
        or dict(input_context) != {"formal": formal, "roots_supplied": True}
    ):
        errors.append("manifest_input_validation_context_invalid")
    source_payload: dict[str, Any] = {}
    common_payload: dict[str, Any] = {}
    population_contract: dict[str, Any] | None = None
    source_path = suite_root / HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME
    common_path = suite_root / HORIZON_SUITE_COMMON_CONFIG_FILENAME
    try:
        source_payload, source_actual_fp, _ = _read_yaml_snapshot(
            source_path,
            "source template",
            HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME,
        )
        _assert_unbound_template(source_payload)
        recorded = manifest.get("source_template_fingerprint")
        if not _fingerprint_matches(recorded, source_actual_fp):
            errors.append("source_template_fingerprint_mismatch")
    except Exception as exc:
        errors.append(f"source_template_invalid:{type(exc).__name__}")
    dt_min: float | None = None
    common_actual_fp: dict[str, Any] | None = None
    selection_support_days = maximum_horizon
    runtime_support_days = maximum_horizon
    expected_modes: tuple[str, ...] = ()
    try:
        common_payload, common_actual_fp, _ = _read_yaml_snapshot(
            common_path,
            "common config",
            HORIZON_SUITE_COMMON_CONFIG_FILENAME,
        )
        recorded = manifest.get("common_config_fingerprint")
        if not _fingerprint_matches(recorded, common_actual_fp):
            errors.append("common_config_fingerprint_mismatch")
        if maximum_horizon is None:
            raise HorizonSuiteError("horizon 空集合")
        dt_min = _read_dt_min_and_validate_common(common_payload, maximum_horizon)
        expected_common = _derive_common_payload(source_payload, maximum_horizon)
        if common_payload != expected_common:
            errors.append("common_config_derived_payload_mismatch")
        population_contract = _arrival_population_contract(common_payload)
        gap_contract = _payload_gap_censoring_contract(common_payload)
        if (gap_contract is not None) != manifest_gap_keys_present:
            # legacy 必須完全沒有新 policy keys；新 suite 必須完整保存三個 exact ID。
            errors.append("manifest_gap_censoring_policy_presence_mismatch")
        elif gap_contract is not None:
            for field, expected in gap_contract.items():
                if manifest.get(field) != expected:
                    errors.append(f"manifest_gap_censoring_policy_mismatch:{field}")
        for field in (
            "forcing_years",
            "observation_years",
            "arrival_selection_policy_id",
            "replicates_per_stratum",
            "arrival_core_count",
            "arrival_event_count",
        ):
            if manifest.get(field) != population_contract[field]:
                errors.append(f"manifest_arrival_population_mismatch:{field}")
        selection_support_days, runtime_support_days, expected_modes = _suite_support_contract(
            common_payload, maximum_horizon
        )
        expected_source_schema_version = (
            GAP_CENSORED_BED_RESIDENCE_INPUT_SCHEMA_VERSION
            if expected_modes and gap_contract is not None
            else BED_RESIDENCE_INPUT_SCHEMA_VERSION
            if expected_modes
            else DERIVED_INPUT_SCHEMA_VERSION
        )
        if manifest.get("source_schema_version") != expected_source_schema_version:
            errors.append("manifest_source_schema_version_invalid")
        if manifest.get("selection_support_days") != selection_support_days:
            errors.append("manifest_selection_support_mismatch")
        if manifest.get("runtime_support_days") != runtime_support_days:
            errors.append("manifest_runtime_support_mismatch")
        if manifest_modes != expected_modes:
            errors.append("manifest_backtrack_modes_mismatch")
        try:
            _assert_suite_topology(
                suite_root,
                horizons,
                expected_modes,
                publication_marker_required=publication_marker_required,
            )
        except Exception as exc:
            errors.append(f"suite_topology_invalid:{type(exc).__name__}")
        manifest_dt = manifest.get("dt_min_seconds")
        if (
            isinstance(manifest_dt, bool)
            or not isinstance(manifest_dt, (int, float))
            or not math.isfinite(float(manifest_dt))
            or float(manifest_dt) != dt_min
        ):
            errors.append("manifest_dt_min_seconds_mismatch")
        if common_payload.get("boundaries", {}).get("max_backtrack_days") != float(maximum_horizon):
            errors.append("common_requested_horizon_mismatch")
        if common_payload.get("boundaries", {}).get("maximum_step_count") != maximum_step_count_for_horizon(
            maximum_horizon, dt_min
        ):
            errors.append("common_step_count_mismatch")
        recorded_steps = manifest.get("maximum_step_count_by_horizon")
        expected_steps = {
            str(days): maximum_step_count_for_horizon(days, dt_min) for days in horizons
        }
        if recorded_steps != expected_steps:
            errors.append("manifest_step_count_mapping_mismatch")
        expected_refs = {
            ("inputs", "ocm_gap_safe_arrival_manifest"): "common-input/ocm_gap_safe_arrival.json",
            ("inputs", "nww_full_hourly_analysis_manifest"): "common-input/nww_full_hourly.json",
            ("inputs", "derived_input_artifact_index"): "common-input/artifact_index.json",
            ("scenarios", "material_manifest"): "common-input/material.json",
            ("scenarios", "receptor_manifest"): "common-input/receptor.json",
            ("scenarios", "arrival_time_manifest"): "common-input/arrival.json",
            ("scenarios", "receptor_arrival_initial_condition_manifest"): (
                "common-input/initial_conditions.json"
            ),
            ("geometry", "domain_manifest"): "common-input/domain.json",
            ("geometry", "local_domain_manifest"): "common-input/local.json",
            ("geometry", "open_boundary_manifest"): "common-input/open.json",
            ("geometry", "receptor_manifest"): "common-input/receptor.json",
            ("physics", "settling", "material_manifest"): "common-input/material.json",
        }
        for path_parts, expected in expected_refs.items():
            current: Any = common_payload
            for key in path_parts:
                current = current.get(key) if isinstance(current, Mapping) else None
            if current != expected:
                errors.append(f"common_reference_mismatch:{'.'.join(path_parts)}")
        for field in ("release_binding", "release_approval", "pilot_execution_binding"):
            if common_payload.get(field) is not None:
                errors.append(f"common_binding_present:{field}")
    except Exception as exc:
        errors.append(f"common_config_invalid:{type(exc).__name__}")

    input_root = suite_root / HORIZON_SUITE_COMMON_INPUT_DIRECTORY
    artifact_source_config_hash: str | None = None
    artifact_support_days: float | None = None
    inventory_payload: Mapping[str, Any] | None = None
    recovery_method = manifest.get("recovery_method")
    if recovery_method not in {"fresh_build_v1", "resume_reuse_validated_common_input_v1"}:
        errors.append("manifest_recovery_method_invalid")
    elif recovery_method == "fresh_build_v1" and manifest.get("recovery_source_fingerprint") is not None:
        errors.append("manifest_fresh_recovery_source_must_be_null")
    try:
        component_fps = _component_fingerprints(input_root)
        artifact_index_payload, artifact_index_fp = _read_artifact_index(input_root)
        _validate_artifact_closure(input_root, component_fps)
        inventory_candidate, _ = read_canonical_json(
            input_root / ARTIFACT_FILENAMES["forcing_inventory"]
        )
        if isinstance(inventory_candidate, Mapping):
            inventory_payload = inventory_candidate
        recorded_index = manifest.get("common_artifact_index_fingerprint")
        if not _fingerprint_matches(recorded_index, component_fps["artifact_index"]):
            errors.append("artifact_index_fingerprint_mismatch")
        if any(
            artifact_index_fp.get(field) != component_fps["artifact_index"].get(field)
            for field in ("sha256", "canonical_sha256", "size_bytes")
        ):
            errors.append("artifact_index_reader_fingerprint_mismatch")
        recorded_components = manifest.get("common_artifact_fingerprints")
        if not isinstance(recorded_components, Mapping):
            errors.append("common_artifact_fingerprints_missing")
        else:
            if set(recorded_components) != set(ARTIFACT_FILENAMES):
                errors.append("common_artifact_fingerprints_set_mismatch")
            for kind in ARTIFACT_FILENAMES:
                if recorded_components.get(kind) != component_fps[kind]:
                    errors.append(f"common_component_fingerprint_mismatch:{kind}")
        if manifest.get("common_identity_fingerprints") != _identity_fingerprints(component_fps):
            errors.append("common_identity_fingerprints_mismatch")
        recorded_recovery_source = manifest.get("recovery_source_fingerprint")
        if recovery_method == "resume_reuse_validated_common_input_v1":
            expected_recovery_source = _recovery_source_fingerprint(
                manifest.get("source_template_fingerprint")
                if isinstance(manifest.get("source_template_fingerprint"), Mapping)
                else {},
                manifest.get("common_config_fingerprint")
                if isinstance(manifest.get("common_config_fingerprint"), Mapping)
                else {},
                component_fps,
            )
            if recorded_recovery_source != expected_recovery_source:
                errors.append("manifest_recovery_source_fingerprint_mismatch")
        elif recovery_method == "fresh_build_v1" and recorded_recovery_source is not None:
            errors.append("manifest_fresh_recovery_source_must_be_null")
        source_bindings = artifact_index_payload.get("source_bindings")
        expected_config_hash = _config_hash_from_payload(common_payload)
        if isinstance(source_bindings, Mapping) and type(source_bindings.get("config_hash")) is str:
            artifact_source_config_hash = source_bindings["config_hash"]
        if (
            not isinstance(source_bindings, Mapping)
            or source_bindings.get("config_hash") != expected_config_hash
        ):
            errors.append("artifact_index_source_config_hash_mismatch")
        gap_payload, _ = read_canonical_json(
            input_root / ARTIFACT_FILENAMES["ocm_gap_safe_arrival_horizon"]
        )
        gap_days = gap_payload.get("max_backtrack_days")
        if (
            isinstance(gap_days, bool)
            or not isinstance(gap_days, (int, float))
            or not math.isfinite(float(gap_days))
            or float(gap_days) <= 0.0
        ):
            raise HorizonSuiteError("gap-safe artifact max_backtrack_days 不合法")
        artifact_support_days = float(gap_days)
    except Exception as exc:
        errors.append(f"common_input_invalid:{type(exc).__name__}")
        component_fps = {}

    stored_input_validation: Mapping[str, Any] | None = None
    try:
        input_validation_path = suite_root / HORIZON_SUITE_VALIDATION_DIRECTORY / "input.json"
        input_validation, input_fp = read_canonical_json(input_validation_path)
        stored_input_validation = input_validation
        if input_validation.get("valid") is not True:
            errors.append("input_validation_not_valid")
        recorded = manifest.get("input_validation_fingerprint")
        if not _fingerprint_matches(recorded, input_fp):
            errors.append("input_validation_fingerprint_mismatch")
    except Exception as exc:
        errors.append(f"input_validation_invalid:{type(exc).__name__}")
    root_count = sum(
        root is not None
        for root in (ocm_native_root, ocm_surface_root, nww_analysis_root)
    )
    roots_supplied = root_count == 3
    if root_count not in (0, 3):
        # 三套 accepted product root 必須一起提供；只給其中一套會讓 input validator
        # 產生看似成功但無法重現完整 forcing evidence 的摘要，因此直接 fail closed。
        errors.append("forcing_roots_partial")
        input_result: Mapping[str, Any] = {}
    else:
        try:
            input_result = validate_input_derivatives(
                input_root,
                config_path=common_path,
                formal=formal,
                ocm_native_root=ocm_native_root,
                ocm_surface_root=ocm_surface_root,
                nww_analysis_root=nww_analysis_root,
            )
            if not isinstance(input_result, Mapping) or input_result.get("valid") is not True:
                errors.append("input_validator_failed")
            if (
                roots_supplied
                and stored_input_validation is not None
                and canonical_json_bytes(stored_input_validation)
                != canonical_json_bytes(input_result)
            ):
                errors.append("input_validation_evidence_mismatch")
        except Exception as exc:
            errors.append(f"input_validator_exception:{type(exc).__name__}")
            input_result = {}
    if not roots_supplied:
        # 缺少三個 forcing root 時仍可驗 sidecar、固定 closure 與 release evidence，但
        # 無法重現 build 時的 canonical-axis／forcing coverage summary；把限制公開留在
        # JSON summary，而非把省略 roots 的結果誤稱為與 build evidence 完全相同。
        input_evidence_warning = "input_validation_evidence_not_replayed_without_roots"
    else:
        input_evidence_warning = None

    mode_slots: tuple[str | None, ...] = expected_modes if expected_modes else (None,)
    expected_release_pairs = {(days, mode) for days in horizons for mode in mode_slots}
    releases = manifest.get("releases")
    if not isinstance(releases, list) or len(releases) != len(expected_release_pairs):
        errors.append("release_manifest_count_invalid")
        releases = []
    records_by_pair: dict[tuple[int, str | None], Mapping[str, Any]] = {}
    for record in releases:
        if not isinstance(record, Mapping) or type(record.get("backtrack_days")) is not int:
            errors.append("release_manifest_record_invalid")
            continue
        if set(record) != _RELEASE_RECORD_KEYS:
            errors.append(f"release_manifest_record_keys_invalid:{record.get('backtrack_days')}")
        days = record["backtrack_days"]
        mode_value = record.get("backtrack_mode")
        mode = mode_value if isinstance(mode_value, str) else None
        if mode_value is not None and not isinstance(mode_value, str):
            errors.append(f"release_manifest_mode_invalid:{days}")
            continue
        key = (days, mode)
        if key not in expected_release_pairs:
            errors.append(f"release_manifest_unexpected_pair:{days}:{mode}")
            continue
        if key in records_by_pair:
            errors.append(f"release_manifest_pair_duplicate:{days}:{mode}")
        records_by_pair[key] = record
    if set(records_by_pair) != expected_release_pairs:
        errors.append("release_manifest_horizon_mode_set_mismatch")
    reference_binding: dict[str, Mapping[str, Any]] | None = None
    reference_index_hash: str | None = None
    for days, mode in sorted(
        expected_release_pairs,
        key=lambda pair: (pair[0], "" if pair[1] is None else pair[1]),
    ):
        record = records_by_pair.get((days, mode))
        mode_label = "legacy" if mode is None else mode
        if record is None:
            errors.append(f"release_manifest_missing:{days}:{mode_label}")
            continue
        stem = _release_stem(days, mode)
        release_relative_path = f"{HORIZON_SUITE_RELEASE_DIRECTORY}/{stem}.yaml"
        validation_relative_path = f"{HORIZON_SUITE_VALIDATION_DIRECTORY}/{stem}.json"
        release_path = suite_root / release_relative_path
        validation_path = suite_root / validation_relative_path
        try:
            release_payload, release_actual_fp, _ = _read_yaml_snapshot(
                release_path,
                f"release {days} 日/{mode_label}",
                release_relative_path,
            )
            if population_contract is not None:
                try:
                    release_population = _arrival_population_contract(release_payload)
                    if release_population != population_contract:
                        errors.append(
                            f"release_arrival_population_mismatch:{days}:{mode_label}"
                        )
                except Exception as exc:
                    errors.append(
                        f"release_arrival_population_invalid:{days}:{mode_label}:{type(exc).__name__}"
                    )
            boundaries = release_payload.get("boundaries")
            inputs = release_payload.get("inputs")
            if not isinstance(boundaries, Mapping) or boundaries.get("max_backtrack_days") != float(days):
                errors.append(f"release_horizon_mismatch:{days}")
            if dt_min is not None and (
                not isinstance(boundaries, Mapping)
                or boundaries.get("maximum_step_count") != maximum_step_count_for_horizon(days, dt_min)
            ):
                errors.append(f"release_step_count_mismatch:{days}")
            if (
                not isinstance(inputs, Mapping)
                or inputs.get("backtrack_support_days") != selection_support_days
            ):
                errors.append(f"release_selection_support_mismatch:{days}:{mode_label}")
            release_scenarios = release_payload.get("scenarios")
            release_bed = (
                release_scenarios.get("bed_residence_time")
                if isinstance(release_scenarios, Mapping)
                else None
            )
            if mode is None:
                if release_bed is not None:
                    errors.append(f"release_bed_residence_unexpected:{days}")
            elif (
                not isinstance(release_bed, Mapping)
                or release_bed.get("backtrack_mode") != mode
                or release_bed.get("runtime_horizon_support_days") != runtime_support_days
            ):
                errors.append(f"release_bed_residence_mode_mismatch:{days}:{mode_label}")
            expected_record_steps = (
                maximum_step_count_for_horizon(days, dt_min) if dt_min is not None else None
            )
            # create_release_config 產生 approval 摘要時使用同一份 release config、
            # 同一個 common-input，但不傳 accepted forcing roots；這裡以相同的 no-root
            # validator contract 重放摘要，避免把 source axis 的重建旗標誤混入 approval
            # evidence。正式模式再重放一次 formal summary；pilot 的 formal summary 必須
            # 固定為 None。任何重放例外都只形成錯誤，不讓 validator 對損壞 YAML 外拋。
            nonformal_input_summary: Mapping[str, Any] | None = None
            formal_input_summary: Mapping[str, Any] | None = None
            try:
                nonformal_replay = validate_input_derivatives(
                    input_root,
                    config_path=release_path,
                    formal=False,
                )
                if isinstance(nonformal_replay, Mapping) and isinstance(
                    nonformal_replay.get("summary"), Mapping
                ):
                    nonformal_input_summary = nonformal_replay["summary"]
                if (
                    not isinstance(nonformal_replay, Mapping)
                    or nonformal_replay.get("valid") is not True
                ):
                    errors.append(f"release_nonformal_replay_failed:{days}:{mode_label}")
            except Exception as exc:
                errors.append(
                    f"release_nonformal_replay_exception:{days}:{mode_label}:{type(exc).__name__}"
                )
            if formal:
                try:
                    formal_replay = validate_input_derivatives(
                        input_root,
                        config_path=release_path,
                        formal=True,
                    )
                    if isinstance(formal_replay, Mapping) and isinstance(
                        formal_replay.get("summary"), Mapping
                    ):
                        formal_input_summary = formal_replay["summary"]
                    if (
                        not isinstance(formal_replay, Mapping)
                        or formal_replay.get("valid") is not True
                    ):
                        errors.append(f"release_formal_replay_failed:{days}:{mode_label}")
                except Exception as exc:
                    errors.append(
                        f"release_formal_replay_exception:{days}:{mode_label}:{type(exc).__name__}"
                    )
            errors.extend(
                _validate_release_registered_metadata(
                    release_payload,
                    expected_status=expected_status,
                    formal=formal,
                    days=days,
                    maximum_step_count=expected_record_steps,
                    common_config_fingerprint=common_actual_fp,
                    artifact_index_sha256=(
                        component_fps.get("artifact_index", {}).get("sha256")
                        if component_fps
                        else None
                    ),
                    artifact_source_config_hash=artifact_source_config_hash,
                    artifact_support_days=artifact_support_days,
                    selection_support_days=selection_support_days,
                    runtime_support_days=runtime_support_days,
                    backtrack_mode=mode,
                    nonformal_input_summary=nonformal_input_summary,
                    formal_input_summary=formal_input_summary,
                )
            )
            if common_payload and expected_record_steps is not None:
                try:
                    expected_release = _derive_expected_release_payload(
                        common_payload,
                        days=days,
                        maximum_step_count=expected_record_steps,
                        inventory_payload=inventory_payload,
                        backtrack_mode=mode,
                        formal=formal,
                    )
                    if _release_payload_without_registered_fields(
                        release_payload
                    ) != _release_payload_without_registered_fields(expected_release):
                        errors.append(f"release_derived_payload_mismatch:{days}:{mode_label}")
                except Exception as exc:
                    errors.append(
                        f"release_expected_payload_invalid:{days}:{mode_label}:{type(exc).__name__}"
                    )
            if release_payload.get("config_status") != expected_status:
                errors.append(f"release_status_invalid:{days}:{mode_label}")
            approval = release_payload.get("release_approval")
            if not isinstance(approval, Mapping) or approval.get("status") != expected_status:
                errors.append(f"release_approval_status_invalid:{days}:{mode_label}")
            binding = _release_records(release_payload)
            release_binding = release_payload.get("release_binding")
            index_hash = (
                release_binding.get("input_directory_artifact_index_sha256")
                if isinstance(release_binding, Mapping)
                else None
            )
            if reference_binding is None:
                reference_binding = dict(binding)
                reference_index_hash = index_hash
            else:
                for kind in ARTIFACT_FILENAMES:
                    for field in ("path", "sha256", "canonical_sha256", "size_bytes"):
                        if binding[kind].get(field) != reference_binding[kind].get(field):
                            errors.append(
                                f"release_binding_mismatch:{days}:{mode_label}:{kind}:{field}"
                            )
                if index_hash != reference_index_hash:
                    errors.append(f"release_artifact_index_mismatch:{days}:{mode_label}")
            if component_fps:
                for kind in ARTIFACT_FILENAMES:
                    for field in ("sha256", "canonical_sha256", "size_bytes"):
                        if binding[kind].get(field) != component_fps[kind].get(field):
                            errors.append(
                                f"release_component_hash_mismatch:{days}:{mode_label}:{kind}:{field}"
                            )
            if component_fps and index_hash != component_fps["artifact_index"].get("sha256"):
                errors.append(f"release_artifact_index_hash_mismatch:{days}:{mode_label}")
            for kind, filename in ARTIFACT_FILENAMES.items():
                expected_binding_path = f"../{HORIZON_SUITE_COMMON_INPUT_DIRECTORY}/{filename}"
                if binding[kind].get("path") != expected_binding_path:
                    errors.append(f"release_binding_path_mismatch:{days}:{mode_label}:{kind}")
            if record.get("backtrack_mode") != mode:
                errors.append(f"release_record_mode_mismatch:{days}:{mode_label}")
            if record.get("config_path") != release_relative_path:
                errors.append(f"release_config_path_mismatch:{days}:{mode_label}")
            if record.get("validation_path") != validation_relative_path:
                errors.append(f"release_validation_path_mismatch:{days}:{mode_label}")
            if (
                type(record.get("maximum_step_count")) is not int
                or record.get("maximum_step_count") != expected_record_steps
            ):
                errors.append(f"release_record_step_count_mismatch:{days}:{mode_label}")
            if record.get("config_status") != expected_status:
                errors.append(f"release_record_status_invalid:{days}:{mode_label}")
            recorded_config = record.get("config_fingerprint")
            if not _fingerprint_matches(recorded_config, release_actual_fp):
                errors.append(f"release_config_fingerprint_mismatch:{days}:{mode_label}")
            release_result = validate_release_config(
                release_path, input_directory=input_root, formal=formal
            )
            if not isinstance(release_result, Mapping) or release_result.get("valid") is not True:
                errors.append(f"release_validator_failed:{days}:{mode_label}")
            identity = _identity_fingerprints(
                {
                    kind: {
                        field: binding[kind].get(field)
                        for field in ("sha256", "canonical_sha256", "size_bytes")
                    }
                    for kind in _IDENTITY_KINDS
                }
            )
            if record.get("identity_fingerprints") != identity:
                errors.append(f"release_record_identity_mismatch:{days}:{mode_label}")
            if reference_binding is not None and identity != _identity_fingerprints(
                {
                    kind: {
                        field: reference_binding[kind].get(field)
                        for field in ("sha256", "canonical_sha256", "size_bytes")
                    }
                    for kind in _IDENTITY_KINDS
                }
            ):
                errors.append(f"release_identity_mismatch:{days}:{mode_label}")
            validation_payload, validation_fp = read_canonical_json(validation_path)
            if validation_payload.get("valid") is not True:
                errors.append(f"release_validation_not_valid:{days}:{mode_label}")
            if isinstance(release_result, Mapping) and canonical_json_bytes(
                validation_payload
            ) != canonical_json_bytes(release_result):
                errors.append(f"release_validation_evidence_mismatch:{days}:{mode_label}")
            recorded_validation = record.get("validation_fingerprint")
            if not _fingerprint_matches(recorded_validation, validation_fp):
                errors.append(f"release_validation_fingerprint_mismatch:{days}:{mode_label}")
        except Exception as exc:
            errors.append(f"release_invalid:{days}:{mode_label}:{type(exc).__name__}")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": [input_evidence_warning] if input_evidence_warning else [],
        "summary": {
            "horizons_days": list(horizons),
            "selection_support_days": selection_support_days,
            "runtime_support_days": runtime_support_days,
            "backtrack_modes": list(expected_modes),
            "release_count": len(expected_release_pairs),
            "input_build_count": manifest.get("input_build_count"),
            "input_valid": input_result.get("valid") is True
            if isinstance(input_result, Mapping)
            else False,
            "formal": formal,
            "source_canonical_axis_rebuilt": roots_supplied,
        },
    }


def validate_horizon_suite(
    path: str | Path,
    *,
    formal: bool = True,
    ocm_native_root: str | Path | None = None,
    ocm_surface_root: str | Path | None = None,
    nww_analysis_root: str | Path | None = None,
    _allow_unpublished_partial: bool = False,
) -> dict[str, Any]:
    """唯讀驗證完整 suite；任何缺檔、竄改或 downstream exception 都回 valid=false。

    common input 固定位於 suite 的 ``common-input``，不接受外部 input override，避免
    release YAML 的相對 binding 與另一份看似相同但未經同一套 closure 的輸入混用。若
    需要搬移成果，應連同整個 suite 一起搬移，或重新建立新 suite。公開入口預設要求
    immutable publication marker；``_allow_unpublished_partial`` 僅供本模組 builder 在
    marker 尚未寫入的自有 partial 內部 gate 使用，外部 CLI 不得把該模式當成正式成功。
    """

    try:
        suite_root = _assert_regular_directory(path)
        manifest, manifest_fp = read_canonical_json(suite_root / HORIZON_SUITE_MANIFEST_FILENAME)
    except Exception as exc:
        return {"valid": False, "errors": [f"suite_read_invalid:{type(exc).__name__}"], "warnings": []}
    try:
        result = _validate_suite_contents(
            suite_root,
            manifest,
            formal=formal,
            publication_marker_required=not _allow_unpublished_partial,
            ocm_native_root=ocm_native_root,
            ocm_surface_root=ocm_surface_root,
            nww_analysis_root=nww_analysis_root,
        )
    except Exception as exc:
        result = {
            "valid": False,
            "errors": [f"suite_validator_exception:{type(exc).__name__}"],
            "warnings": [],
        }
    if not _allow_unpublished_partial:
        marker_errors = _validate_publication_marker(suite_root, manifest_fp)
        if marker_errors:
            result.setdefault("errors", []).extend(marker_errors)
            result["valid"] = False
    result.setdefault("summary", {})["manifest_fingerprint"] = {
        field: manifest_fp[field] for field in ("sha256", "canonical_sha256", "size_bytes")
    }
    return result


def _recovery_source_fingerprint(
    source_template_fingerprint: Mapping[str, Any],
    common_config_fingerprint: Mapping[str, Any],
    component_fingerprints: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """建立不含絕對路徑的 recovery source fingerprint。

    recovery 只允許重用已驗證的 source-template、common-config 與 common-input；因此
    manifest 保存三個來源 bytes hash，以及由所有 immutable component fingerprint
    canonical 化後再雜湊的 closure 摘要。這些欄位足以在不暴露 SERVER 絕對路徑的前提下
    重現「哪一批來源被重用」的稽核鏈，且不把原 partial 的 basename 當成可信內容。
    """

    component_snapshot = {
        kind: {
            field: component_fingerprints[kind][field]
            for field in ("sha256", "canonical_sha256", "size_bytes")
        }
        for kind in sorted(component_fingerprints)
    }
    return {
        "source_template_sha256": str(source_template_fingerprint["sha256"]),
        "common_config_sha256": str(common_config_fingerprint["sha256"]),
        "common_input_artifact_index_sha256": str(
            component_fingerprints["artifact_index"]["sha256"]
        ),
        "common_input_components_sha256": sha256(
            canonical_json_bytes(component_snapshot)
        ).hexdigest(),
    }


def _build_suite_manifest(
    *,
    source_template_fingerprint: Mapping[str, Any],
    common_config_fingerprint: Mapping[str, Any],
    common_payload: Mapping[str, Any],
    input_validation_fingerprint: Mapping[str, Any],
    component_fingerprints: Mapping[str, Mapping[str, Any]],
    release_records: Sequence[Mapping[str, Any]],
    release_identity: Mapping[str, Any],
    horizons: tuple[int, ...],
    population_contract: Mapping[str, Any],
    selection_support_days: int,
    runtime_support_days: int,
    backtrack_modes: tuple[str, ...],
    dt_min: float,
    step_counts: Mapping[int, int],
    formal: bool,
    recovery_method: str = "fresh_build_v1",
    recovery_source_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """以同一組 component／release evidence 組裝 suite root manifest。

    fresh build 與 recovery 必須使用完全相同的 schema、topology 與來源綁定；差異只
    記錄在 ``recovery_method`` 與可稽核 fingerprint。函式不讀取檔案、不產生 input，
    由 caller 先完成 closure、formal gate 與 release validator 後再寫入 manifest。
    """

    manifest: dict[str, Any] = {
        "manifest_kind": "horizon_suite_manifest",
        "schema_version": HORIZON_SUITE_SCHEMA_VERSION,
        "source_schema_version": (
            GAP_CENSORED_BED_RESIDENCE_INPUT_SCHEMA_VERSION
            if backtrack_modes and _payload_gap_censoring_contract(common_payload) is not None
            else BED_RESIDENCE_INPUT_SCHEMA_VERSION
            if backtrack_modes
            else DERIVED_INPUT_SCHEMA_VERSION
        ),
        "status": "approved" if formal else "generated",
        "gate_mode": "formal" if formal else "pilot",
        "policy_id": HORIZON_SUITE_POLICY_ID,
        "method_id": HORIZON_SUITE_METHOD_ID,
        "horizons_days": list(horizons),
        "forcing_years": population_contract["forcing_years"],
        "observation_years": population_contract["observation_years"],
        "arrival_selection_policy_id": population_contract["arrival_selection_policy_id"],
        "replicates_per_stratum": population_contract["replicates_per_stratum"],
        "arrival_core_count": population_contract["arrival_core_count"],
        "arrival_event_count": population_contract["arrival_event_count"],
        "selection_support_days": selection_support_days,
        "runtime_support_days": runtime_support_days,
        "backtrack_modes": list(backtrack_modes),
        "step_count_formula": "ceil(days*86400/dt_min_seconds)+1",
        "dt_min_seconds": float(dt_min),
        "maximum_step_count_by_horizon": {str(days): step_counts[days] for days in horizons},
        "source_template_fingerprint": dict(source_template_fingerprint),
        "common_config_fingerprint": dict(common_config_fingerprint),
        "common_artifact_index_fingerprint": dict(component_fingerprints["artifact_index"]),
        "common_artifact_fingerprints": {
            kind: dict(component_fingerprints[kind]) for kind in sorted(ARTIFACT_FILENAMES)
        },
        "common_identity_fingerprints": release_identity["identity"],
        "input_validation_fingerprint": dict(input_validation_fingerprint),
        "input_validation_context": {"formal": formal, "roots_supplied": True},
        "releases": [dict(record) for record in release_records],
        "paths": _expected_suite_paths(),
        "input_build_count": 1,
        "recovery_method": recovery_method,
        "recovery_source_fingerprint": (
            dict(recovery_source_fingerprint) if recovery_source_fingerprint is not None else None
        ),
    }
    gap_contract = _payload_gap_censoring_contract(common_payload)
    if gap_contract is not None:
        # 六份 release 共用同一份 gap-censored common input；policy snapshot 放在
        # suite root，讓後續 validator 不必從任一 release 猜測科學分母語意。
        manifest.update(gap_contract)
    return manifest


def build_horizon_suite(
    config_template_path: str | Path,
    backtrack_days: Sequence[Any],
    destination: str | Path,
    ocm_native_root: str | Path,
    ocm_surface_root: str | Path,
    nww_analysis_root: str | Path,
    formal: bool = True,
) -> dict[str, Any]:
    """以最大選時包絡建立一次 common input，再原子發布 horizon×mode releases。

    ``backtrack_days`` 可為任意數量的唯一正整數；函式先排序，並把最大值寫入
    ``inputs.backtrack_support_days``。對 bed-residence template，selection support 是
    ``最大 horizon + maximum_age_days``，runtime support 仍是最大 horizon，並為每個
    horizon 產生兩種合法模式；例如 H30/H60/H90 產生六份 release。來源 template 僅讀取，
    套件內的 common config 是深拷貝；accepted forcing roots 只傳給 builder，不進入
    manifest。每個 release 的 step budget 只依 H 計算，即
    ``ceil(days*86400/dt_min_seconds)+1``；固定日曆模式的成員有效期間由 runtime 個別解析，
    不縮小全域 budget。中途任何一步失敗都保留 ``.partial-*`` 供人工稽核，且不發布
    destination；避免共享檔案系統上的自動清理競態刪除其他資料。
    """

    horizons = normalize_horizons(backtrack_days)
    maximum_horizon = max(horizons)
    # ``_read_yaml_snapshot`` 會以同一個 no-follow descriptor 驗證並讀取 template；
    # 不在此先做一次 path-based regular-file check，避免 assert 與真正 bytes 讀取之間
    # 出現可被替換的時間窗。
    template = Path(config_template_path)
    source_payload, source_fp, source_bytes = _read_yaml_snapshot(
        template,
        "config template",
        HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME,
    )
    common_payload = _derive_common_payload(source_payload, maximum_horizon)
    population_contract = _arrival_population_contract(common_payload)
    selection_support_days, runtime_support_days, backtrack_modes = _suite_support_contract(
        common_payload, maximum_horizon
    )
    integration = common_payload.get("integration")
    dt_min = integration.get("dt_min_seconds") if isinstance(integration, Mapping) else None

    destination_path = Path(destination)
    partial, parent_identity, partial_identity = _prepare_partial(destination_path)
    published = False
    try:
        source_copy = partial / HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME
        _atomic_write_bytes(source_copy, source_bytes, suffix=".yaml")
        common_config = partial / HORIZON_SUITE_COMMON_CONFIG_FILENAME
        common_bytes = yaml.safe_dump(
            common_payload, allow_unicode=True, sort_keys=False, default_flow_style=False
        ).encode("utf-8")
        _atomic_write_bytes(common_config, common_bytes, suffix=".yaml")
        common_fp = _yaml_fingerprint_from_snapshot(
            common_bytes, common_payload, HORIZON_SUITE_COMMON_CONFIG_FILENAME
        )
        common_input = partial / HORIZON_SUITE_COMMON_INPUT_DIRECTORY
        build_input_derivatives(
            config_path=common_config,
            destination=common_input,
            ocm_native_root=ocm_native_root,
            ocm_surface_root=ocm_surface_root,
            nww_analysis_root=nww_analysis_root,
            formal=formal,
            strict=True,
        )
        input_validation = validate_input_derivatives(
            common_input,
            config_path=common_config,
            formal=formal,
            ocm_native_root=ocm_native_root,
            ocm_surface_root=ocm_surface_root,
            nww_analysis_root=nww_analysis_root,
        )
        if not isinstance(input_validation, Mapping) or input_validation.get("valid") is not True:
            raise HorizonSuiteError("共同 input validator 未通過")
        validation_dir = partial / HORIZON_SUITE_VALIDATION_DIRECTORY
        validation_dir.mkdir(parents=True, exist_ok=True)
        input_validation_fp = write_canonical_json(
            validation_dir / "input.json", dict(input_validation)
        )
        component_fps = _component_fingerprints(common_input)
        artifact_index_payload, _ = _read_artifact_index(common_input)
        _validate_artifact_closure(common_input, component_fps)
        source_bindings = artifact_index_payload.get("source_bindings")
        expected_config_hash = _config_hash_from_payload(common_payload)
        if (
            not isinstance(source_bindings, Mapping)
            or source_bindings.get("config_hash") != expected_config_hash
        ):
            raise HorizonSuiteError("artifact_index source_bindings.config_hash 不符")
        step_counts = {days: maximum_step_count_for_horizon(days, dt_min) for days in horizons}
        release_records, release_identity = _build_releases(
            common_config=common_config,
            common_input=common_input,
            partial=partial,
            horizons=horizons,
            step_counts=step_counts,
            backtrack_modes=backtrack_modes,
            selection_support_days=selection_support_days,
            runtime_support_days=runtime_support_days,
            formal=formal,
        )
        manifest = _build_suite_manifest(
            source_template_fingerprint=source_fp,
            common_config_fingerprint=common_fp,
            common_payload=common_payload,
            input_validation_fingerprint=input_validation_fp,
            component_fingerprints=component_fps,
            release_records=release_records,
            release_identity=release_identity,
            horizons=horizons,
            population_contract=population_contract,
            selection_support_days=selection_support_days,
            runtime_support_days=runtime_support_days,
            backtrack_modes=backtrack_modes,
            dt_min=float(dt_min),
            step_counts=step_counts,
            formal=formal,
        )
        manifest_path = partial / HORIZON_SUITE_MANIFEST_FILENAME
        write_canonical_json(manifest_path, manifest)
        validation = validate_horizon_suite(
            partial,
            formal=formal,
            ocm_native_root=ocm_native_root,
            ocm_surface_root=ocm_surface_root,
            nww_analysis_root=nww_analysis_root,
            _allow_unpublished_partial=True,
        )
        if validation.get("valid") is not True:
            detail = ";".join(str(item) for item in validation.get("errors", [])[:8])
            raise HorizonSuiteError(f"horizon suite partial validator 未通過：{detail}")
        for directory_name in (
            HORIZON_SUITE_COMMON_INPUT_DIRECTORY,
            HORIZON_SUITE_RELEASE_DIRECTORY,
            HORIZON_SUITE_VALIDATION_DIRECTORY,
        ):
            _fsync_directory(partial / directory_name)
        _fsync_directory(partial)
        publication_started = False
        publication_result = _publish_partial(
            partial, destination_path, parent_identity, partial_identity
        )
        publication_started = True
        try:
            publication_method, destination_identity = _normalise_publication_result(
                publication_result, partial_identity
            )
            _write_publication_marker(
                destination_path,
                publication_method,
                parent_identity,
                destination_identity,
            )
            _fsync_directory(destination_path, expected_identity=destination_identity)
            # rename／mkdir、所有內容、marker 與 sidecar 都完成後才同步 parent directory。
            # parent identity 變動時保留成果但不宣稱耐久性已確認。
            _fsync_directory(destination_path.parent, expected_identity=parent_identity)
        except Exception as exc:
            # native rename 或 NFS reserved destination 已經對外佔用 basename；不能刪除
            # 或覆寫它來掩蓋 marker／durability 失敗。缺 marker 的目錄必由公開 validator
            # 拒絕，交由操作員依 runbook 選新 destination 或明確清理。
            if publication_started:
                raise RuntimeError("已發布但 durability 未確認（publication marker 亦未確認）") from exc
            raise
        published = True
        partial = Path()
        return {
            "destination": str(destination_path),
            "status": manifest["status"],
            "horizons_days": list(horizons),
            "selection_support_days": selection_support_days,
            "runtime_support_days": runtime_support_days,
            "backtrack_modes": list(backtrack_modes),
            "maximum_step_count_by_horizon": manifest["maximum_step_count_by_horizon"],
            "common_input_build_count": 1,
            "release_count": len(release_records),
            "publication_method": publication_method,
            "publication_marker": HORIZON_SUITE_PUBLICATION_MARKER_FILENAME,
            "validation": validation,
        }
    except Exception:
        if not published and partial != Path():
            _cleanup_partial(partial, partial_identity, parent_identity)
        raise


def _assert_tree_has_no_symlink_or_special_file(root: Path) -> None:
    """遞迴確認 recovery source tree 只含普通檔案與目錄。

    ``_assert_no_symlink_components`` 只能保護路徑的父層；partial 內部若藏有
    symbolic link，直接 ``copytree`` 可能把 caller 未授權的資料帶入新 release。因此
    recovery 在讀取與複製前逐項使用不追 symlink 的目錄掃描，並拒絕 FIFO、socket 與
    其他特殊節點。此檢查只讀取 metadata，不改寫原 partial。
    """

    _assert_regular_directory(root)
    for entry in os.scandir(root):
        entry_path = Path(entry.path)
        if entry.is_symlink():
            raise HorizonSuiteError("recovery partial 不得包含 symbolic link")
        if entry.is_dir(follow_symlinks=False):
            _assert_tree_has_no_symlink_or_special_file(entry_path)
        elif not entry.is_file(follow_symlinks=False):
            raise HorizonSuiteError("recovery partial 不得包含特殊檔案")


def _assert_recovery_partial_identity(partial: Path, destination: Path) -> None:
    """確認 caller 指定的是與 destination 同 parent 的 exact preserved partial。

    partial basename 必須符合本模組建立的 ``.<destination>.partial-*`` 形狀；不接受
    任意目錄、symbolic link 或另一個 parent 下的同名資料，避免 recovery 被用來把
    未核准來源偷偷搬入新的正式成果。
    """

    if os.path.abspath(os.fspath(partial.parent)) != os.path.abspath(
        os.fspath(destination.parent)
    ):
        raise HorizonSuiteError("recovery partial 與 destination 必須位於同一 parent")
    prefix = f".{destination.name}.partial-"
    if not partial.name.startswith(prefix) or partial.name == prefix:
        raise HorizonSuiteError("recovery partial basename 不符合 preserved partial 契約")
    _assert_no_symlink_components(partial, allow_missing_leaf=False)
    _assert_tree_has_no_symlink_or_special_file(partial)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("不可覆寫既有 horizon suite destination")


def _copy_verified_common_input(source: Path, destination: Path) -> None:
    """複製已通過 closure gate 的 common-input，絕不呼叫 input builder。

    source 已先由 ``_validate_artifact_directory``、component fingerprint 與
    ``validate_input_derivatives`` 驗證；這裡仍以 ``symlinks=False`` 複製普通檔案，並
    只把 immutable common-input 帶到新的 owned partial，不複製舊 releases 或 partial
    manifest。若來源在驗證後遭替換，後續新目錄中的 sidecar／validator 會 fail closed。
    """

    if destination.exists() or destination.is_symlink():
        raise FileExistsError("recovery 新 partial 的 common-input 已存在")
    shutil.copytree(source, destination, symlinks=False)
    _assert_tree_has_no_symlink_or_special_file(destination)


def resume_horizon_suite(
    preserved_partial: str | Path,
    destination: str | Path,
    backtrack_days: Sequence[Any],
    ocm_native_root: str | Path,
    ocm_surface_root: str | Path,
    nww_analysis_root: str | Path,
    formal: bool = True,
) -> dict[str, Any]:
    """從 exact preserved partial 重建 suite，重用 common-input 且不再次 input-build。

    recovery 先唯讀驗證 source-template、common-config、artifact index／closure、來源
    config hash 與三套 accepted forcing root；成功後才在 destination parent 建立另一個
    owned partial，複製已驗證 common-input，重新產生所有 release、validation 與 root
    manifest，再以既有 exclusive rename 原子發布。原 preserved partial 永不寫入、改名或
    刪除；destination 已存在、partial 有 symlink／tampering、caller 日數與 common
    config 不一致、或任何 formal gate 失敗都 fail closed。manifest 固定記錄
    ``input_build_count=1``、``resume_reuse_validated_common_input_v1`` 與不含絕對路徑的
    source fingerprint。
    """

    horizons = normalize_horizons(backtrack_days)
    maximum_horizon = max(horizons)
    partial_source = Path(preserved_partial)
    destination_path = Path(destination)
    _assert_recovery_partial_identity(partial_source, destination_path)

    source_path = partial_source / HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME
    common_path = partial_source / HORIZON_SUITE_COMMON_CONFIG_FILENAME
    common_input_source = partial_source / HORIZON_SUITE_COMMON_INPUT_DIRECTORY
    source_payload, source_fp, source_bytes = _read_yaml_snapshot(
        source_path,
        "recovery source template",
        HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME,
    )
    _assert_unbound_template(source_payload)
    common_payload, common_fp, common_bytes = _read_yaml_snapshot(
        common_path,
        "recovery common config",
        HORIZON_SUITE_COMMON_CONFIG_FILENAME,
    )
    expected_common = _derive_common_payload(source_payload, maximum_horizon)
    if common_payload != expected_common:
        raise HorizonSuiteError("recovery common config 與 source template／日數不一致")
    population_contract = _arrival_population_contract(common_payload)
    selection_support_days, runtime_support_days, backtrack_modes = _suite_support_contract(
        common_payload, maximum_horizon
    )
    integration = common_payload.get("integration")
    dt_min = integration.get("dt_min_seconds") if isinstance(integration, Mapping) else None
    if dt_min is None:
        raise HorizonSuiteError("recovery common config 缺少 integration.dt_min_seconds")
    _assert_regular_directory(common_input_source)
    expected_common_files = {filename for filename in ARTIFACT_FILENAMES.values()}
    expected_common_files.update(_COMMON_INPUT_EXTRA_FILENAMES)
    expected_common_files.update(
        f"{filename}.sha256" for filename in tuple(expected_common_files)
    )
    _assert_exact_entries(
        common_input_source,
        expected_files=expected_common_files,
        expected_directories=set(),
        label="recovery common-input",
    )
    input_directory_errors, _, _ = _validate_artifact_directory(common_input_source)
    if input_directory_errors:
        raise HorizonSuiteError("recovery common-input artifact directory 未通過")
    component_fps = _component_fingerprints(common_input_source)
    artifact_index_payload, _ = _read_artifact_index(common_input_source)
    _validate_artifact_closure(common_input_source, component_fps)
    expected_config_hash = _config_hash_from_payload(common_payload)
    source_bindings = artifact_index_payload.get("source_bindings")
    if not isinstance(source_bindings, Mapping) or source_bindings.get("config_hash") != expected_config_hash:
        raise HorizonSuiteError("recovery artifact_index source config hash 不符")

    # 若 preserved partial 已有 manifest，僅核對其日數與 common fingerprint；不採用其中
    # 的 releases／validation，避免把中斷時殘留或被竄改的 metadata 帶入新發布。
    old_manifest_path = partial_source / HORIZON_SUITE_MANIFEST_FILENAME
    if old_manifest_path.exists() or old_manifest_path.is_symlink():
        old_manifest, _ = read_canonical_json(old_manifest_path)
        if old_manifest.get("horizons_days") != list(horizons):
            raise HorizonSuiteError("recovery preserved manifest 日數與 caller 不一致")
        if old_manifest.get("common_config_fingerprint") != common_fp:
            raise HorizonSuiteError("recovery preserved manifest common config fingerprint 不符")
        if old_manifest.get("source_template_fingerprint") != source_fp:
            raise HorizonSuiteError("recovery preserved manifest source fingerprint 不符")
        if old_manifest.get("backtrack_modes") != list(backtrack_modes):
            raise HorizonSuiteError("recovery preserved manifest backtrack mode 不符")
        if old_manifest.get("selection_support_days") != selection_support_days:
            raise HorizonSuiteError("recovery preserved manifest selection support 不符")
        if old_manifest.get("runtime_support_days") != runtime_support_days:
            raise HorizonSuiteError("recovery preserved manifest runtime support 不符")
        if old_manifest.get("input_build_count") != 1:
            raise HorizonSuiteError("recovery preserved manifest input_build_count 不符")
        if old_manifest.get("common_artifact_index_fingerprint") != component_fps["artifact_index"]:
            raise HorizonSuiteError("recovery preserved manifest artifact index fingerprint 不符")

    input_validation = validate_input_derivatives(
        common_input_source,
        config_path=common_path,
        formal=formal,
        ocm_native_root=ocm_native_root,
        ocm_surface_root=ocm_surface_root,
        nww_analysis_root=nww_analysis_root,
    )
    if not isinstance(input_validation, Mapping) or input_validation.get("valid") is not True:
        raise HorizonSuiteError("recovery common-input formal validator 未通過")

    partial, parent_identity, partial_identity = _prepare_partial(destination_path)
    published = False
    try:
        _atomic_write_bytes(partial / HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME, source_bytes, suffix=".yaml")
        _atomic_write_bytes(partial / HORIZON_SUITE_COMMON_CONFIG_FILENAME, common_bytes, suffix=".yaml")
        _copy_verified_common_input(
            common_input_source,
            partial / HORIZON_SUITE_COMMON_INPUT_DIRECTORY,
        )
        validation_dir = partial / HORIZON_SUITE_VALIDATION_DIRECTORY
        validation_dir.mkdir(parents=True, exist_ok=True)
        input_validation_fp = write_canonical_json(
            validation_dir / "input.json", dict(input_validation)
        )
        step_counts = {days: maximum_step_count_for_horizon(days, dt_min) for days in horizons}
        release_records, release_identity = _build_releases(
            common_config=partial / HORIZON_SUITE_COMMON_CONFIG_FILENAME,
            common_input=partial / HORIZON_SUITE_COMMON_INPUT_DIRECTORY,
            partial=partial,
            horizons=horizons,
            step_counts=step_counts,
            backtrack_modes=backtrack_modes,
            selection_support_days=selection_support_days,
            runtime_support_days=runtime_support_days,
            formal=formal,
        )
        recovery_fingerprint = _recovery_source_fingerprint(source_fp, common_fp, component_fps)
        manifest = _build_suite_manifest(
            source_template_fingerprint=source_fp,
            common_config_fingerprint=common_fp,
            common_payload=common_payload,
            input_validation_fingerprint=input_validation_fp,
            component_fingerprints=component_fps,
            release_records=release_records,
            release_identity=release_identity,
            horizons=horizons,
            population_contract=population_contract,
            selection_support_days=selection_support_days,
            runtime_support_days=runtime_support_days,
            backtrack_modes=backtrack_modes,
            dt_min=float(dt_min),
            step_counts=step_counts,
            formal=formal,
            recovery_method="resume_reuse_validated_common_input_v1",
            recovery_source_fingerprint=recovery_fingerprint,
        )
        write_canonical_json(partial / HORIZON_SUITE_MANIFEST_FILENAME, manifest)
        validation = validate_horizon_suite(
            partial,
            formal=formal,
            ocm_native_root=ocm_native_root,
            ocm_surface_root=ocm_surface_root,
            nww_analysis_root=nww_analysis_root,
            _allow_unpublished_partial=True,
        )
        if validation.get("valid") is not True:
            detail = ";".join(str(item) for item in validation.get("errors", [])[:8])
            raise HorizonSuiteError(f"horizon suite recovery validator 未通過：{detail}")
        for directory_name in (
            HORIZON_SUITE_COMMON_INPUT_DIRECTORY,
            HORIZON_SUITE_RELEASE_DIRECTORY,
            HORIZON_SUITE_VALIDATION_DIRECTORY,
        ):
            _fsync_directory(partial / directory_name)
        _fsync_directory(partial)
        publication_started = False
        publication_result = _publish_partial(
            partial, destination_path, parent_identity, partial_identity
        )
        publication_started = True
        try:
            publication_method, destination_identity = _normalise_publication_result(
                publication_result, partial_identity
            )
            _write_publication_marker(
                destination_path,
                publication_method,
                parent_identity,
                destination_identity,
            )
            _fsync_directory(destination_path, expected_identity=destination_identity)
            _fsync_directory(destination_path.parent, expected_identity=parent_identity)
        except Exception as exc:
            if publication_started:
                raise RuntimeError("已發布但 durability 未確認（publication marker 亦未確認）") from exc
            raise
        published = True
        partial = Path()
        return {
            "destination": str(destination_path),
            "status": manifest["status"],
            "horizons_days": list(horizons),
            "selection_support_days": selection_support_days,
            "runtime_support_days": runtime_support_days,
            "backtrack_modes": list(backtrack_modes),
            "maximum_step_count_by_horizon": manifest["maximum_step_count_by_horizon"],
            "common_input_build_count": 1,
            "release_count": len(release_records),
            "recovery_method": manifest["recovery_method"],
            "publication_method": publication_method,
            "publication_marker": HORIZON_SUITE_PUBLICATION_MARKER_FILENAME,
            "validation": validation,
        }
    except Exception:
        if not published and partial != Path():
            _cleanup_partial(partial, partial_identity, parent_identity)
        raise


__all__ = [
    "HORIZON_SUITE_COMMON_CONFIG_FILENAME",
    "HORIZON_SUITE_COMMON_INPUT_DIRECTORY",
    "HORIZON_SUITE_MANIFEST_FILENAME",
    "HORIZON_SUITE_METHOD_ID",
    "HORIZON_SUITE_NATIVE_PUBLICATION_METHOD",
    "HORIZON_SUITE_NFS_PUBLICATION_METHOD",
    "HORIZON_SUITE_POLICY_ID",
    "HORIZON_SUITE_PUBLICATION_MARKER_FILENAME",
    "HORIZON_SUITE_PUBLICATION_POLICY_ID",
    "HORIZON_SUITE_PUBLICATION_POLICY_VERSION",
    "HORIZON_SUITE_RELEASE_DIRECTORY",
    "HORIZON_SUITE_SCHEMA_VERSION",
    "HORIZON_SUITE_SOURCE_TEMPLATE_FILENAME",
    "HORIZON_SUITE_VALIDATION_DIRECTORY",
    "HorizonSuiteError",
    "build_horizon_suite",
    "resume_horizon_suite",
    "maximum_step_count_for_horizon",
    "normalize_horizons",
    "validate_horizon_suite",
]
