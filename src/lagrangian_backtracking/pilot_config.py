"""由 calibration evidence 建立可供 pilot runtime 使用的 generated config。

本模組只處理已驗證 release YAML、Slice 3B1 calibration evidence 與 caller 明示工程
scalar 的綁定，不讀取 OCM／NWW 大型陣列、不建立 accepted product，也不執行粒子軌跡。
輸出的設定仍標記為 ``generated`` 與 ``candidate_pending_dt_and_member_convergence``；
它只是把候選值與執行參數封裝成現有 ``run-create``／``run-shard`` 可讀的 immutable
設定，不能被解讀成已通過 trajectory、time-step 或 ensemble convergence。

所有外部 component path 都沿用 input derivation 已登錄的固定檔名，並以相對於 target
config parent 的路徑寫入既有 ``release_binding``。新的 ``pilot_execution_binding`` 只
保存 hash、候選值、套用欄位與 scalar snapshot，不保存任何檔案或目錄 path；因此 target
config 搬到不同 parent 時，runtime 參照會重建，而 calibration/source identity 仍由 hash
固定。建立採 hidden temporary YAML、file fsync 與 atomic ``os.replace``，失敗時不留下
destination。
"""

from __future__ import annotations

import copy
import math
import os
import tempfile
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import yaml

from .config import ProjectConfig, load_config
from .input_derivation import (
    ARTIFACT_FILENAMES,
    DERIVED_INPUT_SCHEMA_VERSION,
    _assert_no_symlink_components,
    _assert_regular_directory,
    _assert_regular_file,
    _replace_manifest_references,
    read_canonical_json,
    validate_release_config,
)
from .pilot_calibration import (
    PILOT_CALIBRATION_ARTIFACT_KIND,
    PILOT_CALIBRATION_SCHEMA_VERSION,
    _read_json,
    validate_pilot_calibration,
)

PILOT_EXECUTION_BINDING_SCHEMA_VERSION: Final[str] = "1.0.0"
"""pilot execution binding 的固定 schema 版本；與 input/calibration schema 分開管理。"""

PILOT_EXECUTION_BINDING_STATUS: Final[str] = "candidate_pending_dt_and_member_convergence"
"""候選參數已綁定、但尚未通過 dt 與 member convergence 的狀態。"""

_CANDIDATE_NAMES: Final[tuple[str, ...]] = (
    "constant_kh_m2ps",
    "constant_kz_m2ps",
    "floor_m2ps",
    "cap_m2ps",
)
_CANDIDATE_TARGET_FIELDS: Final[dict[str, str]] = {
    "constant_kh_m2ps": "physics.horizontal_diffusion.constant_kh_m2ps",
    "constant_kz_m2ps": "physics.vertical_diffusion.constant_kz_m2ps",
    "floor_m2ps": "physics.horizontal_diffusion.smagorinsky.kh_floor_m2ps",
    "cap_m2ps": "physics.horizontal_diffusion.smagorinsky.kh_cap_m2ps",
}
_EXECUTION_SCALAR_NAMES: Final[tuple[str, ...]] = (
    "dt_min_seconds",
    "dt_max_seconds",
    "output_interval_seconds",
    "max_backtrack_days",
    "maximum_step_count",
    "members_per_scenario",
    "master_seed",
    "shard_scenario_count",
    "checkpoint_interval_sweeps",
    "active_chunk_size",
    "max_resident_forcing_months",
)
_PILOT_EXECUTION_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "source_config_hash",
        "input_artifact_index_sha256",
        "calibration_schema_version",
        "calibration_manifest_sha256",
        "calibration_report_sha256",
        "pair_samples_sha256",
        "candidate_values",
        "applied_fields",
        "execution_scalar_snapshot",
        "status",
    }
)


def _strict_float(value: object, *, label: str, positive: bool = False) -> float:
    """驗證 caller／YAML scalar 是原生有限數值並轉為 Python ``float``。

    ``bool`` 不可沿用 Python 的整數子類別語意；NumPy scalar 也不在本 binding 的原生
    scalar 契約內。所有時間、尺度與擴散值最後以有限 Python float 保存，避免 YAML
    round-trip 將不可序列化或非有限值帶入 runtime。
    """

    if type(value) not in {int, float}:
        raise ValueError(f"{label} 必須是原生 int/float")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} 必須是有限數值")
    if positive and normalized <= 0.0:
        raise ValueError(f"{label} 必須大於 0")
    return normalized


def _strict_int(value: object, *, label: str, positive: bool = False) -> int:
    """驗證 caller／YAML scalar 是原生整數，拒絕 bool、NumPy 整數與非法符號。"""

    if type(value) is not int:
        raise ValueError(f"{label} 必須是原生 int")
    if positive and value < 1:
        raise ValueError(f"{label} 必須是正整數")
    if not positive and value < 0:
        raise ValueError(f"{label} 必須是非負整數")
    return value


def _normalise_execution_scalars(
    *,
    dt_min_seconds: object,
    dt_max_seconds: object,
    output_interval_seconds: object,
    max_backtrack_days: object,
    maximum_step_count: object,
    members_per_scenario: object,
    master_seed: object,
    shard_scenario_count: object,
    checkpoint_interval_sweeps: object,
    active_chunk_size: object,
    max_resident_forcing_months: object,
) -> dict[str, float | int | None]:
    """驗證並正規化全部 pilot 執行 scalar，回傳可直接寫入 YAML 的 snapshot。

    ``dt``、輸出間隔與回溯期採有限正浮點數；M、shard、checkpoint、active chunk 與
    resident month 採正整數，master seed 採非負整數，active chunk 的 ``None`` 必須由
    caller 明示。步數下限只保證在 ``dt_max`` 下有足夠步數覆蓋 horizon，並非充分的
    numerical convergence proof，因此此函式不替 caller 決定更大的 maximum step count。
    """

    dt_min = _strict_float(dt_min_seconds, label="dt_min_seconds", positive=True)
    dt_max = _strict_float(dt_max_seconds, label="dt_max_seconds", positive=True)
    output_interval = _strict_float(
        output_interval_seconds,
        label="output_interval_seconds",
        positive=True,
    )
    horizon_days = _strict_float(max_backtrack_days, label="max_backtrack_days", positive=True)
    maximum_steps = _strict_int(maximum_step_count, label="maximum_step_count", positive=True)
    members = _strict_int(members_per_scenario, label="members_per_scenario", positive=True)
    seed = _strict_int(master_seed, label="master_seed", positive=False)
    shard_count = _strict_int(shard_scenario_count, label="shard_scenario_count", positive=True)
    checkpoint_sweeps = _strict_int(
        checkpoint_interval_sweeps,
        label="checkpoint_interval_sweeps",
        positive=True,
    )
    if active_chunk_size is None:
        active_chunk: int | None = None
    else:
        active_chunk = _strict_int(active_chunk_size, label="active_chunk_size", positive=True)
    resident_months = _strict_int(
        max_resident_forcing_months,
        label="max_resident_forcing_months",
        positive=True,
    )
    if not (dt_min <= dt_max <= output_interval):
        raise ValueError("dt_min_seconds、dt_max_seconds、output_interval_seconds 順序無效")
    required_seconds = horizon_days * 86_400.0
    if not math.isfinite(required_seconds):
        raise ValueError("max_backtrack_days 產生不可表示的秒數")
    required_steps = math.ceil(required_seconds / dt_max)
    if maximum_steps < required_steps:
        raise ValueError("maximum_step_count 不足以覆蓋 max_backtrack_days")
    return {
        "dt_min_seconds": dt_min,
        "dt_max_seconds": dt_max,
        "output_interval_seconds": output_interval,
        "max_backtrack_days": horizon_days,
        "maximum_step_count": maximum_steps,
        "members_per_scenario": members,
        "master_seed": seed,
        "shard_scenario_count": shard_count,
        "checkpoint_interval_sweeps": checkpoint_sweeps,
        "active_chunk_size": active_chunk,
        "max_resident_forcing_months": resident_months,
    }


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """以 safe YAML 讀取 mapping root；不允許以任意 YAML object 取代設定資料。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config YAML root 必須是 mapping")
    return payload


def _sha256_regular_file(path: Path) -> str:
    """以固定 chunk 讀取普通檔案，回傳 raw bytes SHA-256，不載入大型檔案。"""

    source = _assert_regular_file(path)
    digest = sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_input_artifact_fingerprints(input_root: Path) -> tuple[dict[str, dict[str, Any]], str]:
    """依固定 ``ARTIFACT_FILENAMES`` 讀取全部 component fingerprint 與 index hash。

    此函式不採用 artifact index 內的任意 path；固定檔名來自 Slice 1 的資料契約，且
    ``read_canonical_json`` 會同步檢查每個 component 的 sidecar binding。回傳值供 target
    release binding 重建，確保 config 搬家後仍指向同一份 accepted input。
    """

    fingerprints: dict[str, dict[str, Any]] = {}
    for kind, filename in sorted(ARTIFACT_FILENAMES.items()):
        _payload, fingerprint = read_canonical_json(input_root / filename)
        fingerprints[kind] = dict(fingerprint)
    _index_payload, index_fingerprint = read_canonical_json(input_root / "artifact_index.json")
    return fingerprints, str(index_fingerprint["sha256"])


def _relative_artifact_path(input_root: Path, config_parent: Path, filename: str) -> str:
    """建立相對於 target config parent 的固定 component path，不產生絕對 path。"""

    relative_directory = Path(os.path.relpath(input_root, config_parent))
    return (relative_directory / filename).as_posix()


def _rebuild_release_binding(
    payload: dict[str, Any],
    *,
    input_root: Path,
    target_path: Path,
    fingerprints: Mapping[str, Mapping[str, Any]],
    artifact_index_sha256: str,
) -> dict[str, Any]:
    """重建既有 release binding 的相對 path 與 exact component fingerprint。

    release binding 仍由 input derivation 的 schema 1.0.0 管理；本模組只把來源 config 的
    binding 搬到 target parent，並以實際 accepted component fingerprint 覆蓋 path／hash。
    candidate 與 execution scalar 不寫入此 mapping，而是集中在獨立 pilot binding，讓
    downstream release validator 與 pilot validator 的責任邊界清楚。
    """

    source_binding = payload.get("release_binding")
    if not isinstance(source_binding, Mapping):
        raise ValueError("source config 缺少 release_binding")
    binding = copy.deepcopy(dict(source_binding))
    artifact_records: list[dict[str, Any]] = []
    for kind, filename in sorted(ARTIFACT_FILENAMES.items()):
        fingerprint = dict(fingerprints[kind])
        fingerprint["kind"] = kind
        fingerprint["path"] = _relative_artifact_path(input_root, target_path.parent, filename)
        artifact_records.append(fingerprint)
    binding.update(
        {
            "schema_version": DERIVED_INPUT_SCHEMA_VERSION,
            "input_directory_artifact_index_sha256": artifact_index_sha256,
            "artifacts": artifact_records,
            "approved_only_after_exact_hash_validation": True,
        }
    )
    return binding


def _candidate_values_from_report(report: Mapping[str, Any]) -> dict[str, float]:
    """從已驗證 1.1.0 report 取出四個候選值，拒絕缺值、非有限或負值。

    candidate value 直接來自 calibration report；不在這裡重新估計 quantile，也不使用
    source config 的 placeholder。這樣 target config 的物理欄位可以由 binding hash 與
    validator 精確追溯到 pair table 統計結果。
    """

    raw_candidates = report.get("candidates")
    if not isinstance(raw_candidates, Mapping):
        raise ValueError("calibration report candidates 不合法")
    result: dict[str, float] = {}
    for name in _CANDIDATE_NAMES:
        candidate = raw_candidates.get(name)
        if not isinstance(candidate, Mapping) or candidate.get("available") is not True:
            raise ValueError(f"calibration candidate unavailable: {name}")
        value = _strict_float(candidate.get("value"), label=f"candidate.{name}")
        if value < 0.0:
            raise ValueError(f"candidate.{name} 必須是非負值")
        result[name] = value
    if result["floor_m2ps"] > result["cap_m2ps"]:
        raise ValueError("calibration floor_m2ps 不可大於 cap_m2ps")
    return result


def _read_gap_safe_horizon(
    input_root: Path,
    *,
    requested_max_backtrack_days: float,
) -> dict[str, Any]:
    """驗證 accepted gap-safe arrival root 與每筆 record 足以支援 requested horizon。

    這裡只讀取既有 ``ocm_gap_safe_arrival.json``，不重建時間軸、不補值，也不把
    ``crossed_gap`` 或 ``missing_utc`` 轉成可接受狀態。root 與每筆 record 都必須明示
    至少 requested horizon，且每個 expected step 均有 support；任一缺口立即拒絕。
    """

    path = input_root / ARTIFACT_FILENAMES["ocm_gap_safe_arrival_horizon"]
    payload, _fingerprint = read_canonical_json(path)
    if payload.get("manifest_kind") != "ocm_gap_safe_arrival_horizon_manifest":
        raise ValueError("gap-safe manifest kind 不符")
    if payload.get("schema_version") != DERIVED_INPUT_SCHEMA_VERSION:
        raise ValueError("gap-safe schema 不符")
    root_days = _strict_float(
        payload.get("max_backtrack_days"),
        label="gap_safe.root.max_backtrack_days",
        positive=True,
    )
    if requested_max_backtrack_days > root_days:
        raise ValueError("requested max_backtrack_days 超過 gap-safe root horizon")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("gap-safe records 不可為空")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("gap-safe record 必須是 mapping")
        record_days = _strict_float(
            record.get("max_backtrack_days"),
            label="gap_safe.record.max_backtrack_days",
            positive=True,
        )
        if requested_max_backtrack_days > record_days:
            raise ValueError("requested max_backtrack_days 超過 gap-safe record horizon")
        if record.get("crossed_gap") is not False:
            raise ValueError("gap-safe record crossed_gap 必須是 false")
        if record.get("missing_utc") != []:
            raise ValueError("gap-safe record missing_utc 必須是空清單")
        expected_steps = record.get("expected_step_count")
        supported_steps = record.get("supported_step_count")
        if (
            type(expected_steps) is not int
            or expected_steps < 1
            or type(supported_steps) is not int
            or supported_steps != expected_steps
        ):
            raise ValueError("gap-safe record step count 必須完整支援")
    return payload


def _build_pilot_payload(
    source_payload: Mapping[str, Any],
    *,
    target_path: Path,
    input_root: Path,
    fingerprints: Mapping[str, Mapping[str, Any]],
    artifact_index_sha256: str,
    source_config_hash: str,
    calibration_report: Mapping[str, Any],
    calibration_hashes: Mapping[str, str],
    scalar_snapshot: Mapping[str, float | int | None],
    candidate_values: Mapping[str, float],
) -> dict[str, Any]:
    """將 source release payload、calibration candidate 與 scalar 組成 target YAML mapping。"""

    # input derivation 的 helper 以固定 ARTIFACT_FILENAMES 重新寫 runtime component refs；
    # 因此 target config 即使與 source 不同 parent，也不會繼續讀 source-relative path。
    payload = _replace_manifest_references(
        copy.deepcopy(dict(source_payload)),
        config_output=target_path,
        input_directory=input_root,
    )
    payload["config_status"] = "generated"

    integration = payload.get("integration")
    boundaries = payload.get("boundaries")
    scenarios = payload.get("scenarios")
    execution = payload.get("execution")
    physics = payload.get("physics")
    if not all(isinstance(item, dict) for item in (integration, boundaries, scenarios, execution, physics)):
        raise ValueError("source config 缺少 runtime mapping")
    horizontal = physics.get("horizontal_diffusion")
    vertical = physics.get("vertical_diffusion")
    if not isinstance(horizontal, dict) or not isinstance(vertical, dict):
        raise ValueError("source config 缺少 horizontal/vertical diffusion mapping")
    smagorinsky = horizontal.get("smagorinsky")
    if not isinstance(smagorinsky, dict):
        raise ValueError("source config 缺少 Smagorinsky mapping")

    horizontal["constant_kh_m2ps"] = candidate_values["constant_kh_m2ps"]
    smagorinsky["kh_floor_m2ps"] = candidate_values["floor_m2ps"]
    smagorinsky["kh_cap_m2ps"] = candidate_values["cap_m2ps"]
    vertical["constant_kz_m2ps"] = candidate_values["constant_kz_m2ps"]

    integration.update(
        {
            "dt_min_seconds": scalar_snapshot["dt_min_seconds"],
            "dt_max_seconds": scalar_snapshot["dt_max_seconds"],
            "output_interval_seconds": scalar_snapshot["output_interval_seconds"],
        }
    )
    boundaries.update(
        {
            "max_backtrack_days": scalar_snapshot["max_backtrack_days"],
            "maximum_step_count": scalar_snapshot["maximum_step_count"],
        }
    )
    scenarios.update(
        {
            "members_per_scenario": scalar_snapshot["members_per_scenario"],
            "master_seed": scalar_snapshot["master_seed"],
        }
    )
    execution.update(
        {
            "shard_scenario_count": scalar_snapshot["shard_scenario_count"],
            "checkpoint_interval_sweeps": scalar_snapshot["checkpoint_interval_sweeps"],
            "active_chunk_size": scalar_snapshot["active_chunk_size"],
            "max_resident_forcing_months": scalar_snapshot["max_resident_forcing_months"],
        }
    )

    # release_binding 保留 source 的 provenance 欄位，但所有可執行 component path/hash
    # 都以 target parent 與目前 input root 重建；不把 calibration candidate 混進 input binding。
    payload["release_binding"] = _rebuild_release_binding(
        payload,
        input_root=input_root,
        target_path=target_path,
        fingerprints=fingerprints,
        artifact_index_sha256=artifact_index_sha256,
    )
    approval = payload.get("release_approval")
    approval_payload = copy.deepcopy(dict(approval)) if isinstance(approval, Mapping) else {}
    approval_payload.update(
        {
            "status": "generated",
            "pilot_execution_status": PILOT_EXECUTION_BINDING_STATUS,
            "blockers": [
                "calibration_candidate_bound",
                "trajectory_convergence_pending",
                "dt_and_member_convergence_pending",
            ],
        }
    )
    payload["release_approval"] = approval_payload
    payload["pilot_execution_binding"] = {
        "schema_version": PILOT_EXECUTION_BINDING_SCHEMA_VERSION,
        "source_config_hash": source_config_hash,
        "input_artifact_index_sha256": artifact_index_sha256,
        "calibration_schema_version": calibration_report["schema_version"],
        "calibration_manifest_sha256": calibration_hashes["manifest_sha256"],
        "calibration_report_sha256": calibration_hashes["report_sha256"],
        "pair_samples_sha256": calibration_hashes["pair_samples_sha256"],
        "candidate_values": dict(candidate_values),
        "applied_fields": dict(_CANDIDATE_TARGET_FIELDS),
        "execution_scalar_snapshot": dict(scalar_snapshot),
        "status": PILOT_EXECUTION_BINDING_STATUS,
    }
    return payload


def _write_hidden_yaml(destination: Path, payload: Mapping[str, Any]) -> Path:
    """在 destination 同一 parent 寫入已 fsync 的 hidden YAML temporary file。"""

    _assert_no_symlink_components(destination.parent, allow_missing_leaf=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("destination 已存在")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(destination.parent, allow_missing_leaf=False)
    rendered = yaml.safe_dump(
        dict(payload),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.partial-",
            suffix=".yaml",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        assert temporary is not None
        return temporary
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _fsync_file(path: Path) -> None:
    """在 atomic rename 後再次同步 target file，確保 caller 收到的檔案已落盤。"""

    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _strict_mapping_equal(left: object, right: object) -> bool:
    """以型別與值共同比較 binding snapshot，避免 1、1.0 或 bool 隱式相等。"""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _strict_mapping_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _strict_mapping_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _binding_has_path(value: object) -> bool:
    """遞迴拒絕 pilot binding 中的絕對、Unix 或 Windows path 字串。"""

    if isinstance(value, Mapping):
        return any(_binding_has_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_binding_has_path(item) for item in value)
    if isinstance(value, str):
        return Path(value).is_absolute() or "/" in value or "\\" in value
    return False


def _target_scalar_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """從 raw YAML target 取出所有必填 scalar，保留 None 以檢查 active chunk 是否明示。"""

    integration = payload.get("integration")
    boundaries = payload.get("boundaries")
    scenarios = payload.get("scenarios")
    execution = payload.get("execution")
    if not all(isinstance(item, Mapping) for item in (integration, boundaries, scenarios, execution)):
        raise ValueError("target config runtime mapping 不完整")
    required_fields = {
        "integration": ("dt_min_seconds", "dt_max_seconds", "output_interval_seconds"),
        "boundaries": ("max_backtrack_days", "maximum_step_count"),
        "scenarios": ("members_per_scenario", "master_seed"),
        "execution": (
            "shard_scenario_count",
            "checkpoint_interval_sweeps",
            "active_chunk_size",
            "max_resident_forcing_months",
        ),
    }
    for label, names in required_fields.items():
        section = payload[label]
        if any(name not in section for name in names):
            raise ValueError(f"target config {label} scalar 欄位不完整")
    return {
        "dt_min_seconds": integration.get("dt_min_seconds"),
        "dt_max_seconds": integration.get("dt_max_seconds"),
        "output_interval_seconds": integration.get("output_interval_seconds"),
        "max_backtrack_days": boundaries.get("max_backtrack_days"),
        "maximum_step_count": boundaries.get("maximum_step_count"),
        "members_per_scenario": scenarios.get("members_per_scenario"),
        "master_seed": scenarios.get("master_seed"),
        "shard_scenario_count": execution.get("shard_scenario_count"),
        "checkpoint_interval_sweeps": execution.get("checkpoint_interval_sweeps"),
        "active_chunk_size": execution.get("active_chunk_size"),
        "max_resident_forcing_months": execution.get("max_resident_forcing_months"),
    }


def _target_candidate_values(payload: Mapping[str, Any]) -> dict[str, object]:
    """從 raw YAML 讀取四個 runtime candidate target fields，避免只信任 Pydantic coercion。"""

    physics = payload.get("physics")
    if not isinstance(physics, Mapping):
        raise ValueError("target config physics mapping 不完整")
    horizontal = physics.get("horizontal_diffusion")
    vertical = physics.get("vertical_diffusion")
    if not isinstance(horizontal, Mapping) or not isinstance(vertical, Mapping):
        raise ValueError("target config diffusion mapping 不完整")
    smagorinsky = horizontal.get("smagorinsky")
    if not isinstance(smagorinsky, Mapping):
        raise ValueError("target config Smagorinsky mapping 不完整")
    return {
        "constant_kh_m2ps": horizontal.get("constant_kh_m2ps"),
        "constant_kz_m2ps": vertical.get("constant_kz_m2ps"),
        "floor_m2ps": smagorinsky.get("kh_floor_m2ps"),
        "cap_m2ps": smagorinsky.get("kh_cap_m2ps"),
    }


def _hashes_for_calibration(calibration_root: Path) -> dict[str, str]:
    """計算 calibration 三個固定 payload 的 raw SHA-256。"""

    return {
        "manifest_sha256": _sha256_regular_file(calibration_root / "manifest.json"),
        "report_sha256": _sha256_regular_file(calibration_root / "calibration_report.json"),
        "pair_samples_sha256": _sha256_regular_file(calibration_root / "pair_samples.parquet"),
    }


def _load_calibration_report(calibration_root: Path) -> dict[str, Any]:
    """嚴格讀取 calibration report，保留 report validator 已確認的 JSON mapping。"""

    return _read_json(calibration_root / "calibration_report.json")


def _calibration_metadata_gate(report: Mapping[str, Any]) -> dict[str, float]:
    """檢查只有完整 1.1.0 candidate evidence 才能進入 pilot config builder。"""

    if report.get("schema_version") != PILOT_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("calibration schema 必須是目前 1.1.0")
    if report.get("artifact_kind") != PILOT_CALIBRATION_ARTIFACT_KIND:
        raise ValueError("calibration artifact kind 不符")
    if report.get("evidence_class") != "server_real_data_pilot_candidate":
        raise ValueError("calibration evidence class 不符")
    if report.get("completion_status") != "complete":
        raise ValueError("calibration evidence 尚未 complete")
    if report.get("recommendation_status") != (
        "candidate_pending_trajectory_convergence_and_scientific_validation"
    ):
        raise ValueError("calibration recommendation status 不符")
    return _candidate_values_from_report(report)


def _source_and_calibration_gate(
    *,
    source_config: Path,
    input_root: Path,
    calibration_root: Path,
) -> tuple[ProjectConfig, dict[str, Any], dict[str, float], dict[str, str], str, str]:
    """執行 source release、1.1 calibration、candidate 與可重算的 hash gate。

    source semantic config hash 由 calibration report 的
    ``input_binding.config_hash`` 驗證；其餘保存的 hash 分別來自 input artifact index
    與三個 calibration payload。這裡不保存無法由公開 validator 重新取得的 source
    config 檔案雜湊，避免 binding 產生不可驗證的證據欄位。
    """

    source = load_config(source_config, formal_release=False)
    source_release = validate_release_config(
        source_config,
        input_directory=input_root,
        formal=False,
    )
    if source_release.get("valid") is not True:
        raise ValueError("source release config gate failed")
    calibration_validation = validate_pilot_calibration(
        calibration_root,
        config_path=source_config,
        input_directory=input_root,
    )
    if calibration_validation.get("valid") is not True:
        raise ValueError("calibration artifact gate failed")
    report = _load_calibration_report(calibration_root)
    candidate_values = _calibration_metadata_gate(report)
    source_config_hash = source.config_hash()
    calibration_hashes = _hashes_for_calibration(calibration_root)
    report_binding = report.get("input_binding")
    if not isinstance(report_binding, Mapping):
        raise ValueError("calibration report input binding 缺失")
    if report_binding.get("config_hash") != source_config_hash:
        raise ValueError("calibration report source config hash 不一致")
    _input_payload, input_fingerprint = read_canonical_json(input_root / "artifact_index.json")
    if report_binding.get("input_artifact_index_sha256") != input_fingerprint.get("sha256"):
        raise ValueError("calibration report input artifact hash 不一致")
    return (
        source,
        report,
        candidate_values,
        calibration_hashes,
        source_config_hash,
        str(input_fingerprint["sha256"]),
    )


def create_pilot_execution_config(
    source_config_path: str | Path,
    input_directory: str | Path,
    calibration_directory: str | Path,
    destination: str | Path,
    *,
    dt_min_seconds: float,
    dt_max_seconds: float,
    output_interval_seconds: float,
    max_backtrack_days: float,
    maximum_step_count: int,
    members_per_scenario: int,
    master_seed: int,
    shard_scenario_count: int,
    checkpoint_interval_sweeps: int,
    active_chunk_size: int | None,
    max_resident_forcing_months: int,
) -> dict[str, Any]:
    """由完整 source release 與 1.1.0 calibration 建立 generated pilot execution config。

    三類輸入都在寫檔前驗證：source release 必須通過 input binding、calibration 必須
    是完整 1.1.0 candidate evidence、gap-safe component 必須支援 caller 指定 horizon，
    並且所有 engineering scalar 必須通過原生型別與步數下限。候選值只從 report 讀入，
    不人工重算或複製 Kh/Kz/floor/cap。target 先在 hidden temporary YAML 上以
    ``load_config``、``validate_release_config`` 與本模組 validator 重算；全部成功後才
    以同一 parent 的 atomic rename 發布。回傳 mapping 不包含大型資料，只含 JSON-safe
    summary 與 target hash。
    """

    source_config = _assert_regular_file(source_config_path)
    input_root = _assert_regular_directory(input_directory)
    calibration_root = _assert_regular_directory(calibration_directory)
    destination_path = Path(destination)
    _assert_no_symlink_components(destination_path.parent, allow_missing_leaf=True)
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError("pilot config destination 已存在")

    scalar_snapshot = _normalise_execution_scalars(
        dt_min_seconds=dt_min_seconds,
        dt_max_seconds=dt_max_seconds,
        output_interval_seconds=output_interval_seconds,
        max_backtrack_days=max_backtrack_days,
        maximum_step_count=maximum_step_count,
        members_per_scenario=members_per_scenario,
        master_seed=master_seed,
        shard_scenario_count=shard_scenario_count,
        checkpoint_interval_sweeps=checkpoint_interval_sweeps,
        active_chunk_size=active_chunk_size,
        max_resident_forcing_months=max_resident_forcing_months,
    )
    requested_days = float(scalar_snapshot["max_backtrack_days"])
    _read_gap_safe_horizon(input_root, requested_max_backtrack_days=requested_days)
    (
        _source,
        calibration_report,
        candidate_values,
        calibration_hashes,
        source_config_hash,
        artifact_index_sha256,
    ) = _source_and_calibration_gate(
        source_config=source_config,
        input_root=input_root,
        calibration_root=calibration_root,
    )
    source_payload = _load_yaml_mapping(source_config)
    fingerprints, rebuilt_index_sha256 = _read_input_artifact_fingerprints(input_root)
    if rebuilt_index_sha256 != artifact_index_sha256:
        raise ValueError("input artifact index hash changed during build")
    payload = _build_pilot_payload(
        source_payload,
        target_path=destination_path,
        input_root=input_root,
        fingerprints=fingerprints,
        artifact_index_sha256=artifact_index_sha256,
        source_config_hash=source_config_hash,
        calibration_report=calibration_report,
        calibration_hashes=calibration_hashes,
        scalar_snapshot=scalar_snapshot,
        candidate_values=candidate_values,
    )

    temporary: Path | None = None
    published = False
    try:
        temporary = _write_hidden_yaml(destination_path, payload)
        # temporary 與 destination 同 parent，故 release binding 的相對 path 與最終檔案
        # 一致；先驗證 hidden file 可避免半成品出現在 operator 可見的 final 名稱。
        load_config(temporary, formal_release=False)
        release_validation = validate_release_config(
            temporary,
            input_directory=input_root,
            formal=False,
        )
        if release_validation.get("valid") is not True:
            raise ValueError("target release binding gate failed")
        own_validation = validate_pilot_execution_config(
            temporary,
            input_directory=input_root,
            calibration_directory=calibration_root,
        )
        if own_validation.get("valid") is not True:
            raise ValueError("target pilot execution validator failed")
        if destination_path.exists() or destination_path.is_symlink():
            raise FileExistsError("pilot config destination appeared during publish")
        os.replace(temporary, destination_path)
        temporary = None
        published = True
        _fsync_file(destination_path)
        # rename 後再走一次公開 loader／兩個 validator，確保 caller 取得的 final leaf
        # 與 hidden file 使用同一份 bytes；若此 deterministic recheck 失敗，只清除本次
        # newly-published leaf，不碰事前已存在的檔案。
        final_config = load_config(destination_path, formal_release=False)
        final_release = validate_release_config(
            destination_path,
            input_directory=input_root,
            formal=False,
        )
        final_own = validate_pilot_execution_config(
            destination_path,
            input_directory=input_root,
            calibration_directory=calibration_root,
        )
        if final_release.get("valid") is not True or final_own.get("valid") is not True:
            raise ValueError("published pilot config validation failed")
        return {
            "output": str(destination_path),
            "valid": True,
            "config_status": final_config.config_status,
            "schema_version": final_config.schema_version,
            "target_config_hash": final_config.config_hash(),
            "source_config_hash": source_config_hash,
            "calibration_schema_version": calibration_report["schema_version"],
            "candidate_values": dict(candidate_values),
            "members_per_scenario": scalar_snapshot["members_per_scenario"],
            "dt_min_seconds": scalar_snapshot["dt_min_seconds"],
            "dt_max_seconds": scalar_snapshot["dt_max_seconds"],
            "max_backtrack_days": scalar_snapshot["max_backtrack_days"],
            "status": PILOT_EXECUTION_BINDING_STATUS,
        }
    except Exception:
        if published and destination_path.exists() and not destination_path.is_symlink():
            destination_path.unlink()
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _safe_error(errors: list[str], code: str) -> None:
    """加入固定 validator error code；不把第三方 exception 或外部 path 寫入回傳值。"""

    if code not in errors:
        errors.append(code)


def _validate_pilot_binding_shape(binding: object, errors: list[str]) -> Mapping[str, Any] | None:
    """驗證 pilot binding root/nested key topology，unknown 或 missing 一律 fail closed。"""

    if not isinstance(binding, Mapping):
        _safe_error(errors, "pilot_execution_binding_missing")
        return None
    if set(binding) != _PILOT_EXECUTION_BINDING_KEYS:
        _safe_error(errors, "pilot_execution_binding_keys_invalid")
    if binding.get("schema_version") != PILOT_EXECUTION_BINDING_SCHEMA_VERSION:
        _safe_error(errors, "pilot_execution_binding_schema_invalid")
    if binding.get("status") != PILOT_EXECUTION_BINDING_STATUS:
        _safe_error(errors, "pilot_execution_binding_status_invalid")
    if _binding_has_path(binding):
        _safe_error(errors, "pilot_execution_binding_path_forbidden")
    candidate_values = binding.get("candidate_values")
    if not isinstance(candidate_values, Mapping) or set(candidate_values) != set(_CANDIDATE_NAMES):
        _safe_error(errors, "pilot_execution_binding_candidates_invalid")
    applied_fields = binding.get("applied_fields")
    if not isinstance(applied_fields, Mapping) or dict(applied_fields) != _CANDIDATE_TARGET_FIELDS:
        _safe_error(errors, "pilot_execution_binding_applied_fields_invalid")
    snapshot = binding.get("execution_scalar_snapshot")
    if not isinstance(snapshot, Mapping) or set(snapshot) != set(_EXECUTION_SCALAR_NAMES):
        _safe_error(errors, "pilot_execution_binding_scalar_snapshot_invalid")
    for field in (
        "source_config_hash",
        "input_artifact_index_sha256",
        "calibration_manifest_sha256",
        "calibration_report_sha256",
        "pair_samples_sha256",
    ):
        value = binding.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            _safe_error(errors, f"pilot_execution_binding_hash_invalid:{field}")
    if binding.get("calibration_schema_version") != PILOT_CALIBRATION_SCHEMA_VERSION:
        _safe_error(errors, "pilot_execution_binding_calibration_schema_invalid")
    return binding


def _validate_target_candidates(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
    binding: Mapping[str, Any] | None,
    errors: list[str],
) -> dict[str, float] | None:
    """精確比對 report candidate、pilot binding candidate 與 target physics 欄位。"""

    try:
        report_values = _candidate_values_from_report(report)
        target_values_raw = _target_candidate_values(payload)
    except Exception:
        _safe_error(errors, "target_candidate_fields_invalid")
        return None
    for name, expected in report_values.items():
        try:
            actual = _strict_float(target_values_raw[name], label=f"target.{name}")
        except Exception:
            _safe_error(errors, f"target_candidate_invalid:{name}")
            continue
        if actual != expected:
            _safe_error(errors, f"target_candidate_mismatch:{name}")
    if binding is not None:
        raw_binding_values = binding.get("candidate_values")
        if not isinstance(raw_binding_values, Mapping):
            _safe_error(errors, "pilot_execution_binding_candidates_invalid")
        else:
            for name, expected in report_values.items():
                try:
                    actual = _strict_float(raw_binding_values.get(name), label=f"binding.{name}")
                except Exception:
                    _safe_error(errors, f"pilot_binding_candidate_invalid:{name}")
                    continue
                if actual != expected:
                    _safe_error(errors, f"pilot_binding_candidate_mismatch:{name}")
    return report_values


def _validate_target_scalars(
    payload: Mapping[str, Any],
    binding: Mapping[str, Any] | None,
    errors: list[str],
) -> dict[str, float | int | None] | None:
    """驗證 target runtime scalar 與 binding snapshot，包含 horizon/step-budget gate。"""

    try:
        raw = _target_scalar_snapshot(payload)
        normalised = _normalise_execution_scalars(**raw)
    except Exception:
        _safe_error(errors, "target_execution_scalar_gate_invalid")
        return None
    if binding is not None:
        snapshot = binding.get("execution_scalar_snapshot")
        if not isinstance(snapshot, Mapping) or not _strict_mapping_equal(dict(snapshot), normalised):
            _safe_error(errors, "pilot_execution_binding_scalar_snapshot_mismatch")
    return normalised


def validate_pilot_execution_config(
    path: str | Path,
    *,
    input_directory: str | Path,
    calibration_directory: str | Path,
) -> dict[str, Any]:
    """唯讀驗證 generated pilot config、release binding、calibration hash 與 scalar contract。

    validator 先檢查 target 的 release/input binding，再以明示 ``input_directory`` 驗證
    calibration artifact；它不把 target 當成 calibration source config，因 target config
    hash 必然不同。所有失敗都收斂成固定、不含絕對路徑或第三方例外的 error code。除了
    target YAML／release binding 外，還會檢查 calibration 必須是完整 1.1.0 candidate、
    report source config hash 與 pilot binding 一致、三個 calibration payload SHA 對應、
    gap-safe root/records 支援 horizon，以及 runtime candidate／scalar exact binding。
    """

    errors: list[str] = []
    config_path: Path | None = None
    input_root: Path | None = None
    calibration_root: Path | None = None
    payload: dict[str, Any] | None = None
    config: ProjectConfig | None = None
    report: dict[str, Any] | None = None
    binding: Mapping[str, Any] | None = None
    scalar_snapshot: dict[str, float | int | None] | None = None
    candidate_values: dict[str, float] | None = None
    actual_artifact_index_sha256: str | None = None
    calibration_hashes: dict[str, str] = {}

    try:
        config_path = _assert_regular_file(path)
        payload = _load_yaml_mapping(config_path)
    except Exception:
        _safe_error(errors, "target_config_unreadable")

    if payload is not None:
        try:
            config = load_config(config_path, formal_release=False)
        except Exception:
            _safe_error(errors, "target_config_schema_invalid")
        if payload.get("config_status") != "generated":
            _safe_error(errors, "target_config_status_invalid")
        binding = _validate_pilot_binding_shape(payload.get("pilot_execution_binding"), errors)
        scalar_snapshot = _validate_target_scalars(payload, binding, errors)

    # release/input binding 是第一個外部 gate；錯誤只保留固定 code，避免 validator 把
    # source path、YAML exception 或 SERVER 目錄洩漏給 stdout／排程器。
    try:
        input_root = _assert_regular_directory(input_directory)
    except Exception:
        _safe_error(errors, "input_directory_invalid")
    if config_path is not None and input_root is not None:
        try:
            release = validate_release_config(
                config_path,
                input_directory=input_root,
                formal=False,
            )
            if release.get("valid") is not True:
                _safe_error(errors, "target_release_binding_invalid")
        except Exception:
            _safe_error(errors, "target_release_binding_invalid")
    try:
        calibration_root = _assert_regular_directory(calibration_directory)
    except Exception:
        _safe_error(errors, "calibration_directory_invalid")

    if input_root is not None:
        try:
            _index_payload, fingerprint = read_canonical_json(input_root / "artifact_index.json")
            actual_artifact_index_sha256 = str(fingerprint["sha256"])
        except Exception:
            _safe_error(errors, "input_artifact_index_unreadable")

    if calibration_root is not None and input_root is not None:
        # 不提供 config_path：target hash 不應被當成 calibration source hash；兩者的
        # identity 由下方 report.input_binding.config_hash 與 binding.source_config_hash
        # 明確比較。
        try:
            calibration_validation = validate_pilot_calibration(
                calibration_root,
                input_directory=input_root,
            )
            if calibration_validation.get("valid") is not True:
                _safe_error(errors, "calibration_artifact_invalid")
        except Exception:
            _safe_error(errors, "calibration_artifact_invalid")
        try:
            report = _load_calibration_report(calibration_root)
            manifest = _read_json(calibration_root / "manifest.json")
            calibration_hashes = _hashes_for_calibration(calibration_root)
            if report.get("schema_version") != PILOT_CALIBRATION_SCHEMA_VERSION:
                _safe_error(errors, "calibration_schema_invalid")
            if manifest.get("schema_version") != PILOT_CALIBRATION_SCHEMA_VERSION:
                _safe_error(errors, "calibration_manifest_schema_invalid")
            if report.get("artifact_kind") != PILOT_CALIBRATION_ARTIFACT_KIND:
                _safe_error(errors, "calibration_kind_invalid")
            if report.get("evidence_class") != "server_real_data_pilot_candidate":
                _safe_error(errors, "calibration_evidence_class_invalid")
            if report.get("completion_status") != "complete":
                _safe_error(errors, "calibration_completion_invalid")
            if report.get("recommendation_status") != (
                "candidate_pending_trajectory_convergence_and_scientific_validation"
            ):
                _safe_error(errors, "calibration_recommendation_invalid")
            candidate_values = _validate_target_candidates(payload or {}, report, binding, errors)
            report_binding = report.get("input_binding")
            if not isinstance(report_binding, Mapping):
                _safe_error(errors, "calibration_input_binding_invalid")
            else:
                if binding is not None and report_binding.get("config_hash") != binding.get(
                    "source_config_hash"
                ):
                    _safe_error(errors, "source_config_hash_mismatch")
                if actual_artifact_index_sha256 is not None and report_binding.get(
                    "input_artifact_index_sha256"
                ) != actual_artifact_index_sha256:
                    _safe_error(errors, "calibration_input_artifact_hash_mismatch")
        except Exception:
            _safe_error(errors, "calibration_payload_unreadable")

    if binding is not None:
        if report is not None:
            if binding.get("calibration_schema_version") != report.get("schema_version"):
                _safe_error(errors, "pilot_binding_calibration_schema_mismatch")
            if calibration_hashes:
                # binding 對 manifest/report 使用 calibration_ 前綴以避免與 input artifact
                # index 混淆；calibration_hashes 則沿用三個 payload 的短欄位名稱。兩者在
                # 此處明示映射，避免新增欄位時靠模糊字串拼接而漏驗證。
                hash_fields = {
                    "calibration_manifest_sha256": "manifest_sha256",
                    "calibration_report_sha256": "report_sha256",
                    "pair_samples_sha256": "pair_samples_sha256",
                }
                for binding_field, hash_field in hash_fields.items():
                    if binding.get(binding_field) != calibration_hashes.get(hash_field):
                        _safe_error(errors, f"pilot_binding_{binding_field}_mismatch")
        if (
            actual_artifact_index_sha256 is not None
            and binding.get("input_artifact_index_sha256") != actual_artifact_index_sha256
        ):
            _safe_error(errors, "pilot_binding_input_artifact_hash_mismatch")
        if candidate_values is not None:
            raw_binding_candidates = binding.get("candidate_values")
            if isinstance(raw_binding_candidates, Mapping):
                for name, expected in candidate_values.items():
                    try:
                        if (
                            _strict_float(raw_binding_candidates.get(name), label=f"binding.{name}")
                            != expected
                        ):
                            _safe_error(errors, f"pilot_binding_candidate_mismatch:{name}")
                    except Exception:
                        _safe_error(errors, f"pilot_binding_candidate_invalid:{name}")

    if payload is not None and config_path is not None and input_root is not None:
        release_binding = payload.get("release_binding")
        if not isinstance(release_binding, Mapping):
            _safe_error(errors, "target_release_binding_missing")
        elif actual_artifact_index_sha256 is not None and release_binding.get(
            "input_directory_artifact_index_sha256"
        ) != actual_artifact_index_sha256:
            _safe_error(errors, "target_release_input_artifact_hash_mismatch")

    if scalar_snapshot is not None and input_root is not None:
        try:
            _read_gap_safe_horizon(
                input_root,
                requested_max_backtrack_days=float(scalar_snapshot["max_backtrack_days"]),
            )
        except Exception:
            _safe_error(errors, "gap_safe_horizon_invalid")

    target_hash: str | None = None
    if config is not None:
        try:
            target_hash = config.config_hash()
        except Exception:
            _safe_error(errors, "target_config_hash_invalid")
    summary = {
        "schema_version": PILOT_EXECUTION_BINDING_SCHEMA_VERSION,
        "source_calibration_schema_version": report.get("schema_version") if report else None,
        "calibration_schema_version": report.get("schema_version") if report else None,
        "target_config_hash": target_hash,
        "candidate_values": candidate_values,
        "members_per_scenario": scalar_snapshot.get("members_per_scenario")
        if scalar_snapshot
        else None,
        "dt_min_seconds": scalar_snapshot.get("dt_min_seconds") if scalar_snapshot else None,
        "dt_max_seconds": scalar_snapshot.get("dt_max_seconds") if scalar_snapshot else None,
        "max_backtrack_days": scalar_snapshot.get("max_backtrack_days")
        if scalar_snapshot
        else None,
        "status": binding.get("status") if binding is not None else None,
    }
    return {"valid": not errors, "errors": errors, "summary": summary}


__all__ = [
    "PILOT_EXECUTION_BINDING_SCHEMA_VERSION",
    "PILOT_EXECUTION_BINDING_STATUS",
    "create_pilot_execution_config",
    "validate_pilot_execution_config",
]
