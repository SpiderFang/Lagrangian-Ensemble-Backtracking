"""跨區 pilot 試跑的一致性驗證。

本模組把 A、B、C、D 區的單次試跑視為同一個工程實驗矩陣：每個 run workspace
必須由既有的 ``run_plan.json`` 與 ``normalized_config.json`` 提供完整的 immutable
身分、數值方法、隨機系集、分片／checkpoint 及部署環境證據。驗證器不讀取
trajectory shard、forcing 或大型 Parquet；它只比較啟動時已發布的兩份 JSON，因此可在
SERVER 上快速檢查四區是否真的沿用同一組試跑設定。

區域研究本來就可能有不同的 study site、flow domain、arrival/scenario 識別碼、幾何與
輸入產品 hash/path，以及由區域校準得到的 Kh、Kz、Smagorinsky cap。這些差異以明確
白名單移除後才比較；其餘 integration、boundaries、Stokes 開關與方法、無效波政策、
材質行為／沉降速度、pilot scalar snapshot、M／seed、分片／checkpoint 與程式部署
版本都必須逐字一致。任何缺欄位、非標準 JSON、symlink、過大文件或型別錯誤都
fail closed，不能以缺值或預設值補齊。
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PILOT_MATRIX_SCHEMA_VERSION = "1.0.0"
"""跨區 pilot matrix 報告的版本；只描述比較報告，不代表科學成果版本。"""

MAX_PILOT_MATRIX_JSON_BYTES = 8 * 1024 * 1024
"""單一輸入 JSON 的大小上限；避免 validator 以意外大型文件耗盡記憶體。"""

# run plan 中會改變執行排程或結果可重現性的欄位；區域識別與 scenario hash 另以
# topology／selection 摘要保存，不把可預期的區域差異誤判為設定不一致。
_REQUIRED_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "run_kind",
        "experiment_case_id",
        "master_seed",
        "seed_policy",
        "members_per_scenario",
        "shard_scenario_count",
        "checkpoint_interval_sweeps",
        "active_chunk_size",
        "scenario_count",
        "particle_count",
        "shard_count",
        "scenario_selection",
        "shards",
        "code_provenance",
    }
)
_REQUIRED_SELECTION_FIELDS = frozenset(
    {"schema_version", "mode", "source_scenario_count", "selected_scenario_count"}
)
_REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "deployment_tree_sha256",
        "uv_lock_sha256",
        "package_version",
        "numpy_version",
        "numba_version",
        "pyarrow_version",
        "python_version",
    }
)
_REQUIRED_CONFIG_FIELDS = frozenset(
    {"schema_version", "design_version", "integration", "boundaries", "physics", "pilot_execution_binding"}
)
_REQUIRED_SCALAR_SNAPSHOT_FIELDS = frozenset(
    {
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
    }
)
_MATERIAL_FIELDS = frozenset({"material_id", "behavior_class", "settling_velocity_mps"})
_PROVENANCE_FIELDS = tuple(sorted(_REQUIRED_PROVENANCE_FIELDS))
_HORIZONTAL_REGIONAL_DIFFUSION_FIELDS = frozenset(
    {"constant_kh_m2ps", "kh_cap_m2ps"}
)
_VERTICAL_REGIONAL_DIFFUSION_FIELDS = frozenset({"constant_kz_m2ps"})


def _reject_json_constant(value: str) -> None:
    """拒絕 JSON 的 NaN、Infinity 與 -Infinity，避免非有限值繞過比較。"""

    raise ValueError(f"json_non_finite_constant:{value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """拒絕重複 JSON key；若採最後一筆會讓相同 bytes 之外的語意被悄悄改寫。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"json_duplicate_key:{key}")
        result[key] = value
    return result


def _path_has_symlink(path: Path) -> bool:
    """檢查 caller 明示的 root 或 JSON leaf 是否為 symbolic link。

    macOS 的 ``/var`` 通常本身是系統相容 symlink（實際暫存目錄會位於
    ``/private/var``）；若無條件沿著整條絕對路徑向上檢查，會把所有合法的 pytest
    temporary path 都誤判為外部資料樹。因此安全邊界由 ``_read_run`` 先拒絕 run root，
    再由本函式拒絕該 root 內的 JSON leaf；不對系統父層做不必要的猜測。
    """

    return path.is_symlink()


def _strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """讀取有限大小且不跟隨 symlink 的 JSON object。

    ``run_plan.json`` 與 ``normalized_config.json`` 都是 immutable 產物；本函式只接受
    普通檔案、UTF-8、無重複 key、無非有限常數的 object。錯誤會保留穩定 code 前綴，供
    per-run 與全域報告使用，而不把作業系統路徑或第三方 parser 例外輸出成契約內容。
    """

    if _path_has_symlink(path):
        raise ValueError(f"symlink_forbidden:{label}")
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise ValueError(f"file_unreadable:{label}") from exc
    if not path.is_file() or not os.path.isfile(path):
        raise ValueError(f"file_not_regular:{label}")
    if stat_result.st_size > MAX_PILOT_MATRIX_JSON_BYTES:
        raise ValueError(f"file_too_large:{label}")
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("json_"):
            raise
        raise ValueError(f"json_invalid:{label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"json_root_not_object:{label}")
    return value


def _required_mapping(value: object, *, label: str, errors: list[str]) -> Mapping[str, Any] | None:
    """驗證 object mapping，並以固定錯誤碼回報缺失或型別錯誤。"""

    if not isinstance(value, Mapping):
        errors.append(f"field_type_invalid:{label}")
        return None
    return value


def _require_fields(
    value: Mapping[str, Any], required: frozenset[str], *, label: str, errors: list[str]
) -> None:
    """要求指定 object 的必要欄位；unknown 欄位保留，因既有 schema 可能向前擴充。"""

    for field in sorted(required - set(value)):
        errors.append(f"field_missing:{label}.{field}")


def _strict_nonempty_string(value: object, *, label: str, errors: list[str]) -> str | None:
    """驗證不含首尾空白的非空字串，拒絕把數字或 null 轉成文字。"""

    if not isinstance(value, str) or not value.strip() or value != value.strip():
        errors.append(f"field_value_invalid:{label}")
        return None
    return value


def _strict_int(value: object, *, label: str, errors: list[str], minimum: int = 0) -> int | None:
    """驗證原生 JSON integer，避免 bool 被 Python 當成 int。"""

    if type(value) is not int or value < minimum:
        errors.append(f"field_value_invalid:{label}")
        return None
    return value


def _strict_scalar(value: object, *, label: str, errors: list[str]) -> object:
    """驗證 scalar 沒有 NaN 或巢狀容器，並保留原值供 canonical 比較。"""

    if isinstance(value, (dict, list, tuple)) or value is None:
        # active_chunk_size 可明示為 null；其他 scalar 的 null 仍由 caller 的缺值語意
        # 判斷，這裡只避免把 object/list 當作可比較的單值。
        if value is not None:
            errors.append(f"field_type_invalid:{label}")
        return value
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"field_value_invalid:{label}")
    return value


def _first_values(rows: object, key: str) -> list[object]:
    """從 scenario/shard list 收集唯一 scalar，供 per-run identity 顯示。"""

    if not isinstance(rows, list):
        return []
    values: list[object] = []
    for row in rows:
        if isinstance(row, Mapping) and key in row and row[key] not in values:
            values.append(row[key])
    return values


def _material_identity(config: Mapping[str, Any]) -> list[dict[str, object]]:
    """提取 material_id、behavior_class、settling velocity，供 identity 與報告稽核。"""

    physics = config.get("physics")
    settling = physics.get("settling") if isinstance(physics, Mapping) else None
    records = settling.get("material_classes") if isinstance(settling, Mapping) else None
    if not isinstance(records, list):
        return []
    result: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        result.append(
            {
                key: record.get(key)
                for key in sorted(_MATERIAL_FIELDS)
                if key in record
            }
        )
    return sorted(result, key=lambda item: str(item.get("material_id", "")))


def _run_identity(plan: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, object]:
    """由兩份 immutable 文件建立不含本機路徑的 run 身分摘要。"""

    selection = plan.get("scenario_selection")
    selection_map = selection if isinstance(selection, Mapping) else {}
    shards = plan.get("shards")
    shard_rows = shards if isinstance(shards, list) else []
    sites = _first_values(shard_rows, "study_site_id")
    regions = _first_values(shard_rows, "analysis_region_id")
    arrivals = _first_values(shard_rows, "arrival_time_id")
    if not arrivals:
        arrivals = _first_values(shard_rows, "arrival_time_utc_ns")
    identity: dict[str, object] = {
        "run_id": plan.get("run_id"),
        "run_kind": plan.get("run_kind"),
        "experiment_case_id": plan.get("experiment_case_id"),
        "members_per_scenario": plan.get("members_per_scenario"),
        "master_seed": plan.get("master_seed"),
        "seed_policy": plan.get("seed_policy"),
        "scenario_count": plan.get("scenario_count"),
        "particle_count": plan.get("particle_count"),
        "scenario_selection_mode": selection_map.get("mode"),
        "source_scenario_count": selection_map.get("source_scenario_count"),
        "selected_scenario_count": selection_map.get("selected_scenario_count"),
        "study_site_ids": sorted(str(item) for item in sites),
        "analysis_region_ids": sorted(str(item) for item in regions),
        "arrival_time_ids": sorted(str(item) for item in arrivals),
        "material_selection": _material_identity(config),
    }
    # normalized config 可能有完整五站 identity；將它加入摘要可讓 operator 在不讀
    # workspace 其他檔案的前提下知道本次比較涵蓋哪些區域與 domain。
    config_sites = config.get("study_sites")
    if isinstance(config_sites, list):
        identity["configured_study_site_ids"] = sorted(
            str(item.get("study_site_id"))
            for item in config_sites
            if isinstance(item, Mapping) and item.get("study_site_id") is not None
        )
    config_domains = config.get("domains")
    if isinstance(config_domains, list):
        identity["configured_flow_domain_ids"] = sorted(
            str(item.get("flow_domain_id"))
            for item in config_domains
            if isinstance(item, Mapping) and item.get("flow_domain_id") is not None
        )
    return identity


def _validate_plan(plan: Mapping[str, Any], *, errors: list[str]) -> None:
    """檢查 matrix 比較所需的 run plan 欄位與 scalar 型別。"""

    _require_fields(plan, _REQUIRED_PLAN_FIELDS, label="run_plan", errors=errors)
    _strict_nonempty_string(plan.get("schema_version"), label="run_plan.schema_version", errors=errors)
    run_kind = _strict_nonempty_string(plan.get("run_kind"), label="run_plan.run_kind", errors=errors)
    # 這個命令只比較四區「相同設定的第一次試跑」；formal 已進入正式研究生命週期，
    # synthetic 則沒有可與真實 pilot 對照的輸入證據，兩者若混入會讓報告失去明確語意。
    # 因此即使其他欄位完全相同，也必須在單一 run 階段拒絕非 pilot 的 run_kind。
    if run_kind != "pilot":
        errors.append("field_value_invalid:run_plan.run_kind")
    _strict_nonempty_string(plan.get("run_id"), label="run_plan.run_id", errors=errors)
    _strict_nonempty_string(
        plan.get("experiment_case_id"), label="run_plan.experiment_case_id", errors=errors
    )
    _strict_nonempty_string(plan.get("seed_policy"), label="run_plan.seed_policy", errors=errors)
    scalar_fields = {
        "master_seed": 0,
        "members_per_scenario": 1,
        "shard_scenario_count": 1,
        "checkpoint_interval_sweeps": 1,
        "scenario_count": 1,
        "particle_count": 1,
        "shard_count": 1,
    }
    scalar_values: dict[str, int] = {}
    for field, minimum in scalar_fields.items():
        result = _strict_int(plan.get(field), label=f"run_plan.{field}", errors=errors, minimum=minimum)
        if result is not None:
            scalar_values[field] = result
    active_chunk = plan.get("active_chunk_size")
    if active_chunk is not None:
        _strict_int(active_chunk, label="run_plan.active_chunk_size", errors=errors, minimum=1)
    if (
        "particle_count" in scalar_values
        and "scenario_count" in scalar_values
        and scalar_values["particle_count"]
        != scalar_values["scenario_count"] * scalar_values.get("members_per_scenario", -1)
    ):
        errors.append("run_plan_particle_count_mismatch")

    selection = _required_mapping(
        plan.get("scenario_selection"), label="run_plan.scenario_selection", errors=errors
    )
    if selection is not None:
        _require_fields(
            selection,
            _REQUIRED_SELECTION_FIELDS,
            label="run_plan.scenario_selection",
            errors=errors,
        )
        _strict_nonempty_string(
            selection.get("mode"), label="run_plan.scenario_selection.mode", errors=errors
        )
        if "selection_policy" not in selection and "ranking_policy" not in selection:
            errors.append("field_missing:run_plan.scenario_selection.selection_policy")
        else:
            _strict_nonempty_string(
                selection.get("selection_policy", selection.get("ranking_policy")),
                label="run_plan.scenario_selection.selection_policy",
                errors=errors,
            )
        if selection.get("mode") == "pilot_exact":
            _strict_nonempty_string(
                selection.get("material_id"),
                label="run_plan.scenario_selection.material_id",
                errors=errors,
            )
        for field in ("source_scenario_count", "selected_scenario_count"):
            _strict_int(
                selection.get(field),
                label=f"run_plan.scenario_selection.{field}",
                errors=errors,
                minimum=1,
            )
        if (
            isinstance(selection.get("selected_scenario_count"), int)
            and selection.get("selected_scenario_count") != plan.get("scenario_count")
        ):
            errors.append("run_plan_selection_count_mismatch")

    # shards 是 list 而非 Mapping，這裡特別處理以便錯誤碼指出 topology 而非泛用型別錯誤。
    if plan.get("shards") is not None and not isinstance(plan.get("shards"), list):
        errors.append("field_type_invalid:run_plan.shards")
    elif isinstance(plan.get("shards"), list):
        rows = plan["shards"]
        if len(rows) != plan.get("shard_count"):
            errors.append("run_plan_shard_count_mismatch")
        total = 0
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                errors.append(f"field_type_invalid:run_plan.shards[{index}]")
                continue
            for field in (
                "scenario_start_index",
                "scenario_stop_index",
                "scenario_count",
                "particle_count",
                "group_part_index",
                "group_part_count",
            ):
                _strict_int(
                    row.get(field),
                    label=f"run_plan.shards[{index}].{field}",
                    errors=errors,
                    minimum=0 if field in {"scenario_start_index", "group_part_index"} else 1,
                )
            if isinstance(row.get("scenario_count"), int):
                total += row["scenario_count"]
            if (
                isinstance(row.get("scenario_count"), int)
                and isinstance(row.get("particle_count"), int)
                and isinstance(plan.get("members_per_scenario"), int)
                and row["particle_count"] != row["scenario_count"] * plan["members_per_scenario"]
            ):
                errors.append(f"run_plan_shard_particle_count_mismatch:{index}")
        if isinstance(plan.get("scenario_count"), int) and total != plan["scenario_count"]:
            errors.append("run_plan_shard_scenario_count_mismatch")

    provenance = _required_mapping(
        plan.get("code_provenance"), label="run_plan.code_provenance", errors=errors
    )
    if provenance is not None:
        _require_fields(
            provenance,
            _REQUIRED_PROVENANCE_FIELDS,
            label="run_plan.code_provenance",
            errors=errors,
        )
        for field in _PROVENANCE_FIELDS:
            _strict_nonempty_string(
                provenance.get(field), label=f"run_plan.code_provenance.{field}", errors=errors
            )


def _validate_config(config: Mapping[str, Any], *, errors: list[str]) -> None:
    """檢查 normalized config 的必要區塊與材料／scalar binding。"""

    _require_fields(config, _REQUIRED_CONFIG_FIELDS, label="normalized_config", errors=errors)
    for section in ("integration", "boundaries", "physics"):
        _required_mapping(config.get(section), label=f"normalized_config.{section}", errors=errors)
    physics = config.get("physics")
    physics_map = _required_mapping(physics, label="normalized_config.physics", errors=errors)
    if physics_map is not None:
        _required_mapping(
            physics_map.get("stokes"), label="normalized_config.physics.stokes", errors=errors
        )
        _required_mapping(
            physics_map.get("horizontal_diffusion"),
            label="normalized_config.physics.horizontal_diffusion",
            errors=errors,
        )
        _required_mapping(
            physics_map.get("vertical_diffusion"),
            label="normalized_config.physics.vertical_diffusion",
            errors=errors,
        )
    settling = physics.get("settling") if isinstance(physics, Mapping) else None
    settling_map = _required_mapping(settling, label="normalized_config.physics.settling", errors=errors)
    if settling_map is not None:
        materials = settling_map.get("material_classes")
        if not isinstance(materials, list) or not materials:
            errors.append("field_missing:normalized_config.physics.settling.material_classes")
        else:
            for index, material in enumerate(materials):
                if not isinstance(material, Mapping):
                    errors.append(f"field_type_invalid:normalized_config.physics.settling.material_classes[{index}]")
                    continue
                _require_fields(
                    material,
                    _MATERIAL_FIELDS,
                    label=f"normalized_config.physics.settling.material_classes[{index}]",
                    errors=errors,
                )
                _strict_nonempty_string(
                    material.get("material_id"),
                    label=f"normalized_config.physics.settling.material_classes[{index}].material_id",
                    errors=errors,
                )
                _strict_nonempty_string(
                    material.get("behavior_class"),
                    label=f"normalized_config.physics.settling.material_classes[{index}].behavior_class",
                    errors=errors,
                )
                velocity = material.get("settling_velocity_mps")
                if type(velocity) not in {int, float} or not math.isfinite(float(velocity)):
                    errors.append(
                        f"field_value_invalid:normalized_config.physics.settling.material_classes[{index}].settling_velocity_mps"
                    )
    binding = _required_mapping(
        config.get("pilot_execution_binding"),
        label="normalized_config.pilot_execution_binding",
        errors=errors,
    )
    if binding is not None:
        snapshot = _required_mapping(
            binding.get("execution_scalar_snapshot"),
            label="normalized_config.pilot_execution_binding.execution_scalar_snapshot",
            errors=errors,
        )
        if snapshot is not None:
            _require_fields(
                snapshot,
                _REQUIRED_SCALAR_SNAPSHOT_FIELDS,
                label="normalized_config.pilot_execution_binding.execution_scalar_snapshot",
                errors=errors,
            )


def _validate_selected_material(
    plan: Mapping[str, Any], config: Mapping[str, Any], *, errors: list[str]
) -> None:
    """確認 ``pilot_exact`` 指定的材質在 normalized config 中恰好存在一次。

    精確 pilot 的 ``scenario_selection.material_id`` 是執行身分的一部分，不能只依
    plan 內的字串宣告。若該材質不存在，或同一 ``material_id`` 重複出現，後續
    ``_material_contract`` 可能會得到空集合或多筆而無法表示唯一的物性，這會讓兩份
    都錯誤的 run 意外通過跨區比較。因此在單一 run validation 階段先 fail closed；
    非 ``pilot_exact`` 的分層／完整 pilot 沒有單一選定材質，維持原有全材質 contract。
    """

    selection = plan.get("scenario_selection")
    if not isinstance(selection, Mapping) or selection.get("mode") != "pilot_exact":
        return
    selected_material_id = selection.get("material_id")
    if not isinstance(selected_material_id, str) or not selected_material_id.strip():
        # _validate_plan 已回報 material_id 缺失或型別錯誤；此處避免再產生一筆無法
        # 指向實際材質表的重複錯誤。
        return

    physics = config.get("physics")
    settling = physics.get("settling") if isinstance(physics, Mapping) else None
    materials = settling.get("material_classes") if isinstance(settling, Mapping) else None
    if not isinstance(materials, list):
        # _validate_config 已指出 material_classes 的結構錯誤；保留一個明確的
        # selected-material leaf，讓 operator 知道 pilot_exact binding 也沒有落地。
        errors.append(
            "selected_material_not_found:normalized_config.physics.settling.material_classes"
        )
        return

    matches = [
        index
        for index, material in enumerate(materials)
        if isinstance(material, Mapping) and material.get("material_id") == selected_material_id
    ]
    if not matches:
        errors.append(
            "selected_material_not_found:normalized_config.physics.settling.material_classes"
        )
    elif len(matches) > 1:
        errors.append(
            "selected_material_not_unique:normalized_config.physics.settling.material_classes"
        )


def _canonical_value(value: object) -> object:
    """建立排序穩定的 JSON value 副本，保留資料內容而不擴大比較範圍。

    normalized config 來自嚴格 JSON reader，已經沒有 tuple、NumPy scalar 或非有限值；
    這裡只固定 mapping key 的順序，讓 report 內部 contract 在不同 JSON 排版下仍有相同
    語意。列表順序仍然保留，因為 integration、boundary 與 Stokes sensitivity case 的
    順序是設定契約的一部分。
    """

    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    return value


def _selection_contract(selection: Mapping[str, Any]) -> dict[str, object]:
    """抽取 scenario selection 的共同設定欄位。

    site、arrival、receptor、scenario 識別碼及其內容／來源 hash 是區域資料 identity，
    刻意不進共同 contract；schema、模式、版本化選擇政策、數量與選定 material_id 才是
    需要跨區一致的執行設定。分層模式使用 ``ranking_policy``，精確模式使用
    ``selection_policy``，兩者在此轉成同一個比較欄位。
    """

    return {
        "schema_version": selection.get("schema_version"),
        "mode": selection.get("mode"),
        "selection_policy": selection.get("selection_policy", selection.get("ranking_policy")),
        "source_scenario_count": selection.get("source_scenario_count"),
        "selected_scenario_count": selection.get("selected_scenario_count"),
        "material_id": selection.get("material_id"),
    }


def _is_external_binding_key(key: str) -> bool:
    """判定物理 section 內只代表外部檔案 provenance 的欄位。

    本函式只服務 physics/diffusion projection；整個 inputs、geometry、release binding
    根本不會進入 contract。路徑、hash、size 與 manifest 名稱不是物理方法，因此即使
    四區由不同產品或不同發布位置產生，也不能干擾共同設定比較。
    """

    lower = key.lower()
    return (
        key in {"material_manifest", "manifest_path"}
        or lower.endswith(("_path", "_root", "_sha256", "_hash"))
        or lower in {"path", "root", "sha256", "hash", "size_bytes"}
    )


def _diffusion_contract(
    section: object, *, regional_keys: frozenset[str]
) -> dict[str, object]:
    """抽取擴散方法與非區域物理欄位，排除 Kh/Kz/Smagorinsky cap。

    Kh、Kz 與 Smagorinsky cap 是本次跨區校準允許不同的三個數值；floor、公式、
    sensitivity、方法與開關仍保留，因此不能用「整個 diffusion section 不比較」掩蓋
    數值方法差異。
    """

    if not isinstance(section, Mapping):
        return {}
    projected = _project_diffusion_value(section, regional_keys=regional_keys)
    # section 已由 caller 驗證為 mapping；此型別分支只讓回傳契約保持明確，避免未來
    # 修改遞迴投影時把 scalar 意外帶進 diffusion contract。
    return projected if isinstance(projected, dict) else {}


def _project_diffusion_value(value: object, *, regional_keys: frozenset[str]) -> object:
    """遞迴建立擴散共同設定投影，移除所有層級的區域校準與外部 binding。

    真實 normalized config 的區域校準資訊可能包在 ``smagorinsky`` 或其他版本化
    calibration mapping 內；只處理一層會把內層 ``kh_cap_m2ps``／``constant_kz_m2ps``
    留在 contract，造成合法的跨區差異被拒絕。這裡只在明示的 regional key 或外部
    provenance key 上做投影，其餘未知欄位完整保留，讓新增的非區域物理設定仍會在
    leaf comparison 中 fail closed，而不會被寬泛的 metadata 清理掩蓋。
    """

    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        for key, nested_value in value.items():
            key_text = str(key)
            if key_text in regional_keys or _is_external_binding_key(key_text):
                continue
            projected[key_text] = _project_diffusion_value(
                nested_value, regional_keys=regional_keys
            )
        return projected
    if isinstance(value, list):
        return [
            _project_diffusion_value(item, regional_keys=regional_keys)
            for item in value
        ]
    return _canonical_value(value)


def _material_contract(
    settling: object, *, selected_material_id: object
) -> dict[str, object]:
    """抽取選定材質的 behavior 與 settling velocity。

    精確 pilot 有一個 ``material_id``，只比較該材質；完整／分層 pilot 沒有單一材質
    時則排序比較所有材質的三欄。分類來源、校準狀態、shape 描述與 manifest path 屬於
    component metadata，不冒充物性 contract。
    """

    if not isinstance(settling, Mapping):
        return {}
    records = settling.get("material_classes")
    if not isinstance(records, list):
        return {}
    selected = [
        record
        for record in records
        if isinstance(record, Mapping)
        and (selected_material_id is None or record.get("material_id") == selected_material_id)
    ]
    result: list[dict[str, object]] = []
    for record in selected:
        result.append(
            {
                "material_id": record.get("material_id"),
                "behavior_class": record.get("behavior_class"),
                "settling_velocity_mps": record.get("settling_velocity_mps"),
            }
        )
    result.sort(key=lambda item: str(item.get("material_id")))
    return {"materials": result}


def _physics_contract(physics: object, *, selected_material_id: object) -> dict[str, object]:
    """抽取 Stokes、擴散與非區域物理欄位。

    ``physics`` 的完整設定同時含輸入 manifest、校準狀態與區域 candidate metadata；
    本投影只留下會改變同一試跑數值語意的欄位，並以顯式 regional exclusions 保留
    跨區 Kh/Kz/cap 校準彈性。Stokes mapping 則完整保留，確保 no_stokes、formulation
    或 invalid wave policy 的差異都在 leaf path 被指出。
    """

    if not isinstance(physics, Mapping):
        return {}
    result: dict[str, object] = {}
    for key, value in physics.items():
        key_text = str(key)
        if key_text == "stokes":
            result[key_text] = _canonical_value(value)
        elif key_text == "settling":
            settling = value if isinstance(value, Mapping) else {}
            policy_fields = {
                policy_key: _canonical_value(settling[policy_key])
                for policy_key in (
                    "z_positive_up_sign_convention",
                    "require_strictly_negative_velocity",
                    "positive_or_zero_velocity_policy",
                    "velocity_unit",
                )
                if policy_key in settling
            }
            policy_fields.update(_material_contract(value, selected_material_id=selected_material_id))
            result[key_text] = policy_fields
        elif key_text == "horizontal_diffusion":
            result[key_text] = _diffusion_contract(
                value, regional_keys=_HORIZONTAL_REGIONAL_DIFFUSION_FIELDS
            )
        elif key_text == "vertical_diffusion":
            result[key_text] = _diffusion_contract(
                value, regional_keys=_VERTICAL_REGIONAL_DIFFUSION_FIELDS
            )
        elif not _is_external_binding_key(key_text):
            result[key_text] = _canonical_value(value)
    return result


def _config_contract(config: Mapping[str, Any], *, selection: Mapping[str, Any]) -> dict[str, object]:
    """建立 normalized config 的明確共同 contract。

    只比較 schema/design、完整 integration/boundaries、physics 物理投影及 pilot scalar
    snapshot；study_sites、domains、inputs、geometry、release_binding、release_approval、
    outputs 與其他發布 metadata 不讀入 contract。這個邊界直接對應跨區試跑的科學設定，
    避免把區域資料 identity 誤判成數值方法不一致。
    """

    binding = config.get("pilot_execution_binding")
    snapshot = binding.get("execution_scalar_snapshot") if isinstance(binding, Mapping) else None
    physics = config.get("physics")
    return {
        "schema_version": config.get("schema_version"),
        "design_version": config.get("design_version"),
        "integration": _canonical_value(config.get("integration")),
        "boundaries": _canonical_value(config.get("boundaries")),
        "physics": _physics_contract(
            physics,
            selected_material_id=selection.get("material_id"),
        ),
        "pilot_execution_binding": {
            "execution_scalar_snapshot": _canonical_value(snapshot),
        },
    }


def _read_run(root: Path) -> dict[str, Any]:
    """讀取單一 run 並回傳 plan/config/identity/contract；失敗由 caller 收斂。"""

    if _path_has_symlink(root):
        raise ValueError("symlink_forbidden:run_root")
    if not root.is_dir():
        raise ValueError("run_root_not_directory")
    plan = _strict_json_object(root / "run_plan.json", label="run_plan")
    config = _strict_json_object(root / "normalized_config.json", label="normalized_config")
    errors: list[str] = []
    _validate_plan(plan, errors=errors)
    _validate_config(config, errors=errors)
    _validate_selected_material(plan, config, errors=errors)
    identity = _run_identity(plan, config)
    return {
        "root": str(root),
        "valid": not errors,
        "identity": identity,
        "errors": sorted(set(errors)),
        "_plan": plan,
        "_config": config,
    }


def _comparison_contract(run: Mapping[str, Any]) -> dict[str, object]:
    """建立單一 run 的四層共同 contract。

    contract 以欄位投影取代「整份 normalized config scrub」：plan 的 raw input/config/
    geometry hash、發布 path 與 metadata 根本不會進入比較；code provenance 只保留
    部署樹與依賴／Python 版本；config 只保留研究數值方法與 pilot scalar。這個明確邊界
    是跨區通過的必要條件，也讓差異能回報到實際 leaf。
    """

    plan = run["_plan"]
    config = run["_config"]
    assert isinstance(plan, Mapping)
    assert isinstance(config, Mapping)
    provenance = plan.get("code_provenance")
    shards = plan.get("shards")
    selection = plan.get("scenario_selection")
    shard_topology: list[dict[str, object]] = []
    if isinstance(shards, list):
        for row in shards:
            if isinstance(row, Mapping):
                shard_topology.append(
                    {
                        "scenario_start_index": row.get("scenario_start_index"),
                        "scenario_stop_index": row.get("scenario_stop_index"),
                        "scenario_count": row.get("scenario_count"),
                        "particle_count": row.get("particle_count"),
                        "group_part_index": row.get("group_part_index"),
                        "group_part_count": row.get("group_part_count"),
                    }
                )
    plan_contract = {
        "schema_version": plan.get("schema_version"),
        "run_kind": plan.get("run_kind"),
        "experiment_case_id": plan.get("experiment_case_id"),
        "master_seed": plan.get("master_seed"),
        "seed_policy": plan.get("seed_policy"),
        "members_per_scenario": plan.get("members_per_scenario"),
        "shard_scenario_count": plan.get("shard_scenario_count"),
        "checkpoint_interval_sweeps": plan.get("checkpoint_interval_sweeps"),
        "active_chunk_size": plan.get("active_chunk_size"),
        "scenario_count": plan.get("scenario_count"),
        "particle_count": plan.get("particle_count"),
        "shard_count": plan.get("shard_count"),
        "scenario_selection": _selection_contract(selection) if isinstance(selection, Mapping) else {},
        "shard_topology": shard_topology,
    }
    provenance_contract = {
        field: provenance.get(field)
        for field in _PROVENANCE_FIELDS
    } if isinstance(provenance, Mapping) else {}
    return {
        "plan": _canonical_value(plan_contract),
        "code_provenance": _canonical_value(provenance_contract),
        "config": _config_contract(
            config,
            selection=selection if isinstance(selection, Mapping) else {},
        ),
    }


def _leaf_differences(reference: object, candidate: object, path: str) -> list[str]:
    """遞迴列出兩份 contract 的實際 leaf 差異，不用 section 名稱掩蓋根因。"""

    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        differences: list[str] = []
        for key in sorted(set(reference) | set(candidate)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in reference or key not in candidate:
                differences.append(child_path)
            else:
                differences.extend(_leaf_differences(reference[key], candidate[key], child_path))
        return differences
    if isinstance(reference, list) and isinstance(candidate, list):
        differences = []
        if len(reference) != len(candidate):
            differences.append(f"{path}.length")
        for index, (left, right) in enumerate(zip(reference, candidate, strict=False)):
            differences.extend(_leaf_differences(left, right, f"{path}[{index}]"))
        return differences
    return [] if reference == candidate else [path]


def _diff_contracts(reference: Mapping[str, Any], candidate: Mapping[str, Any], *, label: str) -> list[str]:
    """以 contract leaf path 回報不同設定；允許差異因未投影而不會出現在此處。"""

    return [
        f"pilot_matrix_contract_mismatch:{label}.{path}"
        for path in _leaf_differences(reference, candidate, "")
    ]


def validate_pilot_matrix(run_roots: Sequence[str | Path]) -> dict[str, Any]:
    """唯讀驗證兩個以上 pilot run 是否使用相同工程設定。

    回傳值固定包含 ``schema_version``、``valid``、``run_count``、逐 run ``runs`` 與
    全域 ``errors``。``runs`` 只公開 caller 可稽核的 root、identity 與錯誤碼；內部
    contract 不會寫入輸出。函式本身不建立、修改或刪除任何檔案，CLI 以 ``valid`` 映射
    shell exit code 0（通過）或 2（拒絕）。
    """

    errors: list[str] = []
    raw_roots = list(run_roots)
    if len(raw_roots) < 2:
        errors.append("pilot_matrix_requires_at_least_two_runs")
    roots: list[Path] = []
    seen: set[str] = set()
    for raw in raw_roots:
        root = Path(raw)
        identity_key = os.fspath(root)
        if identity_key in seen:
            errors.append("pilot_matrix_duplicate_run_root")
        seen.add(identity_key)
        roots.append(root)

    runs: list[dict[str, Any]] = []
    for root in roots:
        try:
            run = _read_run(root)
        except Exception as exc:
            code = str(exc).split(":", 1)[0] or "run_unreadable"
            run = {
                "root": str(root),
                "valid": False,
                "identity": {"run_id": None},
                "errors": [code],
                "_plan": None,
                "_config": None,
            }
        runs.append(run)

    comparable = [run for run in runs if run.get("valid") is True]
    if len(comparable) != len(runs):
        errors.append("pilot_matrix_run_invalid")
    if len(comparable) >= 2:
        reference = _comparison_contract(comparable[0])
        for index, run in enumerate(comparable[1:], start=1):
            errors.extend(_diff_contracts(reference, _comparison_contract(run), label=f"run_{index}"))

    for run in runs:
        errors.extend(f"{run['root']}:{error}" for error in run.get("errors", []))
    public_runs = [
        {
            "root": run["root"],
            "valid": run["valid"],
            "identity": run["identity"],
            "errors": sorted(set(run.get("errors", []))),
        }
        for run in runs
    ]
    report = {
        "schema_version": PILOT_MATRIX_SCHEMA_VERSION,
        "valid": not errors,
        "run_count": len(runs),
        "runs": public_runs,
        "errors": sorted(set(errors)),
    }
    # 先 canonicalize 一次，確保浮點／特殊值與遞迴 mapping 不會在 CLI 輸出階段才失敗。
    json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return report


def canonical_pilot_matrix_json(report: Mapping[str, Any]) -> str:
    """將 matrix report 輸出成排序 key、無多餘空白的 canonical JSON。"""

    return json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "MAX_PILOT_MATRIX_JSON_BYTES",
    "PILOT_MATRIX_SCHEMA_VERSION",
    "canonical_pilot_matrix_json",
    "validate_pilot_matrix",
]
