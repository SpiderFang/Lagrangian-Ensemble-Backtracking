"""run workspace 的只讀完整性、seed、分片與 engineering benchmark 驗證。

``validate_run`` 不會取得 lock、修改 progress、latest pointer 或任何輸出；它會檢查 schema
2 immutable plan、固定 lock topology、檔案 checksum、scenario/seed table 順序與 row count、
seed 導出規則、shard range 及每個 COMPLETE trajectory shard。驗證失敗一律回傳 JSON-safe
``valid=false``，方便 SERVER shell 在不吞掉現場檔案的情況下停止。``benchmark_report``
只彙總已驗證的 progress metadata，並明確標記工程量測不是科學成果，也不允許用 pilot 的
縮減數量重新定義五站 50,000 基礎情境。
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from .engine import ParticleResult
from .outputs import (
    TRAJECTORY_SHARD_SCHEMA_VERSION,
    read_trajectory_shard,
    sha256_file,
    validate_trajectory_shard,
)
from .run_control import (
    _FAILURE_RE,
    _SCENARIO_COLUMNS,
    _SEED_COLUMNS,
    _SHA256_RE,
    RunController,
    _cross_check_plan_progress,
    _read_json,
    _safe_slug,
    _scenario_from_row,
    _scenario_hash,
    load_run_plan,
    load_run_progress,
)
from .runner import ScenarioShard, _scenario_shard_id, iter_run_units, scenario_execution_sort_key
from .scenarios import Scenario

_RUN_FILE_NAMES = frozenset(
    {"normalized_config.json", "input_inventory.json", "scenario_table.parquet", "seed_table.parquet"}
)


@dataclass(frozen=True, slots=True)
class ValidatedTrajectoryShard:
    """一個已通過完整 run binding 的唯讀軌跡分片。

    ``scenarios`` 來自 immutable run plan 所繫結的 scenario table；``results`` 則是依
    plan shard 順序讀回、且逐筆對齊 ``iter_run_units`` 的粒子結果。所有位置都是公尺制
    的 ``x/y/z``，時間欄位使用世界協調時間（UTC）奈秒，回溯年齡使用秒。這個容器只
    保存單一分片，讓後續聚合或 release writer 可以串流處理大型正式 run，而不必把全 run
    的 50,000×M 軌跡同時載入記憶體。

    ``trajectory_manifest_sha256`` 是已通過 trajectory validator 的
    ``output/manifest.json`` 位元組內容雜湊；它與 shard 範圍、結果計數共同作為下游
    結果發布的輸入繫結證據。此類別不提供任何寫入或修復方法，也不代表絕對來源機率；
    下游仍須依正式聚合規格產生條件式來源足跡與相對來源權重。

    ``trajectory_schema_version`` 緊接在 manifest 雜湊之後，保存同一份已驗證
    ``output/manifest.json`` 的原生 ``schema_version`` 字串。它描述目前 shard 的
    固定 payload 契約；一般工程 iterator 保留 validator 已接受的 v1／v2／v3 實際版本，
    交給下游辨識，不在此處把 reader 相容性誤當成正式報告資格。正式 report gate 由
    ``report_pipeline`` 另行要求全 run 單一 v2 或 v3，避免把相容讀取誤當成正式科學
    輸入資格。
    """

    shard_id: str
    scenario_start_index: int
    scenario_stop_index: int
    scenarios: tuple[Scenario, ...]
    results: tuple[ParticleResult, ...]
    output_relative_path: str
    trajectory_manifest_sha256: str
    trajectory_schema_version: str
    particle_count: int
    observation_count: int
    event_count: int


def _relative_file(root: Path, token: Any, *, label: str, directory: bool = False) -> Path:
    """解析 run 內相對 token，拒絕絕對路徑、``..`` 與 symlink。

    ``directory=True`` 供 trajectory shard 目錄使用；其餘 run file 必須是普通檔案。
    """

    if not isinstance(token, str) or not token or Path(token).is_absolute():
        raise ValueError(f"{label} 必須是非空相對路徑")
    path = Path(token)
    if any(part in {"", ".", ".."} for part in path.parts) or "\\" in token:
        raise ValueError(f"{label} 含不安全 path token")
    resolved = root.joinpath(*path.parts)
    if resolved.is_symlink() or (not resolved.is_dir() if directory else not resolved.is_file()):
        expected = "目錄" if directory else "普通檔案"
        raise ValueError(f"{label} 必須指向 run 內{expected}")
    return resolved


def _build_shards(plan: Mapping[str, Any], scenarios: tuple[Any, ...]) -> tuple[ScenarioShard, ...]:
    """依 plan range 重建 shard，並確認全域 range 無缺口或重疊。"""

    rows = plan["shards"]
    if not isinstance(rows, list):
        raise ValueError("run plan shards 必須是 array")
    # 不依 start index 重新排序 plan rows；若有人竄改 row 順序，必須讓 contiguous 契約
    # 失敗，而不是由 validator 幫忙「修正」後靜默接受另一個 execution ordering。
    normalized = rows
    result: list[ScenarioShard] = []
    expected_start = 0
    seen_ids: set[str] = set()
    for index, row in enumerate(normalized):
        if not isinstance(row, dict):
            raise ValueError(f"plan.shards[{index}] 必須是 object")
        required = {
            "shard_id",
            "scenario_start_index",
            "scenario_stop_index",
            "scenario_count",
            "particle_count",
            "scenario_hash",
            "execution_group_id",
            "analysis_region_id",
            "arrival_time_utc_ns",
            "group_part_index",
            "group_part_count",
        }
        if set(row) != required:
            raise ValueError(f"plan.shards[{index}] 欄位集合不符")
        shard_id = _safe_slug(row["shard_id"], label="plan.shard_id")
        if shard_id in seen_ids:
            raise ValueError(f"plan shard_id 重複：{shard_id}")
        seen_ids.add(shard_id)
        start = row["scenario_start_index"]
        stop = row["scenario_stop_index"]
        count = row["scenario_count"]
        particles = row["particle_count"]
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in (start, stop, count, particles)
        ):
            raise ValueError(f"plan.shards[{index}] range/count 必須是整數")
        if start != expected_start or stop <= start or stop - start != count:
            raise ValueError(f"shard range 有缺口或重疊：{shard_id}")
        subset = tuple(scenarios[start:stop])
        if len(subset) != count or _scenario_hash(subset) != row["scenario_hash"]:
            raise ValueError(f"shard scenario hash 不符：{shard_id}")
        shard = ScenarioShard(
            shard_id=shard_id,
            experiment_case_id=str(plan["experiment_case_id"]),
            scenario_start_index=start,
            scenario_stop_index=stop,
            scenarios=subset,
            members_per_scenario=int(plan["members_per_scenario"]),
            execution_group_id=row["execution_group_id"],
            analysis_region_id=row["analysis_region_id"],
            arrival_time_utc_ns=row["arrival_time_utc_ns"],
            group_part_index=row["group_part_index"],
            group_part_count=row["group_part_count"],
        )
        if any(
            scenario.analysis_region_id != shard.analysis_region_id
            or scenario.arrival_time_utc_ns != shard.arrival_time_utc_ns
            for scenario in subset
        ):
            raise ValueError(f"shard scenarios 與 execution group 不一致：{shard_id}")
        expected_shard_id = _scenario_shard_id(
            experiment_case_id=str(plan["experiment_case_id"]),
            execution_group_id=shard.execution_group_id,
            analysis_region_id=shard.analysis_region_id,
            arrival_time_utc_ns=shard.arrival_time_utc_ns,
            group_part_index=shard.group_part_index,
            group_part_count=shard.group_part_count,
            scenarios=subset,
            start=start,
            stop=stop,
        )
        if shard_id != expected_shard_id:
            raise ValueError(f"shard ID 不符合 execution ordering/range contract：{shard_id}")
        if shard.particle_count != particles:
            raise ValueError(f"shard particle count 不符：{shard_id}")
        result.append(shard)
        expected_start = stop
    if expected_start != len(scenarios):
        raise ValueError("shard ranges 未完整覆蓋 scenario table")
    if len(result) != int(plan["shard_count"]):
        raise ValueError("shard_count 與 plan 不一致")
    return tuple(result)


def _validate_run_files(root: Path, plan: Mapping[str, Any], errors: list[str]) -> None:
    """檢查 plan files contract 與四個 immutable run input files。"""

    files = plan.get("files")
    if not isinstance(files, dict) or set(files) != _RUN_FILE_NAMES:
        errors.append("run_plan.files: fixed_set_mismatch")
        return
    for filename, contract in files.items():
        if not isinstance(contract, dict) or set(contract) - {"size_bytes", "sha256", "row_count"}:
            errors.append(f"{filename}: invalid_file_contract")
            continue
        try:
            path = _relative_file(root, filename, label=filename)
        except ValueError as exc:
            errors.append(f"{filename}: {exc}")
            continue
        expected_size = contract.get("size_bytes")
        expected_sha = contract.get("sha256")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            errors.append(f"{filename}: invalid_size")
        elif path.stat().st_size != expected_size:
            errors.append(f"{filename}: size")
        if not isinstance(expected_sha, str) or _SHA256_RE.fullmatch(expected_sha) is None:
            errors.append(f"{filename}: invalid_sha256")
        elif sha256_file(path) != expected_sha:
            errors.append(f"{filename}: checksum")
        if "row_count" in contract:
            try:
                rows = pq.read_metadata(path).num_rows
                if rows != contract["row_count"]:
                    errors.append(f"{filename}: row_count")
            except Exception as exc:  # noqa: BLE001 - validator 必須轉為 JSON-safe error
                errors.append(f"{filename}: invalid_parquet:{type(exc).__name__}")


def _validate_lock_topology(root: Path, plan: Mapping[str, Any], errors: list[str]) -> None:
    """只讀驗證固定 lock topology，不取得鎖也不修復任何檔案。

    validator 與 worker 可同時讀取 workspace；它不應為了檢查而阻塞在 ``flock``。這裡只
    驗證 run plan 宣告的 ``locks/`` 是普通目錄，且恰有 run gate、progress 與每個 shard
    的零長度普通 lock file；active worker 造成的 crash-window 仍由 checkpoint/output
    契約回報，不在 validator 內偷偷 reconcile。
    """

    lock_root = root / str(plan.get("lock_root", ""))
    if lock_root.is_symlink() or not lock_root.is_dir():
        errors.append("locks: missing_not_directory_or_symlink")
        return
    expected = {"run_gate.lock", "progress.lock"} | {
        f"{row['shard_id']}.lock" for row in plan.get("shards", []) if isinstance(row, dict)
    }
    try:
        entries = tuple(lock_root.iterdir())
    except OSError as exc:
        errors.append(f"locks: unreadable:{type(exc).__name__}")
        return
    if {entry.name for entry in entries} != expected:
        errors.append("locks: fixed_set_mismatch")
    for entry in entries:
        try:
            if entry.is_symlink() or not entry.is_file() or entry.stat().st_size != 0:
                errors.append(f"locks: invalid_entry={entry.name}")
        except OSError as exc:
            errors.append(f"locks: invalid_entry={entry.name}:{type(exc).__name__}")


def _load_scenarios(root: Path, plan: Mapping[str, Any], errors: list[str]) -> tuple[Any, ...]:
    """讀 scenario table 並驗證欄位、stable ID、順序與 count。"""

    path = root / "scenario_table.parquet"
    if path.is_symlink() or not path.is_file():
        errors.append("scenario_table.parquet: missing_or_symlink")
        return ()
    try:
        table = pq.read_table(path)
        if tuple(table.column_names) != _SCENARIO_COLUMNS:
            errors.append("scenario_table.parquet: columns")
            return ()
        values = tuple(
            _scenario_from_row(row, label=f"scenario[{index}]") for index, row in enumerate(table.to_pylist())
        )
        if len(values) != int(plan["scenario_count"]):
            errors.append("scenario_table.parquet: plan_count")
        if len({item.scenario_id for item in values}) != len(values):
            errors.append("scenario_table.parquet: duplicate_id")
        if tuple(sorted(values, key=scenario_execution_sort_key)) != values:
            errors.append("scenario_table.parquet: execution_order")
        return values
    except Exception as exc:  # noqa: BLE001 - 壞資料必須回報而不是 crash
        errors.append(f"scenario_table.parquet: invalid:{type(exc).__name__}")
        return ()


def _validate_seed_table(
    root: Path, plan: Mapping[str, Any], shards: Sequence[ScenarioShard], errors: list[str]
) -> None:
    """逐列核對 seed table 的粒子順序與 128-bit deterministic seed。"""

    path = root / "seed_table.parquet"
    if path.is_symlink() or not path.is_file():
        errors.append("seed_table.parquet: missing_or_symlink")
        return
    try:
        parquet = pq.ParquetFile(path)
        if tuple(parquet.schema_arrow.names) != _SEED_COLUMNS:
            errors.append("seed_table.parquet: columns")
            return
        expected_count = sum(shard.particle_count for shard in shards)
        if parquet.metadata.num_rows != expected_count or expected_count != int(plan["particle_count"]):
            errors.append("seed_table.parquet: row_count")
        expected_units = iter(
            chain.from_iterable(
                iter_run_units(shard, master_seed=int(plan["master_seed"])) for shard in shards
            )
        )
        index = 0
        for batch in parquet.iter_batches(batch_size=8192, columns=list(_SEED_COLUMNS)):
            # 每批最多 8192 rows；正式 50,000×M seed table 不會被一次轉成 Python list。
            for row in batch.to_pylist():
                unit = next(expected_units, None)
                if unit is None:
                    errors.append("seed_table.parquet: unexpected_extra_row")
                    return
                expected = {
                    "scenario_id": unit.scenario.scenario_id,
                    "experiment_case_id": unit.experiment_case_id,
                    "member_id": unit.member_id,
                    "particle_id": unit.particle_id,
                    "seed_128_hex": f"{unit.seed:032x}",
                }
                if row != expected:
                    errors.append(f"seed_table.parquet: row_{index}_mismatch")
                    # 繼續掃描可顯示多個錯誤，但每欄錯誤只保留第一筆以免壞表灌滿 stdout。
                    if len(errors) > 100:
                        return
                index += 1
        if next(expected_units, None) is not None:
            errors.append("seed_table.parquet: missing_rows")
    except Exception as exc:  # noqa: BLE001 - validator 必須完整回報壞 Parquet
        errors.append(f"seed_table.parquet: invalid:{type(exc).__name__}")


def _validate_checkpoint_tree(
    controller: RunController,
    plan: Mapping[str, Any],
    progress: Mapping[str, Any],
    shards: Sequence[ScenarioShard],
    errors: list[str],
) -> None:
    """唯讀核對 default/external checkpoint tree、latest 與 progress cross-link。

    controller 的 scanner 與 restore 共用相同 schema 2 binding/order 檢查，但此處傳入
    ``repair_latest=False``，所以 missing/stale latest 只回報 machine-readable recoverable
    error，不修改 SERVER 現場。operator 必須明確執行 ``RunController.reconcile`` 才會修復。
    """

    for shard in shards:
        row = progress["shards"][shard.shard_id]
        try:
            selected = controller._scan_checkpoint_generations(shard, repair_latest=False)
        except Exception as exc:  # noqa: BLE001 - 壞 checkpoint 是 validation result
            errors.append(f"checkpoint[{shard.shard_id}]: invalid:{type(exc).__name__}")
            continue
        sequence = int(row["checkpoint_sequence"])
        token = row["checkpoint_relative_path"]
        if row["lifecycle"] == "PLANNED" and selected is not None:
            errors.append(f"checkpoint[{shard.shard_id}]: planned_has_generation")
            continue
        if selected is None:
            if sequence > 0 or token is not None:
                errors.append(f"checkpoint[{shard.shard_id}]: progress_generation_missing")
            continue
        expected_token = f"{plan['run_id']}/{shard.shard_id}/{selected.path.name}"
        if selected.pointer_state in {"missing", "stale"}:
            errors.append(
                f"checkpoint[{shard.shard_id}]: "
                f"recoverable_latest_{selected.pointer_state}_requires_reconcile"
            )
        if selected.sequence == sequence:
            if token != expected_token:
                errors.append(f"checkpoint[{shard.shard_id}]: progress_path_mismatch")
            counters_invalid = (
                int(row["particle_steps"]) < selected.particle_steps
                or int(row["sweeps_completed"]) < selected.sweeps_completed
                if row["lifecycle"] == "COMPLETE"
                else int(row["particle_steps"]) != selected.particle_steps
                or int(row["sweeps_completed"]) != selected.sweeps_completed
            )
            if counters_invalid:
                errors.append(f"checkpoint[{shard.shard_id}]: progress_counter_mismatch")
        elif selected.sequence > sequence and row["lifecycle"] == "RUNNING":
            errors.append(f"checkpoint[{shard.shard_id}]: recoverable_orphan_requires_reconcile")
        else:
            errors.append(f"checkpoint[{shard.shard_id}]: progress_sequence_mismatch")


def _expected_output_metadata(plan: Mapping[str, Any], shard: ScenarioShard) -> dict[str, Any]:
    """建立 output manifest 必須與 run plan 一致的 immutable metadata 子集合。"""

    provenance = cast(Mapping[str, Any], plan["code_provenance"])
    return {
        "run_id": plan["run_id"],
        "run_kind": plan["run_kind"],
        "config_hash": plan["config_hash"],
        "input_inventory_sha256": plan["raw_input_inventory_sha256"],
        "checkpoint_input_binding_hash": plan["checkpoint_input_binding_hash"],
        "component_canonical_hashes": plan["component_canonical_hashes"],
        "geometry_canonical_hashes": plan["geometry_canonical_hashes"],
        "code_commit": provenance.get("git_commit"),
        "deployment_tree_sha256": provenance["deployment_tree_sha256"],
        "uv_lock_sha256": provenance["uv_lock_sha256"],
        "dirty_flag": provenance.get("git_dirty"),
        "seed_policy": plan["seed_policy"],
        "shard_id": shard.shard_id,
        "experiment_case_id": shard.experiment_case_id,
    }


def _validate_output_tree(
    root: Path,
    plan: Mapping[str, Any],
    progress: Mapping[str, Any],
    shards: Sequence[ScenarioShard],
    errors: list[str],
) -> None:
    """驗證 output topology、published-before-progress 與完整 RunUnit identity/order。

    非 COMPLETE shard 若已有合法 output，代表程序在原子發布 shard 後、更新 progress 前
    中斷；只讀 validator 會明示 recoverable error，不能靜默視為 valid。controller
    ``reconcile`` 才有權在 binding、metadata、checksum 與完整身分通過後採認。
    """

    progress_shards = progress.get("shards")
    if not isinstance(progress_shards, dict):
        errors.append("progress.shards: not_object")
        return
    expected_by_id = {shard.shard_id: shard for shard in shards}
    if set(progress_shards) != set(expected_by_id):
        errors.append("progress.shards: id_set_mismatch")
    output_root = root / str(plan["output_root"])
    if output_root.is_symlink() or not output_root.is_dir():
        errors.append("outputs: missing_not_directory_or_symlink")
        return
    for entry in output_root.iterdir():
        if entry.is_symlink() or not entry.is_dir() or entry.name not in expected_by_id:
            errors.append(f"outputs: unknown_or_symlink={entry.name}")
    for shard in shards:
        row = progress_shards.get(shard.shard_id)
        if not isinstance(row, dict):
            errors.append(f"progress[{shard.shard_id}]: missing")
            continue
        lifecycle = row.get("lifecycle")
        if lifecycle not in {"PLANNED", "RUNNING", "PAUSED", "COMPLETE", "FAILED"}:
            errors.append(f"progress[{shard.shard_id}]: lifecycle")
            continue
        token = row.get("output_relative_path")
        output = output_root / shard.shard_id
        if lifecycle != "COMPLETE" and not output.exists() and not output.is_symlink():
            continue
        try:
            if lifecycle == "COMPLETE":
                output = _relative_file(
                    root, token, label=f"progress[{shard.shard_id}].output", directory=True
                )
            validation = validate_trajectory_shard(
                output,
                require_formal_metadata=plan["run_kind"] == "formal",
                strict_run_metadata=plan["run_kind"] in {"formal", "pilot"},
                expected_metadata=_expected_output_metadata(plan, shard),
            )
            if not validation["valid"]:
                errors.extend(f"output[{shard.shard_id}]: {error}" for error in validation["errors"])
                continue
            if lifecycle != "COMPLETE":
                errors.append(f"output[{shard.shard_id}]: recoverable_published_before_progress")
                continue
            particle_table = pq.read_table(output / "particle_table.parquet")
            rows = particle_table.to_pylist()
            expected_identities = [
                {
                    "particle_id": unit.particle_id,
                    "scenario_id": unit.scenario.scenario_id,
                    "member_id": unit.member_id,
                    "study_site_id": unit.scenario.study_site_id,
                    "analysis_region_id": unit.scenario.analysis_region_id,
                    "receptor_id": unit.scenario.receptor_id,
                }
                for unit in iter_run_units(shard, master_seed=int(plan["master_seed"]))
            ]
            identity_keys = tuple(expected_identities[0]) if expected_identities else ()
            actual_identities = [{key: row.get(key) for key in identity_keys} for row in rows]
            if actual_identities != expected_identities:
                errors.append(f"output[{shard.shard_id}]: run_unit_identity_or_order")
            if len(rows) != shard.particle_count:
                errors.append(f"output[{shard.shard_id}]: particle_count")
        except Exception as exc:  # noqa: BLE001 - validator cannot crash on missing/corrupt output
            errors.append(f"output[{shard.shard_id}]: invalid:{type(exc).__name__}")


def _validate_failure_tree(
    root: Path,
    plan: Mapping[str, Any],
    progress: Mapping[str, Any],
    shards: Sequence[ScenarioShard],
    errors: list[str],
) -> None:
    """驗證 immutable failure artifacts 拓撲、identity 與 progress cross-link。

    成功 resume 後可保留歷史 failure artifacts；只有目前 FAILED shard 必須由 progress 指向
    其中一筆。artifact 不得含 symlink、未知檔名、絕對路徑或額外 JSON 欄位。
    """

    failure_root = root / str(plan["failure_root"])
    if failure_root.is_symlink() or not failure_root.is_dir():
        errors.append("failures: missing_not_directory_or_symlink")
        return
    expected_ids = {shard.shard_id for shard in shards}
    valid_tokens: set[str] = set()
    for entry in failure_root.iterdir():
        if entry.is_symlink() or not entry.is_dir() or entry.name not in expected_ids:
            errors.append(f"failures: unknown_or_symlink={entry.name}")
            continue
        for artifact in entry.iterdir():
            if (
                artifact.is_symlink()
                or not artifact.is_file()
                or _FAILURE_RE.fullmatch(artifact.name) is None
            ):
                errors.append(f"failures[{entry.name}]: unknown_or_symlink={artifact.name}")
                continue
            token = f"{plan['failure_root']}/{entry.name}/{artifact.name}"
            try:
                payload = _read_json(artifact, label="failure artifact")
                if set(payload) != {"schema_version", "run_id", "shard_id", "error_code", "message"}:
                    raise ValueError("failure keys")
                if (
                    payload["schema_version"] != "1.0.0"
                    or payload["run_id"] != plan["run_id"]
                    or payload["shard_id"] != entry.name
                    or type(payload["error_code"]) is not str
                    or not payload["error_code"]
                    or type(payload["message"]) is not str
                ):
                    raise ValueError("failure identity/type")
                if re.search(r"(?:[A-Za-z]:)?/[^\s]+", payload["message"]):
                    raise ValueError("failure absolute path")
                valid_tokens.add(token)
            except Exception as exc:  # noqa: BLE001 - malformed failure must be JSON-safe
                errors.append(f"failures[{entry.name}]: invalid:{type(exc).__name__}")
    for shard in shards:
        row = progress["shards"].get(shard.shard_id, {})
        token = row.get("failure_relative_path") if isinstance(row, dict) else None
        if row.get("lifecycle") == "FAILED" and token not in valid_tokens:
            errors.append(f"failures[{shard.shard_id}]: progress_artifact_missing_or_invalid")


def validate_run(
    path: str | Path,
    *,
    require_complete: bool = False,
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    """只讀驗證一個 run workspace，所有壞資料轉成 JSON-safe ``valid=false``。

    ``require_complete=False`` 允許 operator 驗證進行中的 PLANNED/RUNNING/PAUSED run；
    ``require_complete=True`` 另外要求 run lifecycle 與全部 shard 都是 COMPLETE。此函式
    不會修復 latest 或 progress，修復／採認責任保留給 ``RunController.reconcile``。
    ``checkpoint_root`` 可指定 runtime external root；其絕對路徑只用於本次讀取，不寫入
    result JSON 或 run plan。若 run 曾用 external root 而 caller 省略此參數，progress
    cross-link 會因 default root 找不到 generation 而 fail-closed。
    """

    root = Path(path)
    errors: list[str] = []
    plan: dict[str, Any] | None = None
    progress: dict[str, Any] | None = None
    try:
        if root.is_symlink() or not root.is_dir():
            return {"valid": False, "errors": ["run root missing/not_directory/symlink"], "summary": {}}
        allowed = {
            "normalized_config.json",
            "input_inventory.json",
            "scenario_table.parquet",
            "seed_table.parquet",
            "run_plan.json",
            "run_progress.json",
            "shards",
            "checkpoints",
            "failures",
            "locks",
        }
        for entry in root.iterdir():
            if entry.is_symlink() or entry.name not in allowed:
                errors.append(f"run_root: unknown_or_symlink={entry.name}")
            elif entry.name in {"shards", "checkpoints", "failures", "locks"} and not entry.is_dir():
                errors.append(f"run_root: {entry.name}_not_directory")
        try:
            plan = load_run_plan(root)
            progress = load_run_progress(root)
            _cross_check_plan_progress(plan, progress)
        except Exception as exc:  # noqa: BLE001 - top-level API must not crash
            errors.append(f"plan_or_progress: invalid:{type(exc).__name__}")
            return {"valid": False, "errors": errors, "summary": {}}
        if plan["run_id"] != progress["run_id"]:
            errors.append("plan_progress: run_id")
        _validate_lock_topology(root, plan, errors)
        _validate_run_files(root, plan, errors)
        scenarios = _load_scenarios(root, plan, errors)
        try:
            shards = _build_shards(plan, scenarios)
        except Exception as exc:  # noqa: BLE001 - report range corruption
            errors.append(f"shards: invalid:{type(exc).__name__}")
            shards = ()
        if shards:
            _validate_seed_table(root, plan, shards, errors)
            _validate_output_tree(root, plan, progress, shards, errors)
            _validate_failure_tree(root, plan, progress, shards, errors)
            try:
                # dummy factory 絕不可被只讀 validator 呼叫；controller constructor 與 scanner
                # 都只驗 plan/progress/checkpoint。若未來 constructor 改變，此 lambda 會立即
                # 暴露違反 read-only 邊界的 regression。
                def reject_request_factory(unit: Any) -> Any:
                    """防止 validator 意外建立物理 request。"""

                    del unit
                    raise RuntimeError("validate_run 不可呼叫 request_factory")

                controller = RunController(
                    root,
                    request_factory=reject_request_factory,
                    checkpoint_root=checkpoint_root,
                )
                _validate_checkpoint_tree(controller, plan, progress, shards, errors)
            except Exception as exc:  # noqa: BLE001 - external topology error 必須 JSON-safe
                errors.append(f"checkpoint_root: invalid:{type(exc).__name__}")
        all_complete = bool(shards) and all(
            isinstance(progress["shards"].get(shard.shard_id), dict)
            and progress["shards"][shard.shard_id].get("lifecycle") == "COMPLETE"
            for shard in shards
        )
        if progress["run_lifecycle"] == "COMPLETE" and not all_complete:
            errors.append("run_lifecycle COMPLETE 但不是全部 shard COMPLETE")
        if require_complete and (progress["run_lifecycle"] != "COMPLETE" or not all_complete):
            errors.append("require_complete: run 尚未完成")
        summary = {
            "run_id": plan["run_id"],
            "run_kind": plan["run_kind"],
            "run_lifecycle": progress["run_lifecycle"],
            "scenario_count": plan["scenario_count"],
            "particle_count": plan["particle_count"],
            "shard_count": plan["shard_count"],
            "completed_shard_count": sum(
                1
                for row in progress["shards"].values()
                if isinstance(row, dict) and row.get("lifecycle") == "COMPLETE"
            ),
        }
        return {"valid": not errors, "errors": errors, "summary": summary}
    except Exception as exc:  # noqa: BLE001 - malformed external state is a validation result
        errors.append(f"validator_exception:{type(exc).__name__}")
        return {"valid": False, "errors": errors, "summary": {}}


def iter_complete_run_trajectory_shards(
    path: str | Path,
    *,
    checkpoint_root: str | Path | None = None,
) -> Iterator[ValidatedTrajectoryShard]:
    """依 immutable run 順序串流讀取已完整驗證的 trajectory shards。

    這是聚合與結果發布共用的唯讀輸入邊界：先要求整個 run 通過
    ``validate_run(require_complete=True)``，再重新載入 plan、progress、scenario table
    與 shard range，逐一依 progress 保存的相對路徑驗證並讀回一個分片。每次只把目前
    shard 的 ``ParticleResult`` tuple 放在記憶體中，避免正式的 50,000×M 軌跡被一次展開。
    讀回結果必須逐筆符合 ``iter_run_units`` 的 scenario、member、particle、研究站點、
    分析區域與 receptor 身分及順序；manifest 的粒子、觀測、事件計數與 manifest JSON
    位元組 SHA-256 也會一併核對並保存於回傳容器。這些資料只是通過完整性檢查的條件式
    來源足跡輸入，不代表絕對來源機率。

    Args:
        path: run workspace 目錄。輸入路徑只用於本次讀取，不會放入例外訊息。
        checkpoint_root: 可選的 external checkpoint 根目錄，語意與 ``validate_run`` 相同。

    Yields:
        依 immutable plan shard 順序產生的 ``ValidatedTrajectoryShard``。

    Raises:
        ValueError: run 未完成、任何 plan/progress/output binding 失敗、manifest 計數或
            checksum 不符，或 trajectory 結果的 identity/order 不符。例外訊息刻意不含
            絕對路徑，避免把 SERVER 部署位置洩漏到上層日志。
    """

    try:
        root = Path(path)
        if checkpoint_root is None:
            validation = validate_run(root, require_complete=True)
        else:
            validation = validate_run(
                root,
                require_complete=True,
                checkpoint_root=checkpoint_root,
            )
        if not isinstance(validation, Mapping) or validation.get("valid") is not True:
            raise ValueError("run 未通過 require_complete 驗證")

        plan = load_run_plan(root)
        progress = load_run_progress(root)
        _cross_check_plan_progress(plan, progress)
        errors: list[str] = []
        scenarios = _load_scenarios(root, plan, errors)
        if errors:
            raise ValueError("scenario table 驗證失敗")
        shards = _build_shards(plan, scenarios)
        progress_shards = progress.get("shards")
        if not isinstance(progress_shards, dict):
            raise ValueError("progress.shards 必須是 object")

        for shard in shards:
            row = progress_shards.get(shard.shard_id)
            if not isinstance(row, dict) or row.get("lifecycle") != "COMPLETE":
                raise ValueError("shard 尚未完成")
            output_relative_path = row.get("output_relative_path")
            output = _relative_file(
                root,
                output_relative_path,
                label=f"progress[{shard.shard_id}].output",
                directory=True,
            )
            expected_metadata = _expected_output_metadata(plan, shard)
            require_formal_metadata = plan["run_kind"] == "formal"
            strict_run_metadata = plan["run_kind"] in {"formal", "pilot"}
            validation = validate_trajectory_shard(
                output,
                require_formal_metadata=require_formal_metadata,
                strict_run_metadata=strict_run_metadata,
                expected_metadata=expected_metadata,
            )
            if not isinstance(validation, Mapping) or validation.get("valid") is not True:
                raise ValueError("trajectory shard 驗證失敗")
            manifest = validation.get("manifest")
            if not isinstance(manifest, Mapping):
                raise ValueError("trajectory manifest 缺少解析結果")
            schema_version = manifest.get("schema_version")
            # 這裡只確認已通過 validator 的 manifest 邊界仍是原生非空版本字串；刻意
            # 保留實際 v1／v2／v3 版本，因為一般工程聚合需維持 reader 的舊檔相容性。
            # 正式 report gate 會在更上層要求全 run 單一 v2 或 v3，避免把相容讀取與
            # 正式報告資格混為一談。不以 str() 或 trim 進行可能掩蓋壞 manifest 的修補。
            if type(schema_version) is not str or not schema_version:
                raise ValueError("trajectory manifest schema_version 不合法")
            manifest_counts = tuple(
                manifest.get(name) for name in ("particle_count", "observation_count", "event_count")
            )
            if any(type(value) is not int or value < 0 for value in manifest_counts):
                raise ValueError("trajectory manifest count 不合法")

            results = read_trajectory_shard(
                output,
                require_formal_metadata=require_formal_metadata,
                strict_run_metadata=strict_run_metadata,
                expected_metadata=expected_metadata,
            )
            if not isinstance(results, tuple):
                raise ValueError("trajectory reader 必須回傳 tuple")
            particle_count = int(manifest_counts[0])
            observation_count = int(manifest_counts[1])
            event_count = int(manifest_counts[2])
            if particle_count != shard.particle_count or len(results) != particle_count:
                raise ValueError("trajectory particle count 不符")
            observed_count = sum(len(result.observations) for result in results)
            emitted_count = sum(len(result.events) for result in results)
            if observed_count != observation_count or emitted_count != event_count:
                raise ValueError("trajectory manifest count 與內容不符")

            for index, (result, unit) in enumerate(
                zip(
                    results,
                    iter_run_units(shard, master_seed=int(plan["master_seed"])),
                    strict=True,
                )
            ):
                state = result.final_state
                actual_identity = (
                    state.particle_id,
                    state.scenario_id,
                    state.member_id,
                    state.study_site_id,
                    state.analysis_region_id,
                    state.receptor_id,
                )
                expected_identity = (
                    unit.particle_id,
                    unit.scenario.scenario_id,
                    unit.member_id,
                    unit.scenario.study_site_id,
                    unit.scenario.analysis_region_id,
                    unit.scenario.receptor_id,
                )
                if actual_identity != expected_identity:
                    raise ValueError(f"trajectory run unit identity/order 不符：index={index}")

            yield ValidatedTrajectoryShard(
                shard_id=shard.shard_id,
                scenario_start_index=shard.scenario_start_index,
                scenario_stop_index=shard.scenario_stop_index,
                scenarios=shard.scenarios,
                results=results,
                output_relative_path=output_relative_path,
                trajectory_manifest_sha256=sha256_file(output / "manifest.json"),
                trajectory_schema_version=schema_version,
                particle_count=particle_count,
                observation_count=observation_count,
                event_count=event_count,
            )
    except Exception:
        # Path、Arrow、JSON、NumPy 或上層驗證例外可能附帶絕對部署路徑；公開 iterator
        # 統一轉成固定訊息，讓 SERVER 日志只保留可判讀的 fail-closed 結果。
        raise ValueError("完整 run trajectory shards 讀取失敗") from None


def benchmark_report(
    path: str | Path,
    *,
    require_complete: bool = False,
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    """彙總已通過 run/checkpoint 驗證的工程 metrics，不宣稱科學結果。

    external checkpoint root 只傳給 ``validate_run`` 做完整性檢查，不保存於報告；因此報告
    可攜且不洩漏 SERVER 絕對部署路徑。
    """

    validation = validate_run(
        path,
        require_complete=require_complete,
        checkpoint_root=checkpoint_root,
    )
    base = {
        "valid": bool(validation["valid"]),
        "errors": list(validation["errors"]),
        "engineering_measurement_not_scientific_result": True,
        "pilot_cannot_redefine_five_site_50000_baseline": True,
    }
    if not validation["valid"]:
        return {**base, "summary": validation.get("summary", {})}
    root = Path(path)
    progress = load_run_progress(root)
    plan = load_run_plan(root)
    rows = [row for row in progress["shards"].values() if isinstance(row, dict)]
    metrics_rows = [row.get("metrics", {}) for row in rows if isinstance(row.get("metrics"), dict)]
    sum_keys = ("wall_seconds", "process_cpu_seconds", "output_bytes", "checkpoint_bytes", "particle_steps")
    metrics = {key: sum(float(item.get(key, 0)) for item in metrics_rows) for key in sum_keys}
    metrics["max_rss_bytes"] = max((int(item.get("max_rss_bytes", 0)) for item in metrics_rows), default=0)
    forcing_stats: dict[str, float] = {}
    for item in metrics_rows:
        cache = item.get("forcing_cache_stats")
        if isinstance(cache, Mapping):
            for key, value in cache.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    forcing_stats[key] = forcing_stats.get(key, 0.0) + float(value)
    completed = sum(1 for row in rows if row.get("lifecycle") == "COMPLETE")
    summary = {
        "run_id": plan["run_id"],
        "run_lifecycle": progress["run_lifecycle"],
        "scenario_count": plan["scenario_count"],
        "particle_count": plan["particle_count"],
        "completed_shard_count": completed,
        "shard_count": plan["shard_count"],
        "completed_fraction": completed / plan["shard_count"] if plan["shard_count"] else 0.0,
        "scenario_count_completed": sum(
            int(row.get("scenario_stop_index", 0)) - int(row.get("scenario_start_index", 0))
            for row in rows
            if row.get("lifecycle") == "COMPLETE"
        ),
        "particle_steps": int(metrics["particle_steps"]),
        "wall_seconds": metrics["wall_seconds"],
        "process_cpu_seconds": metrics["process_cpu_seconds"],
        "max_rss_bytes": metrics["max_rss_bytes"],
        "output_bytes": int(metrics["output_bytes"]),
        "checkpoint_bytes": int(metrics["checkpoint_bytes"]),
        "forcing_cache_stats": forcing_stats,
        "baseline_contract": "five study sites, 50,000 base scenarios; pilot reduction is not a new baseline",
    }
    return {**base, "summary": summary}


__all__ = [
    "TRAJECTORY_SHARD_SCHEMA_VERSION",
    "ValidatedTrajectoryShard",
    "benchmark_report",
    "iter_complete_run_trajectory_shards",
    "validate_run",
]
