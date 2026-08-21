"""安全寫出軌跡、事件與檢查資料，避免留下半套結果。

所有粒子的觀測點放在連續的一維陣列；``trajectory_offsets`` 記錄每條軌跡在陣列中的起點
和終點，所以不需要難以處理的巢狀陣列。每個結果分片先寫到暫存目錄，確認檔案大小、筆數
和 SHA-256 檔案指紋都正確後，才一次改成正式名稱。已有同名結果時一律拒絕覆寫。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .engine import ParticleResult
from .models import ParticleStatus


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """分段讀取檔案並計算 SHA-256 指紋，不把大型資料一次放進記憶體。"""

    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """遞迴把 Enum 轉 value，使 dataclass 事件可交給 Arrow/JSON。"""

    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: object) -> None:
    """以 UTF-8、排序 key 與禁止 NaN 的格式寫可稽核 JSON。"""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_trajectory_shard(
    destination: str | Path,
    results: Sequence[ParticleResult],
    *,
    run_metadata: dict[str, Any],
) -> Path:
    """安全發布一個已完成的粒子結果分片。

    ``run_metadata`` 至少要寫下設定與輸入資料的指紋、程式版本、是否有未提交修改、亂數
    種子規則和此分片涵蓋的情境範圍。函式不替呼叫端猜這些資訊。空分片沒有可用分母，
    不能產生可解釋的比例，因此直接拒絕。
    """

    if not results:
        raise ValueError("trajectory shard 不可為空")
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"不可覆寫既有 shard：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{target.name}.partial-{uuid4().hex}"
    partial.mkdir()
    try:
        particle_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        offsets = [0]
        time_values: list[int] = []
        age_values: list[float] = []
        x_values: list[float] = []
        y_values: list[float] = []
        z_values: list[float] = []
        status_values: list[str] = []
        for result in results:
            state = result.final_state
            particle_rows.append(
                {
                    "particle_id": state.particle_id,
                    "scenario_id": state.scenario_id,
                    "member_id": state.member_id,
                    "study_site_id": state.study_site_id,
                    "analysis_region_id": state.analysis_region_id,
                    "receptor_id": state.receptor_id,
                    "final_status": state.status.value,
                    "step_count": result.step_count,
                    "minimum_clamp_count": result.minimum_clamp_count,
                }
            )
            for observation in result.observations:
                time_values.append(observation.time_utc_ns)
                age_values.append(observation.age_seconds)
                x_values.append(observation.x_m)
                y_values.append(observation.y_m)
                z_values.append(observation.z_m)
                status_values.append(observation.status.value)
            offsets.append(len(time_values))
            for event in result.events:
                row = {key: _json_safe(value) for key, value in asdict(event).items()}
                row["attributes_json"] = json.dumps(row.pop("attributes"), ensure_ascii=False, sort_keys=True)
                event_rows.append(row)

        pq.write_table(pa.Table.from_pylist(particle_rows), partial / "particle_table.parquet")
        event_table = (
            pa.Table.from_pylist(event_rows)
            if event_rows
            else pa.table({"particle_id": pa.array([], pa.string())})
        )
        pq.write_table(event_table, partial / "events.parquet")
        arrays = {
            "trajectory_offsets.npy": np.asarray(offsets, dtype=np.int64),
            "time_utc_ns.npy": np.asarray(time_values, dtype=np.int64),
            "age_seconds.npy": np.asarray(age_values, dtype=np.float64),
            "x_m.npy": np.asarray(x_values, dtype=np.float64),
            "y_m.npy": np.asarray(y_values, dtype=np.float64),
            "z_m.npy": np.asarray(z_values, dtype=np.float64),
            "status_code.npy": np.asarray(status_values, dtype="U32"),
        }
        for filename, values in arrays.items():
            np.save(partial / filename, values, allow_pickle=False)
        files = sorted(path for path in partial.iterdir() if path.is_file())
        manifest = {
            "schema_version": "1.0.0",
            "particle_count": len(results),
            "observation_count": len(time_values),
            "event_count": len(event_rows),
            "run_metadata": run_metadata,
            "files": {
                path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files
            },
        }
        _write_json(partial / "manifest.json", manifest)
        os.replace(partial, target)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return target


def validate_trajectory_shard(path: str | Path, *, require_formal_metadata: bool = False) -> dict[str, Any]:
    """重新確認每個檔案、時間順序和粒子狀態都正確。

    ``require_formal_metadata`` 只在準備做正式彙整或發布時啟用；合成測試可少部分正式
    資訊。只要陣列長度不一致、座標不是有限數字、回溯時間沒有增加、UTC 時間沒有往過去
    走，或最後觀測狀態和粒子表不同，就列為錯誤；能打開檔案不代表結果可用。
    """

    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    for filename, contract in manifest["files"].items():
        file_path = root / filename
        if not file_path.is_file() or file_path.stat().st_size != contract["size_bytes"]:
            errors.append(f"{filename}: missing_or_size")
        elif sha256_file(file_path) != contract["sha256"]:
            errors.append(f"{filename}: checksum")
    offsets = np.load(root / "trajectory_offsets.npy", mmap_mode="r", allow_pickle=False)
    observations = np.load(root / "time_utc_ns.npy", mmap_mode="r", allow_pickle=False)
    if (
        offsets.shape != (manifest["particle_count"] + 1,)
        or offsets[0] != 0
        or offsets[-1] != observations.size
    ):
        errors.append("trajectory_offsets: csr_contract")
    if np.any(np.diff(offsets) <= 0):
        errors.append("trajectory_offsets: empty_or_nonmonotonic")
    particle_table = pq.read_table(root / "particle_table.parquet")
    if particle_table.num_rows != manifest["particle_count"]:
        errors.append("particle_table: row_count")
    if pq.read_metadata(root / "events.parquet").num_rows != manifest["event_count"]:
        errors.append("events: row_count")
    arrays = {
        "age_seconds": np.load(root / "age_seconds.npy", mmap_mode="r", allow_pickle=False),
        "x_m": np.load(root / "x_m.npy", mmap_mode="r", allow_pickle=False),
        "y_m": np.load(root / "y_m.npy", mmap_mode="r", allow_pickle=False),
        "z_m": np.load(root / "z_m.npy", mmap_mode="r", allow_pickle=False),
        "status_code": np.load(root / "status_code.npy", mmap_mode="r", allow_pickle=False),
    }
    for name, values in arrays.items():
        if values.shape != observations.shape:
            errors.append(f"{name}: observation_shape")
    for name in ("age_seconds", "x_m", "y_m", "z_m"):
        if not np.all(np.isfinite(arrays[name])):
            errors.append(f"{name}: nonfinite")
    known_statuses = {status.value for status in ParticleStatus}
    if any(str(value) not in known_statuses for value in arrays["status_code"]):
        errors.append("status_code: unknown")
    particle_rows = particle_table.to_pylist()
    particle_ids = [row["particle_id"] for row in particle_rows]
    if len(set(particle_ids)) != len(particle_ids):
        errors.append("particle_table: duplicate_particle_id")
    if offsets.shape == (manifest["particle_count"] + 1,):
        for index, row in enumerate(particle_rows):
            start = int(offsets[index])
            stop = int(offsets[index + 1])
            age = arrays["age_seconds"][start:stop]
            utc = observations[start:stop]
            if age.size == 0 or not np.isclose(age[0], 0.0) or np.any(np.diff(age) <= 0):
                errors.append(f"particle[{index}]: age_contract")
            if utc.size == 0 or np.any(np.diff(utc) >= 0):
                errors.append(f"particle[{index}]: backward_time_contract")
            if stop > start and str(arrays["status_code"][stop - 1]) != row["final_status"]:
                errors.append(f"particle[{index}]: final_status")
    if require_formal_metadata:
        required = {
            "config_hash",
            "input_inventory_hash",
            "code_commit",
            "dirty_flag",
            "seed_policy",
            "shard_id",
            "experiment_case_id",
        }
        missing = sorted(required - set(manifest.get("run_metadata", {})))
        if missing:
            errors.append("run_metadata: missing_formal_fields=" + ",".join(missing))
    return {"valid": not errors, "errors": errors, "manifest": manifest}


def temporary_output_directory(prefix: str = "lbt-") -> Path:
    """建立明示位於系統 temporary root 的測試／smoke 目錄。"""

    return Path(tempfile.mkdtemp(prefix=prefix))
