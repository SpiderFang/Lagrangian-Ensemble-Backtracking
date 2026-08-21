"""安全寫入與讀回中途計算狀態，避免混用不同設定或輸入資料。

中途保存的資料雖未正式發布，每一代仍使用不可覆寫的新資料夾。呼叫端以遞增編號建立
新一代，成功完成後才更新外部的「最新版本」指標。這能避免程序中斷破壞上一代可恢復
狀態，也能在續跑前逐項確認設定、輸入資料清單、實驗案例、批次和亂數規則相同，拒絕
以不同流速資料或不同工作配置接續同一批粒子。
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from .models import ParticleState, ParticleStatus
from .outputs import sha256_file


@dataclass(frozen=True, slots=True)
class CheckpointBinding:
    """判定中途計算狀態是否可安全續跑的固定識別欄位。"""

    config_hash: str
    input_inventory_hash: str
    experiment_case_id: str
    shard_id: str
    seed_policy: str
    code_commit: str


def write_checkpoint(
    destination: str | Path,
    *,
    binding: CheckpointBinding,
    states: Sequence[ParticleState],
    sequence: int,
) -> Path:
    """以完整寫入後再更名的方式保存一代粒子狀態，禁止覆寫既有資料夾。"""

    if sequence < 0 or not states:
        raise ValueError("checkpoint sequence 必須非負且 states 不可空")
    if len({state.particle_id for state in states}) != len(states):
        raise ValueError("checkpoint particle_id 必須唯一")
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"不可覆寫 checkpoint：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{target.name}.partial-{uuid4().hex}"
    partial.mkdir()
    try:
        rows = []
        for state in states:
            row = asdict(state)
            row["status"] = state.status.value
            rows.append(row)
        state_path = partial / "particle_states.parquet"
        pq.write_table(pa.Table.from_pylist(rows), state_path)
        metadata = {
            "schema_version": "1.0.0",
            "sequence": sequence,
            "particle_count": len(states),
            "binding": asdict(binding),
            "files": {
                state_path.name: {
                    "size_bytes": state_path.stat().st_size,
                    "sha256": sha256_file(state_path),
                }
            },
        }
        with (partial / "checkpoint.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(partial, target)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return target


def load_checkpoint(
    path: str | Path,
    *,
    expected_binding: CheckpointBinding,
) -> tuple[list[ParticleState], int]:
    """檢查設定綁定、檔案摘要與資料筆數後，讀回不可修改的粒子狀態。

    任一綁定欄位不同都拒絕續跑；即使科學設定看似相同，程式提交版本不同也必須先有
    明確的相容性決定，避免未審查的程式行為改變混入長時間批次計算。
    """

    root = Path(path)
    metadata = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    actual_binding = CheckpointBinding(**metadata["binding"])
    if actual_binding != expected_binding:
        raise ValueError(f"checkpoint binding 不相容：actual={actual_binding}, expected={expected_binding}")
    state_path = root / "particle_states.parquet"
    contract = metadata["files"][state_path.name]
    if state_path.stat().st_size != contract["size_bytes"] or sha256_file(state_path) != contract["sha256"]:
        raise ValueError("checkpoint state checksum 或 size 不符")
    rows = pq.read_table(state_path).to_pylist()
    if len(rows) != metadata["particle_count"]:
        raise ValueError("checkpoint particle row count 不符")
    states = [ParticleState(**{**row, "status": ParticleStatus(row["status"])}) for row in rows]
    if len({state.particle_id for state in states}) != len(states):
        raise ValueError("checkpoint 含重複 particle_id")
    return states, int(metadata["sequence"])
