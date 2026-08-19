"""不可混用設定／輸入／seed 的 immutable checkpoint 寫入與恢復。

checkpoint 是未發布工作資料，但每一代仍採不可變目錄；caller 以遞增 sequence 建立新
目錄，成功後才更新外部 latest pointer。這避免程序中斷時破壞上一代可恢復狀態，也
讓 restart 能逐項驗證 config hash、input inventory hash、experiment、shard 與 seed
policy，拒絕以不同 forcing 或 worker 配置接續同一粒子集合。
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
    """決定 checkpoint 是否可安全續跑的不可變識別欄位。"""

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
    """原子發布一代 particle state checkpoint，禁止覆寫既有目錄。"""

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
    """驗證 binding/checksum/row count 後重建不可變 particle states。

    任一 binding 欄位不同都拒絕恢復；即使科學設定相同，code commit 不同也需另建立
    migration/compatibility 決策，避免未審查的程式行為改變混入長批次。
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
