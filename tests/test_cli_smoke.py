"""整合 CLI 的 constant-flow shard 垂直切片測試。"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from lagrangian_backtracking.cli import main


def test_integrated_synthetic_smoke_and_validator(tmp_path: Path) -> None:
    """同一 shard 應可由 synthetic-smoke 產生，再由 validate-shard 獨立驗證。"""

    output = tmp_path / "synthetic-shard"
    assert main(["synthetic-smoke", "--output", str(output)]) == 0
    assert main(["validate-shard", str(output)]) == 0
    events = pq.read_table(output / "events.parquet", columns=["event_type"]).column(0).to_pylist()
    assert events == [
        "local_domain_first_exit",
        "other_site_local_domain_enter",
        "other_site_local_domain_exit",
        "flow_domain_open_exit",
    ]


def test_behavior_manifest_has_ten_records(tmp_path: Path) -> None:
    """CLI 產生的 material/behavior manifest 必須固定為十筆。"""

    output = tmp_path / "behaviors.json"
    assert main(["behavior-manifest", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 10
    assert payload["records"][0]["settling_velocity_mps"] < 0
    assert payload["records"][-1]["settling_velocity_mps"] > 0
