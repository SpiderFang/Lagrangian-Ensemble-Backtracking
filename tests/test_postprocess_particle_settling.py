"""WP-54：後處理在 v7 逐粒子沉降速度身分表下的輸出欄位與 manifest。

- representative 身分表：離開紀錄表欄位與 manifest 完全沒有新欄位／新鍵（與舊版一致）。
- balanced 身分表：離開紀錄表多出 ``particle_settling_velocity_mps``／
  ``settling_candidate_index`` 兩欄、值逐列等於身分表；``settling_velocity_mps`` 欄仍是
  類別代表速度（後處理與圖面依類別分組不變）；manifest 帶抽樣中繼資料。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyarrow.parquet as pq

from lagrangian_backtracking.native_runner.backward_driver import NATIVE_STATUS_LEFT_INTEREST_REGION
from lagrangian_backtracking.native_runner.postprocess import (
    DEPARTURE_TABLE_SCHEMA,
    PARTICLE_SETTLING_VELOCITY_COLUMN,
    SETTLING_CANDIDATE_INDEX_COLUMN,
    PostprocessAccumulator,
    PostprocessConfig,
)

from test_native_postprocess import (  # noqa: F401
    _JAN_2024_UTC_NS,
    _make_identity_table,
    _make_result,
)


def _identity(n: int = 6):
    return _make_identity_table(
        particle_ids=[f"p{i}" for i in range(n)],
        study_site_id=["A"] * n,
        settling_velocity_mps=[-0.0482] * n,
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * n,
    )


def test_representative_identity_adds_no_columns_or_manifest_keys(tmp_path) -> None:
    identity = _identity()
    result = _make_result(rows=list(range(6)), final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION] * 6)
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path, config=PostprocessConfig())
    accumulator.add_batch("b0", result)
    manifest = accumulator.finalize()

    table = pq.read_table(tmp_path / "departures.parquet")
    assert table.column_names == [f.name for f in DEPARTURE_TABLE_SCHEMA]
    assert PARTICLE_SETTLING_VELOCITY_COLUMN not in table.column_names
    assert "settling_velocity_sampling" not in manifest.config


def test_balanced_identity_writes_per_particle_speed_columns_and_manifest(tmp_path) -> None:
    identity = _identity()
    candidates = np.array([-0.005274, -0.03, -0.0539, -0.0697, -0.1145, -0.187049])
    index = np.array([0, 1, 2, 3, 4, 5], dtype=np.int16)
    sampling = {"mode": "class_candidates_balanced", "groups": [{"study_site_id": "A"}], "master_seed": 42}
    balanced = replace(
        identity,
        particle_settling_velocity_mps=candidates,
        settling_candidate_index=index,
        settling_sampling=sampling,
    )
    result = _make_result(rows=list(range(6)), final_status=[NATIVE_STATUS_LEFT_INTEREST_REGION] * 6)
    accumulator = PostprocessAccumulator(balanced, output_dir=tmp_path, config=PostprocessConfig())
    accumulator.add_batch("b0", result)
    manifest = accumulator.finalize()

    table = pq.read_table(tmp_path / "departures.parquet")
    assert table.column_names[-2:] == [PARTICLE_SETTLING_VELOCITY_COLUMN, SETTLING_CANDIDATE_INDEX_COLUMN]
    rows = {row["row_particle_index"]: row for row in table.to_pylist()}
    for i in range(6):
        assert rows[i][PARTICLE_SETTLING_VELOCITY_COLUMN] == candidates[i]
        assert rows[i][SETTLING_CANDIDATE_INDEX_COLUMN] == index[i]
        assert rows[i]["settling_velocity_mps"] == -0.0482  # 類別代表速度，分組鍵不變
    assert manifest.config["settling_velocity_sampling"] == sampling
    assert manifest.identity_table_content_sha256 == balanced.content_sha256
