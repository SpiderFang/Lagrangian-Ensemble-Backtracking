"""schema 3 checkpoint segment chain 的垂直整合測試。

本檔沿用 ``test_checkpoint_execution`` 的 synthetic shard、request factory 與 binding，
只驗證 checkpoint 拓撲、cursor、checksum、chain 及重啟後的逐欄結果；資料不是 OCM／NWW3
產品，也不能解讀成正式研究成果。
"""

from __future__ import annotations

import gzip
import json
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_checkpoint_execution import _binding, _factory, _shard, _write_legacy_batch_checkpoint

import lagrangian_backtracking.checkpoint as checkpoint_module
from lagrangian_backtracking.checkpoint import (
    inspect_execution_checkpoint,
    load_execution_checkpoint,
    write_execution_checkpoint,
)
from lagrangian_backtracking.engine import EnvironmentSampleStatus
from lagrangian_backtracking.models import ParticleStatus
from lagrangian_backtracking.outputs import sha256_file
from lagrangian_backtracking.production import ProductionBatch


def _write_chain(tmp_path: Path, *, generation_count: int = 5) -> tuple[Path, list[Path]]:
    """建立多代 v3 synthetic chain，回傳最後一代與全部 generation 路徑。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    previous: Path | None = None
    paths: list[Path] = []
    for sequence in range(1, generation_count + 1):
        batch.advance()
        current = batch.write_checkpoint(
            tmp_path / f"checkpoint-{sequence:08d}",
            binding=_binding(),
            sequence=sequence,
            previous_checkpoint=previous,
        )
        paths.append(current)
        previous = current
    assert previous is not None
    return previous, paths


def _refresh_manifest(
    root: Path, filename: str, *, update_uncompressed_size: bool = True
) -> None:
    """測試篡改後只重算指定 payload manifest，保留 semantic gate 的可觀測性。"""

    payload = root / filename
    metadata_path = root / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = metadata["files"][filename]
    contract["size_bytes"] = payload.stat().st_size
    contract["sha256"] = sha256_file(payload)
    if filename.endswith(".gz") and update_uncompressed_size:
        with gzip.open(payload, "rb") as handle:
            contract["uncompressed_size_bytes"] = len(handle.read())
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_payload(root: Path, stem: str) -> dict[str, object]:
    """讀取 schema 3.1 gzip payload，讓測試不依賴已淘汰的未壓縮檔名。"""

    with gzip.open(root / f"{stem}.json.gz", "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    """以與 production writer 相同的 deterministic gzip 參數改寫測試 payload。"""

    encoder = json.JSONEncoder(ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("wb") as raw_handle, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw_handle, compresslevel=1, mtime=0
    ) as gzip_handle:
        for fragment in encoder.iterencode(payload):
            gzip_handle.write(fragment.encode("utf-8"))
        gzip_handle.write(b"\n")


def _convert_v31_root_to_v30(root: Path) -> None:
    """將測試 root 轉為已發布的 schema 3.0 拓撲，模擬既有舊 chain。"""

    metadata_path = root / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    files: dict[str, dict[str, object]] = {}
    for stem in ("compact_state", "history_segment"):
        payload = _read_payload(root, stem)
        payload["schema_version"] = "3.0.0"
        raw_path = root / f"{stem}.json"
        raw_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        files[raw_path.name] = {
            "size_bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
        }
        (root / f"{stem}.json.gz").unlink()
    metadata["schema_version"] = "3.0.0"
    metadata["files"] = files
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_schema31_gzip_payload_is_deterministic_and_manifest_uses_compressed_size(
    tmp_path: Path,
) -> None:
    """相同 payload 的 gzip bytes 必須固定，manifest size 則是壓縮檔實際長度。"""

    first, _ = _write_chain(tmp_path / "first", generation_count=1)
    second, _ = _write_chain(tmp_path / "second", generation_count=1)
    first_metadata = json.loads((first / "checkpoint.json").read_text(encoding="utf-8"))
    second_metadata = json.loads((second / "checkpoint.json").read_text(encoding="utf-8"))
    assert first_metadata["schema_version"] == second_metadata["schema_version"] == "3.1.0"
    for filename in ("compact_state.json.gz", "history_segment.json.gz"):
        first_path = first / filename
        second_path = second / filename
        assert first_path.read_bytes() == second_path.read_bytes()
        contract = first_metadata["files"][filename]
        assert contract["content_encoding"] == "gzip"
        assert contract["size_bytes"] == first_path.stat().st_size
        with gzip.open(first_path, "rb") as handle:
            assert contract["uncompressed_size_bytes"] == len(handle.read())


def test_schema30_to_schema31_mixed_chain_preserves_history(tmp_path: Path) -> None:
    """既有 3.0 root 接續 3.1 generation 時，loader 必須保留完整 history。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    first = batch.write_checkpoint(
        tmp_path / "checkpoint-00000001", binding=_binding(), sequence=1
    )
    _convert_v31_root_to_v30(first)
    first_bytes_before = {
        path.name: path.read_bytes() for path in first.iterdir() if path.is_file()
    }
    batch.advance()
    second = batch.write_checkpoint(
        tmp_path / "checkpoint-00000002",
        binding=_binding(),
        sequence=2,
        previous_checkpoint=first,
    )
    loaded = load_execution_checkpoint(second, expected_binding=_binding())
    assert loaded.schema_version == "3.1.0"
    assert loaded.executions == [runtime.execution for runtime in batch.runtimes]
    assert (first / "compact_state.json").is_file()
    assert (second / "compact_state.json.gz").is_file()
    assert first_bytes_before == {
        path.name: path.read_bytes() for path in first.iterdir() if path.is_file()
    }


def test_schema31_to_schema30_downgrade_is_rejected(tmp_path: Path) -> None:
    """3.1 chain 不得以舊 3.0 generation 續接，避免回降切斷 gzip 契約。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    first = batch.write_checkpoint(
        tmp_path / "checkpoint-00000001", binding=_binding(), sequence=1
    )
    batch.advance()
    second = batch.write_checkpoint(
        tmp_path / "checkpoint-00000002",
        binding=_binding(),
        sequence=2,
        previous_checkpoint=first,
    )
    _convert_v31_root_to_v30(second)
    with pytest.raises(ValueError, match=r"3\.1.*降回.*3\.0"):
        load_execution_checkpoint(second, expected_binding=_binding())


def test_schema31_writer_rejects_downgraded_middle_generation_before_partial(
    tmp_path: Path,
) -> None:
    """續寫前必須掃描完整鏈，拒絕自洽 3.0 中間代切斷既有 3.1 歷史。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    first = batch.write_checkpoint(
        tmp_path / "checkpoint-00000001", binding=_binding(), sequence=1
    )
    batch.advance()
    second = batch.write_checkpoint(
        tmp_path / "checkpoint-00000002",
        binding=_binding(),
        sequence=2,
        previous_checkpoint=first,
    )
    _convert_v31_root_to_v30(second)
    batch.advance()
    target = tmp_path / "checkpoint-00000003"
    with pytest.raises(ValueError, match=r"3\.1.*降回.*3\.0"):
        batch.write_checkpoint(
            target,
            binding=_binding(),
            sequence=3,
            previous_checkpoint=second,
        )
    assert not target.exists()
    assert not tuple(tmp_path.glob(f".{target.name}.partial-*"))


@pytest.mark.parametrize("damage", ["truncate", "bit"])
def test_schema31_gzip_corruption_is_fail_closed(tmp_path: Path, damage: str) -> None:
    """gzip trailer 截斷或壓縮 bytes 篡改即使重算 manifest 也不得被採認。"""

    target, _ = _write_chain(tmp_path, generation_count=1)
    payload = target / "history_segment.json.gz"
    corrupted = bytearray(payload.read_bytes())
    if damage == "truncate":
        del corrupted[-8:]
    else:
        corrupted[len(corrupted) // 2] ^= 0x01
    payload.write_bytes(corrupted)
    _refresh_manifest(target, payload.name, update_uncompressed_size=False)
    with pytest.raises(ValueError, match="checksum|gzip|JSON|uncompressed_size"):
        load_execution_checkpoint(target, expected_binding=_binding())


def test_schema31_gzip_writer_failures_leave_no_adoptable_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gzip write、close 與 generation rename 失敗都不得留下 target 或 partial。"""

    def write_failure(*args: object, **kwargs: object) -> int:
        """模擬 payload 尚未完成時的寫入失敗。"""

        del args, kwargs
        raise OSError("simulated gzip write failure")

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    target = tmp_path / "write" / "checkpoint-00000001"
    monkeypatch.setattr(checkpoint_module, "_write_json_gzip", write_failure)
    with pytest.raises(OSError, match="gzip write"):
        batch.write_checkpoint(target, binding=_binding(), sequence=1)
    assert not target.exists()
    assert not tuple(target.parent.glob(f".{target.name}.partial-*"))

    monkeypatch.undo()
    original_close = checkpoint_module.gzip.GzipFile.close

    def close_failure(gzip_file: object) -> None:
        """先完成 trailer 再回報 close 失敗，驗證 writer cleanup 邊界。"""

        original_close(gzip_file)  # type: ignore[arg-type]
        raise OSError("simulated gzip close failure")

    monkeypatch.setattr(checkpoint_module.gzip.GzipFile, "close", close_failure)
    target = tmp_path / "close" / "checkpoint-00000001"
    with pytest.raises(OSError, match="gzip close"):
        batch.write_checkpoint(target, binding=_binding(), sequence=1)
    assert not target.exists()
    assert not tuple(target.parent.glob(f".{target.name}.partial-*"))

    monkeypatch.undo()

    def rename_failure(*args: object, **kwargs: object) -> None:
        """模擬 payload 完整後 atomic rename 失敗。"""

        del args, kwargs
        raise OSError("simulated generation rename failure")

    monkeypatch.setattr(checkpoint_module.os, "replace", rename_failure)
    target = tmp_path / "rename" / "checkpoint-00000001"
    with pytest.raises(OSError, match="generation rename"):
        batch.write_checkpoint(target, binding=_binding(), sequence=1)
    assert not target.exists()
    assert not tuple(target.parent.glob(f".{target.name}.partial-*"))


def test_v3_segment_chain_restores_exact_result_and_only_appends_rows(tmp_path: Path) -> None:
    """多次 checkpoint/resume 必須逐欄等於不中斷執行，且每代只帶新增 history rows。"""

    final_path, paths = _write_chain(tmp_path)
    uninterrupted = ProductionBatch(_shard(), master_seed=123, request_factory=_factory).complete()
    loaded = load_execution_checkpoint(
        final_path,
        expected_binding=_binding(),
        expected_run_units=ProductionBatch(_shard(), master_seed=123, request_factory=_factory).units,
    )
    assert loaded.schema_version == "3.1.0"
    assert loaded.sequence == len(paths)

    total_segment_observations = 0
    total_segment_events = 0
    previous_compact: dict[str, object] | None = None
    for sequence, path in enumerate(paths, start=1):
        metadata = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
        compact = _read_payload(path, "compact_state")
        segment = _read_payload(path, "history_segment")
        assert metadata["schema_version"] == "3.1.0"
        assert set(compact) == {"schema_version", "sequence", "binding", "particle_order", "records"}
        assert all("observations" not in record for record in compact["records"])
        for index, record in enumerate(segment["records"]):
            total_segment_observations += len(record["observations"])
            total_segment_events += len(record["events"])
            if previous_compact is not None:
                previous_record = previous_compact["records"][index]
                assert record["observation_start_cursor"] == previous_record["observation_cursor"]
                assert record["event_start_cursor"] == previous_record["event_cursor"]
        previous_compact = compact
        assert metadata["segment"]["sequence"] == sequence
    assert total_segment_observations + len(loaded.executions) == sum(
        len(execution.observations) for execution in loaded.executions
    )
    assert total_segment_events == sum(len(execution.events) for execution in loaded.executions)

    restored = ProductionBatch.from_checkpoint(
        final_path,
        shard=_shard(),
        master_seed=123,
        request_factory=_factory,
        expected_binding=_binding(),
        active_chunk_size=1,
    )
    assert restored.complete() == uninterrupted


def test_v3_inspection_uses_compact_counters(tmp_path: Path) -> None:
    """generation scan 的 counter inspection 應能讀取 v3 而不要求先還原完整 history。"""

    final_path, paths = _write_chain(tmp_path, generation_count=3)
    del final_path
    for sequence, path in enumerate(paths, start=1):
        metadata = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
        compact = _read_payload(path, "compact_state")
        step_counts = [record["step_count"] for record in compact["records"]]
        assert inspect_execution_checkpoint(path, expected_binding=_binding()) == (
            sequence,
            max(step_counts),
            sum(step_counts),
        )
        assert metadata["observation_count"] >= len(compact["records"])


def test_v3_generation_storage_grows_linearly_with_checkpoint_count(tmp_path: Path) -> None:
    """generation 數加倍時，累計 payload 不應呈現完整 history 重寫的平方成長。"""

    _, four_paths = _write_chain(tmp_path / "four", generation_count=4)
    _, eight_paths = _write_chain(tmp_path / "eight", generation_count=8)

    def stored_bytes(paths: list[Path]) -> int:
        """計算 generation 目錄內所有檔案的實際位元組，不含外層 latest pointer。"""

        return sum(path.stat().st_size for root in paths for path in root.iterdir())

    four_bytes = stored_bytes(four_paths)
    eight_bytes = stored_bytes(eight_paths)
    # JSON manifest／compact 會帶來固定常數，故不要求恰好二倍；小於三倍可排除
    # 每代重寫累積 history 所產生的近似四倍成長，同時保留測試對實作的寬容度。
    assert eight_bytes < four_bytes * 3


def test_v3_missing_generation_is_rejected(tmp_path: Path) -> None:
    """刪除 chain 中間 generation 時，loader 不得退回較舊狀態繼續執行。"""

    final_path, paths = _write_chain(tmp_path, generation_count=3)
    shutil.rmtree(paths[1])
    with pytest.raises(ValueError, match="缺少前一代"):
        load_execution_checkpoint(final_path, expected_binding=_binding())


def test_v3_cursor_tampering_is_rejected_after_checksum_refresh(tmp_path: Path) -> None:
    """即使同步更新 checksum，篡改 segment cursor 仍必須被 chain continuity gate 拒絕。"""

    final_path, paths = _write_chain(tmp_path, generation_count=3)
    segment_path = paths[1] / "history_segment.json.gz"
    segment = _read_payload(paths[1], "history_segment")
    record = segment["records"][0]
    record["observation_start_cursor"] = 0
    record["observation_end_cursor"] = len(record["observations"])
    _write_payload(segment_path, segment)
    _refresh_manifest(paths[1], "history_segment.json.gz")
    with pytest.raises(ValueError, match="cursor"):
        load_execution_checkpoint(final_path, expected_binding=_binding())


def test_v3_identity_and_binding_tampering_are_rejected_after_checksum_refresh(
    tmp_path: Path,
) -> None:
    """compact 的 particle order 或 binding 被修改時，必須在 restore 前 fail closed。"""

    _, paths = _write_chain(tmp_path / "order", generation_count=2)
    compact_path = paths[1] / "compact_state.json.gz"
    compact = _read_payload(paths[1], "compact_state")
    compact["records"] = list(reversed(compact["records"]))
    _write_payload(compact_path, compact)
    _refresh_manifest(paths[1], "compact_state.json.gz")
    with pytest.raises(ValueError, match="order"):
        load_execution_checkpoint(paths[1], expected_binding=_binding())

    _, binding_paths = _write_chain(tmp_path / "binding", generation_count=2)
    binding_compact_path = binding_paths[1] / "compact_state.json.gz"
    binding_compact = _read_payload(binding_paths[1], "compact_state")
    binding_compact["binding"]["config_hash"] = "tampered"
    _write_payload(binding_compact_path, binding_compact)
    _refresh_manifest(binding_paths[1], "compact_state.json.gz")
    with pytest.raises(ValueError, match="binding"):
        load_execution_checkpoint(binding_paths[1], expected_binding=_binding())


def test_v3_child_links_previous_checkpoint_manifest_including_rng_and_compact(
    tmp_path: Path,
) -> None:
    """子代必須鏈前代 checkpoint.json，前代 RNG/compact 被重寫後不可繼續通過。"""

    _, paths = _write_chain(tmp_path, generation_count=2)
    previous_compact_path = paths[0] / "compact_state.json.gz"
    previous_compact = _read_payload(paths[0], "compact_state")
    previous_compact["records"][0]["step_count"] += 1
    _write_payload(previous_compact_path, previous_compact)
    _refresh_manifest(paths[0], "compact_state.json.gz")
    with pytest.raises(ValueError, match="previous checkpoint.json SHA"):
        load_execution_checkpoint(paths[1], expected_binding=_binding())


def test_schema3_rejects_higher_generation_forged_as_independent_root(tmp_path: Path) -> None:
    """sequence>1 不得藉由清空 previous hash 偽裝成可採認的獨立 root。"""

    _, paths = _write_chain(tmp_path, generation_count=1)
    forged = tmp_path / "checkpoint-00000002"
    shutil.copytree(paths[0], forged)
    for stem in ("compact_state", "history_segment"):
        filename = f"{stem}.json.gz"
        payload_path = forged / filename
        payload = _read_payload(forged, stem)
        payload["sequence"] = 2
        _write_payload(payload_path, payload)
        _refresh_manifest(forged, filename)
    metadata_path = forged / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sequence"] = 2
    metadata["chain_root_sequence"] = 2
    metadata["segment"]["sequence"] = 2
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="root 必須是 sequence 1"):
        load_execution_checkpoint(forged, expected_binding=_binding())


def test_v3_migration_root_preserves_legacy_events_without_rewriting_legacy_tree(
    tmp_path: Path,
) -> None:
    """從舊 schema 續跑時，v3 多代 chain 必須保留事件且不得改寫舊目錄。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    uninterrupted = ProductionBatch(_shard(), master_seed=123, request_factory=_factory).complete()
    batch.advance()
    legacy_root = _write_legacy_batch_checkpoint(
        batch,
        tmp_path / "checkpoint-00000001",
        sequence=1,
    )
    legacy_metadata_before = (legacy_root / "checkpoint.json").read_bytes()
    batch.advance()

    migrated_root = batch.write_checkpoint(
        tmp_path / "checkpoint-00000002",
        binding=_binding(),
        sequence=2,
        previous_checkpoint=legacy_root,
    )
    loaded = load_execution_checkpoint(migrated_root, expected_binding=_binding())
    assert loaded.executions == [runtime.execution for runtime in batch.runtimes]
    assert (legacy_root / "checkpoint.json").read_bytes() == legacy_metadata_before

    migrated_metadata = json.loads(
        (migrated_root / "checkpoint.json").read_text(encoding="utf-8")
    )
    migrated_segment = _read_payload(migrated_root, "history_segment")
    assert migrated_metadata["chain_root_sequence"] == 2
    assert migrated_metadata["segment"]["previous_checkpoint_json_sha256"] is None
    assert all(record["event_start_cursor"] == 0 for record in migrated_segment["records"])
    assert sum(len(record["events"]) for record in migrated_segment["records"]) == sum(
        len(execution.events) for execution in loaded.executions
    )

    # 遷移 root 之後再連續發布兩代 v3，確認 root sequence 不是依目前 generation
    # 重新計算；最終從第三代 v3 restore 後必須與不中斷批次逐欄相同。
    batch.advance()
    second_v3 = batch.write_checkpoint(
        tmp_path / "checkpoint-00000003",
        binding=_binding(),
        sequence=3,
        previous_checkpoint=migrated_root,
    )
    batch.advance()
    third_v3 = batch.write_checkpoint(
        tmp_path / "checkpoint-00000004",
        binding=_binding(),
        sequence=4,
        previous_checkpoint=second_v3,
    )
    for path in (migrated_root, second_v3, third_v3):
        metadata = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
        assert metadata["chain_root_sequence"] == 2
    restored = ProductionBatch.from_checkpoint(
        third_v3,
        shard=_shard(),
        master_seed=123,
        request_factory=_factory,
        expected_binding=_binding(),
        active_chunk_size=1,
    )
    assert restored.complete() == uninterrupted


def test_v3_migration_pending_age_uses_engine_tolerance(tmp_path: Path) -> None:
    """schema 2 migration 也應接受 engine 同點 pending age 的微小浮點誤差。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    legacy_root = _write_legacy_batch_checkpoint(
        batch,
        tmp_path / "checkpoint-00000001",
        sequence=1,
    )
    pending = batch.runtimes[1].execution.observations[-1]
    batch.runtimes[1].execution.observations[-1] = replace(
        pending,
        age_seconds=pending.age_seconds + 5.0e-13,
    )
    migrated_root = batch.write_checkpoint(
        tmp_path / "checkpoint-00000002",
        binding=_binding(),
        sequence=2,
        previous_checkpoint=legacy_root,
    )
    loaded = load_execution_checkpoint(migrated_root, expected_binding=_binding())
    assert loaded.executions[1].observations[-1].age_seconds == pytest.approx(
        pending.age_seconds + 5.0e-13
    )


@pytest.mark.parametrize("mutation", ["state", "rng"])
def test_v3_migration_freezes_legacy_terminal_particle(
    tmp_path: Path, mutation: str
) -> None:
    """schema 2→v3 遷移時，legacy 已終止粒子不可改變 state 或 RNG。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    runtime = batch.runtimes[0]
    runtime.execution.state = replace(
        runtime.execution.state,
        status=ParticleStatus.MAX_AGE,
    )
    legacy_root = _write_legacy_batch_checkpoint(
        batch,
        tmp_path / "checkpoint-00000001",
        sequence=1,
    )
    if mutation == "state":
        runtime.execution.state = replace(
            runtime.execution.state,
            x_m=runtime.execution.state.x_m + 1.0,
        )
        expected_error = "terminal state"
    else:
        runtime.rng.random()
        expected_error = "terminal RNG"

    with pytest.raises(ValueError, match=expected_error):
        write_execution_checkpoint(
            tmp_path / "checkpoint-00000002",
            binding=_binding(),
            run_units=batch.units,
            executions=[item.execution for item in batch.runtimes],
            rngs=[item.rng for item in batch.runtimes],
            triangle_hints=[item.triangle_hint for item in batch.runtimes],
            sequence=2,
            previous_checkpoint=legacy_root,
        )


def test_v3_migration_chain_missing_generation_is_rejected(tmp_path: Path) -> None:
    """遷移 root 後的 v3 chain 缺少中間 generation 時不得退回舊狀態。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    legacy_root = _write_legacy_batch_checkpoint(
        batch,
        tmp_path / "checkpoint-00000001",
        sequence=1,
    )
    batch.advance()
    migrated_root = batch.write_checkpoint(
        tmp_path / "checkpoint-00000002",
        binding=_binding(),
        sequence=2,
        previous_checkpoint=legacy_root,
    )
    batch.advance()
    second_v3 = batch.write_checkpoint(
        tmp_path / "checkpoint-00000003",
        binding=_binding(),
        sequence=3,
        previous_checkpoint=migrated_root,
    )
    batch.advance()
    third_v3 = batch.write_checkpoint(
        tmp_path / "checkpoint-00000004",
        binding=_binding(),
        sequence=4,
        previous_checkpoint=second_v3,
    )
    shutil.rmtree(second_v3)
    with pytest.raises(ValueError, match="缺少前一代"):
        load_execution_checkpoint(third_v3, expected_binding=_binding())


def test_v3_migration_chain_root_tampering_is_rejected(tmp_path: Path) -> None:
    """遷移 root 或後續 generation 的 chain root 被篡改時必須 fail-closed。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    legacy_root = _write_legacy_batch_checkpoint(
        batch,
        tmp_path / "checkpoint-00000001",
        sequence=1,
    )
    batch.advance()
    migrated_root = batch.write_checkpoint(
        tmp_path / "checkpoint-00000002",
        binding=_binding(),
        sequence=2,
        previous_checkpoint=legacy_root,
    )
    batch.advance()
    second_v3 = batch.write_checkpoint(
        tmp_path / "checkpoint-00000003",
        binding=_binding(),
        sequence=3,
        previous_checkpoint=migrated_root,
    )

    # 先篡改遷移 root；後代仍宣告 root=2，loader 必須檢出 root metadata 不一致。
    root_metadata_path = migrated_root / "checkpoint.json"
    root_metadata = json.loads(root_metadata_path.read_text(encoding="utf-8"))
    root_metadata["chain_root_sequence"] = 1
    root_metadata_path.write_text(
        json.dumps(root_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="hash chain root metadata|chain_root_sequence|previous checkpoint.json SHA|legacy source",
    ):
        load_execution_checkpoint(second_v3, expected_binding=_binding())

    # 重新建立獨立鏈，改篡改 continuation metadata；這可區分 root 與後代的一致性 gate。
    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    continuation_root = tmp_path / "continuation"
    legacy_root = _write_legacy_batch_checkpoint(
        batch,
        continuation_root / "checkpoint-00000001",
        sequence=1,
    )
    batch.advance()
    migrated_root = batch.write_checkpoint(
        continuation_root / "checkpoint-00000002",
        binding=_binding(),
        sequence=2,
        previous_checkpoint=legacy_root,
    )
    batch.advance()
    second_v3 = batch.write_checkpoint(
        continuation_root / "checkpoint-00000003",
        binding=_binding(),
        sequence=3,
        previous_checkpoint=migrated_root,
    )
    continuation_metadata_path = second_v3 / "checkpoint.json"
    continuation_metadata = json.loads(
        continuation_metadata_path.read_text(encoding="utf-8")
    )
    continuation_metadata["chain_root_sequence"] = 1
    continuation_metadata_path.write_text(
        json.dumps(continuation_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash chain root metadata|chain root|chain_root_sequence"):
        load_execution_checkpoint(second_v3, expected_binding=_binding())


def test_public_schema3_load_can_continue_writing_without_private_setup(tmp_path: Path) -> None:
    """公開 loader 回傳的 execution 可直接交給下一代 writer，仍保留 stable prefix gate。"""

    first, _ = _write_chain(tmp_path, generation_count=1)
    loaded = load_execution_checkpoint(first, expected_binding=_binding())
    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    rngs = [np.random.Generator(np.random.PCG64DXSM()) for _ in loaded.executions]
    for rng, state in zip(rngs, loaded.rng_states, strict=True):
        rng.bit_generator.state = state
    second = write_execution_checkpoint(
        tmp_path / "checkpoint-00000002",
        binding=_binding(),
        run_units=batch.units,
        executions=loaded.executions,
        rngs=rngs,
        triangle_hints=loaded.triangle_hints,
        sequence=2,
        previous_checkpoint=first,
    )
    assert load_execution_checkpoint(second, expected_binding=_binding()).sequence == 2


def test_schema3_writer_rejects_noncanonical_target_and_previous_paths(tmp_path: Path) -> None:
    """writer 必須拒絕 scanner 無法依固定 parent／basename 還原的 checkpoint 路徑。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    with pytest.raises(ValueError, match="target 名稱"):
        batch.write_checkpoint(
            tmp_path / "noncanonical",
            binding=_binding(),
            sequence=1,
        )
    first = batch.write_checkpoint(
        tmp_path / "checkpoint-00000001",
        binding=_binding(),
        sequence=1,
    )
    with pytest.raises(ValueError, match="同一 parent"):
        batch.write_checkpoint(
            tmp_path / "other" / "checkpoint-00000002",
            binding=_binding(),
            sequence=2,
            previous_checkpoint=first,
        )
    with pytest.raises(ValueError, match="同一 parent"):
        batch.write_checkpoint(
            tmp_path / "checkpoint-00000002",
            binding=_binding(),
            sequence=2,
            previous_checkpoint=tmp_path / "checkpoint-not-standard",
        )


def test_schema2_sequence_zero_migrates_to_v3_and_continues(tmp_path: Path) -> None:
    """保留舊 schema 合法 sequence=0，遷移到 v3 後仍可連續讀取兩代。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    legacy = _write_legacy_batch_checkpoint(
        batch,
        tmp_path / "checkpoint-00000000",
        sequence=0,
    )
    batch.advance()
    first = batch.write_checkpoint(
        tmp_path / "checkpoint-00000001",
        binding=_binding(),
        sequence=1,
        previous_checkpoint=legacy,
    )
    batch.advance()
    second = batch.write_checkpoint(
        tmp_path / "checkpoint-00000002",
        binding=_binding(),
        sequence=2,
        previous_checkpoint=first,
    )
    metadata = json.loads((first / "checkpoint.json").read_text(encoding="utf-8"))
    source = metadata["legacy_source"]
    assert metadata["chain_root_sequence"] == 1
    assert source["sequence"] == 0
    assert source["relative_directory"] == "checkpoint-00000000"
    assert load_execution_checkpoint(second, expected_binding=_binding()).sequence == 2
def test_schema3_pending_context_update_is_allowed(tmp_path: Path) -> None:
    """同一 particle／UTC／age 的 pending observation 可更新 status/context。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    first = batch.write_checkpoint(
        tmp_path / "checkpoint-00000001",
        binding=_binding(),
        sequence=1,
    )
    runtime = batch.runtimes[1]
    pending = runtime.execution.observations[-1]
    runtime.execution.observations[-1] = replace(
        pending,
        environment_sample_status=EnvironmentSampleStatus.INVALID,
        environment_qc_flags=7,
        # engine 以 age 的 1e-12 秒絕對容差判定同一 pending row；checkpoint 必須
        # 使用相同契約，不能因浮點累積的 5e-13 秒差異誤拒合法 context 更新。
        age_seconds=pending.age_seconds + 5.0e-13,
        x_m=pending.x_m + 0.25,
        y_m=pending.y_m - 0.5,
        z_m=pending.z_m + 0.75,
    )
    second = batch.write_checkpoint(
        tmp_path / "checkpoint-00000002",
        binding=_binding(),
        sequence=2,
        previous_checkpoint=first,
    )
    loaded = load_execution_checkpoint(second, expected_binding=_binding())
    loaded_pending = loaded.executions[1].observations[-1]
    assert loaded_pending.environment_sample_status is EnvironmentSampleStatus.INVALID
    assert loaded_pending.environment_qc_flags == 7
    assert (loaded_pending.x_m, loaded_pending.y_m, loaded_pending.z_m) == (
        pending.x_m + 0.25,
        pending.y_m - 0.5,
        pending.z_m + 0.75,
    )
    assert loaded_pending.age_seconds == pytest.approx(pending.age_seconds + 5.0e-13)


def test_schema3_pending_time_replacement_is_rejected(tmp_path: Path) -> None:
    """pending observation 換成不同 UTC 時間時不可透過 cursor 靜默遺失。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    first = batch.write_checkpoint(
        tmp_path / "checkpoint-00000001",
        binding=_binding(),
        sequence=1,
    )
    runtime = batch.runtimes[1]
    pending = runtime.execution.observations[-1]
    runtime.execution.observations[-1] = replace(
        pending,
        time_utc_ns=pending.time_utc_ns + 1,
    )
    with pytest.raises(ValueError, match="pending boundary.*core"):
        batch.write_checkpoint(
            tmp_path / "checkpoint-00000002",
            binding=_binding(),
            sequence=2,
            previous_checkpoint=first,
        )


def test_schema3_pending_age_beyond_engine_tolerance_is_rejected(tmp_path: Path) -> None:
    """pending age 超過 engine 的 1e-12 秒容差時，必須拒絕替換造成的歷史缺列。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    first = batch.write_checkpoint(
        tmp_path / "checkpoint-00000001",
        binding=_binding(),
        sequence=1,
    )
    runtime = batch.runtimes[1]
    pending = runtime.execution.observations[-1]
    runtime.execution.observations[-1] = replace(
        pending,
        age_seconds=pending.age_seconds + 2.0e-12,
    )
    with pytest.raises(ValueError, match="pending boundary.*core"):
        batch.write_checkpoint(
            tmp_path / "checkpoint-00000002",
            binding=_binding(),
            sequence=2,
            previous_checkpoint=first,
        )


def test_schema3_terminal_particle_is_frozen_across_continuations(tmp_path: Path) -> None:
    """前代已終止粒子不可在後代改變 current、cursor、RNG、hint 或 history。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    runtime = batch.runtimes[0]
    runtime.execution.state = replace(
        runtime.execution.state,
        status=ParticleStatus.MAX_AGE,
    )

    def write_direct(path: Path, sequence: int, previous: Path) -> Path:
        """以公開底層 writer 保留人工修改，隔離 ProductionBatch alignment gate。"""

        return write_execution_checkpoint(
            path,
            binding=_binding(),
            run_units=batch.units,
            executions=[item.execution for item in batch.runtimes],
            rngs=[item.rng for item in batch.runtimes],
            triangle_hints=[item.triangle_hint for item in batch.runtimes],
            sequence=sequence,
            previous_checkpoint=previous,
        )

    # 直接從 terminal 狀態建立 root，才能測到「前代已終止」的後代凍結契約；若先以
    # ACTIVE root 寫出再改成 terminal，該次 transition 仍屬合法的最後一次狀態更新。
    first = write_execution_checkpoint(
        tmp_path / "checkpoint-00000001",
        binding=_binding(),
        run_units=batch.units,
        executions=[item.execution for item in batch.runtimes],
        rngs=[item.rng for item in batch.runtimes],
        triangle_hints=[item.triangle_hint for item in batch.runtimes],
        sequence=1,
    )
    second = write_direct(tmp_path / "checkpoint-00000002", 2, first)
    third = write_direct(tmp_path / "checkpoint-00000003", 3, second)
    assert load_execution_checkpoint(third, expected_binding=_binding()).sequence == 3

    # 保持 terminal status 但改變 state 的位置；這不是 engine 合法的同點 pending
    # context 更新，而是已停止粒子的 current-state 篡改，writer 必須在建立 partial 前拒絕。
    runtime.execution.state = replace(runtime.execution.state, x_m=runtime.execution.state.x_m + 1.0)
    with pytest.raises(ValueError, match="terminal state"):
        write_direct(tmp_path / "checkpoint-00000004", 4, third)


def test_schema3_stable_history_and_nested_event_attributes_are_rejected(tmp_path: Path) -> None:
    """已發布 observation prefix 與 BoundaryEvent.attributes 的巢狀修改都必須 fail-closed。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    first = batch.write_checkpoint(
        tmp_path / "checkpoint-00000001",
        binding=_binding(),
        sequence=1,
    )
    runtime = batch.runtimes[0]
    runtime.execution.observations[0] = replace(runtime.execution.observations[0], x_m=1.0)
    with pytest.raises(ValueError, match="stable observation/event prefix"):
        batch.write_checkpoint(
            tmp_path / "checkpoint-00000002",
            binding=_binding(),
            sequence=2,
            previous_checkpoint=first,
        )

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    first = batch.write_checkpoint(
        tmp_path / "events" / "checkpoint-00000001",
        binding=_binding(),
        sequence=1,
    )
    event = batch.runtimes[0].execution.events[0]
    attributes = event.attributes
    attributes |= {"tampered": True}
    with pytest.raises(ValueError, match="stable observation/event prefix"):
        batch.write_checkpoint(
            tmp_path / "events" / "checkpoint-00000002",
            binding=_binding(),
            sequence=2,
            previous_checkpoint=first,
        )

    # 把已發布 event 以 slice 方式再次 append 也不得把原 tracker 降級成未發布；原
    # stable object 的巢狀 attributes 變更仍應被 writer 偵測。
    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    runtime = batch.runtimes[0]
    active_state = replace(runtime.execution.state, status=ParticleStatus.ACTIVE)
    runtime.execution.state = active_state
    ProductionBatch._write_state_to_batch(
        batch.particle_batch,
        0,
        active_state,
        runtime.triangle_hint,
    )
    first = batch.write_checkpoint(
        tmp_path / "alias" / "checkpoint-00000001",
        binding=_binding(),
        sequence=1,
    )
    event = batch.runtimes[0].execution.events[0]
    batch.runtimes[0].execution.events[len(batch.runtimes[0].execution.events) :] = (event,)
    event.attributes["aliased_tamper"] = True
    with pytest.raises(ValueError, match="stable observation/event prefix"):
        batch.write_checkpoint(
            tmp_path / "alias" / "checkpoint-00000002",
            binding=_binding(),
            sequence=2,
            previous_checkpoint=first,
        )


def test_schema3_deepcopy_preserves_stable_event_tracking(tmp_path: Path) -> None:
    """snapshot/deepcopy 後修改已發布 event attributes 必須仍被下一代 writer 拒絕。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    first = batch.write_checkpoint(
        tmp_path / "checkpoint-00000001",
        binding=_binding(),
        sequence=1,
    )

    # snapshot 是公開 API 的獨立複本；deepcopy 也代表外部工作流常用的複製方式。兩者
    # 都必須保留已發布事件 attributes 的 stable tracker，否則 nested mutation 會避開
    # history dirty gate，下一代只依 cursor 寫入時便會靜默遺失篡改內容。
    copied_execution = deepcopy(batch.runtimes[0].execution)
    copied_execution.events[0].attributes["copied_tamper"] = True
    executions = [copied_execution] + [runtime.execution for runtime in batch.runtimes[1:]]
    rngs: list[np.random.Generator] = []
    for runtime in batch.runtimes:
        rng = np.random.Generator(np.random.PCG64DXSM())
        rng.bit_generator.state = runtime.rng.bit_generator.state
        rngs.append(rng)
    with pytest.raises(ValueError, match="stable observation/event prefix"):
        write_execution_checkpoint(
            tmp_path / "checkpoint-00000002",
            binding=_binding(),
            run_units=batch.units,
            executions=executions,
            rngs=rngs,
            triangle_hints=[runtime.triangle_hint for runtime in batch.runtimes],
            sequence=2,
            previous_checkpoint=first,
        )


@pytest.mark.parametrize(
    "mutation",
    ["slice_at_prefix", "insert_at_prefix", "append", "extend", "iadd", "imul"],
)
def test_schema3_rejects_adjacent_observation_engine_key_duplicates(
    tmp_path: Path, mutation: str
) -> None:
    """各種 list 變更都不得把同一 engine observation key 留成兩筆資料。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    first = batch.write_checkpoint(
        tmp_path / "checkpoint-00000001",
        binding=_binding(),
        sequence=1,
    )
    # member 1 在 synthetic sweep 後仍為 active 且只有一筆 pending；使用它可將測試
    # 集中在 history writer 的 engine-key gate，不混入 member 0 的 terminal freeze。
    observations = batch.runtimes[1].execution.observations
    pending = observations[-1]
    replacement = replace(pending, x_m=pending.x_m + 123.0)
    if mutation == "slice_at_prefix":
        observations[0:0] = [replacement]
    elif mutation == "insert_at_prefix":
        observations.insert(0, replacement)
    elif mutation == "append":
        observations.append(pending)
    elif mutation == "extend":
        observations.extend([pending])
    elif mutation == "iadd":
        observations += [pending]
    else:
        observations *= 2

    # 空 slice insertion／insert 會先由 tracker 標記 prefix；其餘追加 API 由 writer
    # 的增量相鄰 key gate 攔截。兩條閘門都必須存在，才能封住不同的 Python list 入口。
    if mutation in {"slice_at_prefix", "insert_at_prefix"}:
        assert observations.checkpoint_history_dirty is True
    with pytest.raises(ValueError, match="stable observation/event prefix|相鄰 observation engine key"):
        batch.write_checkpoint(
            tmp_path / "checkpoint-00000002",
            binding=_binding(),
            sequence=2,
            previous_checkpoint=first,
        )
    assert not (tmp_path / "checkpoint-00000002").exists()
    restored = load_execution_checkpoint(first, expected_binding=_binding())
    assert restored.sequence == 1
    assert len(restored.executions[1].observations) == 1


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("state_own_local_exit_recorded_int", "own_local_exit_recorded.*boolean"),
        ("state_time_utc_ns_float", "time_utc_ns.*整數"),
        ("observation_particle_id", "particle_id.*RunUnit identity"),
        ("observation_particle_id_stable", "particle_id.*RunUnit identity"),
        ("observation_time_utc_ns_float", "time_utc_ns.*整數"),
        ("event_particle_id", "identity.*RunUnit identity"),
        ("event_fraction_out_of_range", "fraction.*介於 0 與 1"),
        ("event_time_utc_ns_bool", "time_utc_ns.*整數"),
    ],
)
def test_schema3_writer_rejects_payloads_loader_would_reject(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    """writer 必須在 atomic publish 前拒絕 loader 必定無法還原的 payload。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    first = batch.write_checkpoint(
        tmp_path / "checkpoint-00000001",
        binding=_binding(),
        sequence=1,
    )
    runtime = batch.runtimes[1]
    if mutation == "state_own_local_exit_recorded_int":
        runtime.execution.state = replace(
            runtime.execution.state,
            own_local_exit_recorded=0,
        )
    elif mutation == "state_time_utc_ns_float":
        runtime.execution.state = replace(
            runtime.execution.state,
            time_utc_ns=float(runtime.execution.state.time_utc_ns),
        )
    elif mutation == "observation_particle_id":
        pending = runtime.execution.observations[-1]
        # 追加成新的 pending row，讓上一代 boundary 仍正確；這專門驗證 writer 對
        # 本代新增列的 particle identity gate，而不是只依既有 pending 比對攔截。
        runtime.execution.observations.append(
            replace(pending, particle_id="wrong-particle")
        )
    elif mutation == "observation_particle_id_stable":
        pending = runtime.execution.observations[-1]
        # 再追加一筆合法 pending，讓錯誤列落在本代 stable segment 中；這確認 gate
        # 不只驗最後一列，而是驗證 observation_start 之後的所有新增 rows。
        runtime.execution.observations.extend(
            [replace(pending, particle_id="wrong-particle"), pending]
        )
    elif mutation == "observation_time_utc_ns_float":
        pending = runtime.execution.observations[-1]
        runtime.execution.observations[-1] = replace(
            pending,
            time_utc_ns=float(pending.time_utc_ns),
        )
    else:
        source_event = batch.runtimes[0].execution.events[0]
        event = replace(
            source_event,
            particle_id=runtime.execution.state.particle_id,
            scenario_id=runtime.execution.state.scenario_id,
            member_id=runtime.execution.state.member_id,
            study_site_id=runtime.execution.state.study_site_id,
            analysis_region_id=runtime.execution.state.analysis_region_id,
            receptor_id=runtime.execution.state.receptor_id,
        )
        if mutation == "event_particle_id":
            event = replace(event, particle_id="wrong-event")
        elif mutation == "event_fraction_out_of_range":
            event = replace(event, fraction=1.5)
        else:
            event = replace(event, time_utc_ns=True)
        runtime.execution.events.append(event)

    target = tmp_path / "checkpoint-00000002"
    with pytest.raises(ValueError, match=expected_error):
        batch.write_checkpoint(
            target,
            binding=_binding(),
            sequence=2,
            previous_checkpoint=first,
        )
    assert not target.exists()
    assert not tuple(tmp_path.glob(".checkpoint-00000002.partial-*"))
    assert load_execution_checkpoint(first, expected_binding=_binding()).sequence == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("particle_id", ""),
        ("particle_id", True),
        ("scenario_id", ""),
        ("study_site_id", ""),
        ("analysis_region_id", ""),
        ("receptor_id", ""),
        ("experiment_case_id", ""),
        ("experiment_case_id", True),
        ("member_id", True),
        ("member_id", -1),
        ("member_id", 1.5),
        ("arrival_time_utc_ns", True),
        ("arrival_time_utc_ns", 1.5),
        ("seed", True),
        ("seed", -1),
        ("seed", 1.5),
    ],
)
def test_schema3_writer_rejects_invalid_run_unit_identity(
    tmp_path: Path, field: str, replacement: object
) -> None:
    """RunUnit identity 的空值、bool、負值與錯誤數值型別必須在 partial 前拒絕。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    units = list(batch.units)
    unit = units[0]
    if field in {
        "scenario_id",
        "study_site_id",
        "analysis_region_id",
        "receptor_id",
        "arrival_time_utc_ns",
    }:
        units[0] = replace(
            unit,
            scenario=replace(unit.scenario, **{field: replacement}),
        )
    else:
        units[0] = replace(unit, **{field: replacement})

    target = tmp_path / "checkpoint-00000001"
    with pytest.raises(ValueError, match=r"run_units\[0\]\.identity"):
        write_execution_checkpoint(
            target,
            binding=_binding(),
            run_units=units,
            executions=[runtime.execution for runtime in batch.runtimes],
            rngs=[runtime.rng for runtime in batch.runtimes],
            triangle_hints=[runtime.triangle_hint for runtime in batch.runtimes],
            sequence=1,
        )
    assert not target.exists()
    assert not tuple(tmp_path.glob(".checkpoint-00000001.partial-*"))

    # 所有失敗案例共用同一個正常 root 回歸，確保 strict identity gate 不改變合法
    # RunUnit 的 canonical payload 與 loader round-trip。
    valid = batch.write_checkpoint(
        target,
        binding=_binding(),
        sequence=1,
    )
    assert load_execution_checkpoint(valid, expected_binding=_binding()).sequence == 1


def test_schema3_writer_rejects_duplicate_run_unit_particle_id(
    tmp_path: Path,
) -> None:
    """不同 member／seed 不得共用同一 particle_id，避免 loader particle order 重複。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    units = list(batch.units)
    units[1] = replace(units[1], particle_id=units[0].particle_id)
    target = tmp_path / "checkpoint-00000001"
    with pytest.raises(ValueError, match="particle_id 必須唯一"):
        write_execution_checkpoint(
            target,
            binding=_binding(),
            run_units=units,
            executions=[runtime.execution for runtime in batch.runtimes],
            rngs=[runtime.rng for runtime in batch.runtimes],
            triangle_hints=[runtime.triangle_hint for runtime in batch.runtimes],
            sequence=1,
        )
    assert not target.exists()
    assert not tuple(tmp_path.glob(".checkpoint-00000001.partial-*"))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("config_hash", ("config",)),
        ("input_inventory_hash", ["inventory"]),
        ("experiment_case_id", True),
        ("shard_id", ""),
        ("seed_policy", None),
        ("code_commit", 123),
    ],
)
def test_schema3_writer_rejects_noncanonical_binding(
    tmp_path: Path, field: str, replacement: object
) -> None:
    """binding 的 tuple、list、bool、空值與數值不得在 partial 後才由 loader 發現。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    invalid_binding = replace(_binding(), **{field: replacement})
    target = tmp_path / "checkpoint-00000001"
    with pytest.raises(ValueError, match=rf"binding\.{field}"):
        batch.write_checkpoint(target, binding=invalid_binding, sequence=1)
    assert not target.exists()
    assert not tuple(tmp_path.glob(".checkpoint-00000001.partial-*"))

    # 驗證 strict binding preflight 只拒絕不可持久化輸入；合法 binding 仍可立即被 loader
    # 還原，且不需要改變同一批次的 execution、RNG 或 particle history。
    valid = batch.write_checkpoint(target, binding=_binding(), sequence=1)
    loaded = load_execution_checkpoint(valid, expected_binding=_binding())
    assert loaded.binding == _binding()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("config_hash", ["config"]),
        ("input_inventory_hash", True),
        ("experiment_case_id", ""),
        ("shard_id", None),
        ("seed_policy", 123),
        ("code_commit", {"commit": "value"}),
        ("random_stream_id", "   "),
        ("random_stream_id", False),
    ],
)
def test_schema3_loader_rejects_noncanonical_binding_payload(
    tmp_path: Path, field: str, replacement: object
) -> None:
    """loader 必須以同一 binding parser 拒絕 JSON 正規化後仍不合法的欄位。"""

    batch = ProductionBatch(_shard(), master_seed=123, request_factory=_factory)
    batch.advance()
    target = batch.write_checkpoint(
        tmp_path / "checkpoint-00000001",
        binding=_binding(),
        sequence=1,
    )
    metadata_path = target / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["binding"][field] = replacement
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=rf"schema 3 binding\.{field}"):
        load_execution_checkpoint(target, expected_binding=_binding())
