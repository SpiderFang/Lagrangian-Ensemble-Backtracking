"""trajectory shard strict tracked metadata 與壞 payload fail-safe 測試。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lagrangian_backtracking.engine import EnvironmentSampleStatus, Observation, ParticleResult
from lagrangian_backtracking.models import BoundaryEvent, EventType, ParticleState, ParticleStatus
from lagrangian_backtracking.outputs import (
    TRAJECTORY_SHARD_SCHEMA_VERSION,
    read_trajectory_shard,
    sha256_file,
    validate_trajectory_shard,
    write_trajectory_shard,
)

_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_LEGACY_SCHEMA_VERSION = "1.0.0"
_BASE_PAYLOAD_FILES = {
    "particle_table.parquet",
    "events.parquet",
    "trajectory_offsets.npy",
    "time_utc_ns.npy",
    "age_seconds.npy",
    "x_m.npy",
    "y_m.npy",
    "z_m.npy",
    "status_code.npy",
}
_ENVIRONMENT_PAYLOAD_FILES = {
    "environment_sample_status_code.npy",
    "eta_m.npy",
    "bed_z_m.npy",
    "forcing_month_yyyymm.npy",
    "environment_qc_flags.npy",
}


def _result() -> ParticleResult:
    """建立一條具有 backward UTC 與終止狀態的最小合法軌跡。"""

    state = ParticleState(
        particle_id="particle-0",
        scenario_id="scenario-0",
        member_id=0,
        study_site_id="gongliao",
        analysis_region_id="A",
        receptor_id="receptor-0",
        x_m=1.0,
        y_m=2.0,
        z_m=-3.0,
        time_utc_ns=10,
        age_seconds=10.0,
        status=ParticleStatus.MAX_AGE,
    )
    return ParticleResult(
        state,
        [
            Observation("particle-0", 20, 0.0, 0.0, 0.0, -3.0, ParticleStatus.ACTIVE),
            Observation("particle-0", 10, 10.0, 1.0, 2.0, -3.0, ParticleStatus.MAX_AGE),
        ],
        [],
        1,
        0,
    )


def _metadata(run_kind: str, *, commit: str | None, dirty: bool | None) -> dict:
    """建立 formal/pilot 共用的 strict tracked-run metadata。"""

    return {
        "run_id": "tracked-run",
        "run_kind": run_kind,
        "config_hash": "1" * 64,
        "input_inventory_sha256": "2" * 64,
        "checkpoint_input_binding_hash": "3" * 64,
        "component_canonical_hashes": {"material": "4" * 64},
        "geometry_canonical_hashes": {"domain": "5" * 64},
        "code_commit": commit,
        "deployment_tree_sha256": "6" * 64,
        "uv_lock_sha256": "7" * 64,
        "dirty_flag": dirty,
        "seed_policy": "sha256_v1",
        "shard_id": "shard-0",
        "experiment_case_id": "baseline",
        "resource_usage": {
            "wall_seconds": 1.0,
            "process_cpu_seconds": 0.5,
            "max_rss_bytes": 1024,
            "output_bytes": 0,
            "checkpoint_bytes": 0,
            "particle_steps": 1,
        },
    }


def test_strict_formal_and_pilot_metadata_rules(tmp_path: Path) -> None:
    """formal 必須 clean commit；pilot 可明示 nullable commit/dirty，但欄位不可缺。"""

    formal = tmp_path / "formal"
    write_trajectory_shard(formal, [_result()], run_metadata=_metadata("formal", commit=_COMMIT, dirty=False))
    assert validate_trajectory_shard(
        formal,
        strict_run_metadata=True,
        require_formal_metadata=True,
    )["valid"]

    pilot = tmp_path / "pilot"
    write_trajectory_shard(pilot, [_result()], run_metadata=_metadata("pilot", commit=None, dirty=None))
    assert validate_trajectory_shard(pilot, strict_run_metadata=True)["valid"]

    dirty = tmp_path / "dirty"
    write_trajectory_shard(dirty, [_result()], run_metadata=_metadata("formal", commit=_COMMIT, dirty=True))
    assert not validate_trajectory_shard(
        dirty,
        strict_run_metadata=True,
        require_formal_metadata=True,
    )["valid"]


@pytest.mark.parametrize(
    "damage", ["bad_json", "nonfinite_json", "traversal", "npy", "unknown", "symlink"]
)
def test_bad_shard_payloads_return_json_safe_invalid(tmp_path: Path, damage: str) -> None:
    """壞 JSON/NPY/path/unknown/symlink 一律回 valid=false，不拋 reader 例外。"""

    shard = tmp_path / f"shard-{damage}"
    write_trajectory_shard(shard, [_result()], run_metadata={"run_kind": "synthetic"})
    if damage == "bad_json":
        (shard / "manifest.json").write_text("{bad", encoding="utf-8")
    elif damage == "nonfinite_json":
        manifest_path = shard / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["run_metadata"]["unsafe_number"] = float("inf")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif damage == "traversal":
        manifest_path = shard / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = manifest["files"].pop("x_m.npy")
        manifest["files"]["../x_m.npy"] = contract
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif damage == "npy":
        (shard / "x_m.npy").write_bytes(b"not-a-npy")
    elif damage == "unknown":
        (shard / "unknown.bin").write_bytes(b"unknown")
    else:
        payload = shard / "x_m.npy"
        payload.unlink()
        payload.symlink_to(shard / "y_m.npy")
    result = validate_trajectory_shard(shard)
    assert result["valid"] is False
    assert isinstance(result["errors"], list)


def _event_result(particle_id: str, *, with_event: bool) -> ParticleResult:
    """建立含多筆 observation 的結果，供 reader 驗證事件欄位與原始順序。"""

    state = ParticleState(
        particle_id=particle_id,
        scenario_id=f"scenario-{particle_id}",
        member_id=7,
        study_site_id="gongliao",
        analysis_region_id="A",
        receptor_id="receptor-0",
        x_m=30.0,
        y_m=40.0,
        z_m=-5.0,
        time_utc_ns=10,
        age_seconds=20.0,
        status=ParticleStatus.MAX_AGE,
    )
    observations = [
        Observation(particle_id, 30, 0.0, 10.0, 20.0, -1.0, ParticleStatus.ACTIVE),
        Observation(particle_id, 20, 10.0, 20.0, 30.0, -3.0, ParticleStatus.ACTIVE),
        Observation(particle_id, 10, 20.0, 30.0, 40.0, -5.0, ParticleStatus.MAX_AGE),
    ]
    events = []
    if with_event:
        events.append(
            BoundaryEvent(
                particle_id=particle_id,
                scenario_id=f"scenario-{particle_id}",
                member_id=7,
                study_site_id="gongliao",
                analysis_region_id="A",
                receptor_id="receptor-0",
                event_type=EventType.LOCAL_DOMAIN_FIRST_EXIT,
                time_utc_ns=15,
                x_m=25.0,
                y_m=35.0,
                z_m=-4.0,
                fraction=0.25,
                related_study_site_id="guishan",
                boundary_segment_id="open-east",
                boundary_s_m=12.5,
                source_face_id=3,
                triangle_id=4,
                forcing_month_id="202401",
                attributes={
                    "accepted": True,
                    "crossing_count": 2,
                    "label": "first-exit",
                    "score": 0.75,
                },
            )
        )
    return ParticleResult(state, observations, events, 9, 2)


def _environment_result() -> ParticleResult:
    """建立 not_sampled、valid、invalid 三種環境狀態供 v2 round-trip 使用。

    三筆 observation 的位置與時間仍是固定的公尺制／UTC 奈秒基準；valid 那筆以
    ``bed_z_m <= z_m <= eta_m`` 和 ``202401`` 示範完整 context，invalid 那筆只保存
    非零品質旗標。這些是本機合成資料契約測試，不代表任何 OCM／NWW3 科學樣本。
    """

    state = ParticleState(
        particle_id="environment-p0",
        scenario_id="scenario-environment-p0",
        member_id=3,
        study_site_id="gongliao",
        analysis_region_id="A",
        receptor_id="receptor-0",
        x_m=30.0,
        y_m=40.0,
        z_m=-3.0,
        time_utc_ns=10,
        age_seconds=20.0,
        status=ParticleStatus.MAX_AGE,
    )
    observations = [
        Observation(
            "environment-p0",
            30,
            0.0,
            10.0,
            20.0,
            -3.0,
            ParticleStatus.ACTIVE,
        ),
        Observation(
            "environment-p0",
            20,
            10.0,
            20.0,
            30.0,
            -3.0,
            ParticleStatus.ACTIVE,
            EnvironmentSampleStatus.VALID,
            1.0,
            -10.0,
            "202401",
            0,
        ),
        Observation(
            "environment-p0",
            10,
            20.0,
            30.0,
            40.0,
            -3.0,
            ParticleStatus.MAX_AGE,
            EnvironmentSampleStatus.INVALID,
            None,
            None,
            None,
            7,
        ),
    ]
    return ParticleResult(state, observations, [], 9, 2)


def _downgrade_to_v1(shard: Path) -> None:
    """把已發布的 v2 合成 shard 降為 legacy v1 fixture，不改動九個 base payload。"""

    manifest_path = shard / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename in _ENVIRONMENT_PAYLOAD_FILES:
        (shard / filename).unlink()
        manifest["files"].pop(filename)
    manifest["schema_version"] = _LEGACY_SCHEMA_VERSION
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _refresh_checksum(shard: Path, filename: str) -> None:
    """在測試刻意竄改 payload 後，只更新該 payload 的 manifest checksum。"""

    manifest_path = shard / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = shard / filename
    manifest["files"][filename] = {
        "size_bytes": payload.stat().st_size,
        "sha256": sha256_file(payload),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def test_writer_publishes_v2_manifest_and_exact_payload_topology(tmp_path: Path) -> None:
    """writer 永遠發布 v2，manifest 與實際根目錄都只能含固定十四個 payload。"""

    shard = tmp_path / "v2-topology"
    write_trajectory_shard(shard, [_result()], run_metadata={"run_kind": "synthetic"})

    manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
    expected_payload = _BASE_PAYLOAD_FILES | _ENVIRONMENT_PAYLOAD_FILES
    assert manifest["schema_version"] == TRAJECTORY_SHARD_SCHEMA_VERSION
    assert set(manifest["files"]) == expected_payload
    assert {entry.name for entry in shard.iterdir()} == expected_payload | {"manifest.json"}
    assert np.load(shard / "environment_sample_status_code.npy", allow_pickle=False).dtype == np.dtype(
        np.uint8
    )
    assert np.load(shard / "eta_m.npy", allow_pickle=False).dtype == np.dtype(np.float64)
    assert np.load(shard / "bed_z_m.npy", allow_pickle=False).dtype == np.dtype(np.float64)
    assert np.load(shard / "forcing_month_yyyymm.npy", allow_pickle=False).dtype == np.dtype(np.int32)
    assert np.load(shard / "environment_qc_flags.npy", allow_pickle=False).dtype == np.dtype(np.uint32)
    assert validate_trajectory_shard(shard)["valid"]


def test_v2_round_trip_preserves_all_environment_states_and_sentinels(tmp_path: Path) -> None:
    """v2 reader 依 code 還原三種狀態、None、月份與品質旗標，不靠 NaN 猜測。"""

    shard = tmp_path / "environment-round-trip"
    source = _environment_result()
    write_trajectory_shard(shard, [source], run_metadata={"run_kind": "synthetic"})

    assert np.array_equal(
        np.load(shard / "environment_sample_status_code.npy", allow_pickle=False),
        np.array([0, 1, 2], dtype=np.uint8),
    )
    assert np.isnan(np.load(shard / "eta_m.npy", allow_pickle=False)[[0, 2]]).all()
    assert np.isnan(np.load(shard / "bed_z_m.npy", allow_pickle=False)[[0, 2]]).all()
    assert np.array_equal(
        np.load(shard / "forcing_month_yyyymm.npy", allow_pickle=False),
        np.array([0, 202401, 0], dtype=np.int32),
    )
    assert np.array_equal(
        np.load(shard / "environment_qc_flags.npy", allow_pickle=False),
        np.array([0, 0, 7], dtype=np.uint32),
    )

    loaded = read_trajectory_shard(shard)
    assert loaded[0].observations == source.observations
    assert loaded[0].observations[0].environment_sample_status is EnvironmentSampleStatus.NOT_SAMPLED
    assert loaded[0].observations[1].environment_sample_status is EnvironmentSampleStatus.VALID
    assert loaded[0].observations[2].environment_sample_status is EnvironmentSampleStatus.INVALID


def test_v1_fixture_is_read_only_compatible_and_gets_not_sampled_defaults(tmp_path: Path) -> None:
    """移除 v2 context 後的 v1 仍可讀，但每筆 observation 明示為無環境樣本。"""

    shard = tmp_path / "legacy-v1"
    source = _environment_result()
    write_trajectory_shard(shard, [source], run_metadata={"run_kind": "synthetic"})
    _downgrade_to_v1(shard)

    validation = validate_trajectory_shard(shard)
    assert validation["valid"]
    assert validation["manifest"]["schema_version"] == _LEGACY_SCHEMA_VERSION
    loaded = read_trajectory_shard(shard)
    assert all(
        observation.environment_sample_status is EnvironmentSampleStatus.NOT_SAMPLED
        and observation.eta_m is None
        and observation.bed_z_m is None
        and observation.forcing_month_id is None
        and observation.environment_qc_flags is None
        for observation in loaded[0].observations
    )


@pytest.mark.parametrize(
    "mutation",
    ["v1_extra_file", "v1_extra_manifest_entry", "v2_missing_file", "unknown_schema"],
)
def test_schema_version_selects_exact_fixed_topology(tmp_path: Path, mutation: str) -> None:
    """v1/v2 版本、manifest files 與實體檔案拓撲不一致時一律拒絕。"""

    shard = tmp_path / mutation
    write_trajectory_shard(shard, [_result()], run_metadata={"run_kind": "synthetic"})
    manifest_path = shard / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "v1_extra_file":
        _downgrade_to_v1(shard)
        np.save(shard / "eta_m.npy", np.array([np.nan, np.nan], dtype=np.float64), allow_pickle=False)
    elif mutation == "v1_extra_manifest_entry":
        _downgrade_to_v1(shard)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["eta_m.npy"] = {"size_bytes": 0, "sha256": "0" * 64}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "v2_missing_file":
        (shard / "eta_m.npy").unlink()
        manifest["files"].pop("eta_m.npy")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        manifest["schema_version"] = "9.9.9"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_trajectory_shard(shard)["valid"] is False


@pytest.mark.parametrize(
    ("filename", "mutation"),
    [
        ("environment_sample_status_code.npy", "code"),
        ("environment_sample_status_code.npy", "dtype"),
        ("eta_m.npy", "shape"),
        ("eta_m.npy", "nan"),
        ("eta_m.npy", "inf"),
        ("forcing_month_yyyymm.npy", "month"),
        ("environment_qc_flags.npy", "qc"),
        ("bed_z_m.npy", "geometry"),
    ],
)
def test_v2_environment_payload_tampering_is_rejected_after_checksum_refresh(
    tmp_path: Path, filename: str, mutation: str
) -> None:
    """逐項竄改 code/dtype/shape/NaN/Inf/month/QC/geometry 即使重算 checksum 仍拒絕。"""

    shard = tmp_path / f"tamper-{mutation}"
    write_trajectory_shard(shard, [_environment_result()], run_metadata={"run_kind": "synthetic"})
    payload = np.load(shard / filename, allow_pickle=False).copy()
    if mutation == "code":
        payload[0] = 3
    elif mutation == "dtype":
        payload = payload.astype(np.int16)
    elif mutation == "shape":
        payload = payload[:-1]
    elif mutation == "nan":
        payload[1] = np.nan
    elif mutation == "inf":
        payload[2] = np.inf
    elif mutation == "month":
        payload[1] = 202413
    elif mutation == "qc":
        payload[1] = 1
    else:
        payload[1] = 0.0
    np.save(shard / filename, payload, allow_pickle=False)
    _refresh_checksum(shard, filename)

    validation = validate_trajectory_shard(shard)
    assert validation["valid"] is False
    assert not any(f"{filename}: checksum" == error for error in validation["errors"])
    with pytest.raises(ValueError, match="驗證失敗"):
        read_trajectory_shard(shard)


def test_read_trajectory_shard_round_trips_observations_events_and_order(tmp_path: Path) -> None:
    """reader 保留 particle table、observation offsets 與事件屬性的原始順序。"""

    shard = tmp_path / "round-trip"
    source = (_event_result("p-z", with_event=True), _event_result("p-a", with_event=True))
    write_trajectory_shard(shard, source, run_metadata={"run_kind": "synthetic"})

    loaded = read_trajectory_shard(shard)

    assert isinstance(loaded, tuple)
    assert [result.final_state.particle_id for result in loaded] == ["p-z", "p-a"]
    assert loaded[0].observations == list(source[0].observations)
    assert loaded[1].observations == list(source[1].observations)
    assert loaded[0].step_count == 9
    assert loaded[0].minimum_clamp_count == 2
    assert loaded[0].events == list(source[0].events)
    assert loaded[1].events == list(source[1].events)
    assert loaded[0].final_state == source[0].final_state


def test_read_trajectory_shard_supports_empty_event_parquet(tmp_path: Path) -> None:
    """events.parquet 只有空的 particle_id 欄位時仍能正常得到空事件清單。"""

    shard = tmp_path / "no-events"
    source = (_event_result("p0", with_event=False),)
    write_trajectory_shard(shard, source, run_metadata={"run_kind": "synthetic"})

    loaded = read_trajectory_shard(shard)

    assert loaded[0].events == []
    assert loaded[0].observations == source[0].observations
    assert pq.read_table(shard / "events.parquet").column_names == ["particle_id"]


def test_read_trajectory_shard_forwards_validation_options_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reader 必須原封不動轉送三個驗證選項，且 invalid 結果不能繼續讀檔。"""

    calls: list[tuple[object, bool, bool, object]] = []

    def invalid_validator(path, *, require_formal_metadata, strict_run_metadata, expected_metadata):
        calls.append((path, require_formal_metadata, strict_run_metadata, expected_metadata))
        return {"valid": False, "errors": ["deliberately invalid"], "manifest": None}

    monkeypatch.setattr("lagrangian_backtracking.outputs.validate_trajectory_shard", invalid_validator)
    expected = {"run_id": "expected"}
    with pytest.raises(ValueError, match="驗證失敗") as error:
        read_trajectory_shard(
            tmp_path / "absolute-location-is-not-exposed",
            require_formal_metadata=True,
            strict_run_metadata=True,
            expected_metadata=expected,
        )

    assert calls == [
        (tmp_path / "absolute-location-is-not-exposed", True, True, expected)
    ]
    assert str(tmp_path) not in str(error.value)


def test_read_trajectory_shard_rejects_corrupt_checksum_without_exposing_path(tmp_path: Path) -> None:
    """payload checksum 損壞時 reader 不會進入反序列化，也不把絕對路徑放進例外。"""

    shard = tmp_path / "checksum-failure"
    write_trajectory_shard(
        shard, [_event_result("p0", with_event=False)], run_metadata={"run_kind": "synthetic"}
    )
    payload = shard / "x_m.npy"
    payload.write_bytes(payload.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="驗證失敗") as error:
        read_trajectory_shard(shard)

    assert str(shard) not in str(error.value)


def test_read_trajectory_shard_rejects_final_status_semantic_mismatch_after_checksum_refresh(
    tmp_path: Path,
) -> None:
    """同步更新 checksum 後竄改最後狀態，仍應由 final_status 語意 gate 拒絕。"""

    shard = tmp_path / "status-mismatch"
    write_trajectory_shard(
        shard, [_event_result("p0", with_event=False)], run_metadata={"run_kind": "synthetic"}
    )
    status_path = shard / "status_code.npy"
    statuses = np.load(status_path, allow_pickle=False).copy()
    statuses[-1] = "active"
    np.save(status_path, statuses, allow_pickle=False)
    _refresh_checksum(shard, "status_code.npy")

    validation = validate_trajectory_shard(shard)
    assert validation["valid"] is False
    assert any("final_status" in item for item in validation["errors"])
    assert not any("checksum" in item for item in validation["errors"])
    with pytest.raises(ValueError, match="驗證失敗"):
        read_trajectory_shard(shard)


@pytest.mark.parametrize("attributes_json", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": null}', '{"a": []}'])
def test_read_trajectory_shard_rejects_unsafe_event_attributes(
    tmp_path: Path, attributes_json: str
) -> None:
    """事件屬性遇到重複鍵、非有限值或非允許 scalar 時必須 fail closed。"""

    shard = tmp_path / "unsafe-attributes"
    write_trajectory_shard(
        shard, [_event_result("p0", with_event=True)], run_metadata={"run_kind": "synthetic"}
    )
    table = pq.read_table(shard / "events.parquet")
    rows = table.to_pylist()
    rows[0]["attributes_json"] = attributes_json
    pq.write_table(pa.Table.from_pylist(rows), shard / "events.parquet")
    _refresh_checksum(shard, "events.parquet")

    with pytest.raises(ValueError, match="讀取失敗"):
        read_trajectory_shard(shard)
