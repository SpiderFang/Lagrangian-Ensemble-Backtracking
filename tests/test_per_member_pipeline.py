"""WP-51：per_member（逐成員沉底時刻）從 arrival manifest 到粒子身分表的接線測試。

涵蓋：
1. per_member 的 arrival manifest 可由正式 loader 重算並載入（``250*M`` 筆），
   竄改年齡矩陣、分層排列、成員數、單筆年齡或 UTC 都會被拒絕。
2. ``load_scenario_inputs`` 的 per_member 模式：情境仍是 50,000 個（以 250 個觀測錨點
   建立），初始條件是 ``5,000*M`` 筆，舊 caller 未明示 ``per_member_supported`` 時拒絕。
3. ``build_formal_identity_table``：每顆粒子的起算時刻與起點深度取自自己成員的沉底
   arrival；同一情境的不同成員起點深度可不同；seed 與 particle_id 規則不變。
4. per_observation 的既有雜湊與抽樣輸出在 WP-51 前後位元相同（黃金值）。

測試資料全部合成，不讀 OCM／NWW；``M`` 用 2 以控制 fixture 大小。
"""

from __future__ import annotations

import json
import zlib
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from lagrangian_backtracking import input_derivation as input_derivation_module
from lagrangian_backtracking.bed_residence import (
    BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
    apply_bed_residence_member_sampling,
    sample_bed_residence_member_ages_conditioned_on_availability,
)
from lagrangian_backtracking.config import ProjectConfig
from lagrangian_backtracking.manifests import (
    _load_arrival_document,
    load_scenario_inputs,
    observation_arrival_from_member_arrival,
)
from lagrangian_backtracking.native_runner.production_wiring import build_formal_identity_table
from lagrangian_backtracking.scenarios import ArrivalTime
from test_manifests import (
    EXAMPLE_CONFIG,
    SITES,
    _bed_arrival_payload,
    _bed_config,
    _initial_condition_payload,
    _material_payload,
    _receptor_payload,
    _write_json,
)

_HOUR_NS = 3_600_000_000_000
_M = 2
_GENERATED_OBSERVATION_KEYS = frozenset(
    {
        "observation_time_utc_ns",
        "observation_time_utc",
        "observation_arrival_time_id",
        "observation_tide_class",
        "observation_phase_or_event",
        "observation_season",
        "observation_year",
    }
)


def _member_config(members: int = _M) -> ProjectConfig:
    """per_member 版的正式 loader 測試設定：沿用 ``_bed_config``，改為整段回溯＋逐成員。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    payload["inputs"]["backtrack_support_days"] = 180
    payload["scenarios"]["bed_residence_time"].update(
        {
            "backtrack_mode": BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
            "runtime_horizon_support_days": 90,
            "age_sampling_scope": "per_member",
        }
    )
    payload["scenarios"]["members_per_scenario"] = members
    payload["scenarios"]["master_seed"] = 20261009
    payload["scenarios"]["material_manifest"] = "material.json"
    payload["scenarios"]["receptor_manifest"] = "receptor.json"
    payload["scenarios"]["arrival_time_manifest"] = "arrival.json"
    payload["scenarios"]["receptor_arrival_initial_condition_manifest"] = "initial_conditions.json"
    return ProjectConfig.model_validate(payload)


def _observation_arrivals(config: ProjectConfig) -> dict[str, list[ArrivalTime]]:
    """由既有 ``_bed_arrival_payload`` 還原五站的觀測錨點（去掉沉底轉換產生的欄位）。"""

    base = _bed_arrival_payload(_bed_config())
    by_site: dict[str, list[ArrivalTime]] = {site: [] for site in SITES}
    for record in base["records"]:
        metadata = {
            key: value
            for key, value in record["metadata"].items()
            if not key.startswith("bed_residence_")
            and not key.startswith("deposition_")
            and key not in _GENERATED_OBSERVATION_KEYS
        }
        md = record["metadata"]
        by_site[record["study_site_id"]].append(
            ArrivalTime(
                arrival_time_id=md["observation_arrival_time_id"],
                study_site_id=record["study_site_id"],
                time_utc_ns=md["observation_time_utc_ns"],
                year=md["observation_year"],
                season=md["observation_season"],
                tide_class=record["tide_class"],
                phase_or_event=record["phase_or_event"],
                metadata=metadata,
            )
        )
    return by_site


def _all_hours() -> tuple[int, ...]:
    start = int(datetime(2023, 6, 1, tzinfo=UTC).timestamp()) * 1_000_000_000
    stop = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp()) * 1_000_000_000
    return tuple(range(start, stop, _HOUR_NS))


def _gappy_hours() -> tuple[int, ...]:
    """每三天缺一整天（連續 24 小時）的可用時間軸：讓隨機排列必然配到整層無候選的槽位，
    用來走過「排列後可行性修補」的稽核與載入器驗證。"""

    return tuple(
        value for index, value in enumerate(_all_hours()) if (index // 24) % 3 != 0
    )


def _member_arrival_payload(
    config: ProjectConfig, *, hours: tuple[int, ...] | None = None
) -> dict[str, object]:
    """以正式 sampler／apply／provenance 函式組出 per_member 的 arrival manifest payload。"""

    bed = config.scenarios.bed_residence_time
    members = config.scenarios.members_per_scenario
    observations = _observation_arrivals(config)
    hours = _all_hours() if hours is None else hours
    sampling = sample_bed_residence_member_ages_conditioned_on_availability(
        observation_times_by_site={s: tuple(a.time_utc_ns for a in rows) for s, rows in observations.items()},
        available_time_ns_by_site={s: hours for s in observations},
        maximum_age_days=bed.maximum_age_days,
        sample_count=50,
        member_count=members,
        seed=bed.effective_sampling_seed,
    )
    records: list[ArrivalTime] = []
    for site_id in sorted(observations):
        converted = apply_bed_residence_member_sampling(
            observations[site_id],
            sampling=sampling,
            sampling_seed=bed.effective_sampling_seed,
            maximum_age_days=bed.maximum_age_days,
            design_version=config.design_version,
        )
        for item in converted:
            item.metadata["bed_residence_age_offsets_hours_sha256"] = sampling.age_matrix_sha256()
        records.extend(converted)
    payload = _bed_arrival_payload(_bed_config())
    payload["provenance"]["bed_residence_sampling"] = (
        input_derivation_module._bed_residence_member_provenance(
            bed_config=bed,
            horizon_settings=SimpleNamespace(selection_support_days=180, runtime_support_days=90),
            arrivals=records,
        )
    )
    payload["records"] = [asdict(item) for item in records]
    return payload


@pytest.fixture(scope="module")
def member_dir(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ProjectConfig, dict[str, object]]:
    """寫出 material／receptor／arrival／initial_conditions 四份 per_member 合成 manifest。"""

    tmp_path = tmp_path_factory.mktemp("per_member")
    config = _member_config()
    arrival = _member_arrival_payload(config)
    receptor = _receptor_payload()
    # material／receptor fixture 預設是舊 design；current design 的 formal receptor 需要完整
    # 隨機受體 provenance，與本測試要驗的 per_member 無關，因此下面的 scenario／identity
    # 測試以 formal=False 載入（arrival 的 per_member 驗證與 formal 無關，仍完整執行）。
    material = _material_payload()
    material["design_version"] = config.design_version
    receptor["design_version"] = config.design_version
    (tmp_path / "config.yaml").write_text("placeholder: true\n", encoding="utf-8")
    _write_json(tmp_path / "material.json", material)
    _write_json(tmp_path / "receptor.json", receptor)
    _write_json(tmp_path / "arrival.json", arrival)
    _write_json(
        tmp_path / "initial_conditions.json",
        _initial_condition_payload(config, receptors=receptor, arrivals=arrival),
    )
    return tmp_path, config, arrival


def test_member_arrival_manifest_loads_and_recovers_observation_anchors(member_dir) -> None:
    tmp_path, config, payload = member_dir
    arrivals, _ = _load_arrival_document(tmp_path / "arrival.json", config, formal=True)
    assert len(arrivals) == 250 * _M
    observations = {observation_arrival_from_member_arrival(a).arrival_time_id for a in arrivals}
    assert len(observations) == 250
    for site in SITES:
        site_rows = [a for a in arrivals if a.study_site_id == site]
        assert len(site_rows) == 50 * _M
        months = {
            datetime.fromtimestamp(a.time_utc_ns / 1e9, tz=UTC).month
            for a in site_rows
        }
        # per_observation 的 fixture 沒有 10 月以後的沉底；逐成員隨機配對應該有
        assert months & {10, 11, 12}


def _mutated_arrival_path(tmp_path: Path, payload: dict[str, object], mutate) -> Path:
    clone = deepcopy(payload)
    mutate(clone)
    path = tmp_path / "mutated-arrival.json"
    _write_json(path, clone)
    return path


@pytest.mark.parametrize(
    "name",
    [
        "matrix_swap",
        "permutation_swap",
        "row_age",
        "row_member",
        "row_utc",
        "provenance_member_count",
        "audit_selected_age",
        "audit_initial_stratum",
        "audit_repair_steps_removed",
        "repair_summary",
        "repair_algorithm",
    ],
)
def test_member_arrival_manifest_rejects_tampering(member_dir, name: str) -> None:
    tmp_path, config, payload = member_dir

    def mutate(clone: dict[str, object]) -> None:
        sampling = clone["provenance"]["bed_residence_sampling"]
        records = clone["records"]
        if name == "matrix_swap":
            matrix = sampling["age_matrix_hours"]
            matrix[0][0], matrix[1][0] = matrix[1][0], matrix[0][0]
        elif name == "permutation_swap":
            perm = sampling["slot_permutation"]
            perm[0], perm[1] = perm[1], perm[0]
        elif name == "row_age":
            records[0]["metadata"]["bed_residence_age_hours"] += 1
        elif name == "row_member":
            records[0]["metadata"]["bed_residence_member_id"] = 1
        elif name == "row_utc":
            records[0]["time_utc_ns"] += _HOUR_NS
        elif name == "provenance_member_count":
            sampling["member_count"] = _M + 1
        elif name == "audit_selected_age":
            sampling["rejection_audit"][3]["selected_age_hours"] += 1
        elif name == "audit_initial_stratum":
            # 修補前分層必須等於由種子重算的排列
            audit = sampling["rejection_audit"][3]
            audit["initial_stratum_index"] = (audit["initial_stratum_index"] + 1) % (50 * _M)
        elif name == "audit_repair_steps_removed":
            # 年齡不變，但把某個槽位標成「原本不可行」而沒有任何修補步驟
            sampling["rejection_audit"][3]["initially_infeasible"] = True
        elif name == "repair_summary":
            sampling["permutation_repair"]["changed_slot_count"] += 1
        elif name == "repair_algorithm":
            sampling["permutation_repair"]["algorithm_id"] = "something_else"

    path = _mutated_arrival_path(tmp_path, payload, mutate)
    with pytest.raises(ValueError):
        _load_arrival_document(path, config, formal=True)


def test_member_arrival_manifest_rejects_member_count_config_mismatch(member_dir) -> None:
    tmp_path, _config, _payload = member_dir
    with pytest.raises(ValueError):
        _load_arrival_document(tmp_path / "arrival.json", _member_config(_M + 1), formal=True)


def test_scenario_inputs_per_member_counts_and_legacy_caller_guard(member_dir) -> None:
    tmp_path, config, _payload = member_dir
    inputs = load_scenario_inputs(
        config,
        config_path=tmp_path / "config.yaml",
        formal=False,
        per_member_supported=True,
    )
    assert len(inputs.scenarios) == 50_000, "情境數不隨成員數改變（以 250 個觀測錨點建立）"
    assert len(inputs.arrival_times) == 250
    assert len(inputs.member_arrival_times) == 250 * _M
    assert len(inputs.initial_conditions) == 5_000 * _M
    assert len(inputs.member_arrival_by_key) == 250 * _M
    anchor_ids = {a.arrival_time_id for a in inputs.arrival_times}
    assert {s.arrival_time_id for s in inputs.scenarios} == anchor_ids
    # 每個 (觀測錨點, 成員) 的 arrival 都能在初始條件表以 (receptor, 該 arrival) 查到
    key = next(iter(inputs.member_arrival_by_key))
    member_arrival = inputs.member_arrival_by_key[key]
    assert key == (
        member_arrival.metadata["observation_arrival_time_id"],
        member_arrival.metadata["bed_residence_member_id"],
    )
    receptor_id = next(iter({s.receptor_id for s in inputs.scenarios if s.study_site_id == member_arrival.study_site_id}))
    assert (receptor_id, member_arrival.arrival_time_id) in inputs.initial_conditions_by_pair
    with pytest.raises(ValueError, match="per_member_supported"):
        load_scenario_inputs(config, config_path=tmp_path / "config.yaml", formal=False)


@pytest.fixture(scope="module")
def member_table(member_dir):
    tmp_path, config, _payload = member_dir
    table = build_formal_identity_table(
        config,
        members_per_scenario=_M,
        config_path=tmp_path / "config.yaml",
        formal=False,
    )
    inputs = load_scenario_inputs(
        config, config_path=tmp_path / "config.yaml", formal=False, per_member_supported=True
    )
    return table, inputs, config


def test_identity_table_uses_each_members_own_deposition_time_and_start_depth(member_table) -> None:
    table, inputs, _config = member_table
    assert len(table) == 50_000 * _M
    assert np.unique(table.particle_id).size == len(table)
    # 每顆粒子的起算時刻與起點深度都對得回自己成員的沉底 arrival 與初始條件
    rng = np.random.default_rng(7)
    for row in rng.choice(len(table), size=400, replace=False):
        anchor_id = str(table.arrival_time_id[row])
        member = int(table.member_id[row])
        member_arrival = inputs.member_arrival_by_key[(anchor_id, member)]
        assert int(table.arrival_time_utc_ns[row]) == member_arrival.time_utc_ns
        condition = inputs.initial_conditions_by_pair[
            (str(table.receptor_id[row]), member_arrival.arrival_time_id)
        ]
        assert float(table.z0_m[row]) == condition.z_m_positive_up
        assert int(table.arrival_time_utc_ns[row]) == condition.time_utc_ns
    # 同一情境的不同成員，起算時刻與（因水柱不同的）起點深度可以不同
    scenario_rows = np.flatnonzero(table.scenario_id == table.scenario_id[0])
    assert len(scenario_rows) == _M
    assert len({int(table.arrival_time_utc_ns[r]) for r in scenario_rows}) == _M
    assert len({float(table.z0_m[r]) for r in scenario_rows}) == _M
    # 每站相異起算時刻遠多於 per_observation 的 50
    for site in SITES:
        mask = table.study_site_id == site
        assert np.unique(table.arrival_time_utc_ns[mask]).size > 50


def test_identity_table_labels_separate_deposition_and_observation(member_table) -> None:
    table, inputs, _config = member_table
    labels = table.labels
    assert labels is not None
    for name in (
        "deposition_season",
        "observation_season",
        "observation_tide_class",
        "observation_phase_or_event",
        "observation_time_utc_ns",
    ):
        assert name in labels and labels[name].shape == (len(table),)
    months = (
        table.arrival_time_utc_ns.astype("datetime64[ns]").astype("datetime64[M]").astype(np.int64) % 12 + 1
    )
    expected = np.array(["DJF", "DJF", "MAM", "MAM", "MAM", "JJA", "JJA", "JJA", "SON", "SON", "SON", "DJF"])
    assert np.array_equal(labels["deposition_season"], expected[months - 1])
    # 觀測時刻一定不早於沉底時刻（回溯起點在觀測之前）
    assert np.all(labels["observation_time_utc_ns"] >= table.arrival_time_utc_ns)


def test_member_identity_table_is_reproducible(member_table, member_dir) -> None:
    table, _inputs, config = member_table
    tmp_path, _config, _payload = member_dir
    again = build_formal_identity_table(
        config, members_per_scenario=_M, config_path=tmp_path / "config.yaml", formal=False
    )
    assert again.content_sha256 == table.content_sha256


def test_member_count_mismatch_between_config_and_call_fails(member_dir) -> None:
    tmp_path, config, _payload = member_dir
    with pytest.raises(Exception, match="members_per_scenario"):
        build_formal_identity_table(
            config, members_per_scenario=_M + 1, config_path=tmp_path / "config.yaml", formal=False
        )


# ── per_observation 位元相同的黃金值（以 WP-51 之前的 e8f30a7 原始碼在 .203 算出） ──


def test_per_observation_identity_table_hash_is_unchanged() -> None:
    """沒有標籤與 resolver 時，身分表內容雜湊與 WP-51 之前逐位元相同。"""

    from test_native_particle_input import _build_synthetic_matrix

    from lagrangian_backtracking.native_runner import build_particle_identity_table

    def resolve(unit):
        scenario = unit.scenario
        r = zlib.crc32(scenario.receptor_id.encode()) % 100000
        a = zlib.crc32(scenario.arrival_time_id.encode()) % 100000
        return 300000.0 + r * 0.1, 2500000.0 + a * 0.1, -5.0 - (r % 50) * 0.5

    shards, _ = _build_synthetic_matrix(
        site_count=2, receptor_count=3, arrival_count=2, members_per_scenario=2
    )
    table = build_particle_identity_table(shards, master_seed=12345, resolve_initial_position=resolve)
    assert table.labels is None
    # WP-54（v7）把十類代表速度改為 10 個候選的幾何平均，身分表內容雜湊因此改變；雜湊演算法與欄位
    # 格式沒有變動：以 v6 的舊代表速度重算，仍得到 WP-51 之前的黃金值
    # e4017f21a7fd7b5cd7e1f828962efe8aabafe48aa4e3cdbea4e3c7b8a37d410e（已於 .203 實測核對）。
    assert table.content_sha256 == (
        "3bf9f46f6176c8be84023c98a72ce991ae96759e86cc4f49dd8ae014f2ffb752"
    )


def test_per_observation_sampling_output_is_unchanged() -> None:
    """``_apply_bed_residence_sampling_by_site`` 的 per_observation 輸出（legacy 與條件式兩條）
    與 WP-51 之前逐位元相同。"""

    from test_input_derivation import _synthetic_bed_residence_observation_by_site

    config = input_derivation_module.load_config(EXAMPLE_CONFIG, formal_release=False)
    bed = config.scenarios.bed_residence_time
    assert bed.age_sampling_scope == "per_observation"
    observations = _synthetic_bed_residence_observation_by_site(config)
    start = int(datetime(2023, 1, 1, tzinfo=UTC).timestamp()) * 10**9
    available = {
        site.study_site_id: tuple(range(start, start + 3 * 8760 * _HOUR_NS, _HOUR_NS))
        for site in config.study_sites
    }

    def digest(by_site) -> str:
        payload = {site: [asdict(a) for a in rows] for site, rows in sorted(by_site.items())}
        return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    legacy = input_derivation_module._apply_bed_residence_sampling_by_site(
        observations, bed_residence=bed, design_version=config.design_version
    )
    conditioned = input_derivation_module._apply_bed_residence_sampling_by_site(
        observations,
        bed_residence=bed,
        design_version=config.design_version,
        canonical_available_time_ns_by_site=available,
        gap_censoring_enabled=True,
    )
    assert digest(legacy) == "a589421551937f26c560a953bdc53452db42b2746a54cf9d54bffe161ed5a751"
    assert digest(conditioned) == "a57b79b8e5d8c31c3888ee26cc966fe24f29df82914e8ace38da892f0818f37d"


def test_per_particle_scope_fails_closed_in_input_build() -> None:
    config = input_derivation_module.load_config(EXAMPLE_CONFIG, formal_release=False)
    bed = config.scenarios.bed_residence_time.model_copy(update={"age_sampling_scope": "per_particle"})
    with pytest.raises(input_derivation_module.InputDerivationError, match="per_particle"):
        input_derivation_module._apply_bed_residence_sampling_by_site(
            {}, bed_residence=bed, design_version=config.design_version
        )


@pytest.fixture(scope="module")
def gappy_member_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, ProjectConfig, dict[str, object]]:
    """可用時間軸含整日缺口：隨機排列必然有不可行槽位，抽樣器必須走修補才建得出母體。"""

    tmp_path = tmp_path_factory.mktemp("per_member_gappy")
    config = _member_config()
    arrival = _member_arrival_payload(config, hours=_gappy_hours())
    _write_json(tmp_path / "arrival.json", arrival)
    return tmp_path, config, arrival


def test_gappy_member_arrival_manifest_records_repair_and_loads(gappy_member_dir) -> None:
    """含缺口的母體：provenance 記錄修補摘要，載入器（含種子重算修補前排列）接受。"""

    tmp_path, config, payload = gappy_member_dir
    sampling = payload["provenance"]["bed_residence_sampling"]
    repair = sampling["permutation_repair"]
    assert repair["initially_infeasible_slot_count"] > 0, "測試資料必須真的觸發修補"
    assert repair["repair_step_count"] >= 1
    assert repair["changed_slot_count"] >= repair["initially_infeasible_slot_count"]
    assert "feasibility_repair" in sampling["sampling_method_id"]
    arrivals, _ = _load_arrival_document(tmp_path / "arrival.json", config, formal=True)
    assert len(arrivals) == 250 * _M
    hours = set(_gappy_hours())
    for item in arrivals:
        assert item.time_utc_ns in hours, "沉底整點必須落在可用軸內"


@pytest.mark.parametrize("name", ["audit_initial_stratum", "audit_repair_steps_removed", "repair_summary"])
def test_gappy_member_arrival_manifest_rejects_repair_tampering(gappy_member_dir, name: str) -> None:
    tmp_path, config, payload = gappy_member_dir

    def mutate(clone: dict[str, object]) -> None:
        sampling = clone["provenance"]["bed_residence_sampling"]
        if name == "audit_initial_stratum":
            audit = sampling["rejection_audit"][3]
            audit["initial_stratum_index"] = (audit["initial_stratum_index"] + 1) % (50 * _M)
        elif name == "audit_repair_steps_removed":
            # 把一個被修補過的槽位的修補步驟清空（分層已改變卻沒有紀錄）
            for audit in sampling["rejection_audit"]:
                if audit["repair_steps"]:
                    audit["repair_steps"] = []
                    break
        else:
            sampling["permutation_repair"]["initially_infeasible_slot_count"] -= 1

    path = _mutated_arrival_path(tmp_path, payload, mutate)
    with pytest.raises(ValueError):
        _load_arrival_document(path, config, formal=True)


# ── WP-51 整合：沉底整點「受體可支撐」條件的 provenance 與載入器核對 ─────────────────────


@pytest.fixture(scope="module")
def support_member_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, ProjectConfig, dict[str, object]]:
    """canonical 軸完整、但每三天有一整天「受體不可支撐」（例如 NWW 該日無效）：抽樣器必須把
    這些整點當成不可用，provenance 記錄各站被排除的整點清單。"""

    from lagrangian_backtracking.bed_residence import restrict_available_times_to_supported

    tmp_path = tmp_path_factory.mktemp("per_member_support")
    config = _member_config()
    available, support = restrict_available_times_to_supported(
        {site: _all_hours() for site in SITES}, {site: _gappy_hours() for site in SITES}
    )
    arrival = _member_arrival_payload(config, hours=available[next(iter(SITES))])
    arrival["provenance"]["bed_residence_sampling"]["deposition_support_conditioning"] = support
    _write_json(tmp_path / "arrival.json", arrival)
    return tmp_path, config, arrival


def test_support_conditioned_manifest_loads_and_avoids_excluded_hours(support_member_dir) -> None:
    tmp_path, config, payload = support_member_dir
    block = payload["provenance"]["bed_residence_sampling"]["deposition_support_conditioning"]
    assert block["policy_id"] == "nww_metric_location_exact_hour_support_v1"
    arrivals, _ = _load_arrival_document(tmp_path / "arrival.json", config, formal=True)
    assert len(arrivals) == 250 * _M
    for item in arrivals:
        excluded = set(block["by_site"][item.study_site_id]["excluded_hours_utc_ns"])
        assert excluded and item.time_utc_ns not in excluded


@pytest.mark.parametrize("name", ["policy_id", "unsorted_hours", "hash", "deposition_in_excluded", "missing_site"])
def test_support_conditioned_manifest_rejects_tampering(support_member_dir, name: str) -> None:
    tmp_path, config, payload = support_member_dir

    def mutate(clone: dict[str, object]) -> None:
        block = clone["provenance"]["bed_residence_sampling"]["deposition_support_conditioning"]
        record = block["by_site"][next(iter(SITES))]
        if name == "policy_id":
            block["policy_id"] = "other_policy"
        elif name == "unsorted_hours":
            record["excluded_hours_utc_ns"] = list(reversed(record["excluded_hours_utc_ns"]))
        elif name == "hash":
            record["excluded_hours_sha256"] = "0" * 64
        elif name == "missing_site":
            del block["by_site"][next(iter(SITES))]
        else:
            # 把某筆 arrival 的沉底整點加進排除清單（重算雜湊使其他檢查通過）：載入器必須拒絕
            import hashlib

            target = next(r for r in clone["records"] if r["study_site_id"] == next(iter(SITES)))["time_utc_ns"]
            hours = sorted(set(record["excluded_hours_utc_ns"]) | {target})
            record["excluded_hours_utc_ns"] = hours
            record["excluded_hours_sha256"] = hashlib.sha256(
                json.dumps(hours, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

    path = _mutated_arrival_path(tmp_path, payload, mutate)
    with pytest.raises(ValueError):
        _load_arrival_document(path, config, formal=True)


def test_arrival_payload_injects_support_provenance_only_for_per_member(support_member_dir) -> None:
    """``_arrival_payload`` 把受體可支撐稽核寫進 ``bed_residence_sampling``（只在傳入時）。"""

    tmp_path, config, payload = support_member_dir
    arrivals, _ = _load_arrival_document(tmp_path / "arrival.json", config, formal=True)
    support = payload["provenance"]["bed_residence_sampling"]["deposition_support_conditioning"]
    settings = SimpleNamespace(
        is_generic=False,
        bed_residence_enabled=True,
        selection_support_days=180,
        runtime_support_days=90,
    )
    built = input_derivation_module._arrival_payload(
        config,
        arrivals,
        source_hashes={"x": "0" * 64},
        strict=True,
        horizon_settings=settings,
        deposition_support_provenance=support,
    )
    block = built["provenance"]["bed_residence_sampling"]["deposition_support_conditioning"]
    assert block == json.loads(json.dumps(support))
    without = input_derivation_module._arrival_payload(
        config,
        arrivals,
        source_hashes={"x": "0" * 64},
        strict=True,
        horizon_settings=settings,
    )
    assert "deposition_support_conditioning" not in without["provenance"]["bed_residence_sampling"]


def test_face_support_failure_summary_classifies_nww_and_ocm_and_ranks_times() -> None:
    """face_support 失敗彙總：NWW／OCM 分類、最集中的失敗整點（時刻層級問題的證據）。"""

    t_bad = 1_744_668_000_000_000_000  # 2025-04-14T22:00:00Z
    t_other = t_bad + 3_600_000_000_000
    errors = {
        1: f"hsinchu receptor face NWW runtime 四角支撐失敗：face=1, utc={t_bad}",
        2: f"hsinchu receptor face NWW runtime 四角支撐失敗：face=2, utc={t_bad}",
        3: f"hsinchu receptor face NWW runtime 四角支撐失敗：face=3, utc={t_other}",
        4: "hsinchu receptor face 垂向支撐失敗：zcor 雙側支撐不足",
    }
    summary = input_derivation_module._summarize_face_support_failures(errors)
    assert summary["policy_id"] == "receptor_face_support_failure_summary_v1"
    assert summary["failed_face_count"] == 4
    assert summary["nww_four_corner_failed_faces"] == 3
    assert summary["ocm_vertical_or_other_failed_faces"] == 1
    top = summary["top_nww_first_failure_utc"]
    assert top[0]["utc_ns"] == t_bad and top[0]["failed_faces"] == 2
    assert top[0]["utc"] == "2025-04-14T22:00:00Z"
    assert len(summary["examples"]) == 3
    json.dumps(summary)  # 必須可序列化進 provenance
    empty = input_derivation_module._summarize_face_support_failures({})
    assert empty["failed_face_count"] == 0 and empty["top_nww_first_failure_utc"] == []
