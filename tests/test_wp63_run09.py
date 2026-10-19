"""WP-63（run09 程式整合）：每一項新增功能一個以上的單元測試。

預設行為（設定檔不明示新欄位）必須與 run08 位元相同：config hash、selector 輸入、manifest 驗證都有
對應測試；真實資料的 run08 重現（到達時刻與受體抽樣）在伺服器腳本 ``scripts/wp63/``，不在單元測試內。
"""

from __future__ import annotations

import copy
import json
import os
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import lagrangian_backtracking.config as config_module
from lagrangian_backtracking.bed_residence import (
    load_deposition_exclusion_hours,
    parse_exclusion_hour,
    restrict_available_times_to_supported,
)
from lagrangian_backtracking.config import (
    HORIZONTAL_RECEPTOR_SELECTION_POLICY_AZIMUTH_DISTANCE_LATIN,
    load_config,
)
from lagrangian_backtracking.input_derivation import (
    InputDerivationError,
    _assert_dense_array,
)
from lagrangian_backtracking.manifests import (
    _a_pair_utc_required,
    _validate_external_exclusion_provenance,
    _validate_latin_sampling_record,
    _validate_seam_jump_exclusion_provenance,
)
from lagrangian_backtracking.receptor_stratified import (
    Candidates,
    sample_azimuth_distance_latin,
    select_horizontal_receptors_azimuth_distance_latin,
)
from lagrangian_backtracking.receptors import HorizontalReceptorCandidatePool
from lagrangian_backtracking.scenarios import ArrivalTime
from lagrangian_backtracking.seam_jumps import (
    SeamJumpParameters,
    build_seam_jump_exclusion,
    expand_exclusion_times,
    find_seam_jump_indices,
    sha256_of_time_list,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = ROOT / "configs" / "lagrangian_backtracking.formal_h30_h60_h90_m10.yaml"
EXAMPLE_CONFIG = ROOT / "configs" / "lagrangian_backtracking.example.yaml"
HOUR = 3_600_000_000_000
T0 = int(np.datetime64("2025-01-01T00:00:00", "ns").astype(np.int64))

# run08 的程式碼（wp54-class-velocity@8141b2b）在這兩份設定上的 config hash；新欄位全部預設關閉時必須不變。
RUN08_HASH_FORMAL = "9d3726460353d6e1353753cd35de70198e04271d9bec44419ced045b096fd8ef"
RUN08_HASH_EXAMPLE = "f83c22b9ee5bed9b1cbd898692816bbcf37ebe286f9354adc30b038506688352"


def _write_variant(tmp_path: Path, mutate) -> Path:
    payload = yaml.safe_load(FORMAL_CONFIG.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 接縫跳動排除
def _tide(n: int = 24 * 365, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    t = T0 + np.arange(n, dtype=np.int64) * HOUR
    hours = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    eta = 0.6 * np.sin(2 * np.pi * hours / 12.42) + 0.02 * rng.standard_normal(n)
    return t, eta


def test_seam_jump_detects_injected_spike_and_nothing_else() -> None:
    times, eta = _tide()
    k = 4000
    eta[k] += 0.8  # 單點尖峰：二階差分在 k-1、k、k+1 都異常大
    flagged = find_seam_jump_indices(times, eta, reference_years=[2025])
    assert k in flagged
    assert set(flagged.tolist()) <= {k - 1, k, k + 1}


def test_seam_jump_clean_series_has_no_flags_and_reference_year_filter() -> None:
    times, eta = _tide()
    assert find_seam_jump_indices(times, eta, reference_years=[2025]).size == 0
    eta[100] += 0.8
    assert find_seam_jump_indices(times, eta, reference_years=[2024]).size == 0  # 不在參考年份內


def test_seam_jump_exclusion_union_expands_radius_and_hash_is_deterministic() -> None:
    times, eta_a = _tide(seed=1)
    _, eta_b = _tide(seed=2)
    eta_a[3000] += 0.9
    eta_b[5000] += 0.9
    series = {"a": (times, eta_a), "b": (times, eta_b)}
    first = build_seam_jump_exclusion(series, reference_years=[2025])
    second = build_seam_jump_exclusion(series, reference_years=[2025])
    assert first.provenance == second.provenance
    for site in ("a", "b"):  # union：兩站都含兩個跳動時刻，各自 ±1 小時
        excluded = set(first.excluded_time_ns_by_site[site])
        for k in (3000, 5000):
            assert {int(times[k]) + d * HOUR for d in (-1, 0, 1)} <= excluded
        record = first.provenance["excluded_times_by_site"][site]
        assert record["count"] == len(excluded)
        assert record["sha256"] == sha256_of_time_list(sorted(excluded))
    own = build_seam_jump_exclusion(
        series, reference_years=[2025], parameters=SeamJumpParameters(scope="own_site")
    )
    assert int(times[5000]) not in set(own.excluded_time_ns_by_site["a"])
    assert int(times[5000]) in set(own.excluded_time_ns_by_site["b"])


def test_seam_jump_parameters_validated_and_expand_helper() -> None:
    with pytest.raises(ValueError):
        SeamJumpParameters(scope="nope")
    assert expand_exclusion_times([T0], 1) == (T0 - HOUR, T0, T0 + HOUR)
    assert expand_exclusion_times([T0, T0 + HOUR], 1) == (T0 - HOUR, T0, T0 + HOUR, T0 + 2 * HOUR)


def test_seam_exclusion_provenance_validator_accepts_and_rejects(tmp_path: Path) -> None:
    config = load_config(
        _write_variant(
            tmp_path,
            lambda p: p["arrival_time_selection"].update(seam_jump_exclusion={}),
        )
    )
    times, eta = _tide()
    eta[3000] += 0.9
    sites = [s.study_site_id for s in config.study_sites]
    result = build_seam_jump_exclusion({s: (times, eta.copy()) for s in sites}, reference_years=[2025])
    provenance = {"seam_jump_exclusion": json.loads(json.dumps(result.provenance))}
    ok = ArrivalTime("arr_ok", "gongliao", int(times[10]), 2025, "DJF", "spring_proxy", "slack", {})
    _validate_seam_jump_exclusion_provenance([ok], provenance, config)
    bad = ArrivalTime("arr_bad", "gongliao", int(times[3000]), 2025, "DJF", "spring_proxy", "slack", {})
    with pytest.raises(ValueError, match="排除清單"):
        _validate_seam_jump_exclusion_provenance([bad], provenance, config)
    tampered = copy.deepcopy(provenance)
    tampered["seam_jump_exclusion"]["excluded_times_by_site"]["gongliao"]["utc_ns"].append(1)
    with pytest.raises(ValueError):
        _validate_seam_jump_exclusion_provenance([ok], tampered, config)
    with pytest.raises(ValueError, match="缺少"):
        _validate_seam_jump_exclusion_provenance([ok], {}, config)
    # 未啟用時帶有 provenance 也要拒絕
    plain = load_config(FORMAL_CONFIG)
    with pytest.raises(ValueError, match="沒有啟用"):
        _validate_seam_jump_exclusion_provenance([ok], provenance, plain)
    _validate_seam_jump_exclusion_provenance([ok], {}, plain)


# ---------------------------------------------------------------- 分層抽樣
def _ring_candidates(seed: int = 3, n: int = 4000, radius: float = 10_000.0) -> Candidates:
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 360, n)
    dist = radius * np.sqrt(rng.uniform(0.01, 1.0, n))
    return Candidates(
        face_id=np.arange(n, dtype=np.int64) + 1000,
        azimuth_deg=az,
        distance_m=dist,
        area_m2=rng.uniform(500.0, 900.0, n),
    )


def test_latin_sampling_is_deterministic_and_stratified() -> None:
    cand = _ring_candidates()
    a = sample_azimuth_distance_latin(cand, seed=12345, n=5)
    b = sample_azimuth_distance_latin(cand, seed=12345, n=5)
    assert a == b
    assert sorted(p["azimuth_bin"] for p in a.picks) == [1, 2, 3, 4, 5]
    assert sorted(p["distance_bin"] for p in a.picks) == [1, 2, 3, 4, 5]
    assert sorted(a.permutation) == [1, 2, 3, 4, 5]
    assert a.distance_edges_m[0] == 0.0
    c = sample_azimuth_distance_latin(cand, seed=999, n=5)
    assert c != a  # 不同種子不同結果


def test_latin_sampling_replacement_in_same_cell_and_permutation_fallback() -> None:
    cand = _ring_candidates()
    base = sample_azimuth_distance_latin(cand, seed=7, n=5)
    # (1) 第一個挑中的面被閘門否決：改由同一區塊內隨機順序的下一個遞補，且仍在同一區塊
    first = base.picks[0]
    vetoed = {first["face_id"]}
    again = sample_azimuth_distance_latin(cand, seed=7, n=5, is_supported=lambda f: f not in vetoed)
    assert again.attempts == 1
    assert again.picks[0]["face_id"] != first["face_id"]
    assert again.picks[0]["azimuth_bin"] == first["azimuth_bin"]
    assert again.picks[0]["distance_bin"] == first["distance_bin"]
    assert again.picks[0]["skipped_unsupported"] == 1
    # (2) 整個區塊被否決：換下一個排列
    az_lo, az_hi = base.azimuth_edges_deg[0], base.azimuth_edges_deg[1]
    d_lo, d_hi = base.distance_edges_m[base.permutation[0] - 1], base.distance_edges_m[base.permutation[0]]

    def in_cell(face: int) -> bool:
        i = face - 1000
        az = float(cand.azimuth_deg[i])
        span = (az_hi - az_lo) % 360.0
        inside_az = ((az - az_lo) % 360.0) < span
        return inside_az and d_lo <= float(cand.distance_m[i]) <= d_hi

    fallback = sample_azimuth_distance_latin(cand, seed=7, n=5, is_supported=lambda f: not in_cell(f))
    assert fallback.attempts > 1
    assert len(fallback.attempt_log) == fallback.attempts - 1
    assert fallback.permutation != base.permutation


def test_latin_sampling_fails_closed_when_no_permutation_works() -> None:
    cand = _ring_candidates(n=600)
    with pytest.raises(ValueError, match="所有 120 個排列"):
        sample_azimuth_distance_latin(cand, seed=1, n=5, is_supported=lambda f: False)


def _synthetic_pool(radius: float = 12_000.0, n: int = 3000, seed: int = 4) -> tuple[HorizontalReceptorCandidatePool, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    anchor = (121.9, 25.0)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0, 2 * np.pi, n)
    x, y = r * np.sin(th), r * np.cos(th)
    lon = anchor[0] + x / (111_320.0 * np.cos(np.radians(anchor[1])))
    lat = anchor[1] + y / 110_574.0
    idx = np.arange(n)
    pool = HorizontalReceptorCandidatePool(
        study_site_id="demo",
        anchor_xy=(0.0, 0.0),
        maximum_anchor_snap_distance_m=None,
        candidate_face_local_indices=idx,
        candidate_face_global_indices=idx + 50_000,
        candidate_xy_m=np.column_stack([x, y]),
        candidate_lon=lon,
        candidate_lat=lat,
        tie_break_keys=tuple((0.0, 0.0, int(i)) for i in idx),
        candidate_area_m2=np.full(n, 700.0),
    )
    return pool, anchor


def test_latin_pool_selection_record_passes_manifest_validator(tmp_path: Path) -> None:
    pool, anchor = _synthetic_pool()

    def mutate(p: dict) -> None:
        p["scenarios"]["horizontal_receptor_selection_policy"] = HORIZONTAL_RECEPTOR_SELECTION_POLICY_AZIMUTH_DISTANCE_LATIN
        p["scenarios"]["horizontal_receptor_sampling"] = {"radius_m": 10000}

    config = load_config(_write_variant(tmp_path, mutate))
    receptors, record = select_horizontal_receptors_azimuth_distance_latin(
        pool,
        anchor_lonlat=anchor,
        master_seed=20260916,
        design_version=config.design_version,
        radius_m=10_000.0,
        count=5,
        selection_policy_id=HORIZONTAL_RECEPTOR_SELECTION_POLICY_AZIMUTH_DISTANCE_LATIN,
    )
    assert len(receptors) == 5
    assert all(p["distance_m"] <= 10_000.0 for p in record["picks"])
    selected_global = [r.source_face_global_index for r in receptors]
    _validate_latin_sampling_record("demo", {"stratified_sampling": record}, config, selected_global)
    # 半徑與設定不一致、picks 順序改變都要被拒絕
    wrong = copy.deepcopy(record)
    wrong["radius_m"] = 12_500.0
    with pytest.raises(ValueError, match="半徑"):
        _validate_latin_sampling_record("demo", {"stratified_sampling": wrong}, config, selected_global)
    with pytest.raises(ValueError):
        _validate_latin_sampling_record("demo", {"stratified_sampling": record}, config, selected_global[::-1])
    # 同種子、同池 -> 同結果；半徑縮小後候選只剩圓內
    receptors2, record2 = select_horizontal_receptors_azimuth_distance_latin(
        pool, anchor_lonlat=anchor, master_seed=20260916, design_version=config.design_version,
        radius_m=10_000.0, count=5,
        selection_policy_id=HORIZONTAL_RECEPTOR_SELECTION_POLICY_AZIMUTH_DISTANCE_LATIN,
    )
    assert [r.source_face_global_index for r in receptors2] == selected_global
    assert record2 == record


def test_latin_pool_selection_requires_area_and_enough_candidates() -> None:
    pool, anchor = _synthetic_pool(n=40)
    with pytest.raises(ValueError):
        select_horizontal_receptors_azimuth_distance_latin(
            pool, anchor_lonlat=anchor, master_seed=1, design_version="d", radius_m=1.0, count=5
        )
    import dataclasses

    no_area = dataclasses.replace(pool, candidate_area_m2=None)
    with pytest.raises(ValueError, match="candidate_area_m2"):
        select_horizontal_receptors_azimuth_distance_latin(
            no_area, anchor_lonlat=anchor, master_seed=1, design_version="d", radius_m=10_000.0, count=5
        )


# ---------------------------------------------------------------- 設定檔
def test_new_config_fields_default_off_and_hashes_unchanged() -> None:
    assert load_config(FORMAL_CONFIG).config_hash() == RUN08_HASH_FORMAL
    assert load_config(EXAMPLE_CONFIG).config_hash() == RUN08_HASH_EXAMPLE
    payload = load_config(FORMAL_CONFIG).normalized_payload()
    assert "seam_jump_exclusion" not in payload["arrival_time_selection"]
    assert "horizontal_receptor_sampling" not in payload["scenarios"]
    assert "deposition_exclusion_hours_by_site" not in payload["scenarios"]["bed_residence_time"]
    assert "deposition_exclusion_hours_file" not in payload["scenarios"]["bed_residence_time"]
    assert all("anchor_source" not in site for site in payload["study_sites"])


def test_new_config_fields_enter_hash_only_when_explicit(tmp_path: Path) -> None:
    base = load_config(FORMAL_CONFIG).config_hash()

    def seam(p: dict) -> None:
        p["arrival_time_selection"]["seam_jump_exclusion"] = {}

    cfg = load_config(_write_variant(tmp_path, seam))
    assert cfg.config_hash() != base
    assert cfg.arrival_time_selection.seam_jump_exclusion.to_parameters() == SeamJumpParameters()

    def bad_hours(p: dict) -> None:
        p["scenarios"]["bed_residence_time"]["deposition_exclusion_hours_by_site"] = {
            "gongliao": ["2025-03-01T05:00:00Z"]
        }

    cfg2 = load_config(_write_variant(tmp_path, bad_hours))
    assert cfg2.config_hash() != base
    assert "deposition_exclusion_hours_by_site" in cfg2.normalized_payload()["scenarios"]["bed_residence_time"]


def test_seam_config_rejects_bad_parameters(tmp_path: Path) -> None:
    def bad(p: dict) -> None:
        p["arrival_time_selection"]["seam_jump_exclusion"] = {"scope": "everything"}

    with pytest.raises(Exception):
        load_config(_write_variant(tmp_path, bad))


def test_latin_policy_requires_sampling_block_and_is_accepted(tmp_path: Path) -> None:
    def ok(p: dict) -> None:
        p["scenarios"]["horizontal_receptor_selection_policy"] = HORIZONTAL_RECEPTOR_SELECTION_POLICY_AZIMUTH_DISTANCE_LATIN
        p["scenarios"]["horizontal_receptor_sampling"] = {"radius_m": 10000}

    cfg = load_config(_write_variant(tmp_path, ok))
    assert cfg.scenarios.horizontal_receptor_sampling.radius_m == 10000.0

    def unknown(p: dict) -> None:
        p["scenarios"]["horizontal_receptor_selection_policy"] = "some_other_policy"

    with pytest.raises(Exception):
        load_config(_write_variant(tmp_path, unknown))

    def negative(p: dict) -> None:
        p["scenarios"]["horizontal_receptor_sampling"] = {"radius_m": -1}

    with pytest.raises(Exception):
        load_config(_write_variant(tmp_path, negative))


def test_anchor_override_flag_relaxes_pin_only_when_explicit(tmp_path: Path) -> None:
    def moved(p: dict) -> None:
        site = next(s for s in p["study_sites"] if s["study_site_id"] == "nanwan")
        site["anchor_lonlat"] = [120.77, 21.95]

    with pytest.raises(Exception, match="anchor"):
        load_config(_write_variant(tmp_path, moved))

    def moved_with_flag(p: dict) -> None:
        moved(p)
        site = next(s for s in p["study_sites"] if s["study_site_id"] == "nanwan")
        site["anchor_source"] = "config_override_v1"

    cfg = load_config(_write_variant(tmp_path, moved_with_flag))
    nanwan = next(s for s in cfg.study_sites if s.study_site_id == "nanwan")
    assert tuple(nanwan.anchor_lonlat) == (120.77, 21.95)
    assert nanwan.anchor_source == config_module.ANCHOR_SOURCE_CONFIG_OVERRIDE


# ---------------------------------------------------------------- 配對開關
def test_a_pair_utc_required_default_true_and_off_when_configured(tmp_path: Path) -> None:
    assert _a_pair_utc_required(load_config(FORMAL_CONFIG)) is True

    def off(p: dict) -> None:
        p["arrival_time_selection"]["northeast_pair_utc_when_coverage_allows"] = False

    cfg = load_config(_write_variant(tmp_path, off))
    assert _a_pair_utc_required(cfg) is False
    assert cfg.arrival_time_selection.pairs_northeast_utc is False
    # 沒有 arrival_time_selection 區塊的舊設定視為必須配對
    assert _a_pair_utc_required(SimpleNamespace(arrival_time_selection=None)) is True


# ---------------------------------------------------------------- 壞時刻掛勾
def test_parse_exclusion_hour_formats_and_rejects() -> None:
    expected = int(np.datetime64("2025-03-01T05:00:00", "ns").astype(np.int64))
    assert parse_exclusion_hour("2025-03-01T05:00:00Z") == expected
    assert parse_exclusion_hour("2025-03-01 05:00") == expected
    assert parse_exclusion_hour(expected) == expected
    for bad in ("2025-03-01T05:30:00Z", "not a time", 1.5, True, expected + 1):
        with pytest.raises(ValueError):
            parse_exclusion_hour(bad)


def test_restrict_available_times_default_is_unchanged_and_hook_adds_provenance() -> None:
    hours = [T0 + i * HOUR for i in range(10)]
    canonical = {"a": hours, "b": hours}
    supported = {"a": hours[:9], "b": hours}
    avail0, prov0 = restrict_available_times_to_supported(canonical, supported)
    assert "external_exclusion" not in prov0  # 預設關閉：與 run08 provenance 結構相同
    assert avail0["a"] == tuple(hours[:9]) and avail0["b"] == tuple(hours)
    extra = {"a": [hours[2]], "b": [hours[4], hours[5]]}
    avail1, prov1 = restrict_available_times_to_supported(
        canonical, supported, extra_excluded_time_ns_by_site=extra, extra_exclusion_source={"source": "t"}
    )
    assert hours[2] not in avail1["a"] and hours[4] not in avail1["b"]
    assert prov1["by_site"]["a"]["excluded_hours_utc_ns"] == sorted([hours[9], hours[2]])
    assert prov1["external_exclusion"]["by_site"]["b"]["excluded_hours_utc_ns"] == [hours[4], hours[5]]
    _validate_external_exclusion_provenance(
        prov1, {s: frozenset(r["excluded_hours_utc_ns"]) for s, r in prov1["by_site"].items()}
    )
    tampered = copy.deepcopy(prov1)
    tampered["external_exclusion"]["by_site"]["b"]["excluded_hours_utc_ns"] = [hours[4]]
    with pytest.raises(ValueError):
        _validate_external_exclusion_provenance(
            tampered, {s: frozenset(r["excluded_hours_utc_ns"]) for s, r in prov1["by_site"].items()}
        )
    with pytest.raises(ValueError, match="未知站點"):
        restrict_available_times_to_supported(canonical, supported, extra_excluded_time_ns_by_site={"zzz": [1]})


def test_load_deposition_exclusion_hours_inline_file_and_default(tmp_path: Path) -> None:
    sites = ["a", "b"]
    assert load_deposition_exclusion_hours(SimpleNamespace(), site_ids=sites) is None
    inline = SimpleNamespace(
        deposition_exclusion_hours_by_site={"a": ["2025-01-01T03:00:00Z", "2025-01-01T03:00:00Z"]},
        deposition_exclusion_hours_file=None,
    )
    by_site, source = load_deposition_exclusion_hours(inline, site_ids=sites)
    assert by_site == {"a": (T0 + 3 * HOUR,), "b": ()} and source["source"] == "config_inline"
    path = tmp_path / "bad_hours.json"
    path.write_text(json.dumps({"b": [T0 + HOUR]}), encoding="utf-8")
    from_file = SimpleNamespace(deposition_exclusion_hours_by_site=None, deposition_exclusion_hours_file=str(path))
    by_site2, source2 = load_deposition_exclusion_hours(from_file, site_ids=sites)
    assert by_site2["b"] == (T0 + HOUR,)
    assert source2["file_sha256"] == sha256(path.read_bytes()).hexdigest()
    unknown = SimpleNamespace(deposition_exclusion_hours_by_site={"zzz": [T0]}, deposition_exclusion_hours_file=None)
    with pytest.raises(ValueError, match="未設定的站點"):
        load_deposition_exclusion_hours(unknown, site_ids=sites)


# ---------------------------------------------------------------- 稀疏占位檔防線
def test_dense_guard_rejects_sparse_placeholder_and_accepts_real_file(tmp_path: Path) -> None:
    sparse = tmp_path / "zcor.npy"
    mm = np.lib.format.open_memmap(sparse, mode="w+", dtype=np.float32, shape=(1024, 1024))
    del mm
    status = os.stat(sparse)
    if status.st_blocks * 512 >= 0.05 * status.st_size:
        pytest.skip("此檔案系統不支援稀疏檔，無法產生占位檔")
    with pytest.raises(InputDerivationError, match="稀疏占位檔"):
        _assert_dense_array(np.load(sparse, mmap_mode="r"), label="test")
    real = tmp_path / "real.npy"
    np.save(real, np.random.default_rng(0).standard_normal((1024, 1024)).astype(np.float32))
    _assert_dense_array(np.load(real, mmap_mode="r"), label="test")
    _assert_dense_array(np.zeros((4, 4)), label="in-memory arrays are skipped")


# ---------------------------------------------------------------- 錨點與圖
def test_sampling_range_circle_follows_config_override_and_sampling_radius() -> None:
    from lagrangian_backtracking.native_runner import figures

    anchor = (120.77, 21.95)  # 與常數表的南灣錨點不同
    with pytest.raises(figures.FiguresError):
        figures._sampling_range_circle("nanwan", anchor, 12500.0)
    circle = figures._sampling_range_circle(
        "nanwan", anchor, 12500.0, sampling_radius_m=10000.0, anchor_is_config_override=True
    )
    dist = figures.geodesic_distance_km(anchor[0], anchor[1], circle[:, 0], circle[:, 1])
    assert np.allclose(dist, 10.0, atol=1e-3)
    # run08 預設：錨點與半徑都與表一致才通過
    table_anchor = figures.SAMPLING_RANGE_ANCHORS_LONLAT["nanwan"]
    assert figures._sampling_range_circle("nanwan", table_anchor, 12500.0).shape == (361, 2)


def test_figures_loader_reads_sampling_overrides(tmp_path: Path) -> None:
    from lagrangian_backtracking.native_runner import figures

    def mutate(p: dict) -> None:
        p["scenarios"]["horizontal_receptor_sampling"] = {"radius_m": 10000}
        site = next(s for s in p["study_sites"] if s["study_site_id"] == "nanwan")
        site["anchor_source"] = "config_override_v1"

    path = _write_variant(tmp_path, mutate)
    overrides = figures._load_receptor_sampling_overrides(path)
    assert overrides["nanwan"] == (True, 10000.0)
    assert overrides["gongliao"] == (False, 10000.0)
    plain = figures._load_receptor_sampling_overrides(FORMAL_CONFIG)
    assert all(value == (False, None) for value in plain.values())


# ---------------------------------------------------------------- 到達時刻選取接線
def _series_3h():
    from datetime import UTC, datetime, timedelta

    start = datetime(2024, 1, 1, tzinfo=UTC)
    count = int((datetime(2026, 1, 1, tzinfo=UTC) - start) / timedelta(hours=3))
    time_ns = np.asarray([int((start + i * timedelta(hours=3)).timestamp() * 1e9) for i in range(count)], dtype=np.int64)
    hours = np.arange(count, dtype=float) * 3.0
    elev = (1.0 + 0.5 * np.sin(2 * np.pi * hours / (24 * 14))) * np.sin(2 * np.pi * hours / 12.42)
    wave = np.ones(count)
    speed = np.ones(count)
    wave[1000] = 8.0
    speed[2000] = 3.0
    return time_ns, elev, wave, speed


def test_selector_with_excluded_candidates_never_selects_them_and_recomputes_median() -> None:
    from lagrangian_backtracking.arrival_times import (
        ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V2,
        select_arrival_times,
    )

    time_ns, elev, wave, speed = _series_3h()

    def run(valid: np.ndarray):
        return select_arrival_times(
            study_site_id="gongliao", time_utc_ns=time_ns, elevation_m=elev,
            significant_wave_height_m=wave, current_speed_mps=speed, valid_forcing=valid,
            backward_window_available=np.ones(time_ns.size, dtype=bool),
            design_version="design_baseline_v3_non_rising_a_v3_local20_observation_2025_20260916",
            observation_years=[2025], replicates=2,
            selection_policy=ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V2,
        )

    base = run(np.ones(time_ns.size, dtype=bool))
    removed = expand_exclusion_times([r.time_utc_ns for r in base], 1)
    valid = ~np.isin(time_ns, np.asarray(removed, dtype=np.int64))
    again = run(valid)
    assert len(again) == 50
    assert not ({r.time_utc_ns for r in again} & set(removed))
    # 每季的 spring/neap 仍各 6 筆：門檻（季內中位數）是在排除後的候選集合上重算的
    from collections import Counter

    counts = Counter((r.season, r.tide_class) for r in again if r.tide_class in {"spring_proxy", "neap_proxy"})
    assert set(counts.values()) == {6}


def test_select_arrivals_for_site_applies_exclusion_to_valid_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    import lagrangian_backtracking.input_derivation as derivation

    captured: dict = {}
    monkeypatch.setattr(derivation, "select_arrival_times", lambda **kw: captured.update(kw) or [])
    times = T0 + np.arange(100, dtype=np.int64) * HOUR
    series = {int(t): 0.1 for t in times}
    product = SimpleNamespace(canonical=SimpleNamespace(time_utc_ns=times))

    def call(**extra):
        derivation._select_arrivals_for_site(
            site_id="x", product=product, nww_series=series, ocm_elevation=series, ocm_speed=series,
            max_backtrack_days=1.0, design_version="d", expected_axis=times, strict=True,
            allow_gap_censored_anchor=True, **extra,
        )
        return captured["valid_forcing"].copy()

    assert call().all()  # 預設：與 run08 相同，沒有任何額外排除
    assert call(excluded_times_ns=()).all()
    valid = call(excluded_times_ns=[int(times[5]), int(times[6])])
    assert not valid[5] and not valid[6] and valid.sum() == 98


# ---------------------------------------------------------------- 壞時刻規則（失敗矩陣 → 排除清單）
from lagrangian_backtracking import input_derivation as idv_module  # noqa: E402
from lagrangian_backtracking.bad_hour_rule import (  # noqa: E402
    BAD_HOUR_RULE_POLICY_ID,
    BadHourRuleParameters,
    derivation_audit,
    derive_bad_hours_from_failure_matrix,
    failure_matrix_fingerprint,
    hours_sha256,
)


def _toy_matrix():
    # 6 面 × 10 整點。t2：面 0,1,2 失敗；t5：面 1；t7：面 5；面 3 失敗 8 次（不屬稀少失敗）；面 4 從不失敗。
    failed = np.zeros((6, 10), dtype=bool)
    failed[0, 2] = failed[1, 2] = failed[2, 2] = True
    failed[1, 5] = True
    failed[5, 7] = True
    failed[3, :8] = True
    area = np.array([0.6e6, 0.5e6, 0.7e6, 5.0e6, 0.4e6, 0.2e6])
    times = [T0 + i * HOUR for i in range(10)]
    return failed, area, times


def test_bad_hour_rule_excludes_only_hours_with_large_rare_failure_area() -> None:
    failed, area, times = _toy_matrix()
    params = BadHourRuleParameters(radius_m=10_000.0)
    out = derive_bad_hours_from_failure_matrix(failed, area, times, params)
    # t2 牽連面積 0.6+0.5+0.7=1.8 km2 >= 1；t5 0.5、t7 0.2 不足；面 3（失敗 8 次）不計入牽連面積
    assert out.excluded_time_ns == (times[2],)
    assert out.affected_area_km2[2] == pytest.approx(1.8)
    assert out.affected_area_km2[5] == pytest.approx(0.5) and out.affected_area_km2[7] == pytest.approx(0.2)
    assert out.affected_area_km2[0] == 0.0 and out.affected_face_count[2] == 3
    assert out.rare_failure_face_count == 4 and out.rare_failure_area_km2 == pytest.approx(1.8 + 0.2 - 0.0 + 0.0, abs=1.0)
    # 排除後：面 0、2 只在 t2 失敗而恢復；面 1 仍在 t5 失敗；面 3、5 仍失敗；面 4 本來就有效
    assert out.faces_never_failing == 1 and out.faces_valid_after_exclusion == 3
    assert out.area_valid_before_km2 == pytest.approx(0.4)
    assert out.area_valid_after_km2 == pytest.approx(0.6 + 0.7 + 0.4)


def test_bad_hour_rule_is_deterministic_and_chunk_independent() -> None:
    rng = np.random.default_rng(7)
    faces, hours = 60, 1500  # 超過內部 512 整點分塊
    failed = rng.random((faces, hours)) < 0.002
    area = rng.uniform(0.05e6, 0.4e6, faces)
    times = [T0 + i * HOUR for i in range(hours)]
    params = BadHourRuleParameters(radius_m=10_000.0, area_threshold_km2=0.3)
    a = derive_bad_hours_from_failure_matrix(failed, area, times, params)
    b = derive_bad_hours_from_failure_matrix(failed.copy(), area.copy(), list(times), params)
    assert a.excluded_time_ns == b.excluded_time_ns and len(a.excluded_time_ns) > 0
    counts = failed.sum(axis=1)
    rare = (counts >= 1) & (counts <= 5)
    expected = (area[rare][:, None] * failed[rare]).sum(axis=0) >= 0.3e6
    assert list(a.excluded_time_ns) == [times[i] for i in np.flatnonzero(expected)]


def test_bad_hour_rule_parameters_and_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        BadHourRuleParameters(radius_m=0.0)
    with pytest.raises(ValueError):
        BadHourRuleParameters(radius_m=1.0, min_failed_hours=0)
    with pytest.raises(ValueError):
        BadHourRuleParameters(radius_m=1.0, min_failed_hours=3, max_failed_hours=2)
    with pytest.raises(ValueError):
        BadHourRuleParameters(radius_m=1.0, area_threshold_km2=float("nan"))
    failed, area, times = _toy_matrix()
    params = BadHourRuleParameters(radius_m=1.0)
    with pytest.raises(ValueError, match="嚴格遞增"):
        derive_bad_hours_from_failure_matrix(failed, area, times[::-1], params)
    with pytest.raises(ValueError, match="整點"):
        derive_bad_hours_from_failure_matrix(failed, area, [t + 1 for t in times], params)
    with pytest.raises(ValueError, match="bool"):
        derive_bad_hours_from_failure_matrix(failed.astype(int), area, times, params)


def test_bad_hour_audit_hash_matches_restrict_and_fingerprint_changes() -> None:
    failed, area, times = _toy_matrix()
    params = BadHourRuleParameters(radius_m=10_000.0)
    out = derive_bad_hours_from_failure_matrix(failed, area, times, params)
    audit = derivation_audit(out, params, face_ids=list(range(6)), failed=failed)
    assert audit["policy_id"] == BAD_HOUR_RULE_POLICY_ID
    assert audit["excluded_hours_utc_ns"] == [times[2]]
    assert audit["excluded_hours_sha256"] == hours_sha256([times[2]])
    _a, prov = restrict_available_times_to_supported(
        {"s": times}, {"s": times}, extra_excluded_time_ns_by_site={"s": [times[2]]}
    )
    assert prov["external_exclusion"]["by_site"]["s"]["excluded_hours_sha256"] == audit["excluded_hours_sha256"]
    flipped = failed.copy()
    flipped[4, 9] = True
    assert failure_matrix_fingerprint(flipped, range(6), times) != audit["failure_matrix_sha256"]


def _bed_rule_config(tmp_path: Path, rule: dict | None, **extra):
    def mod(p: dict) -> None:
        bed = p["scenarios"]["bed_residence_time"]
        if rule is not None:
            bed["deposition_bad_hour_rule"] = rule
        bed.update(extra)

    return load_config(_write_variant(tmp_path, mod))


def test_bad_hour_rule_config_default_off_hash_stable_and_exclusive(tmp_path: Path) -> None:
    base = load_config(FORMAL_CONFIG)
    assert base.scenarios.bed_residence_time.deposition_bad_hour_rule is None
    assert "deposition_bad_hour_rule" not in base.normalized_payload()["scenarios"]["bed_residence_time"]
    cfg = _bed_rule_config(tmp_path, {"radius_m": 10000.0})
    rule = cfg.scenarios.bed_residence_time.deposition_bad_hour_rule
    assert rule.to_parameters() == BadHourRuleParameters(radius_m=10000.0)
    assert cfg.normalized_payload()["scenarios"]["bed_residence_time"]["deposition_bad_hour_rule"]["radius_m"] == 10000.0
    with pytest.raises(Exception):
        _bed_rule_config(tmp_path, {"radius_m": 10000.0}, deposition_exclusion_hours_file="x.json")
    with pytest.raises(Exception):
        _bed_rule_config(tmp_path, {"radius_m": 10000.0, "max_failed_hours": 0})
    with pytest.raises(Exception):
        _bed_rule_config(tmp_path, {"min_failed_hours": 1})  # radius_m 必填


def test_derived_rule_provenance_validator(tmp_path: Path) -> None:
    cfg = _bed_rule_config(tmp_path, {"radius_m": 10000.0})
    bed = cfg.scenarios.bed_residence_time
    failed, area, times = _toy_matrix()
    params = bed.deposition_bad_hour_rule.to_parameters()
    out = derive_bad_hours_from_failure_matrix(failed, area, times, params)
    audit = derivation_audit(out, params, face_ids=list(range(6)), failed=failed)
    source = {
        "source": "derived_rule",
        "policy_id": BAD_HOUR_RULE_POLICY_ID,
        "parameters": params.to_dict(),
        "by_site_audit": {"s": {k: v for k, v in audit.items() if k != "excluded_hours_utc_ns"}},
    }
    _a, prov = restrict_available_times_to_supported(
        {"s": times}, {"s": times},
        extra_excluded_time_ns_by_site={"s": list(out.excluded_time_ns)}, extra_exclusion_source=source,
    )
    excluded = {s: frozenset(r["excluded_hours_utc_ns"]) for s, r in prov["by_site"].items()}
    _validate_external_exclusion_provenance(prov, excluded, bed=bed)
    # 設定啟用規則卻沒有 external_exclusion 區塊
    plain = {k: v for k, v in prov.items() if k != "external_exclusion"}
    with pytest.raises(ValueError, match="沒有 external_exclusion"):
        _validate_external_exclusion_provenance(plain, excluded, bed=bed)
    # 參數被改動
    bad = copy.deepcopy(prov)
    bad["external_exclusion"]["source"]["parameters"]["area_threshold_km2"] = 2.0
    with pytest.raises(ValueError, match="參數"):
        _validate_external_exclusion_provenance(bad, excluded, bed=bed)
    # 稽核的雜湊與清單不一致
    bad = copy.deepcopy(prov)
    bad["external_exclusion"]["source"]["by_site_audit"]["s"]["excluded_hours_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="稽核"):
        _validate_external_exclusion_provenance(bad, excluded, bed=bed)
    # 設定沒有規則，provenance 卻宣稱 derived_rule
    with pytest.raises(ValueError, match="沒有 deposition_bad_hour_rule"):
        _validate_external_exclusion_provenance(prov, excluded, bed=load_config(FORMAL_CONFIG).scenarios.bed_residence_time)


def _worker_context(zcor_hours: np.ndarray, design_version: str):
    mesh = SimpleNamespace(
        face_node_count=np.array([4]),
        face_nodes_local=np.array([[0, 1, 2, 3]]),
        source_face_global_index=np.array([17]),
    )
    month = idv_module._OCMMonthPairArrays(
        zcor=zcor_hours, elev=np.zeros(zcor_hours.shape[:2]), wetdry=np.zeros((zcor_hours.shape[0], 1))
    )
    rows = np.arange(zcor_hours.shape[0], dtype=np.int64)
    return dict(
        mesh=mesh,
        depth=np.full(4, 20.0),
        hour_count=zcor_hours.shape[0],
        design_version=design_version,
        master_seed=20260917,
        site_id="s",
        groups=[(month, rows, rows.copy(), "202501")],
    )


@pytest.mark.parametrize("design_version", ["legacy-test-v1", None])
def test_bad_hour_ocm_worker_flags_hours_where_gate_would_fail(design_version) -> None:
    column = np.array([-20.0, -13.333333, -6.666667, 0.0])
    zcor = np.tile(column, (3, 4, 1))  # (hour, node, layer)
    zcor[1, 2, :] = np.nan  # 第 1 個整點某 node 沒有有限 zcor：閘門會失敗
    dv = design_version or idv_module.CURRENT_DESIGN_VERSION
    idv_module._BAD_HOUR_CTX.clear()
    idv_module._BAD_HOUR_CTX.update(_worker_context(zcor, dv))
    try:
        flags = idv_module._bad_hour_ocm_worker(0)
    finally:
        idv_module._BAD_HOUR_CTX.clear()
    assert flags.tolist() == [False, True, False]


def test_bad_hour_group_sources_fail_closed_on_missing_hour_and_sparse_month(tmp_path: Path) -> None:
    month = SimpleNamespace(label="202501")
    values = idv_module._OCMMonthPairArrays(zcor=np.zeros((2, 1, 1)), elev=np.zeros((2, 1)), wetdry=np.zeros((2, 1)))
    cache = idv_module._OCMPairCache(
        flow_domain_id="f", utc_prefer_last_index={T0: (month, 0), T0 + HOUR: (month, 1)},
        depth=np.zeros(1), monthly_arrays={"202501": values},
    )
    groups = idv_module._bad_hour_group_sources(cache, np.array([T0, T0 + HOUR], dtype=np.int64), site_id="s")
    assert groups[0][3] == "202501" and groups[0][1].tolist() == [0, 1]
    with pytest.raises(InputDerivationError, match="不在 OCM source time"):
        idv_module._bad_hour_group_sources(cache, np.array([T0 + 2 * HOUR], dtype=np.int64), site_id="s")


def test_bad_hour_workers_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LBT_BAD_HOUR_WORKERS", "3")
    assert idv_module._bad_hour_workers() == 3
    monkeypatch.setenv("LBT_BAD_HOUR_WORKERS", "0")
    with pytest.raises(InputDerivationError):
        idv_module._bad_hour_workers()
    monkeypatch.delenv("LBT_BAD_HOUR_WORKERS")
    assert idv_module._bad_hour_workers() >= 1


# ---------------------------------------------------------------- 方位扇區分配（龜山島：西半 3、東半 2）
def test_azimuth_sectors_split_west_three_east_two_with_equal_area_bins() -> None:
    from lagrangian_backtracking.receptor_stratified import validate_azimuth_sectors

    cand = _ring_candidates(n=5000)
    sectors = ((0.0, 180.0, 2), (180.0, 360.0, 3))
    res = sample_azimuth_distance_latin(cand, seed=11, n=5, azimuth_sectors=sectors)
    edges = res.azimuth_edges_deg
    assert len(edges) == 6 and edges[0] == 0.0 and edges[2] == 180.0 and edges[5] == 360.0
    assert list(edges) == sorted(edges) and res.start_azimuth_deg == 0.0
    # 前兩個方位區間在東半、後三個在西半
    for pick in res.picks:
        az = pick["azimuth_deg"]
        assert (az < 180.0) == (pick["azimuth_bin"] <= 2)
    assert sum(1 for p in res.picks if p["azimuth_deg"] >= 180.0) == 3
    # 各區間的候選面積在其半區內大致相等（等面積方位切分）
    az = np.asarray(cand.azimuth_deg)
    area = np.asarray(cand.area_m2)
    for lo, hi, a, b in ((0, 180, 0, 2), (180, 360, 2, 5)):
        k = b - a
        for j in range(k):
            e0, e1 = edges[a + j], edges[a + j + 1]
            m = (az >= e0) & (az < e1)
            half = (az >= lo) & (az < hi)
            assert area[m].sum() == pytest.approx(area[half].sum() / k, rel=0.03)
    # 決定性
    again = sample_azimuth_distance_latin(cand, seed=11, n=5, azimuth_sectors=sectors)
    assert again.picks == res.picks and again.azimuth_edges_deg == edges
    # 不給扇區時行為不變（隨機起始方位）
    plain = sample_azimuth_distance_latin(cand, seed=11, n=5)
    assert plain.start_azimuth_deg != 0.0 and plain.azimuth_edges_deg[0] == pytest.approx(plain.azimuth_edges_deg[5])
    with pytest.raises(ValueError):
        validate_azimuth_sectors(((0.0, 180.0, 2), (181.0, 360.0, 3)), 5)
    with pytest.raises(ValueError):
        validate_azimuth_sectors(((0.0, 180.0, 2), (180.0, 360.0, 2)), 5)


def test_azimuth_allocation_config_and_manifest_record(tmp_path: Path) -> None:
    pool, anchor = _synthetic_pool()

    def mutate(p: dict) -> None:
        p["scenarios"]["horizontal_receptor_selection_policy"] = HORIZONTAL_RECEPTOR_SELECTION_POLICY_AZIMUTH_DISTANCE_LATIN
        p["scenarios"]["horizontal_receptor_sampling"] = {
            "radius_m": 10000,
            "azimuth_allocation_by_site": {"demo": {"west_180_360": 3, "east_0_180": 2}},
        }

    config = load_config(_write_variant(tmp_path, mutate))
    sampling = config.scenarios.horizontal_receptor_sampling
    assert sampling.azimuth_sectors_for("demo") == ((0.0, 180.0, 2), (180.0, 360.0, 3))
    assert sampling.azimuth_sectors_for("other") is None
    receptors, record = select_horizontal_receptors_azimuth_distance_latin(
        pool, anchor_lonlat=anchor, master_seed=20260916, design_version=config.design_version,
        radius_m=10_000.0, count=5,
        selection_policy_id=HORIZONTAL_RECEPTOR_SELECTION_POLICY_AZIMUTH_DISTANCE_LATIN,
        azimuth_sectors=sampling.azimuth_sectors_for("demo"),
    )
    assert record["azimuth_sectors"] == [
        {"from_deg": 0.0, "to_deg": 180.0, "bin_count": 2},
        {"from_deg": 180.0, "to_deg": 360.0, "bin_count": 3},
    ]
    assert sum(1 for p in record["picks"] if p["azimuth_deg"] >= 180.0) == 3
    selected_global = [r.source_face_global_index for r in receptors]
    _validate_latin_sampling_record("demo", {"stratified_sampling": record}, config, selected_global)
    # 設定有分配、稽核沒有（或反之）都要被拒絕
    missing = {k: v for k, v in record.items() if k != "azimuth_sectors"}
    with pytest.raises(ValueError, match="方位"):
        _validate_latin_sampling_record("demo", {"stratified_sampling": missing}, config, selected_global)
    with pytest.raises(ValueError, match="沒有方位分配"):
        _validate_latin_sampling_record("other", {"stratified_sampling": record}, config, selected_global)
    # 總和不等於 strata_count
    def bad(p: dict) -> None:
        mutate(p)
        p["scenarios"]["horizontal_receptor_sampling"]["azimuth_allocation_by_site"]["demo"]["east_0_180"] = 1

    with pytest.raises(Exception):
        load_config(_write_variant(tmp_path, bad))
