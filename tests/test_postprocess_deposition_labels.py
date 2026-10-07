"""WP-51 B2：後處理依沉底時刻／觀測時刻標籤分組的測試（全部合成資料）。

涵蓋：
1. 分組正確性：各組粒子數、占比、穿越計數與以純 Python ``Counter`` 獨立算出的期望值一致
   （合併表與逐站表），事件旗標的 ``wave_unavailable`` 獨立成組。
2. 沉底時刻標籤與觀測時刻標籤是兩套獨立分組（故意讓兩者互相錯位）。
3. ``departures.parquet``／``local_domain_departures.parquet`` 附上全部標籤欄、manifest 摘要正確。
4. 沒有標籤的身分表（per_observation 既有流程）：不輸出任何新表／新欄，且既有輸出的 CSV 與
   WP-51 前位元相同（黃金雜湊以 WP-51 前的 postprocess 原始碼算出）。
5. 標籤隨 npz 往返與 CLI 試產子集保留；標籤欄名與既有欄位撞名時 fail closed。
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from test_native_postprocess import (
    _JAN_2024_UTC_NS,
    _JUL_2024_UTC_NS,
    _make_identity_table,
    _make_result,
)

from lagrangian_backtracking import cli
from lagrangian_backtracking.native_runner.backward_driver import (
    NATIVE_STATUS_DEPOSITED,
    NATIVE_STATUS_LEFT_INTEREST_REGION,
    NATIVE_STATUS_MAX_BACKTRACK,
    NATIVE_STATUS_STRANDED,
)
from lagrangian_backtracking.native_runner.particle_input import (
    ParticleIdentityTable,
    load_particle_identity_table_npz,
    save_particle_identity_table_npz,
)
from lagrangian_backtracking.native_runner.postprocess import (
    DEPARTURE_TABLE_SCHEMA,
    LOCAL_DOMAIN_DEPARTURE_TABLE_SCHEMA,
    PostprocessAccumulator,
    PostprocessConfig,
    PostprocessError,
)

_CROSS = NATIVE_STATUS_LEFT_INTEREST_REGION

# ── 合成的 24 顆粒子：站 A 16 顆、站 B 8 顆，分組刻意不平衡 ─────────────────

_N = 24
_SITES = ["A"] * 16 + ["B"] * 8
_TIDE_CLASS = (
    ["spring_proxy"] * 10 + ["neap_proxy"] * 6  # A
    + ["spring_proxy"] * 2 + ["neap_proxy"] * 6  # B
)
_TIDE_PHASE = (
    ["rising"] * 6 + ["falling"] * 6 + ["slack"] * 3 + ["unknown"] * 1  # A
    + ["rising"] * 2 + ["falling"] * 2 + ["slack"] * 4  # B
)
# 高浪旗標與 NWW 可用性：A 站最後 2 顆、B 站最後 1 顆 NWW 無有效值（旗標必為 False）
_HIGH_WAVE = [i in {0, 1, 5} for i in range(16)] + [i in {3} for i in range(8)]
_WAVE_AVAILABLE = [i < 14 for i in range(16)] + [i < 7 for i in range(8)]
_STRONG_CURRENT = [i % 5 == 0 for i in range(_N)]
_DEPOSITION_SEASON = ["JJA"] * 12 + ["SON"] * 4 + ["JJA"] * 3 + ["DJF"] * 5
# 觀測時刻標籤與沉底時刻標籤故意不同（模擬觀測在沉底之後數十天）
_OBS_SEASON = ["SON"] * 12 + ["DJF"] * 4 + ["SON"] * 3 + ["MAM"] * 5
_OBS_TIDE_CLASS = ["neap_proxy"] * 12 + ["spring_proxy"] * 4 + ["event"] * 2 + ["spring_proxy"] * 6
_OBS_PHASE = ["fastest_rising"] * 12 + ["slack_proxy"] * 4 + ["max_hs"] * 2 + ["slack_proxy"] * 6
_STATUS = (
    [_CROSS] * 9 + [NATIVE_STATUS_MAX_BACKTRACK] * 4 + [NATIVE_STATUS_DEPOSITED] * 3  # A
    + [_CROSS] * 3 + [NATIVE_STATUS_STRANDED] * 2 + [NATIVE_STATUS_MAX_BACKTRACK] * 3  # B
)


def _labelled_identity() -> ParticleIdentityTable:
    base = _make_identity_table(
        particle_ids=[f"p{i}" for i in range(_N)],
        study_site_id=_SITES,
        settling_velocity_mps=[-0.001] * _N,
        arrival_time_utc_ns=[_JUL_2024_UTC_NS] * _N,
    )
    labels = {
        "deposition_season": np.asarray(_DEPOSITION_SEASON, dtype="<U32"),
        "deposition_tide_class": np.asarray(_TIDE_CLASS, dtype="<U32"),
        "deposition_tide_phase": np.asarray(_TIDE_PHASE, dtype="<U32"),
        "deposition_high_wave": np.asarray(_HIGH_WAVE, dtype=np.bool_),
        "deposition_strong_current": np.asarray(_STRONG_CURRENT, dtype=np.bool_),
        "deposition_wave_available": np.asarray(_WAVE_AVAILABLE, dtype=np.bool_),
        "observation_season": np.asarray(_OBS_SEASON, dtype="<U32"),
        "observation_tide_class": np.asarray(_OBS_TIDE_CLASS, dtype="<U32"),
        "observation_phase_or_event": np.asarray(_OBS_PHASE, dtype="<U32"),
        "observation_time_utc_ns": np.full(_N, _JUL_2024_UTC_NS + 40 * 86_400 * 10**9, dtype=np.int64),
    }
    return dataclasses.replace(base, labels=labels)


def _run(identity: ParticleIdentityTable, out: Path, statuses: list[int]) -> object:
    accumulator = PostprocessAccumulator(identity, output_dir=out, config=PostprocessConfig())
    accumulator.add_batch("b0", _make_result(rows=list(range(len(statuses))), final_status=statuses))
    return accumulator.finalize()


def _expected_groups(values: list[str], statuses: list[int]) -> dict[str, tuple[int, int]]:
    """純 Python 獨立期望：``{組: (粒子數, 穿越數)}``。"""

    total: Counter[str] = Counter(values)
    crossed: Counter[str] = Counter(v for v, s in zip(values, statuses, strict=True) if s == _CROSS)
    return {group: (total[group], crossed[group]) for group in total}


def _high_wave_states() -> list[str]:
    states = []
    for flag, available in zip(_HIGH_WAVE, _WAVE_AVAILABLE, strict=True):
        if not available:
            states.append("wave_unavailable")
        else:
            states.append("high_wave" if flag else "not_high_wave")
    return states


_EXPECTED_COLUMNS = {
    "deposition_season": _DEPOSITION_SEASON,
    "deposition_tide_class": _TIDE_CLASS,
    "deposition_tide_phase": _TIDE_PHASE,
    "deposition_high_wave": _high_wave_states(),
    "deposition_strong_current": ["strong_current" if f else "not_strong_current" for f in _STRONG_CURRENT],
    "observation_season": _OBS_SEASON,
    "observation_tide_class": _OBS_TIDE_CLASS,
    "observation_phase_or_event": _OBS_PHASE,
}


@pytest.fixture(scope="module")
def labelled_output(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("labelled")
    manifest = _run(_labelled_identity(), out, _STATUS)
    return out, manifest


@pytest.mark.parametrize("column", sorted(_EXPECTED_COLUMNS))
def test_marginal_label_table_counts_and_fractions(labelled_output, column: str) -> None:
    out, _manifest = labelled_output
    table = pq.read_table(out / f"boundary_flux_by_{column}.parquet").to_pylist()
    expected = _expected_groups(_EXPECTED_COLUMNS[column], _STATUS)
    assert {row[column] for row in table} == set(expected)
    for row in table:
        total, crossed = expected[row[column]]
        assert row["total_particle_count"] == total
        assert row["particle_fraction_of_total"] == pytest.approx(total / _N)
        assert row["count_crossed"] == crossed
        assert row["fraction_crossed"] == pytest.approx(crossed / total)
    assert sum(row["total_particle_count"] for row in table) == _N
    assert sum(row["particle_fraction_of_total"] for row in table) == pytest.approx(1.0)
    # CSV 與 Parquet 同時存在
    assert (out / f"boundary_flux_by_{column}.csv").is_file()


@pytest.mark.parametrize("column", sorted(_EXPECTED_COLUMNS))
def test_by_site_label_table_counts_and_within_site_fractions(labelled_output, column: str) -> None:
    out, _manifest = labelled_output
    table = pq.read_table(out / f"boundary_flux_by_site_{column}.parquet").to_pylist()
    for site in ("A", "B"):
        idx = [i for i, s in enumerate(_SITES) if s == site]
        expected = _expected_groups(
            [_EXPECTED_COLUMNS[column][i] for i in idx], [_STATUS[i] for i in idx]
        )
        rows = {row[column]: row for row in table if row["study_site_id"] == site}
        assert set(rows) == set(expected)
        for group, (total, crossed) in expected.items():
            assert rows[group]["total_particle_count"] == total
            assert rows[group]["particle_fraction_within_site"] == pytest.approx(total / len(idx))
            assert rows[group]["count_crossed"] == crossed
        assert sum(r["particle_fraction_within_site"] for r in rows.values()) == pytest.approx(1.0)
    # 欄位順序：站、分組、總數、占比
    names = pq.read_table(out / f"boundary_flux_by_site_{column}.parquet").column_names
    assert names[:4] == ["study_site_id", column, "total_particle_count", "particle_fraction_within_site"]


def test_wave_unavailable_is_its_own_group_and_never_counted_as_calm(labelled_output) -> None:
    out, _manifest = labelled_output
    rows = {
        row["deposition_high_wave"]: row["total_particle_count"]
        for row in pq.read_table(out / "boundary_flux_by_deposition_high_wave.parquet").to_pylist()
    }
    # A 站最後 2 顆、B 站最後 1 顆 NWW 無值
    assert rows["wave_unavailable"] == 3
    assert rows["high_wave"] + rows["not_high_wave"] + rows["wave_unavailable"] == _N


def test_deposition_and_observation_groupings_are_independent(labelled_output) -> None:
    out, _manifest = labelled_output
    deposition = {
        row["deposition_season"]: row["total_particle_count"]
        for row in pq.read_table(out / "boundary_flux_by_deposition_season.parquet").to_pylist()
    }
    observation = {
        row["observation_season"]: row["total_particle_count"]
        for row in pq.read_table(out / "boundary_flux_by_observation_season.parquet").to_pylist()
    }
    assert deposition == {"JJA": 15, "SON": 4, "DJF": 5}
    assert observation == {"SON": 15, "DJF": 4, "MAM": 5}
    # 既有 boundary_flux_by_season 依 arrival_time_utc_ns（此處全為 7 月 -> JJA）計季節，不受標籤影響
    legacy = pq.read_table(out / "boundary_flux_by_season.parquet").to_pylist()
    assert [(row["season"], row["total_particle_count"]) for row in legacy] == [("JJA", _N)]
    legacy_columns = pq.read_table(out / "boundary_flux_by_season.parquet").column_names
    assert "particle_fraction_of_total" not in legacy_columns


def test_departure_tables_carry_label_columns(labelled_output) -> None:
    out, manifest = labelled_output
    table = pq.read_table(out / "departures.parquet")
    expected_label_columns = sorted(_labelled_identity().labels)
    assert table.column_names == [f.name for f in DEPARTURE_TABLE_SCHEMA] + expected_label_columns
    rows = table.to_pylist()
    by_row = {row["row_particle_index"]: row for row in rows}
    assert by_row[0]["deposition_tide_class"] == "spring_proxy"
    assert by_row[0]["deposition_high_wave"] is True
    assert by_row[15]["deposition_wave_available"] is False
    assert by_row[23]["observation_phase_or_event"] == "slack_proxy"
    assert table.schema.field("deposition_high_wave").type == "bool"
    assert table.schema.field("observation_time_utc_ns").type == "int64"
    # manifest 摘要：各組粒子數與占比、最小組
    summary = manifest.config["label_group_summary"]
    assert set(summary) == set(_EXPECTED_COLUMNS)
    tide = summary["deposition_tide_class"]
    assert tide["groups"]["spring_proxy"]["particle_count"] == 12
    assert tide["groups"]["spring_proxy"]["particle_fraction_of_total"] == pytest.approx(0.5)
    assert tide["smallest_group_particle_count"] == 12
    assert summary["deposition_season"]["smallest_group_particle_count"] == 4
    assert manifest.config["label_columns_in_departures"] == expected_label_columns
    # 新表納入 manifest 的 parquet 檢查碼
    assert "boundary_flux_by_deposition_tide_phase.parquet" in manifest.output_files_sha256


def test_local_domain_departures_carry_label_columns(tmp_path: Path) -> None:
    from lagrangian_backtracking.native_runner.backward_driver import LocalDomainFirstExitRecord

    identity = _labelled_identity()
    accumulator = PostprocessAccumulator(identity, output_dir=tmp_path, config=PostprocessConfig())
    records = [
        LocalDomainFirstExitRecord(
            row_particle_index=2, x_m=1.0, y_m=2.0, z_m=-3.0, utc_ns=_JUL_2024_UTC_NS - 3600
        )
    ]
    accumulator.add_batch(
        "b0",
        _make_result(
            rows=list(range(_N)),
            final_status=_STATUS,
            local_domain_departures=records,
        ),
    )
    accumulator.finalize()
    table = pq.read_table(tmp_path / "local_domain_departures.parquet")
    base_columns = [f.name for f in LOCAL_DOMAIN_DEPARTURE_TABLE_SCHEMA]
    assert table.column_names == base_columns + sorted(identity.labels)
    (row,) = table.to_pylist()
    assert row["row_particle_index"] == 2
    assert row["deposition_tide_phase"] == "rising"
    assert row["deposition_wave_available"] is True


# ── 沒有標籤：per_observation 既有輸出不變 ────────────────────────────────

_LEGACY_STATUS = [
    _CROSS, NATIVE_STATUS_MAX_BACKTRACK, _CROSS, NATIVE_STATUS_DEPOSITED,
    _CROSS, NATIVE_STATUS_STRANDED, _CROSS, _CROSS, NATIVE_STATUS_MAX_BACKTRACK,
    _CROSS, NATIVE_STATUS_DEPOSITED, _CROSS,
]


def _legacy_identity() -> ParticleIdentityTable:
    n = len(_LEGACY_STATUS)
    return _make_identity_table(
        particle_ids=[f"q{i}" for i in range(n)],
        study_site_id=["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
        settling_velocity_mps=[-0.001, -0.002, -0.001, -0.002] * 3,
        arrival_time_utc_ns=[_JAN_2024_UTC_NS, _JUL_2024_UTC_NS] * 6,
        z0_m=[-1.0 - i for i in range(n)],
        scenario_id=[f"s{i // 2}" for i in range(n)],
        member_id=[i % 2 for i in range(n)],
    )


def _run_legacy(out: Path) -> None:
    accumulator = PostprocessAccumulator(_legacy_identity(), output_dir=out, config=PostprocessConfig())
    rows = list(range(len(_LEGACY_STATUS)))
    accumulator.add_batch("b0", _make_result(rows=rows, final_status=_LEGACY_STATUS))
    accumulator.finalize()


def _csv_hashes(out: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.glob("*.csv"))}


# 以 WP-51 前（提交 2f3d871）的 postprocess 原始碼對 _run_legacy 算出的 CSV SHA-256。
_LEGACY_CSV_GOLDEN: dict[str, str] = {
    "boundary_flux_by_depth_quartile.csv": "a22d73612484ac5cc434c8fa1cd903ea333bba36527e527028f89e7b5f30019f",
    "boundary_flux_by_season.csv": "3f1089e19711e938170051655e8f05dae066c6a965b4b5d992da2537a7dfbe94",
    "boundary_flux_by_settling_velocity.csv": "c34fe8b6fa2003f667c2119e2ad4bae1c8312612562de9587ab06720209605ee",  # noqa: E501
    "boundary_flux_by_site.csv": "ba93267c911521f1b7999ef640826af5d4bfa83714b3b36fede738cf92958d1f",
    "boundary_summary_by_site.csv": "55301b13500681623e09e95ee0136c4a3c34b0daa3b3581f8da3573acaba46e7",
    "local_domain_first_exit_summary_by_site.csv": "2de9d87bd889a336086cc0a1b3a65182c7ab4d0dc2209c7c2eed761d6ee6a9c5",  # noqa: E501
}


def test_without_labels_no_new_tables_or_columns(tmp_path: Path) -> None:
    _run_legacy(tmp_path)
    names = {path.name for path in tmp_path.iterdir()}
    assert not [name for name in names if "deposition_" in name or "observation_" in name]
    assert not [name for name in names if name.startswith("boundary_flux_by_site_")]
    assert pq.read_table(tmp_path / "departures.parquet").schema.equals(DEPARTURE_TABLE_SCHEMA)
    assert pq.read_table(tmp_path / "local_domain_departures.parquet").schema.equals(
        LOCAL_DOMAIN_DEPARTURE_TABLE_SCHEMA
    )
    import json

    config = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["config"]
    assert "label_group_summary" not in config and "label_columns_in_departures" not in config


def test_per_observation_csv_outputs_are_bit_identical_to_pre_wp51(tmp_path: Path) -> None:
    _run_legacy(tmp_path)
    hashes = _csv_hashes(tmp_path)
    assert hashes, "舊路徑應該有 CSV 輸出"
    assert hashes == _LEGACY_CSV_GOLDEN


# ── 標籤的保存與保護 ─────────────────────────────────────────────────────


def test_labels_survive_npz_roundtrip_and_cli_subset(tmp_path: Path) -> None:
    identity = _labelled_identity()
    path = tmp_path / "identity.npz"
    save_particle_identity_table_npz(identity, path)
    loaded = load_particle_identity_table_npz(path)
    assert loaded.content_sha256 == identity.content_sha256
    assert loaded.labels is not None and set(loaded.labels) == set(identity.labels)
    for name, array in identity.labels.items():
        assert np.array_equal(loaded.labels[name], array)

    keep = np.asarray([1, 5, 20], dtype=np.int64)
    subset = cli._subset_identity_table(loaded, keep)
    assert subset.labels is not None
    for name, array in identity.labels.items():
        assert np.array_equal(subset.labels[name], array[keep])
    assert subset.content_sha256 != loaded.content_sha256


def test_label_name_colliding_with_departure_column_fails_closed(tmp_path: Path) -> None:
    base = _make_identity_table(
        particle_ids=["p0", "p1"],
        study_site_id=["A", "A"],
        settling_velocity_mps=[-0.001, -0.001],
        arrival_time_utc_ns=[_JAN_2024_UTC_NS] * 2,
    )
    clashing = dataclasses.replace(base, labels={"season": np.asarray(["DJF", "DJF"], dtype="<U32")})
    with pytest.raises(PostprocessError, match="同名"):
        PostprocessAccumulator(clashing, output_dir=tmp_path, config=PostprocessConfig())
