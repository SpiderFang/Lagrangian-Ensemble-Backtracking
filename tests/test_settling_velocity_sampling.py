"""WP-54：v7 沉降速度逐粒子均衡抽樣（``native_runner.settling_sampling``）的測試。

涵蓋：

- 全規模（5 站 × 10 類 × 20 受體 × 50 到達時刻 × 10 成員 = 500,000 顆）每站每類 10 個
  候選各恰 1,000 顆；
- 固定主種子可重現、與身分表列順序無關、不同主種子結果不同；
- 種子由 ``derive_member_seed`` 衍生並寫入中繼資料；
- 粒子數不是候選數整數倍時 fail-closed；
- 身分表層級：representative 模式原樣回傳、內容雜湊與 .npz 格式不變；balanced 模式
  帶逐粒子速度、.npz 往返與內容雜湊穩定。
"""

from __future__ import annotations

import numpy as np
import pytest

from lagrangian_backtracking.native_runner.particle_input import (
    ParticleIdentityTable,
    ParticleInputError,
    load_particle_identity_table_npz,
    save_particle_identity_table_npz,
)
from lagrangian_backtracking.native_runner.settling_sampling import (
    SETTLING_SAMPLING_STREAM_ID,
    apply_settling_velocity_sampling,
    assign_balanced_candidates,
)
from lagrangian_backtracking.scenarios import (
    BASELINE_BEHAVIORS,
    SETTLING_CANDIDATE_COUNT,
    SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED,
    SETTLING_VELOCITY_SAMPLING_REPRESENTATIVE,
    derive_member_seed,
)

SITES = ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang")


def _columns(
    *, sites=SITES, materials=None, receptors=20, arrivals=50, members=10
) -> dict[str, np.ndarray]:
    """依站 × 類別 × 受體 × 到達時刻 × 成員展開逐粒子欄位（不建構完整身分表）。"""

    materials = materials or [b.material_id for b in BASELINE_BEHAVIORS]
    site_arr, mat_arr, rec_arr, arr_arr, mem_arr = [], [], [], [], []
    for site in sites:
        for material in materials:
            for r in range(receptors):
                for a in range(arrivals):
                    site_arr.append(np.full(members, site))
                    mat_arr.append(np.full(members, material))
                    rec_arr.append(np.full(members, f"{site}_r{r:02d}"))
                    arr_arr.append(np.full(members, f"{site}_a{a:02d}"))
                    mem_arr.append(np.arange(members, dtype=np.int32))
    return {
        "study_site_id": np.concatenate(site_arr),
        "material_id": np.concatenate(mat_arr),
        "receptor_id": np.concatenate(rec_arr),
        "arrival_time_id": np.concatenate(arr_arr),
        "member_id": np.concatenate(mem_arr),
    }


def _candidates() -> dict[str, tuple[float, ...]]:
    return {b.material_id: b.settling_velocity_candidates_mps for b in BASELINE_BEHAVIORS}


def test_full_scale_each_site_class_gets_every_candidate_exactly_1000_times() -> None:
    cols = _columns()
    assert cols["study_site_id"].shape[0] == 500_000
    result = assign_balanced_candidates(**cols, candidates_by_material=_candidates(), master_seed=20260101)

    assert result.velocity_mps.shape == (500_000,)
    groups = 0
    for site in SITES:
        for behavior in BASELINE_BEHAVIORS:
            rows = (cols["study_site_id"] == site) & (cols["material_id"] == behavior.material_id)
            assert rows.sum() == 10_000
            counts = np.bincount(result.candidate_index[rows], minlength=SETTLING_CANDIDATE_COUNT)
            assert counts.tolist() == [1000] * SETTLING_CANDIDATE_COUNT
            # 指派的速度確實等於該類候選的對應值。
            expected = np.asarray(behavior.settling_velocity_candidates_mps)[result.candidate_index[rows]]
            assert np.array_equal(result.velocity_mps[rows], expected)
            groups += 1
    assert groups == 50
    assert len(result.metadata["groups"]) == 50
    assert all(g["candidate_counts"] == [1000] * 10 for g in result.metadata["groups"])


def test_assignment_is_reproducible_and_independent_of_row_order() -> None:
    cols = _columns(sites=("hsinchu", "nanwan"), receptors=4, arrivals=5, members=10)
    first = assign_balanced_candidates(**cols, candidates_by_material=_candidates(), master_seed=42)
    again = assign_balanced_candidates(**cols, candidates_by_material=_candidates(), master_seed=42)
    assert np.array_equal(first.candidate_index, again.candidate_index)
    assert first.metadata == again.metadata

    # 列順序打亂：同一顆粒子（同站、同類、同受體、同到達時刻、同成員）拿到同一個候選。
    rng = np.random.default_rng(3)
    perm = rng.permutation(cols["member_id"].shape[0])
    shuffled_cols = {k: v[perm] for k, v in cols.items()}
    shuffled = assign_balanced_candidates(
        **shuffled_cols, candidates_by_material=_candidates(), master_seed=42
    )
    assert np.array_equal(shuffled.candidate_index, first.candidate_index[perm])

    other_seed = assign_balanced_candidates(**cols, candidates_by_material=_candidates(), master_seed=43)
    assert not np.array_equal(other_seed.candidate_index, first.candidate_index)
    # 不同主種子仍然是均衡的。
    for group in other_seed.metadata["groups"]:
        assert set(group["candidate_counts"]) == {group["particle_count"] // 10}


def test_group_seed_is_derived_from_member_seed_system_and_recorded() -> None:
    cols = _columns(sites=("hsinchu",), materials=[BASELINE_BEHAVIORS[0].material_id], receptors=2, arrivals=5)
    result = assign_balanced_candidates(**cols, candidates_by_material=_candidates(), master_seed=777)
    (group,) = result.metadata["groups"]
    expected_seed = derive_member_seed(
        master_seed=777,
        scenario_id=f"{SETTLING_SAMPLING_STREAM_ID}|hsinchu|{BASELINE_BEHAVIORS[0].material_id}",
        experiment_case_id=SETTLING_SAMPLING_STREAM_ID,
        member_id=0,
    )
    assert group["seed_hex"] == f"{expected_seed:032x}"
    assert result.metadata["master_seed"] == 777
    assert result.metadata["stream_id"] == SETTLING_SAMPLING_STREAM_ID
    assert result.metadata["mode"] == SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED
    # 位置確實由該種子的 PCG64DXSM 排列決定：獨立重算一次排列並比對。
    generator = np.random.Generator(np.random.PCG64DXSM(expected_seed))
    count = group["particle_count"]
    chosen = (np.arange(count) % 10)[generator.permutation(count)]
    order = np.lexsort((cols["member_id"], cols["arrival_time_id"], cols["receptor_id"]))
    assert np.array_equal(result.candidate_index[order], chosen)


def test_non_multiple_group_size_is_rejected_unless_remainder_allowed() -> None:
    cols = _columns(sites=("hsinchu",), materials=[BASELINE_BEHAVIORS[0].material_id], receptors=3, arrivals=1, members=3)
    assert cols["member_id"].shape[0] == 9
    with pytest.raises(ParticleInputError, match="整數倍"):
        assign_balanced_candidates(**cols, candidates_by_material=_candidates(), master_seed=1)
    result = assign_balanced_candidates(
        **cols, candidates_by_material=_candidates(), master_seed=1, require_exact_balance=False
    )
    counts = np.bincount(result.candidate_index, minlength=10)
    assert counts.max() - counts.min() <= 1


def test_invalid_candidate_sets_are_rejected() -> None:
    cols = _columns(sites=("hsinchu",), materials=[BASELINE_BEHAVIORS[0].material_id], receptors=2, arrivals=5)
    material = BASELINE_BEHAVIORS[0].material_id
    good = list(BASELINE_BEHAVIORS[0].settling_velocity_candidates_mps)
    for bad, pattern in (
        (good[:9], "恰為"),
        ([-v for v in good], "嚴格小於 0"),
        (good[::-1], "由小到大"),
    ):
        with pytest.raises(ParticleInputError, match=pattern):
            assign_balanced_candidates(**cols, candidates_by_material={material: bad}, master_seed=1)


# ══════════════════════════════════════════════════════════════════════════
# 身分表層級
# ══════════════════════════════════════════════════════════════════════════


def _identity_table(*, receptors: int = 2, arrivals: int = 5, members: int = 10) -> ParticleIdentityTable:
    behaviors = BASELINE_BEHAVIORS[:3]
    cols = _columns(
        sites=("hsinchu",), materials=[b.material_id for b in behaviors],
        receptors=receptors, arrivals=arrivals, members=members,
    )
    n = cols["member_id"].shape[0]
    velocity_by_material = {b.material_id: b.settling_velocity_mps for b in behaviors}
    velocity = np.array([velocity_by_material[m] for m in cols["material_id"]], dtype=np.float64)
    rng = np.random.default_rng(1)
    particle_id = np.array([f"p{i:07d}" for i in range(n)])
    return ParticleIdentityTable(
        particle_id=particle_id,
        scenario_id=np.array([f"s{i:07d}" for i in range(n)]),
        member_id=cols["member_id"],
        study_site_id=cols["study_site_id"],
        analysis_region_id=np.full(n, "B"),
        receptor_id=cols["receptor_id"],
        material_id=cols["material_id"],
        arrival_time_id=cols["arrival_time_id"],
        settling_velocity_mps=velocity,
        arrival_time_utc_ns=np.full(n, 1_700_000_000_000_000_000, dtype=np.int64),
        seed_hex=np.array([f"{i:032x}" for i in range(n)]),
        x0_m=rng.uniform(0, 1000, n),
        y0_m=rng.uniform(0, 1000, n),
        z0_m=rng.uniform(-10, -1, n),
        experiment_case_id="no_stokes",
        master_seed=20260101,
        random_stream_id=None,
    )


def test_representative_mode_returns_same_table_and_keeps_hash_and_npz_format(tmp_path) -> None:
    table = _identity_table()
    same = apply_settling_velocity_sampling(
        table, BASELINE_BEHAVIORS, mode=SETTLING_VELOCITY_SAMPLING_REPRESENTATIVE
    )
    assert same is table
    assert table.particle_settling_velocity_mps is None
    path = tmp_path / "rep.npz"
    save_particle_identity_table_npz(table, path)
    with np.load(path, allow_pickle=False) as npz:
        assert "particle_settling_velocity_mps" not in npz.files
        assert "settling_candidate_index" not in npz.files
    loaded = load_particle_identity_table_npz(path)
    assert loaded.content_sha256 == table.content_sha256
    assert loaded.particle_settling_velocity_mps is None


def test_balanced_table_has_per_particle_speed_and_roundtrips(tmp_path) -> None:
    table = _identity_table()
    balanced = apply_settling_velocity_sampling(
        table, BASELINE_BEHAVIORS, mode=SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED
    )
    assert balanced is not table
    assert balanced.content_sha256 != table.content_sha256
    # 類別代表速度欄位原封不動（規劃、分組與後處理仍依類別）。
    assert np.array_equal(balanced.settling_velocity_mps, table.settling_velocity_mps)
    assert balanced.particle_settling_velocity_mps.dtype == np.float64
    assert balanced.settling_candidate_index.dtype == np.int16
    for behavior in BASELINE_BEHAVIORS[:3]:
        rows = balanced.material_id == behavior.material_id
        counts = np.bincount(balanced.settling_candidate_index[rows], minlength=10)
        assert counts.tolist() == [10] * 10  # 100 顆 / 10 個候選

    path = tmp_path / "balanced.npz"
    save_particle_identity_table_npz(balanced, path)
    loaded = load_particle_identity_table_npz(path)
    assert loaded.content_sha256 == balanced.content_sha256
    assert np.array_equal(loaded.particle_settling_velocity_mps, balanced.particle_settling_velocity_mps)
    assert np.array_equal(loaded.settling_candidate_index, balanced.settling_candidate_index)
    assert loaded.settling_sampling == balanced.settling_sampling

    again = apply_settling_velocity_sampling(
        table, BASELINE_BEHAVIORS, mode=SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED
    )
    assert again.content_sha256 == balanced.content_sha256


def test_balanced_table_rejects_double_sampling_and_inconsistent_class_speed() -> None:
    table = _identity_table()
    balanced = apply_settling_velocity_sampling(
        table, BASELINE_BEHAVIORS, mode=SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED
    )
    with pytest.raises(ParticleInputError, match="重複抽樣"):
        apply_settling_velocity_sampling(
            balanced, BASELINE_BEHAVIORS, mode=SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED
        )
    from dataclasses import replace

    wrong = tuple(
        replace(b, settling_velocity_mps=b.settling_velocity_mps * 1.01) for b in BASELINE_BEHAVIORS
    )
    with pytest.raises(ParticleInputError, match="不一致"):
        apply_settling_velocity_sampling(
            table, wrong, mode=SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED
        )
    with pytest.raises(ValueError, match="抽樣模式"):
        apply_settling_velocity_sampling(table, BASELINE_BEHAVIORS, mode="bogus")


def test_table_requires_all_three_per_particle_fields_together() -> None:
    table = _identity_table()
    from dataclasses import replace

    with pytest.raises(ParticleInputError, match="同時提供"):
        replace(table, particle_settling_velocity_mps=table.settling_velocity_mps.copy())


def test_trial_subset_keeps_per_particle_fields_aligned_with_rows() -> None:
    """cli._subset_identity_table（WP-38 試產子集）必須同步取逐粒子速度與候選序號，並保留抽樣中繼資料。"""

    from lagrangian_backtracking.cli import _subset_identity_table

    table = apply_settling_velocity_sampling(
        _identity_table(), BASELINE_BEHAVIORS, mode=SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED
    )
    keep = np.arange(0, len(table), 7, dtype=np.int64)
    sub = _subset_identity_table(table, keep)
    assert len(sub) == keep.size
    assert np.array_equal(sub.particle_settling_velocity_mps, table.particle_settling_velocity_mps[keep])
    assert np.array_equal(sub.settling_candidate_index, table.settling_candidate_index[keep])
    assert sub.settling_sampling == table.settling_sampling
