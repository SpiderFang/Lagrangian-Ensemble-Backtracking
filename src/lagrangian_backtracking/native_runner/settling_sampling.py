"""v7 沉降速度逐粒子抽樣：同站同類別的 10 個候選速度均衡分配給粒子（WP-54）。

設計
----
每個類別（``Behavior``）有 10 個候選沉降速度（五個代表物件各兩個阻力係數界，
由 ``data/settling_velocity_derivation/results.csv`` 推導，見 ``scenarios.
BASELINE_BEHAVIORS``）。``class_candidates_balanced`` 模式下，**同一站、同一類別**
的所有粒子（正式設計為 20 受體 × 50 到達時刻 × M=10 成員 = 10,000 顆）把 10 個
候選值各分配到**完全相同的數量**（每個候選 1,000 顆），哪一顆粒子拿到哪一個
候選，由一個可重現的種子化隨機排列決定。

為什麼用「均衡排列」而不是獨立抽樣
------------------------------------
獨立抽樣（每顆粒子各自從 10 個候選隨機選一個）會使每個候選的粒子數在 1,000
附近有二項分布的波動（標準差約 30 顆），而這個波動會直接進入各候選速度的
來源足跡估計，與沉降速度本身造成的差異混在一起。均衡排列把波動降為零，
讓「候選速度之間的差異」只來自物理而不來自抽樣份額。

可重現性與跨案例一致
--------------------
- 種子由既有種子系統 ``scenarios.derive_member_seed`` 衍生，輸入為主種子、固定
  的抽樣串流識別碼與 ``站|類別``；**不含** ``experiment_case_id``，因此
  ``no_stokes`` 與 ``finite_depth_stokes`` 兩個物理案例對同一顆粒子（同站、同
  類別、同受體、同到達時刻識別碼、同成員）指派完全相同的候選速度，兩案例的
  差異只來自 Stokes 漂移而不來自速度指派。
- 組內粒子先依 ``(receptor_id, arrival_time_id, member_id)`` 排成與身分表列順序
  無關的標準順序，再套用排列，所以身分表列順序改變不會改變指派結果。
- 每組的種子、粒子數與各候選實際計數都寫進回傳的中繼資料，進入身分表的內容
  雜湊與 manifest。

本模組只決定「哪顆粒子用哪個速度」；速度如何進入位移由
``native_runner.settling_kernels`` 與 ``hour_major`` 負責。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..scenarios import (
    SETTLING_CANDIDATE_COUNT,
    SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED,
    SETTLING_VELOCITY_SAMPLING_REPRESENTATIVE,
    Behavior,
    derive_member_seed,
    validate_settling_velocity_sampling_mode,
)
from .particle_input import ParticleIdentityTable, ParticleInputError

__all__ = [
    "SETTLING_SAMPLING_STREAM_ID",
    "BalancedAssignment",
    "assign_balanced_candidates",
    "apply_settling_velocity_sampling",
]

SETTLING_SAMPLING_STREAM_ID = "settling_velocity_sampling_class_candidates_balanced_v1"
"""抽樣串流識別碼，取代 ``derive_member_seed`` 的 ``experiment_case_id`` 輸入。"""


@dataclass(frozen=True, slots=True)
class BalancedAssignment:
    """均衡抽樣結果：逐粒子速度、候選序號與可稽核中繼資料。"""

    velocity_mps: np.ndarray
    candidate_index: np.ndarray
    metadata: dict[str, Any]


def assign_balanced_candidates(
    *,
    study_site_id: np.ndarray,
    material_id: np.ndarray,
    receptor_id: np.ndarray,
    arrival_time_id: np.ndarray,
    member_id: np.ndarray,
    candidates_by_material: Mapping[str, Sequence[float]],
    master_seed: int,
    require_exact_balance: bool = True,
) -> BalancedAssignment:
    """依站與類別，把各類別的候選速度均衡、可重現地分配給粒子。

    Args:
        study_site_id／material_id／receptor_id／arrival_time_id／member_id:
            長度 N 的逐粒子欄位（與身分表同名欄位相同語意）。
        candidates_by_material: ``{material_id: 候選速度序列}``，每類恰
            ``SETTLING_CANDIDATE_COUNT`` 個，嚴格為負且量值由小到大排列。
        master_seed: 主種子（與身分表的 ``master_seed`` 相同）。
        require_exact_balance: 為真（預設，正式用途）時，每個（站, 類別）組的
            粒子數必須是候選數的整數倍，才能做到每個候選份數完全相同；為假時
            允許餘數，餘數粒子隨機落在不同候選上（份數最多差 1），僅供測試與
            非正式規模使用。

    Raises:
        ParticleInputError: 欄位長度不一致、材質缺少候選、候選不合法、或
            ``require_exact_balance`` 為真而某組粒子數不是候選數的整數倍。
    """

    n = int(study_site_id.shape[0])
    for name, array in (
        ("material_id", material_id), ("receptor_id", receptor_id),
        ("arrival_time_id", arrival_time_id), ("member_id", member_id),
    ):
        if array.shape != (n,):
            raise ParticleInputError(f"{name} 長度必須與 study_site_id 相同（{n}）")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ParticleInputError("master_seed 必須是非負整數")

    velocity = np.empty(n, dtype=np.float64)
    index = np.empty(n, dtype=np.int16)
    assigned = np.zeros(n, dtype=bool)
    groups_meta: list[dict[str, Any]] = []

    candidate_arrays: dict[str, np.ndarray] = {}
    for material, values in candidates_by_material.items():
        array = np.asarray(list(values), dtype=np.float64)
        if array.shape != (SETTLING_CANDIDATE_COUNT,):
            raise ParticleInputError(
                f"材質 {material} 的候選沉降速度必須恰為 {SETTLING_CANDIDATE_COUNT} 個，"
                f"實際 {array.shape[0] if array.ndim == 1 else array.shape}"
            )
        if not np.all(np.isfinite(array)) or np.any(array >= 0.0):
            raise ParticleInputError(f"材質 {material} 的候選沉降速度必須全部是嚴格小於 0 的有限值")
        if np.any(np.diff(np.abs(array)) < 0.0):
            raise ParticleInputError(f"材質 {material} 的候選沉降速度必須依量值由小到大排列")
        candidate_arrays[material] = array

    for site in sorted(set(study_site_id.tolist())):
        site_mask = study_site_id == site
        for material in sorted(set(material_id[site_mask].tolist())):
            if material not in candidate_arrays:
                raise ParticleInputError(f"材質 {material} 沒有候選沉降速度，無法做均衡抽樣")
            rows = np.flatnonzero(site_mask & (material_id == material))
            count = int(rows.shape[0])
            if require_exact_balance and count % SETTLING_CANDIDATE_COUNT != 0:
                raise ParticleInputError(
                    f"站 {site} 材質 {material} 的粒子數 {count} 不是候選數 "
                    f"{SETTLING_CANDIDATE_COUNT} 的整數倍，無法做到每個候選份數完全相同"
                )
            # 與身分表列順序無關的標準順序（lexsort 最後一個鍵為主鍵）。
            order = np.lexsort((member_id[rows], arrival_time_id[rows], receptor_id[rows]))
            canonical_rows = rows[order]
            seed = derive_member_seed(
                master_seed=master_seed,
                scenario_id=f"{SETTLING_SAMPLING_STREAM_ID}|{site}|{material}",
                experiment_case_id=SETTLING_SAMPLING_STREAM_ID,
                member_id=0,
            )
            generator = np.random.Generator(np.random.PCG64DXSM(seed))
            slots = np.arange(count, dtype=np.int64) % SETTLING_CANDIDATE_COUNT
            chosen = slots[generator.permutation(count)]
            index[canonical_rows] = chosen.astype(np.int16)
            velocity[canonical_rows] = candidate_arrays[material][chosen]
            assigned[canonical_rows] = True
            counts = np.bincount(chosen, minlength=SETTLING_CANDIDATE_COUNT)
            groups_meta.append(
                {
                    "study_site_id": str(site),
                    "material_id": str(material),
                    "seed_hex": f"{seed:032x}",
                    "particle_count": count,
                    "candidate_counts": [int(value) for value in counts],
                }
            )

    if not bool(np.all(assigned)):
        raise ParticleInputError("內部錯誤：有粒子沒有被指派到任何候選速度")

    metadata = {
        "mode": SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED,
        "stream_id": SETTLING_SAMPLING_STREAM_ID,
        "master_seed": int(master_seed),
        "candidate_count": SETTLING_CANDIDATE_COUNT,
        "exact_balance_required": bool(require_exact_balance),
        "candidates_mps_by_material": {
            material: [float(value) for value in candidate_arrays[material]]
            for material in sorted(candidate_arrays)
        },
        "groups": groups_meta,
    }
    return BalancedAssignment(velocity_mps=velocity, candidate_index=index, metadata=metadata)


def apply_settling_velocity_sampling(
    table: ParticleIdentityTable,
    behaviors: Sequence[Behavior],
    *,
    mode: str,
    require_exact_balance: bool = True,
) -> ParticleIdentityTable:
    """依抽樣模式回傳（可能帶逐粒子沉降速度的）身分表。

    ``representative``：原樣回傳 ``table``（同一個物件），行為與 WP-54 之前完全
    相同。``class_candidates_balanced``：回傳新表，除新增三個選用欄位外與原表
    相同；新表的 ``content_sha256`` 因此改變（包含逐粒子速度與抽樣中繼資料）。

    會核對每個粒子的類別代表速度（``table.settling_velocity_mps``）等於其材質的
    ``Behavior.settling_velocity_mps``，並要求代表值介於該類候選的最小與最大
    量值之間，避免 config、material manifest 與身分表三者對不上而靜默抽樣。
    """

    validate_settling_velocity_sampling_mode(mode)
    if mode == SETTLING_VELOCITY_SAMPLING_REPRESENTATIVE:
        return table
    if table.particle_settling_velocity_mps is not None:
        raise ParticleInputError("身分表已經帶有逐粒子沉降速度，不可重複抽樣")

    by_material = {behavior.material_id: behavior for behavior in behaviors}
    if len(by_material) != len(behaviors):
        raise ParticleInputError("behaviors 的 material_id 不可重複")
    for material in sorted(set(table.material_id.tolist())):
        behavior = by_material.get(material)
        if behavior is None:
            raise ParticleInputError(f"身分表的材質 {material} 不在 behaviors 中")
        rows = table.material_id == material
        if not bool(np.all(table.settling_velocity_mps[rows] == behavior.settling_velocity_mps)):
            raise ParticleInputError(
                f"材質 {material} 的身分表類別速度與 Behavior.settling_velocity_mps 不一致"
            )

    assignment = assign_balanced_candidates(
        study_site_id=table.study_site_id,
        material_id=table.material_id,
        receptor_id=table.receptor_id,
        arrival_time_id=table.arrival_time_id,
        member_id=table.member_id,
        candidates_by_material={
            behavior.material_id: behavior.settling_velocity_candidates_mps
            for behavior in behaviors
            if behavior.material_id in set(table.material_id.tolist())
        },
        master_seed=table.master_seed,
        require_exact_balance=require_exact_balance,
    )
    return replace(
        table,
        particle_settling_velocity_mps=assignment.velocity_mps,
        settling_candidate_index=assignment.candidate_index,
        settling_sampling=assignment.metadata,
    )
