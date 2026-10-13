#!/usr/bin/env python3
"""核對 v7 身分表的逐粒子沉降速度：每個（站位, 類別）的 10 個候選各被指派恰好 1/10 的粒子。

用法：check_identity_table_v7.py IDENTITY_TABLE.npz [--expect-per-candidate N]

檢查項目（任何一項不符就以非零離開碼結束）：
  * 身分表帶有 particle_settling_velocity_mps／settling_candidate_index／settling_sampling；
  * settling_sampling.mode == class_candidates_balanced；
  * 每個（study_site_id, material_id）群內，10 個候選索引的粒子數完全相等（若給 --expect-per-candidate 則必須等於該值）；
  * 每顆粒子的實際速度恰等於其類別候選清單 [candidate_index]，且類別代表值（settling_velocity_mps）與 Behavior 一致。
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from lagrangian_backtracking.native_runner.particle_input import load_particle_identity_table_npz
from lagrangian_backtracking.scenarios import BASELINE_BEHAVIORS, SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table")
    ap.add_argument("--expect-per-candidate", type=int, default=None)
    a = ap.parse_args()

    t = load_particle_identity_table_npz(a.table)
    if t.particle_settling_velocity_mps is None or t.settling_candidate_index is None or t.settling_sampling is None:
        print("身分表沒有逐粒子沉降速度欄位（不是 class_candidates_balanced 產物）", file=sys.stderr)
        return 1
    if t.settling_sampling.get("mode") != SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED:
        print("settling_sampling.mode 不是 class_candidates_balanced：", t.settling_sampling.get("mode"), file=sys.stderr)
        return 1
    by_id = {b.material_id: b for b in BASELINE_BEHAVIORS}
    site = np.asarray(t.study_site_id).astype(str)
    mat = np.asarray(t.material_id).astype(str)
    idx = np.asarray(t.settling_candidate_index).astype(np.int64)
    vel = np.asarray(t.particle_settling_velocity_mps, dtype=np.float64)
    rep = np.asarray(t.settling_velocity_mps, dtype=np.float64)
    bad = 0
    per_cand_seen: set[int] = set()
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for key in sorted({(s, m) for s, m in zip(site.tolist(), mat.tolist())}):
        rows = np.flatnonzero((site == key[0]) & (mat == key[1]))
        b = by_id.get(key[1])
        if b is None:
            print("未知類別", key[1], file=sys.stderr); return 1
        cands = np.asarray(b.settling_velocity_candidates_mps, dtype=np.float64)
        counts = np.bincount(idx[rows], minlength=len(cands))
        if len(counts) != len(cands) or len(set(counts.tolist())) != 1:
            print("候選指派不均衡：", key, counts.tolist(), file=sys.stderr); bad += 1
        per_cand_seen.add(int(counts[0]))
        if a.expect_per_candidate is not None and not np.all(counts == a.expect_per_candidate):
            print("候選粒子數不等於期望值", a.expect_per_candidate, key, counts.tolist(), file=sys.stderr); bad += 1
        if not np.array_equal(vel[rows], cands[idx[rows]]):
            print("逐粒子速度與候選清單不一致：", key, file=sys.stderr); bad += 1
        if not np.all(rep[rows] == float(b.settling_velocity_mps)):
            print("類別代表值與 Behavior 不一致：", key, file=sys.stderr); bad += 1
        groups[key].append(int(counts.sum()))
    print(f"粒子數 {len(site)}；（站位, 類別）群數 {len(groups)}；每候選粒子數集合 {sorted(per_cand_seen)}；"
          f"種子主值 {t.settling_sampling.get('master_seed')}；content_sha256 {t.content_sha256}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
