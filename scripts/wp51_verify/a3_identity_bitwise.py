"""WP-51-V 項目 3：per_observation 路徑與 e8f30a7 位元相同。
以真實 v3 母體（per_observation 設定）建 M=10 身分表，輸出 content_sha256，
並與 run04 已存的 identity_table.npz 內容雜湊比對。用法：python a3_identity_bitwise.py <config.yaml> <run04 identity_table.npz>"""
import sys, hashlib
import numpy as np
from lagrangian_backtracking.native_runner import production_wiring as pw
from lagrangian_backtracking.native_runner.particle_input import load_particle_identity_table_npz
import lagrangian_backtracking
print("source:", lagrangian_backtracking.__file__)
t = pw.build_formal_identity_table(sys.argv[1], members_per_scenario=10)
print("新建 content_sha256:", t.content_sha256, "粒子數", len(t.study_site_id), "labels", None if t.labels is None else sorted(t.labels))
old = load_particle_identity_table_npz(sys.argv[2])
print("run04 content_sha256:", old.content_sha256)
print("相同:", t.content_sha256 == old.content_sha256)
