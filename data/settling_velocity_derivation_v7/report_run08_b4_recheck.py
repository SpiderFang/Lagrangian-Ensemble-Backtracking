#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""report_run08_b4_recheck.py -- 重算附錄中引用的比例、範圍與補充二例子（以 v7 的 50 組、import v7 derive.py）。
輸出 b4_recheck_run08.json / .txt（與本腳本同目錄）。
"""
import json, math, os, sys
sys.path.insert(0, os.environ.get("V7_DIR", os.path.expanduser("~/Workspace/lbt-repo-wp54/data/settling_velocity_derivation")))
import derive as D
HERE = os.path.dirname(os.path.abspath(__file__))
G, RW, NU = D.G, D.RHO_W, D.NU

# 1. 取 stats_run08.json（report_run08_stats.py 輸出，含舊 30 組對照）
S = json.load(open(os.path.join(HERE, 'stats_run08.json')))
out = {}
for tag, label in (('n30_check', '舊版 30 組'), ('n50', '50 組')):
    st = S[tag]['stats']
    out[tag] = dict(
        label=label,
        sphere_ratio_slab=(st['sphere_ratio_slab_min'], st['sphere_ratio_slab_max']),
        sphere_ratio_flat_slabcd=(st['sphere_ratio_flat_slabcd_min'], st['sphere_ratio_flat_slabcd_max']),
        ge1e4_excess_pct=(st['ge1e4_excess_min'], st['ge1e4_excess_max']))

# 2. 補充二例子
def solve(V, A, dA, rho, cd):
    return D.solve(V, A, dA, rho, cd)       # (w, Re, Cd)

def cd_sn(Re):
    return 24 / Re * (1 + 0.15 * Re ** 0.687)

ex = {}
L = W = 0.1; t = 0.02; rho = 1100.0
V = L * W * t; A = L * W; dA = math.sqrt(4 * A / math.pi)
w, Re, Cd = solve(V, A, dA, rho, 1.1)
R = rho / RW - 1
ex['R'] = R; ex['2gRt'] = 2 * G * R * t; ex['w_hi'] = w; ex['Re'] = Re; ex['Cd'] = Cd
ex['Cd_back'] = 2 * G * R * t / w ** 2; ex['SN_term'] = cd_sn(Re); ex['Cd_sum'] = 1.1 + cd_sn(Re)
def plate(Lm, tm, rho, cd):
    A = Lm * Lm; V = A * tm; dA = math.sqrt(4 * A / math.pi)
    return solve(V, A, dA, rho, cd)[0]
ex['edge_0.1_vs_1.0_m_pct'] = (plate(1.0, 0.02, 1100, 1.1) / plate(0.1, 0.02, 1100, 1.1) - 1) * 100
ex['film_5_vs_50_mm_ratio'] = plate(0.05, 3e-5, 1030, 1.1) / plate(0.005, 3e-5, 1030, 1.1)
out['supp2_example'] = ex

# 3. 與 v7 results.csv 的對照（板狀殘片上界）
import csv
rows = list(csv.DictReader(open(os.path.join(os.environ.get('V7_DIR', '.'), 'results.csv'), encoding='utf-8-sig')))
for r in rows:
    if r['class_id'] == '1' and r['scenario'].startswith('大型浮具'):
        out['csv_板狀殘片'] = dict(w_lo=float(r['w_lo_ms']), w_hi=float(r['w_hi_ms']))
json.dump(out, open(os.path.join(HERE, 'b4_recheck_run08.json'), 'w'), ensure_ascii=False, indent=1)
lines = []
for k in ('n30_check', 'n50'):
    o = out[k]
    lines.append('%s: 等效球倍數(片狀) %.3f–%.3f；壓扁以等效球 %.3f–%.3f；Re>=1e4 時 Cd 超出 C_d,inf %.2f%%–%.2f%%' % (
        o['label'], *o['sphere_ratio_slab'], *o['sphere_ratio_flat_slabcd'], *o['ge1e4_excess_pct']))
lines.append('補充二例子: ' + json.dumps(ex, ensure_ascii=False))
open(os.path.join(HERE, 'b4_recheck_run08.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
