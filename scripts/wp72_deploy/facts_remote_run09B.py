# 方法 B：在 .203 唯讀執行；由 postprocess-coastB/departures.parquet 計算報告所需之彙總，輸出 JSON 到標準輸出
# 停止追蹤＝穿越流場範圍邊界（狀態 1）或回溯途中首次抵達海岸（狀態 8）
import os, sys, json, math, numpy as np, pyarrow.parquet as pq
R = os.path.expanduser('~/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes')
PP = sys.argv[1] if len(sys.argv) > 1 else f'{R}/postprocess-coastB'
cols = ['study_site_id', 'material_id', 'final_status_code', 'particle_settling_velocity_mps', 'settling_candidate_index',
        'settling_velocity_mps', 'backtrack_duration_seconds', 'particle_id']
t = pq.read_table(f'{PP}/departures.parquet', columns=cols)
site = np.asarray(t['study_site_id'].to_pylist(), dtype=object)
mat = np.asarray(t['material_id'].to_pylist(), dtype=object)
st = np.asarray(t['final_status_code'])
v = np.asarray(t['particle_settling_velocity_mps'], dtype=np.float64)
dur = np.asarray(t['backtrack_duration_seconds'], dtype=np.float64)
out = {'status': {}, 'dep': [], 'inside': {}, 'stop_status_codes': [1, 8]}
for s in sorted(set(site.tolist())):
    ms = site == s
    out['status'][s] = {int(k): int((st[ms] == k).sum()) for k in np.unique(st[ms])}
    stop = ms & np.isin(st, (1, 8))
    ex = np.sort(dur[stop] / 86400.0)
    n = int(ms.sum())
    def inside_at(d):
        return float(1.0 - np.searchsorted(ex, d, side='right') / n)
    cur = {'n': n, 'n_stop': int(len(ex)), 'max_stop_day': float(ex.max()) if len(ex) else None,
           'at90': inside_at(90.0), 'at180': inside_at(180.0)}
    for th in (0.5, 0.1, 0.01):
        k = int(math.ceil((1.0 - th) * n - 1e-9))
        if k <= len(ex):
            tt = float(ex[k - 1]); cur[f'exact_le_{th}'] = tt; cur[f'first_le_{th}'] = math.ceil(tt * 10 - 1e-9) / 10.0
        else:
            cur[f'exact_le_{th}'] = None; cur[f'first_le_{th}'] = None
    cur['first_zero_day'] = (math.ceil(float(ex.max()) * 10 - 1e-9) / 10.0) if (len(ex) == n) else None
    cur['exact_zero_day'] = float(ex.max()) if (len(ex) == n) else None
    # 剩餘未停止者（觸底）
    cur['n_not_stopped'] = n - len(ex)
    for d in (1, 3, 7, 14, 30, 60):
        cur[f'at{d}'] = inside_at(float(d))
    out['inside'][s] = cur
for i in np.where(st == 4)[0]:
    out['dep'].append({'site': str(site[i]), 'material': str(mat[i]), 'v': float(v[i]), 'dur_s': float(dur[i]), 'dur_h': float(dur[i]) / 3600.0, 'pid': str(t['particle_id'][int(i)])})
out['n_total'] = int(len(site))
print(json.dumps(out, ensure_ascii=False))
