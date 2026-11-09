# 在 .203 唯讀執行：由 run09 departures.parquet 計算報告所需之彙總，輸出 JSON 到標準輸出
import os, json, numpy as np, pyarrow.parquet as pq
R = os.path.expanduser('~/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes')
cols = ['study_site_id', 'material_id', 'final_status_code', 'crossed_interest_region_boundary', 'particle_settling_velocity_mps',
        'settling_candidate_index', 'settling_velocity_mps', 'backtrack_duration_seconds', 'particle_id']
t = pq.read_table(f'{R}/postprocess/departures.parquet', columns=cols)
site = np.asarray(t['study_site_id'].to_pylist(), dtype=object)
mat = np.asarray(t['material_id'].to_pylist(), dtype=object)
st = np.asarray(t['final_status_code'])
cr = np.asarray(t['crossed_interest_region_boundary']).astype(bool)
v = np.asarray(t['particle_settling_velocity_mps'], dtype=np.float64)
rep = np.asarray(t['settling_velocity_mps'], dtype=np.float64)
dur = np.asarray(t['backtrack_duration_seconds'], dtype=np.float64)
cidx = np.asarray(t['settling_candidate_index'])
out = {'class_site': {}, 'class_rep': {}, 'status': {}, 'dep': [], 'cand_per_class': {}, 'inside': {}}
for m in sorted(set(mat.tolist())):
    mm = mat == m
    out['class_rep'][m] = float(rep[mm][0])
    out['cand_per_class'][m] = {'n_rep_unique': int(len(np.unique(rep[mm]))), 'n_v_unique': int(len(np.unique(v[mm]))), 'n_idx_unique': int(len(np.unique(cidx[mm])))}
for s in sorted(set(site.tolist())):
    ms = site == s
    out['status'][s] = {int(k): int((st[ms] == k).sum()) for k in np.unique(st[ms])}
    for m in sorted(set(mat.tolist())):
        mm = ms & (mat == m)
        out['class_site'][f'{s}|{m}'] = {'n': int(mm.sum()), 'crossed': int(cr[mm].sum())}
    # 域內比例曲線（以離開時間之階梯函式精確計算；離開者 final_status_code==1）
    ex = np.sort(dur[ms & (st == 1)] / 86400.0)
    n = int(ms.sum())
    def inside_at(d):
        return float(1.0 - np.searchsorted(ex, d, side='right') / n)
    cur = {'n': n, 'n_exit': int(len(ex)), 'max_exit_day': float(ex.max()) if len(ex) else None,
           'at90': inside_at(90.0), 'at180': inside_at(180.0)}
    import math
    for th in (0.5, 0.1, 0.01):
        # 精確：離開顆數 >= ceil((1-th)*n) 之最早回溯時間（天）；以 0.1 天進位後給報告用
        k = int(math.ceil((1.0 - th) * n - 1e-9))
        if k <= len(ex):
            tt = float(ex[k - 1]); cur[f'exact_le_{th}'] = tt; cur[f'first_le_{th}'] = math.ceil(tt * 10 - 1e-9) / 10.0
        else:
            cur[f'exact_le_{th}'] = None; cur[f'first_le_{th}'] = None
    cur['first_zero_day'] = (math.ceil(float(ex.max()) * 10 - 1e-9) / 10.0) if (len(ex) == n) else None
    cur['exact_zero_day'] = float(ex.max()) if (len(ex) == n) else None
    cur['min_inside_at_end'] = inside_at(180.0)
    # 離開時間分布之關鍵日
    for d in (1, 3, 7, 14, 30, 60):
        cur[f'at{d}'] = inside_at(float(d))
    out['inside'][s] = cur
# 沉底（status 4）顆
for i in np.where(st == 4)[0]:
    out['dep'].append({'site': str(site[i]), 'material': str(mat[i]), 'v': float(v[i]), 'rep': float(rep[i]), 'dur_s': float(dur[i]), 'dur_h': float(dur[i]) / 3600.0, 'cand': int(cidx[i]), 'pid': str(t['particle_id'][int(i)])})
# 達回溯上限（status 2）之回溯時間
out['max_dur'] = {s: float(dur[(site == s) & (st == 2)].max()) / 86400.0 if ((site == s) & (st == 2)).any() else None for s in sorted(set(site.tolist()))}
out['n_total'] = int(len(site))
print(json.dumps(out, ensure_ascii=False))
