# 代表軌跡（水平軌跡總覽所畫之 24 條／地點）之彙總：由 postprocess/trajectory_samples_by_site.parquet 與 departures.parquet 計算
import pyarrow.parquet as pq, numpy as np, json, collections
P='/home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/postprocess/'
man=json.load(open(P+'manifest.json'))
t=pq.read_table(P+'trajectory_samples_by_site.parquet',columns=['study_site_id','row_particle_index','particle_id']).to_pylist()
cols=['study_site_id','material_id','final_status_code','departure_lon_deg','departure_lat_deg','backtrack_duration_seconds','receptor_id','row_particle_index']
d=pq.read_table(P+'departures.parquet',columns=cols).to_pydict()
n=len(d['row_particle_index']); idx={r:i for i,r in enumerate(d['row_particle_index'])}
site=np.array(d['study_site_id']); st=np.array(d['final_status_code']); lon=np.array(d['departure_lon_deg']); lat=np.array(d['departure_lat_deg'])
out={'sample_policy':man['trajectory_sample_seed_policy'],'trajectory_sample_seed':man['config']['trajectory_sample_seed'],'size_per_site':man['config']['trajectory_sample_size_per_group'],'sites':{}}
for s in sorted(set(site)):
    m=(site==s)&(st==1)
    e={'w':float(np.percentile(lon[m],0.01)),'e':float(np.percentile(lon[m],99.99)),'s':float(np.percentile(lat[m],0.01)),'n':float(np.percentile(lat[m],99.99))}
    # 邊界位置：以穿越點最外緣近似（直線邊界）
    e={'w':float(lon[m].min()),'e':float(lon[m].max()),'s':float(lat[m].min()),'n':float(lat[m].max())}
    rows=[r for r in t if r['study_site_id']==s]
    ec=collections.Counter(); sc=collections.Counter(); mats=set(); rc=set(); days=[]
    for r in rows:
        i=idx[r['row_particle_index']]
        x,y=lon[i],lat[i]
        dd={'w':abs(x-e['w'])/0.1,'e':abs(x-e['e'])/0.1,'s':abs(y-e['s'])/0.1,'n':abs(y-e['n'])/0.1}
        k=min(dd,key=dd.get); assert dd[k]<0.05,(s,dd)
        if st[i]==1: ec[k]+=1
        sc[int(st[i])]+=1; mats.add(d['material_id'][i]); rc.add(d['receptor_id'][i]); days.append(d['backtrack_duration_seconds'][i]/86400)
    out['sites'][s]={'n':len(rows),'status_counts':dict(sc),'edge_counts':dict(ec),'n_materials':len(mats),'n_receptors':len(rc),'duration_days_median':float(np.median(days)),'duration_days_max':float(np.max(days)),'edges_deg':e}
print(json.dumps(out,ensure_ascii=False,indent=1))
