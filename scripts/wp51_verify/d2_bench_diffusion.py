"""WP-51-V 任務 D2：真實 lienchiang（D 區）流場、真實身分表與 run04 計畫（唯讀）上，量測每流場小時耗時：
 (1) 擴散組態：gaussian/off（舊）、uniform/off、uniform/constant（新預設）；
 (2) 工作單元結構：per_observation 樣式（同一到達時刻、200 粒子大單元）與 per_member 樣式
     （同樣粒子列拆成 20 粒子小單元、到達時刻分散到 0..23 小時）。
只量時間，不寫成果。用法（環境變數同 prof_env.sh）：python d2_bench_diffusion.py <段前綴> <回溯小時> <單元數>"""
import dataclasses, os, sys, time
import numpy as np
from lagrangian_backtracking.native_runner import hour_major as hm, hour_major_provider as hp, orchestrator as orch

prefix, hours, nunits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
H_NS = 3600 * 10**9
params = hp._resolve_params(os.environ)
base_cfg = dataclasses.replace(params.hour_major_config, max_backtrack_seconds=hours * 3600.0)
print("env 預設擴散組態：", base_cfg.diffusion_noise, base_cfg.vertical_diffusion, base_cfg.kz_constant_m2ps, base_cfg.visser_correction, flush=True)
res = hp._build_worker_resources_entrypoint(params, 0)
units, _ = orch.read_native_plan_file(params.plan_path)
u = [x for x in units if x.unit_id.startswith(prefix)][0]
allm = sorted(res.members[u.unit_id], key=lambda m: (m.arrival_time_utc_ns, m.settling_velocity_mps))
a0 = allm[0].arrival_time_utc_ns
same = [m for m in allm if m.arrival_time_utc_ns == a0][:nunits]
region = same[0].analysis_region_id
dom = hp._domain_resources_for(res, region)
npart = sum(m.particle_count for m in same)
print(f"區域 {region}；單元 {len(same)}（同一到達時刻）、粒子 {npart}", flush=True)

# per_member 樣式：每個單元的粒子列平均拆成 10 份，各份到達時刻 = a0 + (j % 24) 小時
split = []
for m in same:
    rows = m.particle_row_indices
    for j, part in enumerate(np.array_split(rows, 10)):
        if part.size == 0: continue
        split.append(dataclasses.replace(m, unit_id=f"{m.unit_id}#j{j}", particle_row_indices=part.astype(np.int64),
                                         arrival_time_utc_ns=int(a0 + ((j * 7 + len(split)) % 24) * H_NS), input_hash=f"{m.input_hash}#j{j}"))
print(f"per_member 樣式：{len(split)} 個單元、相異到達時刻 {len({s.arrival_time_utc_ns for s in split})}", flush=True)

def run(members, cfg, label, repeats=1):
    out = []
    for rep in range(repeats):
        batch = hm.build_hour_major_domain_batch(res.identity_table, members, interest_regions=dom.interest_regions,
                                                 local_domains=dom.local_domains, local_equals_flow=dom.local_equals_flow)
        kw = dict(forcing_resolver=dom.forcing_resolver_factory(),
                  wave_field_resolver=dom.wave_resolver_factory() if dom.wave_resolver_factory else None)
        t0 = time.time()
        r = hm.run_flow_domain_hour_major(dom.driver_grid, res.handles, batch, config=cfg, **kw)
        el = time.time() - t0
        n = max(1, r.domain_hours_walked)
        out.append((el, n, r))
    el, n, r = min(out, key=lambda x: x[0] / x[1])
    stat = {}
    try:
        fs = np.concatenate([np.asarray(g) for g in [getattr(r, 'final_status', None)] if g is not None]) if hasattr(r, 'final_status') else None
    except Exception:
        fs = None
    print(f"{label:46s} 走過小時 {n:4d} 總秒 {el:8.2f} 每小時 {el/n:7.3f}s kernel/h {r.domain_kernel_call_count/n:5.1f}", flush=True)
    return el / n

if len(sys.argv) > 4 and sys.argv[4] == "alt":
    # 交替重複量測（降低機器負載漂移的影響）：per_observation 樣式，gaussian/off 與 uniform/constant 交替各 3 次
    cfgA = dataclasses.replace(base_cfg, diffusion_noise="gaussian", vertical_diffusion="off")
    cfgB = dataclasses.replace(base_cfg, diffusion_noise="uniform", vertical_diffusion="constant")
    for c_ in (cfgA, cfgB):
        run(same, dataclasses.replace(c_, max_backtrack_seconds=2 * 3600.0), "暖機", repeats=1)
    ta, tb = [], []
    for k in range(3):
        ta.append(run(same, cfgA, f"交替#{k} gaussian/off")); tb.append(run(same, cfgB, f"交替#{k} uniform/constant"))
    print(f"交替量測 中位：gaussian/off {np.median(ta):.3f}s/h；uniform/constant {np.median(tb):.3f}s/h；比值 {np.median(tb)/np.median(ta):.3f}")
    sys.exit(0)
warm = dataclasses.replace(base_cfg, max_backtrack_seconds=2 * 3600.0)
combos = [("gaussian", "off"), ("uniform", "off"), ("uniform", "constant")]
for noise, vd in combos:
    cfg = dataclasses.replace(base_cfg, diffusion_noise=noise, vertical_diffusion=vd)
    hm.validate_diffusion_scheme(cfg)
    run(same, dataclasses.replace(cfg, max_backtrack_seconds=2 * 3600.0), f"暖機 {noise}/{vd}", repeats=1)
res_t = {}
for noise, vd in combos:
    cfg = dataclasses.replace(base_cfg, diffusion_noise=noise, vertical_diffusion=vd)
    res_t[("obs", noise, vd)] = run(same, cfg, f"per_observation 樣式 {noise}/{vd}")
for noise, vd in combos:
    cfg = dataclasses.replace(base_cfg, diffusion_noise=noise, vertical_diffusion=vd)
    res_t[("mem", noise, vd)] = run(split, cfg, f"per_member 樣式 {noise}/{vd}")
b = res_t[("obs", "gaussian", "off")]
print("\n相對 per_observation gaussian/off：")
for k, v in res_t.items():
    print(f"  {k[0]} {k[1]}/{k[2]}: {v / b:.3f}x")
