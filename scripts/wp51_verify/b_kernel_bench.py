"""WP-51-V 任務 D（Numba 迴圈）：融合子步 kernel 在不同擴散組態下的耗時（合成小網格，粒子數放大）。
同一腳本可在 e8f30a7（只跑 legacy）與 wp51 分支（四種組態）下執行。
用法：PYTHONPATH=<tree>/src python b_kernel_bench.py <tree>/tests [N] [ND] [reps]"""
import inspect, os, sys, time
import numpy as np
sys.path.insert(0, sys.argv[1])
import test_substep_numba as tsn
from lagrangian_backtracking.native_runner import backward_driver as bd, build_native_grid_arrays
from lagrangian_backtracking.native_runner.substep_numba import mirror_rk4_substep_loop_numba
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
ND = int(sys.argv[3]) if len(sys.argv) > 3 else 150
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 7
h = bd.load_baytrace_modules(os.environ["LBT_BAYTRACE_ROOT"])
mesh = tsn._build_mesh(); nn = mesh.node_xy.shape[0]
zcor = np.zeros((nn, 3)); zcor[:, 0] = -20.0; zcor[:, 1] = -10.0
g = build_native_grid_arrays(mesh, zcor_reference=zcor, eta_reference=np.zeros(nn))
grid = bd.build_native_backward_driver_grid(g, h, h0=0.0)
composed = tsn._make_composed(grid, u=0.0, v=0.0)
kh = np.full((3, nn), 0.05)
pa = h.particle_arrays_cls(N)
rng = np.random.default_rng(1)
pa.x[:] = 900.0 + rng.uniform(-300, 300, N); pa.y[:] = 900.0; pa.z[:] = -10.0; pa.z0[:] = -10.0; pa.status[:] = 0
bd.locate_initial_elements(grid, h, pa, eta_m=composed.eta_field1, u_current_mps=composed.u_field1, v_current_mps=composed.v_field1, w_current_mps=composed.w_field1)
x0, y0, z0 = pa.x.copy(), pa.y.copy(), pa.z.copy()
seed_lo, seed_hi = bd.derive_particle_diffusion_seed_lo_hi_u64([f"{i:032x}" for i in range(N)], master_seed=1)
has_new = "diffusion_noise" in inspect.signature(mirror_rk4_substep_loop_numba).parameters
cfgs = {"legacy gaussian/off": {}}
if has_new:
    cfgs.update({"uniform/off": dict(diffusion_noise="uniform"),
                 "uniform/constant (新預設)": dict(diffusion_noise="uniform", vertical_diffusion="constant"),
                 "uniform/constant+Visser": dict(diffusion_noise="uniform", vertical_diffusion="constant", visser_correction=True)})
times = {k: [] for k in cfgs}
zfin = {}
for rep in range(REPS + 1):          # 第 0 輪為暖機（含 JIT 編譯），不計
    for name, kw in cfgs.items():
        pa.x[:] = x0; pa.y[:] = y0; pa.z[:] = z0; pa.status[:] = 1
        counter = np.zeros(N, dtype=np.int64)
        t0 = time.perf_counter()
        d = mirror_rk4_substep_loop_numba(grid, h, pa, ibf=-1, ndeltp=ND, dtb=3600.0 / ND, composed=composed, enable_diffusion=True,
                                          hf1_field=kh, hf2_field=kh, particle_seed_lo=seed_lo, particle_seed_hi=seed_hi,
                                          particle_step_counter=counter, **kw)
        wall = time.perf_counter() - t0
        if rep > 0: times[name].append(d.time_move_s)
        zfin[name] = (float(pa.z.std()), float(pa.z.min()), float(pa.z.max()))
base = None
print(f"N={N} 子步={ND}（每次 kernel 呼叫 = 1 流場小時）；{REPS} 次重複")
for name, ts in times.items():
    ts = np.array(ts)
    if base is None: base = np.median(ts)
    print(f"  {name:28s} 中位 {np.median(ts):7.3f}s 最小 {ts.min():7.3f}s 最大 {ts.max():7.3f}s  相對 legacy {np.median(ts)/base:5.3f}x   z 標準差 {zfin[name][0]:.3f} 範圍[{zfin[name][1]:.2f},{zfin[name][2]:.2f}]")
