#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""report_run08_alu_example.py -- 附錄第六節「計算實例：鋁罐」所需之逐步數值（5 種代表物件、10 個解）。

呼叫 v7 derive.py 的 rho_eff_shell、geom_bluff、geom_slab 與 solve；另獨立重寫迭代以記錄迭代次數與前幾次迭代。
輸出 alu_example_run08.json 與 alu_example_run08.txt。
"""
import json, math, os, sys
SRC = os.environ.get("V7_DIR", os.path.expanduser("~/Workspace/lbt-repo-wp54/data/settling_velocity_derivation"))
sys.path.insert(0, SRC)
import derive as D

G, RHO_W, NU = D.G, D.RHO_W, D.NU
M_ALU = 2700.0

objs = [
    # (名稱, 殼重 kg, V_env m3 或 None(實心), 幾何)
    ("罐身碎片", None, None, ("slab", 0.050, 0.030, 0.0001)),
    ("完整進水罐（330 mL）", 0.013, 390e-6, ("bluff",)),
    ("半壓扁", 0.013, 0.100 * 0.060 * 0.025, ("slab", 0.100, 0.060, 0.025)),
    ("重壓扁", 0.013, 0.100 * 0.060 * 0.008, ("slab", 0.100, 0.060, 0.008)),
    ("完整進水罐（500 mL）", 0.017, 590e-6, ("bluff",)),
]


def iterate(num, dA, cd_inf, w0=1e-4, keep=6):
    w = w0
    hist = []
    for k in range(500):
        Re = w * dA / NU
        Cd = 24.0 / Re * (1 + 0.15 * Re ** 0.687) + cd_inf
        wstar = (num / Cd) ** 0.5
        hist.append((k, w, Re, Cd, wstar))
        if abs(wstar - w) < 1e-12 * max(wstar, 1e-12):
            return wstar, k, hist
        w = 0.5 * w + 0.5 * wstar
    return w, 500, hist


out = {"objects": [], "solutions": []}
allw = []
for name, m, Venv, g in objs:
    if g[0] == "bluff":
        V, A, dA = D.geom_bluff(Venv)
        cdlo, cdhi = D.CD_INF["bluff"]
    else:
        V, A, dA = D.geom_slab(g[1], g[2], g[3])
        cdlo, cdhi = D.CD_INF["slab"]
    if m is None:
        rho = M_ALU; Vsolid = V; Vwater = None
    else:
        Vsolid = m / M_ALU; Vwater = Venv - Vsolid; rho = D.rho_eff_shell(m, M_ALU, Venv)
    R = rho / RHO_W - 1
    N = 2 * V * G * R / A
    o = dict(name=name, m=m, V=V, A=A, dA=dA, Vsolid=Vsolid, Vwater=Vwater, rho=rho, R=R, N=N, shape=g[0])
    out["objects"].append(o)
    for tag, cd in (("cd_hi", cdhi), ("cd_lo", cdlo)):
        w_ref, Re_ref, Cd_ref = D.solve(V, A, dA, rho, cd)       # derive.py 之解
        w, k, hist = iterate(N, dA, cd)
        assert abs(w - w_ref) < 1e-9, (name, w, w_ref)
        Re = w * dA / NU
        Cd = 24.0 / Re * (1 + 0.15 * Re ** 0.687) + cd
        s = dict(obj=name, cd_inf=cd, Re=Re, Cd=Cd, w=w, iters=k + 1, converged_at=k, w_derive=w_ref)
        out["solutions"].append(s); allw.append(w)
        if name == "完整進水罐（330 mL）" and cd == 0.6:
            out["iteration_table"] = [dict(k=h[0], w=h[1], Re=h[2], Cd=h[3], wstar=h[4]) for h in hist]
w_geo = math.exp(sum(math.log(x) for x in allw) / len(allw))
out["geo_mean"] = w_geo
out["w_min"] = min(allw); out["w_max"] = max(allw)
out["w_sorted"] = sorted(allw)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "alu_example_run08.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
L = []
for o in out["objects"]:
    L.append("%-14s V=%.4e A=%.4e dA=%.5f Vsolid=%s Vwater=%s rho=%.2f R=%.5f N=%.5f" % (
        o["name"], o["V"], o["A"], o["dA"], "%.4e" % o["Vsolid"], "-" if o["Vwater"] is None else "%.4e" % o["Vwater"], o["rho"], o["R"], o["N"]))
for s in out["solutions"]:
    L.append("%-14s cd=%.1f Re=%.0f Cd=%.4f w=%.5f iters=%d" % (s["obj"], s["cd_inf"], s["Re"], s["Cd"], s["w"], s["iters"]))
L.append("geo mean = %.5f (3sf %.3g)  min=%.5f max=%.5f" % (w_geo, w_geo, out["w_min"], out["w_max"]))
for h in out["iteration_table"][:6]:
    L.append("iter k=%d w=%.6f Re=%.1f Cd=%.4f sqrtN/Cd=%.6f" % (h["k"], h["w"], h["Re"], h["Cd"], h["wstar"]))
L.append("last: %s" % out["iteration_table"][-1])
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "alu_example_run08.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
