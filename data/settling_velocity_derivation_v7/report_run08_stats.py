#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""report_run08_stats.py -- 主報告附錄（50 組代表物件、每類 10 個沉降速度）所需統計量之重算腳本。

只讀取 v7 的 derive.py（以 import 方式呼叫其函式，不改動），輸出：
  stats_run08.json  所有數值（供核對腳本使用）
  stats_run08.txt   人讀版摘要
並對「舊版 30 組」重算同樣的敏感度檢驗，以確認方法與既有附錄數字一致。
"""
import json, math, os, sys

SRC = os.environ.get("V7_DIR", os.path.expanduser("~/Workspace/lbt-repo-wp54/data/settling_velocity_derivation"))
sys.path.insert(0, SRC)
import derive as D   # noqa: E402

G = D.G
RHO_W = D.RHO_W
NU0 = D.NU

# 舊版 30 組：v7 每類 5 種物件中新增的 2 種之索引（0 起算）
NEW_IDX = {1: (1, 4), 2: (1, 4), 3: (1, 4), 4: (1, 3), 5: (1, 3),
           6: (0, 4), 7: (0, 4), 8: (0, 4), 9: (0, 1), 10: (0, 3)}


def scen_list(old_only=False):
    out = []
    for cls in D.CLASSES:
        for i, sc in enumerate(cls["scenarios"]):
            if old_only and i in NEW_IDX[cls["class_id"]]:
                continue
            shape = sc.get("shape", cls["shape"])
            cdlo, cdhi = sc.get("cd_inf", D.CD_INF[shape])
            V, A, dA = sc["geom"]
            shell = sc.get("shell")
            if shell is not None:
                state = "whole" if shape == "bluff" else "flat"
            elif cls["class_id"] in (6, 8, 9) and shape == "slab":
                state = "fragment"
            else:
                state = "other"
            out.append(dict(cid=cls["class_id"], cname=cls["class_zh"], name=sc["scenario"], idx=i,
                            shape=shape, V=V, A=A, dA=dA, rho=sc["rho_p"], cdlo=cdlo, cdhi=cdhi,
                            state=state, size=sc["geom"].size_text))
    return out


def solve_iter(V, A, dA, rho, cd_inf, nu=NU0, first_term=True):
    """與 derive.solve 相同之阻尼固定點迭代，另回傳迭代次數；first_term=False 時 Cd 僅取 cd_inf。"""
    R = rho / RHO_W - 1.0
    num = 2.0 * V * G * R / A
    if not first_term:
        w = (num / cd_inf) ** 0.5
        return w, w * dA / nu, cd_inf, 0
    w = 1e-4
    n = 0
    for _ in range(500):
        n += 1
        Re = max(w * dA / nu, 1e-12)
        Cd = 24.0 / Re * (1.0 + 0.15 * Re ** 0.687) + cd_inf
        w_new = (num / Cd) ** 0.5
        if abs(w_new - w) < 1e-12 * max(w_new, 1e-12):
            w = w_new
            break
        w = 0.5 * w + 0.5 * w_new
    Re = w * dA / nu
    Cd = 24.0 / max(Re, 1e-12) * (1.0 + 0.15 * Re ** 0.687) + cd_inf
    return w, Re, Cd, n


def sols(s, **kw):
    """回傳該物件兩個解 [(w_lo, Re, Cd, n, cd_inf), (w_hi, ...)]；w_lo 對應 cd 上界。"""
    cdhi = kw.pop("cdhi", s["cdhi"]); cdlo = kw.pop("cdlo", s["cdlo"])
    dA = kw.pop("dA", s["dA"])
    a = solve_iter(s["V"], s["A"], dA, s["rho"], cdhi, **kw)
    b = solve_iter(s["V"], s["A"], dA, s["rho"], cdlo, **kw)
    return [a + (cdhi,), b + (cdlo,)]


def gmean(vals, weights=None):
    if weights is None:
        weights = [1.0] * len(vals)
    sw = sum(weights)
    return math.exp(sum(w * math.log(v) for v, w in zip(vals, weights)) / sw)


def class_values(scens, cdsel=None, weight_fn=None, **kw):
    """各類代表值（10 個解之（加權）幾何平均）。"""
    res = {}
    for cid in range(1, 11):
        ss = [s for s in scens if s["cid"] == cid]
        vals, wts = [], []
        for s in ss:
            extra = dict(kw)
            if cdsel is not None:
                lo, hi = cdsel[s["shape"]]
                extra["cdlo"] = lo; extra["cdhi"] = hi
            if "dA_fn" in extra:
                f = extra.pop("dA_fn"); extra["dA"] = f(s)
            for sol in sols(s, **extra):
                vals.append(sol[0]); wts.append(1.0)
        if weight_fn is not None:
            wts = weight_fn(ss, vals)
        res[cid] = gmean(vals, wts)
    return res


def thirds(ss, vals):
    # 完整進水、壓扁、碎片三種狀態各占 1/3；同一狀態內各解等權；非容器類等權
    states = [s["state"] for s in ss for _ in (0, 1)]
    if "whole" not in states:
        return [1.0] * len(vals)
    cnt = {k: states.count(k) for k in set(states)}
    return [1.0 / (len(cnt) * cnt[k]) for k in states]


def dedupe(ss, vals):
    # 速度完全相同的重複解只計一次（其餘各解等權）
    keys = [round(v, 6) for v in vals]
    w = []
    seen = {}
    for k in keys:
        seen[k] = seen.get(k, 0) + 1
    return [1.0 / seen[k] for k in keys]


def pct_changes(base, var):
    return {cid: (var[cid] / base[cid] - 1.0) * 100.0 for cid in base}


def order(d):
    return [cid for cid, _ in sorted(d.items(), key=lambda kv: kv[1])]


def sens(scens):
    out = {}
    base = class_values(scens)
    out["base"] = base
    textbook = {"slab": (1.1, 1.9), "cyl": (0.9, 1.2), "bluff": (0.5, 1.2)}
    variants = {
        "textbook": class_values(scens, cdsel=textbook),
        "no_first_term": class_values(scens, first_term=False),
        "cyl_dn": class_values(scens, dA_fn=lambda s: (6 * s["V"] / math.pi) ** (1 / 3) if s["shape"] == "cyl" else s["dA"]),
        "thirds": class_values(scens, weight_fn=thirds),
        "dedupe": class_values(scens, weight_fn=dedupe),
        "nu10": class_values(scens, nu=1.3604e-6),
        "nu28": class_values(scens, nu=8.7837e-7),
    }
    out["variants"] = {}
    for k, v in variants.items():
        ch = pct_changes(base, v)
        nz = {c: x for c, x in ch.items() if abs(x) > 0.05}
        out["variants"][k] = dict(values=v, pct=ch,
                                  pct_min=min(ch.values()), pct_max=max(ch.values()),
                                  pct_nonzero_min=min(nz.values()) if nz else 0.0,
                                  pct_nonzero_max=max(nz.values()) if nz else 0.0,
                                  changed_classes=sorted(nz), order_same=(order(v) == order(base)))
    out["order_base"] = order(base)
    return out


def stats(scens):
    allsol = []
    for s in scens:
        for sol in sols(s):
            allsol.append(dict(cid=s["cid"], name=s["name"], shape=s["shape"], w=sol[0], Re=sol[1], Cd=sol[2], n=sol[3], cd=sol[4]))
    st = {}
    st["n_scen"] = len(scens); st["n_sol"] = len(allsol)
    st["Re_min"] = min(x["Re"] for x in allsol); st["Re_max"] = max(x["Re"] for x in allsol)
    st["n_iter_max"] = max(x["n"] for x in allsol); st["n_iter_min"] = min(x["n"] for x in allsol)
    def first_share(x):
        return (x["Cd"] - x["cd"]) / x["Cd"]
    for lim, key in ((1e3, "lt1e3"), (1e4, "lt1e4")):
        sel = [x for x in allsol if x["Re"] < lim]
        st["n_sol_" + key] = len(sel)
        st["n_obj_" + key] = len({(x["cid"], x["name"]) for x in sel})          # 至少一解低於界限之物件數
        st["n_obj_both_" + key] = len([1 for s in scens if all(x["Re"] < lim for x in
                                       [dict(Re=y[1]) for y in sols(s)])])
        if sel:
            st["share_" + key + "_min"] = min(first_share(x) for x in sel) * 100
            st["share_" + key + "_max"] = max(first_share(x) for x in sel) * 100
    sel = [x for x in allsol if x["Re"] >= 1e4]
    st["n_sol_ge1e4"] = len(sel)
    st["ge1e4_excess_min"] = min((x["Cd"] - x["cd"]) / x["cd"] for x in sel) * 100
    st["ge1e4_excess_max"] = max((x["Cd"] - x["cd"]) / x["cd"] for x in sel) * 100
    # 等效球倍數
    sl = []
    for s in scens:
        if s["shape"] != "slab":
            continue
        V = s["V"]; Deq = (6 * V / math.pi) ** (1 / 3); Asp = math.pi / 4 * Deq ** 2
        for cd, w_slab in ((s["cdhi"], None), (s["cdlo"], None)):
            wsl = solve_iter(V, s["A"], s["dA"], s["rho"], cd)[0]
            wsp = solve_iter(V, Asp, Deq, s["rho"], cd)[0]
            sl.append(dict(cid=s["cid"], name=s["name"], state=s["state"], cd=cd, ratio=wsp / wsl))
    st["sphere_ratio_slab_min"] = min(x["ratio"] for x in sl); st["sphere_ratio_slab_max"] = max(x["ratio"] for x in sl)
    st["sphere_ratio_slab_min_at"] = min(sl, key=lambda x: x["ratio"])["name"]
    st["sphere_ratio_slab_max_at"] = max(sl, key=lambda x: x["ratio"])["name"]
    # 壓扁瓶罐：以等效球（鈍體阻力係數）／以等效球（薄板阻力係數）
    fl = [x for x in sl if x["state"] == "flat"]
    st["sphere_ratio_flat_slabcd_min"] = min(x["ratio"] for x in fl); st["sphere_ratio_flat_slabcd_max"] = max(x["ratio"] for x in fl)
    fr = [x for x in sl if x["state"] in ("flat", "fragment")]
    st["sphere_ratio_flatfrag_slabcd_min"] = min(x["ratio"] for x in fr); st["sphere_ratio_flatfrag_slabcd_max"] = max(x["ratio"] for x in fr)
    rb = []
    for s in scens:
        if s["state"] != "flat":
            continue
        V = s["V"]; Deq = (6 * V / math.pi) ** (1 / 3); Asp = math.pi / 4 * Deq ** 2
        for k, (cds, cdb) in enumerate(((s["cdhi"], 1.2), (s["cdlo"], 0.6))):
            wsl = solve_iter(V, s["A"], s["dA"], s["rho"], cds)[0]
            wsp = solve_iter(V, Asp, Deq, s["rho"], cdb)[0]
            rb.append(wsp / wsl)
    st["sphere_ratio_flat_bluffcd_min"] = min(rb); st["sphere_ratio_flat_bluffcd_max"] = max(rb)
    # 薄板面積分布
    areas = [(s["name"], s["cid"], s["A"] * 1e4) for s in scens if s["shape"] == "slab"]
    st["n_slab"] = len(areas)
    st["n_slab_25_2500"] = len([1 for a in areas if 25 <= a[2] < 2500])
    st["n_slab_lt25"] = len([1 for a in areas if a[2] < 25])
    st["n_slab_ge2500"] = len([1 for a in areas if a[2] >= 2500])
    st["slab_area_min"] = min(a[2] for a in areas); st["slab_area_max"] = max(a[2] for a in areas)
    st["slab_lt25_names"] = [(a[0], a[1], a[2]) for a in areas if a[2] < 25]
    # 非容器片狀／柱狀物件最長邊
    lens = set()
    for s in scens:
        if s["state"] == "other" and s["shape"] in ("slab", "cyl"):
            lens.add(s["size"])
    st["sizes_other"] = sorted(lens)
    return st, allsol


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    res = {}
    for tag, old in (("n50", False), ("n30_check", True)):
        sc = scen_list(old)
        st, allsol = stats(sc)
        se = sens(sc)
        # 類別彙總
        summ = {}
        for cid in range(1, 11):
            ws = [x["w"] for x in allsol if x["cid"] == cid]
            summ[cid] = dict(name=[s["cname"] for s in sc if s["cid"] == cid][0], w_min=min(ws), w_max=max(ws),
                             w_repr=gmean(ws), n=len(ws),
                             n_distinct=len({round(w, 6) for w in ws}))
        res[tag] = dict(stats=st, sens=se, summary=summ)
    # JSON（鍵轉字串）
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    with open(os.path.join(out_dir, "stats_run08.json"), "w", encoding="utf-8") as f:
        json.dump(conv(res), f, ensure_ascii=False, indent=1)
    lines = []
    for tag in ("n30_check", "n50"):
        r = res[tag]
        lines.append("=" * 70); lines.append(tag)
        for k, v in r["stats"].items():
            lines.append("  %s: %s" % (k, v))
        lines.append("  -- summary")
        for cid, s in r["summary"].items():
            lines.append("  %2d %-8s n=%d distinct=%d min=%.5f repr=%.5f (%.3g) max=%.5f" % (cid, s["name"], s["n"], s["n_distinct"], s["w_min"], s["w_repr"], s["w_repr"], s["w_max"]))
        lines.append("  -- sensitivity (class pct changes)")
        for k, v in r["sens"]["variants"].items():
            lines.append("  %-14s min=%+.2f%% max=%+.2f%% nonzero[%+.2f,%+.2f] changed=%s order_same=%s" % (
                k, v["pct_min"], v["pct_max"], v["pct_nonzero_min"], v["pct_nonzero_max"], v["changed_classes"], v["order_same"]))
            lines.append("      " + " ".join("%d:%+.2f" % (c, p) for c, p in v["pct"].items()))
    with open(os.path.join(out_dir, "stats_run08.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
