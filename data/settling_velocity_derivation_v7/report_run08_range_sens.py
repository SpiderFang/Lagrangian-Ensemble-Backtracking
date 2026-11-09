#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""report_run08_range_sens.py -- 表 A10（實測對照）與表 A11（敏感度）改以「每類 10 個解之範圍」表示所需數值。

沿用 report_run08_stats.py（同資料夾）之 scen_list、sols、solve_iter，其求解器與 derive.py 相同
（阻尼固定點迭代；rho_w=1025、nu=1.05e-6、g=9.81）。不使用任何代表值或幾何平均。

輸出（本腳本所在資料夾）：
  range_sens_run08.json  各類基準範圍、各情境下之下界／上界與相對基準之百分比變化、排序檢查
  range_sens_run08.txt   人讀版
用法：python3 report_run08_range_sens.py
"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('V7_DIR', HERE)
sys.path.insert(0, HERE)
import report_run08_stats as R   # noqa: E402

TEXTBOOK = {"slab": (1.1, 1.9), "cyl": (0.9, 1.2), "bluff": (0.5, 1.2)}


def class_ranges(scens, cdsel=None, **kw):
    """各類 10 個解之 (最小值, 最大值)。cdsel：改以其他 Cd,∞ 上下界；kw：傳給 solve_iter 之選項（nu、first_term）或 dA_fn。"""
    out = {}
    for cid in range(1, 11):
        vals = []
        for s in [x for x in scens if x["cid"] == cid]:
            extra = dict(kw)
            if cdsel is not None:
                lo, hi = cdsel[s["shape"]]
                extra["cdlo"] = lo; extra["cdhi"] = hi
            if "dA_fn" in extra:
                f = extra.pop("dA_fn"); extra["dA"] = f(s)
            for sol in R.sols(s, **extra):
                vals.append(sol[0])
        out[cid] = (min(vals), max(vals), len(vals), len({round(v, 6) for v in vals}))
    return out


def pct(a, b):
    return (a / b - 1.0) * 100.0


def rank(d, k):
    return [cid for cid, _ in sorted(d.items(), key=lambda kv: kv[1][k])]


def main():
    sc = R.scen_list(False)
    assert len(sc) == 50
    base = class_ranges(sc)
    variants = {
        "textbook": class_ranges(sc, cdsel=TEXTBOOK),
        "no_first_term": class_ranges(sc, first_term=False),
        "cyl_dn": class_ranges(sc, dA_fn=lambda s: (6 * s["V"] / math.pi) ** (1 / 3) if s["shape"] == "cyl" else s["dA"]),
        "dedupe": class_ranges(sc),          # 重複解只計一次不改變最小值與最大值，範圍與基準相同
        "nu10": class_ranges(sc, nu=1.3604e-6),
        "nu28": class_ranges(sc, nu=8.7837e-7),
    }
    out = {"base": {str(c): dict(lo=v[0], hi=v[1], n=v[2], n_distinct=v[3]) for c, v in base.items()},
           "order_base_lo": rank(base, 0), "order_base_hi": rank(base, 1), "variants": {}}
    for k, v in variants.items():
        lo = {c: pct(v[c][0], base[c][0]) for c in base}
        hi = {c: pct(v[c][1], base[c][1]) for c in base}
        nzlo = {c: x for c, x in lo.items() if abs(x) > 0.05}
        nzhi = {c: x for c, x in hi.items() if abs(x) > 0.05}
        out["variants"][k] = dict(
            lo={str(c): v[c][0] for c in v}, hi={str(c): v[c][1] for c in v},
            pct_lo={str(c): x for c, x in lo.items()}, pct_hi={str(c): x for c, x in hi.items()},
            pct_lo_min=min(lo.values()), pct_lo_max=max(lo.values()), pct_hi_min=min(hi.values()), pct_hi_max=max(hi.values()),
            changed_lo=sorted(nzlo), changed_hi=sorted(nzhi),
            order_lo_same=(rank(v, 0) == rank(base, 0)), order_hi_same=(rank(v, 1) == rank(base, 1)),
            order_lo=rank(v, 0), order_hi=rank(v, 1))
    with open(os.path.join(HERE, "range_sens_run08.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    L = ["基準（各類 10 個解之範圍，m/s）"]
    for c in range(1, 11):
        L.append("  類 %2d  %.6g – %.6g（相異 %d）" % (c, base[c][0], base[c][1], base[c][3]))
    L.append("  下界排序 %s" % out["order_base_lo"]); L.append("  上界排序 %s" % out["order_base_hi"])
    for k, d in out["variants"].items():
        L.append("-- %s" % k)
        L.append("  下界變化 %+.1f%% 至 %+.1f%%；上界變化 %+.1f%% 至 %+.1f%%" % (d["pct_lo_min"], d["pct_lo_max"], d["pct_hi_min"], d["pct_hi_max"]))
        L.append("  下界有變之類別 %s；上界有變之類別 %s；下界排序不變 %s；上界排序不變 %s" % (d["changed_lo"], d["changed_hi"], d["order_lo_same"], d["order_hi_same"]))
        for c in range(1, 11):
            L.append("    類 %2d 下界 %+.2f%%  上界 %+.2f%%" % (c, d["pct_lo"][str(c)], d["pct_hi"][str(c)]))
    open(os.path.join(HERE, "range_sens_run08.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
