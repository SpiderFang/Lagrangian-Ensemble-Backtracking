#!/usr/bin/env python3
"""WP-53：依 run05 實測的單元耗時，為 run07 的提交順序產生 unit-costs.json（最長優先，只影響動態佇列順序，不改物理結果）。

單元 ID 形如 ``hourmajor::<flow_domain>__seg<i>of<n>__<hash>``；hash 隨母體改變，所以以（案例、流場區域、分段序號）對應
run05 的實測耗時（timing CSV，欄位同 mk_timing_run05.py 輸出）。run05 缺的單元以同案例同區域已測單元的中位數補上。
用法：make_unit_costs_v6.py <timing.csv> <run07 目錄>
"""
import csv, json, re, statistics, sys
from pathlib import Path

RX = re.compile(r"^hourmajor::(?P<d>.+)__seg(?P<i>\d+)of(?P<n>\d+)__[0-9a-f]+$")


def main() -> int:
    timing = Path(sys.argv[1])
    run = Path(sys.argv[2])
    meas = {}
    for r in csv.DictReader(open(timing)):
        meas[(r["experiment_case_id"], r["flow_domain_id"], int(r["segment_index"]), int(r["segment_count"]))] = float(r["elapsed_seconds_this_invocation"])
    for case in ("no_stokes", "finite_depth_stokes"):
        plan = json.loads((run / case / "native-plan.json").read_text(encoding="utf-8"))
        units = list(plan["hour_major_members"].keys())
        costs, filled = {}, 0
        for uid in units:
            g = RX.match(uid)
            key = (case, g["d"], int(g["i"]), int(g["n"]))
            if key in meas:
                costs[uid] = meas[key]
            else:
                same = [v for k, v in meas.items() if k[0] == case and k[1] == g["d"]]
                costs[uid] = statistics.median(same)
                filled += 1
        out = run / case / "unit-costs.json"
        if out.exists():
            raise SystemExit(f"{out} 已存在，拒絕覆寫")
        out.write_text(json.dumps(costs, indent=1, sort_keys=True), encoding="utf-8")
        print(case, "units", len(units), "實測", len(units) - filled, "以中位數補", filled, "總和(h) %.0f" % (sum(costs.values()) / 3600))
    return 0


if __name__ == "__main__":
    sys.exit(main())
