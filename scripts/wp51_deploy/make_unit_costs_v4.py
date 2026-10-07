#!/usr/bin/env python3
"""WP-51：以 run04 的逐段實測耗時，為 v4 計畫檔產生 ``native-run --unit-costs`` 的成本覆寫檔。

背景：計畫檔內建的 ``manifest.extra.unit_costs`` 是「流場小時數 × 節點數 × 層數 × 全域常數」的
估計，不含各區域的實測單位成本差異（run04 實測：東北 25–30、南灣 10–14、連江 1.7–1.9、新竹
2.5 秒／流場小時；新竹估計值卻排在南灣之前）。動態佇列依估計值由大到小提交，結果最長的南灣分段
被排到最後，造成約 20 小時的牆鐘。本工具改用 run04 同一案例、同一區域、同一時序位置（segment_index
對應到達時刻分位數）的實測耗時，取前後各 ``window`` 段的平均（平滑；v4 的分段邊界與 v3 不同，不能
逐段硬對）再乘 ``--scale``（per_member 流場小時約 +9%）得到預估成本，只用來決定提交順序。

只輸出排序用的數字，不改任何工作單元、粒子、seed 或物理結果。

用法：make_unit_costs_v4.py <native-plan.json> <segment_execution_records.csv> <案例 ID> <輸出 JSON>
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

_UNIT_ID = re.compile(r"^hourmajor::(?P<domain>.+)__seg(?P<index>\d+)of(?P<count>\d+)__[0-9a-f]+$")


def build_unit_costs(
    plan_unit_ids: list[str],
    timing_rows: list[dict[str, str]],
    case_id: str,
    *,
    window: int = 2,
    scale: float = 1.09,
) -> dict[str, float]:
    """回傳 ``{unit_id: 預估成本（秒）}``；任一單元對不到實測資料即 ``ValueError``。"""

    elapsed: dict[tuple[str, int], float] = {}
    seg_counts: dict[str, int] = {}
    for row in timing_rows:
        if row["experiment_case_id"] != case_id:
            continue
        domain = row["flow_domain_id"]
        elapsed[(domain, int(row["segment_index"]))] = float(row["elapsed_seconds_this_invocation"])
        seg_counts[domain] = int(row["segment_count"])
    if not elapsed:
        raise ValueError(f"實測資料沒有案例 {case_id!r}")
    costs: dict[str, float] = {}
    for unit_id in plan_unit_ids:
        match = _UNIT_ID.match(unit_id)
        if match is None:
            raise ValueError(f"無法解析小時主軸分段單元 ID：{unit_id!r}")
        domain, index, count = match["domain"], int(match["index"]), int(match["count"])
        if seg_counts.get(domain) != count:
            raise ValueError(
                f"{domain} 的段數與實測資料不同（計畫 {count}、實測 {seg_counts.get(domain)}），無法對應時序位置"
            )
        neighbours = [
            elapsed[(domain, j)]
            for j in range(max(0, index - window), min(count - 1, index + window) + 1)
            if (domain, j) in elapsed
        ]
        if not neighbours:
            raise ValueError(f"{domain} 第 {index} 段在實測資料中沒有對應")
        costs[unit_id] = scale * sum(neighbours) / len(neighbours)
    return costs


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__)
        return 2
    plan_path, csv_path, case_id, out_path = Path(argv[1]), Path(argv[2]), argv[3], Path(argv[4])
    if out_path.exists():
        print(f"拒絕覆寫：{out_path}")
        return 2
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    unit_ids = list(plan["manifest"]["extra"]["unit_costs"])  # 與 native-run 讀到的單元集合相同
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    costs = build_unit_costs(unit_ids, rows, case_id)
    out_path.write_text(json.dumps(costs, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    total = sum(costs.values()) / 3600.0
    print(f"{case_id}：{len(costs)} 個單元，預估機時合計 {total:.1f} 小時 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
