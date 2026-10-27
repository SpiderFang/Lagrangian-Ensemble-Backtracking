# ruff: noqa: E501
"""WP-75：海岸來源分類（本島／島嶼）以最大餘數法湊成總海岸來源比例，並與主檔現行數字對照。

用法：python lr_coast.py <逐段表目錄>
"""

import csv
import sys
from collections import defaultdict

from lagrangian_backtracking.native_runner import segprob_annot as SA
from lagrangian_backtracking.native_runner import segprob_figures as SF

T = sys.argv[1].rstrip("/") + "/"  # 逐段表目錄（coast_segments_<site>.csv 所在）
master = {
    "gongliao": {"臺灣本島": "24.50", "龜山島": "0.88", "其他島嶼": "1.22"},
    "guishan": {"臺灣本島": "11.96", "龜山島": "9.56", "其他島嶼": "0.13"},
    "nanwan": {"臺灣本島": "29.94", "小琉球": "0.58", "蘭嶼": "0.54", "其他島嶼": "0.03"},
    "lienchiang": {"福建沿岸（含近岸島嶼）": "29.68", "馬祖各島": "14.26"},
    "hsinchu": {"臺灣本島": "2.23"},
}
for s in ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang"):
    d = defaultdict(int)
    for r in csv.DictReader(open(T + f"coast_segments_{s}.csv", encoding="utf-8-sig")):  # noqa: SIM115
        c, isl, n = r["class"], r["island_name"], int(r["count_all"])
        if s in ("gongliao", "guishan"):
            k = "臺灣本島" if c == "臺灣本島" else ("龜山島" if isl == "龜山島" else "其他島嶼")
        elif s == "nanwan":
            k = "臺灣本島" if c == "臺灣本島" else (isl if isl in ("小琉球", "蘭嶼") else "其他島嶼")
        elif s == "lienchiang":
            k = "福建沿岸（含近岸島嶼）" if c == "福建沿岸" else "馬祖各島"
        else:
            k = "臺灣本島"
        d[k] += n
    tot = sum(d.values())
    lr = SA.largest_remainder(dict(d), tot, 100000, 2)
    print(s, "海岸合計", SF.pct2(tot, 100000))
    for k in d:
        print(
            "   ", k, d[k], "half-up", SF.pct2(d[k], 100000), "LR", f"{lr[k]:.2f}", "主檔", master[s].get(k)
        )
