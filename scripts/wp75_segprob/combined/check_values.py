# -*- coding: utf-8 -*-
"""核對 v37（括號旁／引線旁 xx.xx%）與 v36（圈號＋數值框）標示的數值：
v36 數值框內每個 ①②…／ⓐⓑ… 後的百分比，必須全部出現在 v37 對應面板的標示中（逐面板、逐項、字串完全相同）；
v37 多出來的項目（v36 只畫括號、沒編號的第 4 個以後集中段）另列。

用法：python check_values.py <v36 圖資料夾（該地點）> <v37 圖資料夾（該地點）>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def load(d: Path):
    fs = sorted(d.glob("manifest_combined_*.json"))
    out = {}
    for f in fs:
        out.update(json.loads(f.read_text(encoding="utf-8")))
    return out


def main(old_dir: str, new_dir: str) -> int:
    old, new = load(Path(old_dir)), load(Path(new_dir))
    bad = 0
    for fig in ("season", "tide", "class"):
        if fig not in old or fig not in new:
            continue
        titles = old[fig]["panel_titles"]
        for i, (bl, ann) in enumerate(zip(old[fig]["box_lines"], new[fig]["annotations"], strict=True)):
            want = [m.group(1) for ln in bl for m in [re.match(r"^[①②③④⑤ⓐⓑⓒⓓⓔⓕⓖ]\s*([0-9.]+)%$", ln)] if m]
            got_b = [st["shown_pct"] for st in ann["stretches"][: ann["n_boundary"]]]
            got_c = [st["pct"] for st in ann["stretches"][ann["n_boundary"]:]]
            got = got_b + got_c
            missing = [w for w in want if w not in got]
            extra = [g for g in got if g not in want]
            tag = "OK " if not missing else "不符"
            bad += bool(missing)
            print(f"{tag} {fig:6s} {titles[i].splitlines()[0]:　<8s} v36 框內 {want} → v37 標示 {got}" + (f"  缺：{missing}" if missing else "") + (f"  v37 新增：{extra}" if extra else ""))
    print("全部 v36 數值皆出現於 v37" if not bad else f"有 {bad} 個面板不符")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
