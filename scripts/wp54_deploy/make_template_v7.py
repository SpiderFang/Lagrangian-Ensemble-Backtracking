#!/usr/bin/env python3
"""由 v6 範本產生 v7（十類逐粒子沉降速度）設定範本。

用法：make_template_v7.py [--check] [--v6-template PATH] [--repo-config PATH] [--output PATH]

只做三件事，其餘逐字複製 v6 範本（所以擴散、成員數、age_sampling_scope 等全部不變）：
  1. 把 physics.settling.material_classes 整段換成本 repo 正式設定檔（含 10 個候選速度與新代表值）；
  2. 在 physics.settling 加上 velocity_sampling: class_candidates_balanced；
  3. 材質區塊以外一律取自 v6 範本；兩邊的 material_id 序列必須相同，否則中止。

安全：輸出檔已存在就中止（絕不覆蓋任何既有範本）；v6 範本與 repo 設定檔只讀。--check 只驗證不寫檔。
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

W = Path.home() / "Workspace"
DEFAULT_V6 = W / "lbt-inputs/formal/template-v6-per-member.yaml"
DEFAULT_OUT = W / "lbt-inputs/formal/template-v7-class-velocity.yaml"
DEFAULT_REPO = Path(__file__).resolve().parents[2] / "configs/lagrangian_backtracking.formal_h30_h60_h90_m10.yaml"

START = "    material_classes:\n"
END_MARK = "\n  horizontal_diffusion:\n"
UNIT_LINE = "    velocity_unit: m_per_s\n"
SAMPLING_LINES = (
    "    # v7：每個粒子從所屬類別的 10 個候選沉降速度各抽 1/10（逐站位、逐類別完全均衡），\n"
    "    # 代表值 settling_velocity_mps 仍作為類別分組鍵與敘述用量。\n"
    "    velocity_sampling: class_candidates_balanced\n"
)


def split(text: str, label: str) -> tuple[str, str, str]:
    i = text.index("\n  settling:\n")
    j = text.index(START, i) + len(START)
    k = text.index(END_MARK, j)
    return text[:j], text[j:k], text[k:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--v6-template", type=Path, default=DEFAULT_V6)
    ap.add_argument("--repo-config", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()

    v6 = a.v6_template.read_text(encoding="utf-8")
    repo = a.repo_config.read_text(encoding="utf-8")
    v6_head, v6_mat, v6_tail = split(v6, "v6")
    _, rp_mat, _ = split(repo, "repo")
    # repo 正式設定檔只提供材質區塊（它的其他區段是較舊的版本，例如潮汐政策與 age_sampling_scope 不同，不可混入）。
    ids = lambda t: re.findall(r"material_id: (\S+)", t)
    if ids(v6_mat) != ids(rp_mat) or len(ids(rp_mat)) != 10:
        print("中止：v6 範本與 repo 設定檔的 material_id 序列不同", file=sys.stderr)
        return 3
    if v6_head.count(UNIT_LINE) != 1:
        print("中止：找不到唯一的 velocity_unit 行", file=sys.stderr)
        return 3
    if rp_mat.count("settling_velocity_candidates_mps:") != 10:
        print("中止：repo 設定檔的 material_classes 沒有 10 個 settling_velocity_candidates_mps", file=sys.stderr)
        return 3
    head = v6_head.replace(UNIT_LINE, UNIT_LINE + SAMPLING_LINES)
    out = head + rp_mat + v6_tail
    sha = hashlib.sha256(out.encode("utf-8")).hexdigest()
    print(f"v6 範本 SHA-256：{hashlib.sha256(v6.encode('utf-8')).hexdigest()}")
    print(f"v7 範本 SHA-256：{sha}（{len(out.splitlines())} 行）")
    if a.check:
        print("檢查模式：未寫檔")
        return 0
    if a.output.exists():
        print(f"中止：{a.output} 已存在，為避免覆蓋而不寫入", file=sys.stderr)
        return 2
    a.output.write_text(out, encoding="utf-8")
    print(f"已寫入 {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
