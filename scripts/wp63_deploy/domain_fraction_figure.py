#!/usr/bin/env python3
"""WP-63：呼叫 native_runner.domain_fraction，對一個案例的 postprocess 產出「仍在流場區域內比例」曲線圖。

用法：domain_fraction_figure.py POSTPROCESS_DIR OUT_FULL OUT_NOTEXT [--max-days 180] [--marks 90 180] [--dpi 300]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lagrangian_backtracking.native_runner.domain_fraction import build_domain_fraction


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("postprocess_dir", type=Path)
    ap.add_argument("out_full", type=Path)
    ap.add_argument("out_notext", type=Path)
    ap.add_argument("--max-days", type=float, default=180.0)
    ap.add_argument("--marks", type=float, nargs="+", default=[90.0, 180.0])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--font-path", type=Path, default=None)
    ap.add_argument("--case-text", default="")
    args = ap.parse_args()
    out = build_domain_fraction(args.postprocess_dir, args.out_full, args.out_notext, max_days=args.max_days,
                                marks_days=args.marks, dpi=args.dpi, font_path=args.font_path, case_text=args.case_text)
    print(f"完成：{len(out)} 張圖")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
