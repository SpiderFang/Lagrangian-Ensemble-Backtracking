"""WP-73：仍在追蹤比例曲線——以舊版面重畫（dpi 40、暫存目錄）並輸出每張圖的線條資料指紋，供與報告版（fs8）manifest 比對（數值未變）。
用法：python curve_fingerprint_old.py POSTPROCESS_DIR OUT_JSON"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from lagrangian_backtracking.native_runner import domain_fraction as df
from lagrangian_backtracking.native_runner.figures import _Labeler, _resolve_font


def main(argv: list[str]) -> int:
    post, out = Path(argv[0]), Path(argv[1])
    curves = df.load_site_curves(post / "departures.parquet", max_days=180.0)
    fr = _resolve_font(None)
    t = _Labeler(fr.cjk_available)
    res: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for sid, c in curves.items():
            for ys, suffix in (("linear", ""), ("log", "_logy")):
                rec = df.render_site_figure(c, Path(tmp) / sid / f"domain_fraction_vs_backtrack_time{suffix}.png", mode="none",
                                            font=fr.font_properties, t=t, marks_days=(90.0, 180.0), dpi=40, yscale=ys)
                res[f"{sid}/domain_fraction_vs_backtrack_time{suffix}.png"] = rec["data_fingerprint"]
    with out.open("x", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
