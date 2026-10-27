# ruff: noqa: E501
"""WP-75 試作：幾何期望極座標圖（單張、季節 2×2、五站縮圖表）。

用法：python run_pe_figs.py <輸出目錄（全新）>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pe_data as D  # noqa: E402
import wp75_common as C  # noqa: E402

from lagrangian_backtracking.native_runner import figures as F  # noqa: E402
from lagrangian_backtracking.native_runner import polar_expected as PE  # noqa: E402
from lagrangian_backtracking.native_runner import polar_expected_figures as PF  # noqa: E402
from lagrangian_backtracking.native_runner import polar_source_figures as psf  # noqa: E402

SEASON_LABEL = {"DJF": "冬", "MAM": "春", "JJA": "夏", "SON": "秋"}


def top2_runs(g) -> list[list[int]]:
    return PE.ranked_runs(g["new"], g["obs"])[:2]


def all_panel(g, *, with_exp: bool = True, with_runs: bool = True, label_prefix: str = "觀測：") -> PF.Panel:
    ln = PF.Line(f"{label_prefix}全年（十類合併）　{g['n']:,} 顆", PF.SINGLE_COLOR, g["obs"])
    return PF.Panel("", [ln], g["exp"] if with_exp else None, top2_runs(g) if with_runs else [])


def main(out: Path) -> int:
    out.mkdir(parents=True)
    (out / "sites_all").mkdir()
    data = D.build_all()
    land = F._load_coastline(C.COASTLINE)  # noqa: SLF001
    fr = F._resolve_font(None)  # noqa: SLF001
    font, cjk = fr.font_properties, fr.cjk_available
    # 共用半徑上限：全年一組、季節一組（同時涵蓋觀測與期望，跨五站）
    peak_all = max(
        max(float(d["groups"]["all"]["obs"].max()), float(d["groups"]["all"]["exp"].max()))
        for d in data.values()
    )
    peak_season = max(
        max(float(d["groups"][f"season_{k}"]["obs"].max()), float(d["groups"][f"season_{k}"]["exp"].max()))
        for d in data.values() for k in D.SEASONS
    )  # fmt: skip
    top_a, ticks_a = PF.pick_scale(peak_all)
    top_s, ticks_s = PF.pick_scale(peak_season)
    print("scale all", top_a, ticks_a, "peak", peak_all, "| season", top_s, ticks_s, "peak", peak_season)
    for sid, d in data.items():
        g = d["groups"]["all"]
        PF.render_single(
            all_panel(g),
            d["anchor"],
            land,
            top_a,
            ticks_a,
            font,
            cjk,
            out / "sites_all" / f"{sid}_all_new.png",
        )
    # (a) 貢寮 全年
    d = data["gongliao"]
    PF.render_single(
        all_panel(d["groups"]["all"]),
        d["anchor"],
        land,
        top_a,
        ticks_a,
        font,
        cjk,
        out / "a_gongliao_all_new.png",
    )
    # 舊規則版式（單線、無期望、無標示）：與主檔同法，供並排比較
    PF.render_single(all_panel(d["groups"]["all"], with_exp=False, with_runs=False, label_prefix=""), d["anchor"], land, top_a, ticks_a, font, cjk,
                     out / "a0_gongliao_all_old_style.png", legend_h=0.42)  # fmt: skip
    # (b) 貢寮 季節 2×2
    panels, legend_lines = [], []
    for k in D.SEASONS:
        g = d["groups"][f"season_{k}"]
        col = psf._SEASON_COLORS[k]  # noqa: SLF001
        ln = PF.Line(f"{g['label']}　{g['n']:,} 顆", col, g["obs"])
        panels.append(PF.Panel(f"{g['label']}　{g['n']:,} 顆", [ln], g["exp"], top2_runs(g)))
        legend_lines.append(ln)
    peak_g = max(max(float(p.exp_pct.max()), float(p.lines[0].obs_pct.max())) for p in panels)
    top_g, ticks_g = PF.pick_scale(peak_g)
    print("gongliao season scale", top_g, ticks_g, "peak", peak_g)
    PF.render_grid(panels, 2, d["anchor"], land, top_g, ticks_g, font, cjk, out / "b_gongliao_season_new.png", legend_lines,
                   legend_title="觀測（沉底時刻的季節）", compact=True)  # fmt: skip
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
