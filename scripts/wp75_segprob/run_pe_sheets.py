# ruff: noqa: E501
"""WP-75 試作：五站縮圖表與貢寮全年並排比較表（只組合已畫好的圖，圖外加說明列）。

用法：python run_pe_sheets.py <圖目錄（run_pe_figs 的輸出）> <輸入目錄（含 master_fig11.jpg、seg_boundary_all.png、seg_coast_all.png）>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from lagrangian_backtracking.native_runner import figures as F  # noqa: E402

SITE_NAME = {
    "gongliao": "貢寮",
    "guishan": "龜山島",
    "hsinchu": "新竹",
    "nanwan": "南灣",
    "lienchiang": "連江",
}


def sheet(items, ncols, path, font, *, cell_w=6.0, bar=0.30, gap=0.12, dpi=150, note=None):
    """items: [(標題, 圖檔)]；各列高度取該列最高者，圖頂端對齊，標題在圖上方（圖外）。"""

    ims = [Image.open(p).convert("RGB") for _t, p in items]
    hs = [cell_w * im.height / im.width for im in ims]
    nrows = -(-len(items) // ncols)
    row_h = [max(hs[r * ncols : (r + 1) * ncols]) for r in range(nrows)]
    W = ncols * cell_w + (ncols + 1) * gap
    H = sum(row_h) + nrows * bar + (nrows + 1) * gap
    fig = plt.figure(figsize=(W, H), dpi=dpi, facecolor="white")
    y = H - gap
    for r in range(nrows):
        for c in range(ncols):
            k = r * ncols + c
            x = gap + c * (cell_w + gap)
            if k < len(items):
                fig.text(
                    x / W,
                    (y - bar * 0.5) / H,
                    items[k][0],
                    fontproperties=font,
                    fontsize=9,
                    va="center",
                    ha="left",
                )
                ax = fig.add_axes([x / W, (y - bar - hs[k]) / H, cell_w / W, hs[k] / H])
                ax.imshow(ims[k])
                ax.axis("off")
            elif note and k == len(items):
                fig.text(
                    x / W,
                    (y - bar) / H,
                    note,
                    fontproperties=font,
                    fontsize=8,
                    va="top",
                    ha="left",
                    linespacing=1.6,
                )
        y -= bar + row_h[r] + gap
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main(figs: Path, inputs: Path) -> int:
    font = F._resolve_font(None).font_properties  # noqa: SLF001
    sites = ["gongliao", "guishan", "hsinchu", "nanwan", "lienchiang"]
    note = (
        "各圖半徑上限相同（25%），圖例同單張圖。\n藍線：觀測方向機率（全年，十類合併，各 100,000 顆）；\n"
        "黑色虛線：沿邊界及海岸均勻時之期望比例；\n淡黃扇形：主要來源方向（觀測高於期望的連續方向，取前兩大）。\n站名只標在本表的圖外標題列。"
    )
    sheet(
        [(f"{SITE_NAME[s]}　全年", figs / "sites_all" / f"{s}_all_new.png") for s in sites],
        3,
        figs / "c_thumbs_all_5sites.png",
        font,
        cell_w=4.2,
        note=note,
    )
    items = [
        ("舊：主檔圖 11 十類十條線（含等深線；規則：>1/36 的連續方向）", inputs / "master_fig11.jpg"),
        ("新：全年單線＋期望虛線＋主要來源方向（觀測>期望）", figs / "a_gongliao_all_new.png"),
        ("邊界穿越分段機率圖（全年）", inputs / "seg_boundary_all.png"),
        ("海岸來源分段機率圖（全年）", inputs / "seg_coast_all.png"),
    ]
    sheet(items, 2, figs / "d_compare_gongliao_all.png", font, cell_w=6.27)
    items[0] = ("舊式全年單線（同新圖版式，無期望、無等深線）", figs / "a0_gongliao_all_old_style.png")
    sheet(items, 2, figs / "d2_compare_gongliao_all_oldstyle_single.png", font, cell_w=6.27)
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
