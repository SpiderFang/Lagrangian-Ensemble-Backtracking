# -*- coding: utf-8 -*-
"""整頁版多面板分段機率圖（貢寮試作）。用法：python fp_run.py <輸出目錄（全新）> [site]"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fp_lib as L  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

PAGE_W, PAGE_H = 11906 / 1440.0, 16838 / 1440.0  # in（A4）
MARGIN = 1.0  # 上下左右 1440 twips
TEXT_W, TEXT_H = PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN  # 6.268 × 9.692
CAP3_PT = 3 * 27 + 2 + 8  # 圖說 3 行：行距 27 pt（同 check 之 K 式）＋段前 2 pt＋段後 8 pt
BEFORE_PT = 12  # 圖段落段前 240 twips
SAFETY_PT = 5
H_FULL = (TEXT_H * 72 - BEFORE_PT - CAP3_PT - SAFETY_PT) / 72.0

# (檔名, kind, name, nrow, ncol, H, slot, class_brackets, shared_titles, 備註)
SPECS = []
for kind in ("boundary", "coast"):
    SPECS += [
        (f"{kind}_season_2x2", kind, "season", 2, 2, H_FULL, False, False, False, "季節 2×2（建議）"),
        (f"{kind}_season_4x1", kind, "season", 4, 1, H_FULL, False, False, False, "季節 4×1 疊排（整頁）"),
        (f"{kind}_tide_2x1", kind, "tide", 2, 1, H_FULL, False, False, False, "大小潮 2×1 疊排（整頁，建議）"),
        (f"{kind}_tide_1x2", kind, "tide", 1, 2, H_FULL, False, False, False, "大小潮 1×2 並排（約半頁）"),
        (f"{kind}_class_4x3slot", kind, "class", 4, 3, H_FULL, True, False, True, "十類 4列×3欄，色階與圖例放備用格（建議）"),
        (f"{kind}_class_5x2", kind, "class", 5, 2, H_FULL, False, False, True, "十類 5列×2欄，色階在右側"),
    ]
SPECS.append(("boundary_class_4x3slot_brackets", "boundary", "class", 4, 3, H_FULL, True, True, True, "十類邊界 4列×3欄＋括號與圈號（變體）"))
SPECS.append(("boundary_class_5x2_brackets", "boundary", "class", 5, 2, H_FULL, False, True, True, "十類邊界 5列×2欄＋括號與圈號（變體）"))


def mockup(png: Path, out: Path, fig_h: float, note: str, cap_lines: int = 3) -> dict:
    sc = 100.0  # px / in
    pw, ph = int(round(PAGE_W * sc)), int(round(PAGE_H * sc))
    foot = 64
    img = Image.new("RGB", (pw + 2, ph + foot + 2), "#e8e8e8")
    d = ImageDraw.Draw(img)
    d.rectangle([1, 1, pw, ph], fill="white", outline="#555555")
    tx0, ty0, tx1, ty1 = MARGIN * sc, MARGIN * sc, (PAGE_W - MARGIN) * sc, (PAGE_H - MARGIN) * sc
    d.rectangle([tx0, ty0, tx1, ty1], outline="#2b6cb0", width=1)  # 文字區範圍
    f = ImageFont.truetype(FONT, 13)
    fs = ImageFont.truetype(FONT, 11)
    im = Image.open(png).convert("RGB")
    wpx, hpx = int(round(L.W_IN * sc)), int(round(fig_h * sc))
    y = ty0 + BEFORE_PT / 72.0 * sc
    img.paste(im.resize((wpx, hpx), Image.LANCZOS), (int(tx0), int(y)))
    d.rectangle([tx0, y, tx0 + wpx, y + hpx], outline="#c53030")  # 圖框
    y2 = y + hpx
    # 圖說 3 行佔位
    cap_pt = cap_lines * 27 + 2 + 8
    for k in range(cap_lines):
        yy = y2 + (2 + 27 * k + 6) / 72.0 * sc
        d.rectangle([tx0 + 20, yy, tx1 - 20 - (0 if k < cap_lines - 1 else 90), yy + 9], fill="#c9c9c9")
    d.text((tx0 + 22, y2 + (2 + 27 * cap_lines + 1) / 72.0 * sc), "圖說（3 行佔位）", fill="#666666", font=fs)
    used_pt = BEFORE_PT + fig_h * 72 + cap_pt
    fill = used_pt / (TEXT_H * 72)
    d.text((6, ph + 8), f"{note}｜A4 {PAGE_W:.2f}×{PAGE_H:.2f} in，文字區 {TEXT_W:.2f}×{TEXT_H:.2f} in（藍框）｜圖 {L.W_IN:.2f}×{fig_h:.2f} in（紅框，原尺寸不縮放）",
           fill="black", font=f)
    d.text((6, ph + 8 + 22), f"圖段前 12 pt ＋ 圖 ＋ 圖說 3 行（{cap_pt} pt）＝ {used_pt / 72:.2f} in，佔文字區高度 {fill * 100:.1f}%",
           fill="black", font=f)
    img.save(out)
    return {"fill": fill, "used_in": used_pt / 72.0}


def main(out_root: Path, site: str) -> int:
    global FONT
    out = out_root / site
    out.mkdir(parents=True, exist_ok=False)
    S = L.prep_site(site)
    FONT = S.font.get_file()
    print("site ready", flush=True)
    rows, recs = [], {}
    for fn, kind, name, nr, nc, H, slot, cb, sh, note in SPECS:
        p = out / f"{fn}.png"
        rec = L.render_grid(S, kind, name, p, nrow=nr, ncol=nc, H=H, slot=slot, class_brackets=cb, shared_titles=sh)
        mk = mockup(p, out / f"mockup_{fn}.png", rec["canvas_in"][1], note)
        recs[fn] = {k: v for k, v in rec.items() if k not in ("text_audit",)}
        recs[fn]["mockup"] = mk
        recs[fn]["note"] = note
        rows.append([fn, note, f"{nr}x{nc}", f"{rec['panel_w_h_in'][0]:.2f}", f"{rec['panel_w_h_in'][1]:.2f}", rec["limit"],
                     f"{rec['canvas_in'][0]:.2f}", f"{rec['canvas_in'][1]:.2f}", f"{mk['fill'] * 100:.1f}", rec["title_max_lines"],
                     rec["text_summary"]])
        print(fn, rec["panel_w_h_in"], rec["canvas_in"], f"fill={mk['fill']:.3f}", flush=True)
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "layout", "rows_x_cols", "panel_w_in", "panel_h_in", "limited_by", "canvas_w_in", "canvas_h_in", "page_fill_pct", "title_max_lines", "text_audit"])
        w.writerows(rows)
    (out / "manifest.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # 候選版面之面板大小（純幾何計算，含未出圖者）
    cand = []
    for nm, n, th in (("季節邊界", 4, 0.60), ("季節海岸", 4, 0.46), ("大小潮", 2, 0.46), ("十類（無括號）", 10, 0.30), ("十類邊界（括號）", 10, 0.60)):
        shapes = {4: [(2, 2), (4, 1), (1, 4)], 2: [(1, 2), (2, 1)], 10: [(2, 5), (5, 2), (4, 3), (3, 4), (10, 1)]}[n]
        for r, c in shapes:
            for slot in ([False, True] if n == 10 else [False]):
                if slot and r * c - n < 2:
                    continue
                g = L.geometry(r, c, n, th, H=H_FULL, slot=slot)
                cand.append([nm, f"{r}列×{c}欄", "備用格" if slot else "右側色階", f"{g.w:.2f}", f"{g.h:.2f}", g.limit, f"{g.fh:.2f}", f"{(g.fh * 72 + BEFORE_PT + CAP3_PT) / (TEXT_H * 72) * 100:.1f}"])
    with (out / "layout_candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "rows_x_cols", "colorbar", "panel_w_in", "panel_h_in", "limited_by", "canvas_h_in", "page_fill_pct"])
        w.writerows(cand)
    print("H_FULL", H_FULL, "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "gongliao"))
