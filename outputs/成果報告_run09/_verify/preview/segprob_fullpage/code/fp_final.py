# -*- coding: utf-8 -*-
"""最終版：五個地點 × {邊界, 海岸} × {十類, 季節, 大小潮} 共 30 張多面板分段機率圖，附檢查與每地點縮圖總覽。
用法：python fp_final.py <輸出根目錄> <site> [site ...]
版面（核准）：單行標題、圖內數值框（每圖固定角落）、十類 4×3（色階左備用格、圖例右備用格；邊界含括號與圈號）、
季節 2×2（每列一條全年樣式直式色階）、大小潮 1×2（單條直式色階）；圖例在格網正下方置中。
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fp_lib as L0  # noqa: E402
import fp_lib4 as L3  # noqa: E402
import fp_run as FR  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

H = FR.H_FULL
LABEL_MAX_PT = 15.0
RATIO = L0.RATIO
W_REF = {"class": 1.87, "season": 2.49, "tide": 2.49}
V8 = json.load(open(Path(__file__).parent / "v8_titles.json", encoding="utf-8"))
LAYOUT = {  # name: (nrow, ncol, mode, shared_titles)
    "class": (4, 3, "slotLR", True),
    "season": (2, 2, "vrow", False),
    "tide": (1, 2, "vrow", False),
}
CIRC = "①②③④⑤"
LADDER = [(0.15, False, False), (0.12, False, False), (0.2, True, False), (0.12, True, False), (0.10, False, False), (0.10, True, False), (0.08, False, False), (0.08, True, False), (0.06, False, False), (0.04, False, False), (0.04, True, False), (0.2, False, True), (0.12, True, True)]  # 備援階梯：縮框內距 → 併圈號項為一行 → 蓋住流場範圍細灰線（僅海岸）


def v8_values(title: str):
    """同 v8data.panel_values：回傳 (組別, 總％, {圈號: ％}, 其餘％)。"""
    lines = title.split("\n")
    tot = float(re.search(r"([\d.]+)%", lines[1]).group(1))
    items, rest = {}, None
    for ln in lines[2:]:
        for mk, v in re.findall(r"([①②③④⑤])\s*([\d.]+)%", ln):
            items[mk] = v
        r = re.search(r"其餘\s*([\d.]+)%", ln)
        if r:
            rest = r.group(1)
    return lines[0], tot, items, rest


def box_values(lines):
    tot = float(re.search(r"([\d.]+)%", lines[0]).group(1))
    items, rest = {}, None
    for ln in lines[1:]:
        for mk, v in re.findall(r"([①②③④⑤])\s*([\d.]+)%", ln):
            items[mk] = v
        r = re.search(r"其餘\s*([\d.]+)%", ln)
        if r:
            rest = r.group(1)
    return tot, items, rest


# ---------------------------------------------------------------- 角落選擇
def group_info(S, kind, name):
    cb = kind == "boundary" and name == "class"
    panels, boxes, _ = L3.make_panels3(S, kind, name, 2.0, class_brackets=cb)
    keys = L3._keys(S, name)
    cnts = [S.bcount[k] if kind == "boundary" else S.ccount[k] for k in keys]
    sizes = L3._measure_boxes_cached(S, boxes)
    return cnts, max(s[0] for s in sizes), max(s[1] for s in sizes)


def select_corner(S, kind):
    """各角落於三組圖（十類、季節、大小潮；以各自預期之面板大小）之可放性：需在緣距 ≥0.07（十類）／≥0.10 in（季節、大小潮）下找到位置。"""
    info = {}
    for name in ("class", "season", "tide"):
        cnts, bw, bh = group_info(S, kind, name)
        w = W_REF[name]
        info[name] = L3.place_box(S, kind, cnts, w, w / RATIO, bw, bh, corners=L3.CORNERS, tag=name)
    rows, best = [], None
    for c in L3.CORNERS:
        res, placed, t0, dmax, mm = {}, True, True, 0.0, 9.0
        for name in ("class", "season", "tide"):
            pl = info[name].get(c)
            w = W_REF[name]
            if pl is None:
                res[name] = None
                placed = False
                continue
            res[name] = (pl["margin"], pl["tier"], round(pl["dist"] / w, 3))
            t0 = t0 and pl["tier"] == 0
            mm = min(mm, pl["margin"])
            dmax = max(dmax, pl["dist"] / w)
        rows.append([S.sd.site_id, kind, c, "全部可放" if placed else "有組別放不下", "完全避開陸地" if (placed and t0) else "需蓋小島",
                     res["class"], res["season"], res["tide"], round(dmax, 3)])
        key = (placed, t0, mm >= 0.05, -dmax)
        if best is None or key > best[0]:
            best = (key, c)
    return best[1], rows


# ---------------------------------------------------------------- 檢查
def check_fig(site, kind, name, rec):
    """回傳 (失敗清單, 檢查列)。"""
    fails, rows = [], []
    fn = f"{kind}_{name}"
    if rec["collision_count"]:
        fails += [f"碰撞：{b}" for b in rec["collisions"]]
    # 括號範圍與標籤距離（同 run_v3 之檢查）
    nb = nl = 0
    for pi, a in enumerate(rec["annotations"], start=1):
        for c in a.get("collisions", []) or []:
            fails.append(f"面板{pi} 標籤碰撞：{c}")
        for rk, t in enumerate(a.get("stretches", []), start=1):
            if t.get("violations"):
                fails.append(f"面板{pi} 集中段{rk} 違規：{t['violations']}")
            if "bracket_side" in t:
                e_s = max(abs(x - y) for x, y in zip(t["bracket_foot_start_lonlat"], t["start_lonlat"], strict=True))
                e_e = max(abs(x - y) for x, y in zip(t["bracket_foot_end_lonlat"], t["end_lonlat"], strict=True))
                ok = max(e_s, e_e) <= 1e-4 and t["bracket_span_err_m"] <= 1.0
                lp, la, cn = t["label_perp_pt"], t["label_along_pt"], t["bracket_has_corner"]
                has = lp is not None
                lab_ok = has and lp <= LABEL_MAX_PT and (cn or la <= LABEL_MAX_PT)
                rows.append([site, fn, pi, rk, "括號範圍", "ok" if ok else "FAIL", f"{max(e_s, e_e):.2e}", t["bracket_span_err_m"]])
                rows.append([site, fn, pi, rk, "標籤距離", "ok" if lab_ok else ("無標籤" if not has else "FAIL"), lp, la])
                nb += 1
                if not ok:
                    fails.append(f"面板{pi} 括號{rk} 範圍與集中段不一致")
                if has and not lab_ok:
                    fails.append(f"面板{pi} 括號{rk} 標籤離括號過遠（{lp}, {la} pt）")
                nl += 1 if has else 0
    # 數值：框內與 v8 標題逐面板比對；各項加總須等於穿越比例
    titles = V8[site].get(fn, [])
    for pi, ls in enumerate(rec["box_lines"], start=1):
        tot, items, rest = box_values(ls)
        s = round(sum(float(v) for v in items.values()) + (float(rest) if rest is not None else 0.0), 2)
        if kind == "boundary" and rest is not None and abs(s - tot) > 0.005:
            fails.append(f"面板{pi} 各項加總 {s} ≠ 總計 {tot}")
        if pi - 1 < len(titles):
            hd, tot8, items8, rest8 = v8_values(titles[pi - 1])
            same = (abs(tot - tot8) < 1e-9)
            if items8 or rest8 is not None or name != "class":
                same = same and items == items8 and (rest == rest8)
            rows.append([site, fn, pi, 0, "與 v8 數值比對", "ok" if same else "FAIL", f"{tot}/{items}/{rest}", f"{tot8}/{items8}/{rest8}"])
            if not same:
                fails.append(f"面板{pi}（{hd}）數值與 v8 不同：新 {tot},{items},{rest}；v8 {tot8},{items8},{rest8}")
        else:
            rows.append([site, fn, pi, 0, "與 v8 數值比對", "無對應", "", ""])
    return fails, rows, nb, nl


# ---------------------------------------------------------------- 總覽
def contact_sheet(outdir: Path, site: str, recs: dict):
    fontf = ImageFont.truetype(str(L0.rs.__dict__.get("FONT", "")) or FR.FONT, 22) if False else ImageFont.truetype(FR.FONT, 22)
    names = ["class", "season", "tide"]
    colw = 1000
    cells = {}
    for k in ("boundary", "coast"):
        for n in names:
            im = Image.open(outdir / f"{k}_{n}.png").convert("RGB")
            h = int(im.height * colw / im.width)
            cells[(k, n)] = im.resize((colw, h), Image.LANCZOS)
    rowh = [max(cells[("boundary", n)].height, cells[("coast", n)].height) + 50 for n in names]
    sheet = Image.new("RGB", (2 * colw + 60, sum(rowh) + 70), "white")
    d = ImageDraw.Draw(sheet)
    d.text((20, 10), f"{site}：邊界（左）／海岸（右）　十類、季節、大小潮", fill="black", font=fontf)
    y = 50
    for n, rh in zip(names, rowh):
        for j, k in enumerate(("boundary", "coast")):
            sheet.paste(cells[(k, n)], (20 + j * (colw + 20), y + 30))
            r = recs[f"{k}_{n}"]
            d.text((20 + j * (colw + 20), y + 4), f"{k}_{n}  面板 {r['panel_w_h_in'][0]:.2f}×{r['panel_w_h_in'][1]:.2f} in  圖高 {r['canvas_in'][1]:.2f} in  框角 {r['box_corner']}",
                   fill="#333333", font=ImageFont.truetype(FR.FONT, 18))
        y += rh
    sheet.save(outdir / "contact_sheet.png")


# ---------------------------------------------------------------- 主程式
def main(out_root: Path, sites) -> int:
    for site in sites:
        t0 = time.time()
        out = out_root / site
        out.mkdir(parents=True, exist_ok=False)
        S = L0.prep_site(site)
        FR.FONT = S.font.get_file()
        corner, crows = {}, []
        for kind in ("boundary", "coast"):
            corner[kind], rr = select_corner(S, kind)
            crows += rr
            print(site, kind, "corner", corner[kind], flush=True)
        recs, allfails, chk, nbr, var = {}, [], [], 0, {}
        for kind in ("boundary", "coast"):
            for name, (nr, nc, mode, sh) in LAYOUT.items():
                p = out / f"{kind}_{name}.png"
                tried = [corner[kind]]  # 放不下時全角落皆放不下（可放位置與角落無關），故只試地點角落
                rec = None
                ladder = [(0.2, False, False)] + LADDER
                for pad, comp, ringok in ladder:
                    L3.BOX_PAD, L3.COMPACT, L3.RING_OK = pad, comp, ringok
                    for c in tried:
                        try:
                            rec = L3.render_fit3(S, kind, name, p, nrow=nr, ncol=nc, H=H, mode=mode, corner=c,
                                                 class_brackets=(kind == "boundary" and name == "class"), shared_titles=sh)
                            break
                        except RuntimeError as e:
                            print("  retry", f"pad={pad} compact={comp} ring_ok={ringok}", c, e, flush=True)
                            if p.exists():
                                p.unlink()
                    if rec is not None:
                        break
                L3.BOX_PAD, L3.COMPACT, L3.RING_OK = 0.2, False, False
                assert rec is not None, (site, kind, name)
                rec["corner_site"] = corner[kind]
                var[f"{kind}_{name}"] = dict(corner=rec["box_corner"], pad=rec["box_pad"], compact=rec["box_compact"], ring_ok=rec["box_ring_ok"], tier=rec["box_land_tier"], margin=rec["box_margin_in"], islets=rec["box_islets_covered"], panel_w=round(rec["panel_w_h_in"][0], 3), canvas_h=round(rec["canvas_in"][1], 3))
                fails, rows, nb, nl = check_fig(site, kind, name, rec)
                chk += rows
                nbr += nb
                recs[f"{kind}_{name}"] = {k: v for k, v in rec.items() if k != "text_audit"}
                recs[f"{kind}_{name}"]["text_summary"] = rec["text_summary"]
                allfails += [f"{kind}_{name}: {f}" for f in fails]
                print(site, f"{kind}_{name}", [round(v, 3) for v in rec["panel_w_h_in"]], [round(v, 2) for v in rec["canvas_in"]],
                      f"corner={rec['box_corner']} m={rec['box_margin_in']} coll={rec['collision_count']} fails={len(fails)}",
                      f"{time.time() - t0:.0f}s", flush=True)
        (out / "manifest.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        with (out / "checks.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["site", "figure", "panel", "rank", "check", "result", "value_a", "value_b"])
            w.writerows(chk)
        with (out / "corner_choice.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["site", "kind", "corner", "placed", "land", "class(margin,tier,dist/w)", "season", "tide", "max_dist_over_w"])
            w.writerows(crows)
        (out / "result.json").write_text(json.dumps(dict(site=site, corner=corner, variants=var, fails=allfails, n_checks=len(chk), n_brackets=nbr),
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
        contact_sheet(out, site, recs)
        print(site, "DONE fails=", len(allfails), allfails[:10], f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), sys.argv[2:]))
