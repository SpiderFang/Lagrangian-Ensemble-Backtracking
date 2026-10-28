# ruff: noqa: E501
"""WP-75：單一地點的全部圖（單張、季節、大小潮、十類；邊界與海岸）與註記表格列。

* 邊界單張：邊小計在邊界內側、集中段括號在外側。
* 邊界多面板：面板標題第三行寫四邊小計（整數、最大餘數法）、不畫括號。
* 海岸單張：引線＋％；連江另有南竿北竿放大小圖，兩個集中段的引線標籤放在小圖內。
* 海岸多面板（季節、大小潮）：小圈號 ①②③ 標在開放海面，數值寫在面板標題最後一行；十類不標。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from lagrangian_backtracking.native_runner import segprob_annot as SA
from lagrangian_backtracking.native_runner import segprob_figures as SF

SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")  # 同舊版圓形圖與主檔文字的順序：冬、春、夏、秋
TIDE_ORDER = ("spring_proxy", "neap_proxy")  # 大潮、小潮
KIND_ZH = {"boundary": "穿越", "coast": "海岸來源"}
INSET_HINT_LC = (119.88, 120.06, 26.10, 26.26)  # 連江：南竿＋北竿所在範圍（用來找上色線段）
INSET_MIN_MAG = 4.0  # 放大小圖至少 4 倍


def inset_window_for(sd, dr, seg_counts, hint=INSET_HINT_LC, pad_deg=0.006):
    """南竿＋北竿放大小圖視窗：包住提示範圍內全部上色海岸線與星號，外加邊距，再向外取到 0.01°。"""

    pts = []
    for i, parts in enumerate(dr.c_lines):
        if seg_counts[i] <= 0:
            continue
        allp = np.vstack(parts)
        c = allp.mean(axis=0)
        if hint[0] <= c[0] <= hint[1] and hint[2] <= c[1] <= hint[3]:
            pts.append(allp)
    pts.append(np.array([sd.anchor]))
    xy = np.vstack(pts)
    lo = np.floor((xy.min(axis=0) - pad_deg) * 100) / 100
    hi = np.ceil((xy.max(axis=0) + pad_deg) * 100) / 100
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


def _in_window(lonlat, w):
    return w[0] <= lonlat[0] <= w[1] and w[2] <= lonlat[1] <= w[3]


def render_site(
    sd, dr, grp, label, mats, bcount, ccount, gn, out: Path, font, *, only=None, lienchiang_inset=True, gaps=True,
):  # fmt: skip
    """輸出該地點的全部圖；回傳 ``(manifest 紀錄, 集中段表格列, 多面板標題數值行紀錄, 附表列（數字對照與零星段）)``。``only`` 可限定 (kind, name) 清單。"""

    out.mkdir(parents=True, exist_ok=True)
    recs: dict = {}
    stretch_rows: list = []
    pct_diff: list = []  # 圖上數字（最大餘數法）與各自四捨五入（pct2）的對照
    tiny_rows: list = []  # 未標示的零星段（單段空檔）位置
    title_log: list = []  # 多面板標題數值行：用了哪個候選、有幾個集中段

    def tot(k, kind):
        cnt = bcount[k] if kind == "boundary" else ccount[k]
        return cnt, SF.pct2(int(cnt.sum()), gn[k])

    def rows_from(kind, k, rec, where):
        for rank, s in enumerate((x for x in rec["stretches"] if x.get("style") != "gap"), start=1):
            stretch_rows.append([sd.site_id, kind, k, rank, s["pct"], s["count"], len(s["segments"]),
                                 f"{s['start_lonlat'][0]:.4f}", f"{s['start_lonlat'][1]:.4f}",
                                 f"{s['end_lonlat'][0]:.4f}", f"{s['end_lonlat'][1]:.4f}", s["segments"][0], s["segments"][-1],
                                 s.get("marker", ""), where])  # fmt: skip

    for kind in ("boundary", "coast"):
        for name, keys, layout in (
            ("all", ["all"], None),
            ("season", [f"season_{k}" for k in SEASON_ORDER], SF.LAYOUT_2x2),
            ("tide", [f"tide_{k}" for k in TIDE_ORDER], SF.LAYOUT_1x2),
            ("class", [f"class_{m}" for m in mats], SF.LAYOUT_2x5),
        ):
            if only is not None and (kind, name) not in only:
                continue
            if name == "all":
                cnt, t = tot("all", kind)
                n = gn["all"]
                total_text = f"{'穿越比例' if kind == 'boundary' else '海岸來源比例'} {t}%"
                if kind == "boundary" and gaps:
                    tt = SA.boundary_gap_summary(sd, cnt, n)["tiny_text"]
                    if tt:
                        total_text += "\n" + tt
                kw: dict = {}
                if kind == "boundary":

                    def ann(ax, fig, f8, avoid, cnt=cnt, n=n):
                        r = SA.annotate_boundary(
                            ax, fig, sd, dr, cnt, n, font=f8, avoid=avoid, mode="label", gaps=gaps
                        )
                        rows_from("boundary", "all", r, "main")
                        gi = r.get("gap_info")
                        if gi:
                            for rk, (a, b) in enumerate(
                                zip(gi["main_pct2"], gi["main_pct"], strict=True), start=1
                            ):
                                pct_diff.append([sd.site_id, "boundary_all", "全年", f"主括號{rk}", a, b])
                            for rk, (a, b) in enumerate(
                                zip(gi["gap_pct2"], gi["gap_pct"], strict=True), start=1
                            ):
                                pct_diff.append([sd.site_id, "boundary_all", "全年", f"空檔{rk}", a, b])
                            for ti in gi["tiny_info"]:
                                tiny_rows.append(
                                    [
                                        sd.site_id,
                                        ti["seg_id"],
                                        ti["pct2"],
                                        f"{ti['lon']:.4f}",
                                        f"{ti['lat']:.4f}",
                                        ti["prev"],
                                        ti["next"],
                                    ]
                                )
                        return r

                else:
                    st = SA.coast_stretches(sd, cnt, n)
                    iw = None
                    if sd.site_id == "lienchiang" and lienchiang_inset:
                        iw = inset_window_for(sd, dr, cnt)
                        kw = {"inset_window": iw, "inset_min_mag": INSET_MIN_MAG}
                    st_in = [s for s in st if iw is not None and _in_window(SA.stretch_mid_lonlat(sd, s), iw)]
                    st_main = [s for s in st if s not in st_in]

                    def ann(ax, fig, f8, avoid, cnt=cnt, n=n, st_main=st_main):
                        r = SA.annotate_coast_labels(
                            ax, fig, sd, dr, cnt, st_main, n, font=f8, star_lonlat=sd.anchor, avoid=avoid
                        )
                        rows_from("coast", "all", r, "main")
                        return r

                    if st_in:

                        def ann_in(iax, fig, f8, avoid, cnt=cnt, n=n, st_in=st_in):
                            r = SA.annotate_coast_labels(
                                iax, fig, sd, dr, cnt, st_in, n, font=f8, star_lonlat=sd.anchor, avoid=avoid
                            )
                            rows_from("coast", "all", r, "inset")
                            return r

                        kw["inset_annotate"] = ann_in
                recs[f"{kind}_all"] = SF.render_single(
                    sd,
                    dr,
                    kind,
                    cnt / n * 100.0,
                    total_text,
                    out / f"{kind}_all.png",
                    font,
                    annotate=ann,
                    **kw,
                )
                continue
            lay = layout
            panels = []
            title_lines: list = []
            for k in keys:
                cnt, t = tot(k, kind)
                n = gn[k]
                base = f"{label[k]}\n{KIND_ZH[kind]} {t}%"
                annotate = None
                variants: tuple[str, ...] = ()
                title = base
                if kind == "boundary" and name == "class":
                    title_lines.append(2)  # 十類各類相近：不畫括號與圈號，標題只有類別名與穿越比例
                elif kind == "boundary":
                    st = SA.boundary_stretches(sd, cnt, n)
                    pw = (
                        SF.rs.REPORT_WIDTH_IN - lay.left - lay.right_strip - lay.gap_x * (lay.ncol - 1)
                    ) / lay.ncol
                    val, vi = SA.pick_value_variant(
                        SA.boundary_value_variants(st, n, crossing=int(cnt.sum())),
                        SF.rs.text_font(font),
                        pw * 72.0 + 3.0,
                    )
                    title = f"{base}\n{val}" if val else base
                    title_lines.append(2 + (val.count("\n") + 1 if val else 0))
                    if st:
                        for nm, p2, plr in SA.title_value_table(st, n, int(cnt.sum())):
                            pct_diff.append([sd.site_id, f"boundary_{name}", label[k], nm, p2, plr])
                    title_log.append([sd.site_id, name, k, label[k], len(st), min(len(st), SA.STRETCH_BOUNDARY_MAX_NUMBERED),
                                      max(0, len(st) - SA.STRETCH_BOUNDARY_MAX_NUMBERED), vi, val.replace("\n", " / ")])  # fmt: skip

                    def annotate(ax, fig, f8, cnt=cnt, n=n, k=k, lay=lay):
                        r = SA.annotate_boundary(
                            ax, fig, sd, dr, cnt, n, font=f8, star_size=lay.star_size, lw_pt=SF.LW_BOUNDARY * lay.lw_scale,
                            mode="marker", levels_pt=SA.BRACKET_LEVELS_PANEL_PT, br_lw=max(0.4, 0.8 * lay.lw_scale), land_pad_pt=1.0,
                        )  # fmt: skip
                        rows_from("boundary", k, r, "panel")
                        return r

                elif name in ("season", "tide"):
                    st = SA.coast_stretches(sd, cnt, n)
                    if st:
                        title = f"{base}\n{SA.marker_title_line(sd, st, n)}"

                        def annotate(ax, fig, f8, cnt=cnt, n=n, st=st, k=k, lay=lay):
                            r = SA.annotate_coast_markers(
                                ax, fig, sd, dr, cnt, st, n, font=f8, star_lonlat=sd.anchor,
                                star_size=lay.star_size, lw_pt=SF.LW_COAST * lay.lw_scale,
                            )  # fmt: skip
                            rows_from("coast", k, r, "panel")
                            return r

                panels.append(SF.Panel(title, cnt / n * 100.0, annotate, variants))
            if kind == "boundary":  # noqa: SIM114
                lay = replace(lay, title_h={2: 0.30, 3: 0.46, 4: 0.60}[max(title_lines)])
            elif name in ("season", "tide"):
                lay = replace(lay, title_h=0.46)
            fname = f"{kind}_{name}.png"
            recs[f"{kind}_{name}"] = SF.render_panels(sd, dr, kind, panels, lay, out / fname, font)
            if kind == "boundary":
                recs[f"{kind}_{name}"]["title_value_choices"] = [r[2:] for r in title_log if r[1] == name]
    return recs, stretch_rows, title_log, {"pct_diff": pct_diff, "tiny": tiny_rows}
