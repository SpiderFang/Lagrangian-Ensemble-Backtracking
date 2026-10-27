# ruff: noqa: E501
"""WP-75：單一地點的全部圖（單張、季節、大小潮、十類；邊界與海岸）與註記表格列。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from lagrangian_backtracking.native_runner import segprob_annot as SA
from lagrangian_backtracking.native_runner import segprob_figures as SF

SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
TIDE_ORDER = ("spring_proxy", "neap_proxy")
KIND_ZH = {"boundary": "穿越", "coast": "海岸來源"}
INSET_LC = (119.9, 120.2, 26.1, 26.3)  # 連江：南竿、北竿放大插圖視窗

# 多面板註記模式：annot＝面板內註記（邊小計＋集中段）；title＝邊小計寫在面板標題、不畫集中段；none＝只有比例
DEFAULT_PANEL_MODE = {
    "boundary": {"season": "title", "tide": "title", "class": "title"},
    "coast": {"season": "annot", "tide": "annot", "class": "none"},
}


def render_site(
    sd,
    dr,
    grp,
    label,
    mats,
    bcount,
    ccount,
    gn,
    out: Path,
    font,
    *,
    panel_mode=None,
    only=None,
    lienchiang_inset=True,
):
    """輸出該地點的全部圖；回傳 ``(manifest 紀錄, 集中段表格列)``。``only`` 可限定 (kind, name) 清單。"""

    mode = panel_mode or DEFAULT_PANEL_MODE
    out.mkdir(parents=True, exist_ok=True)
    recs: dict = {}
    stretch_rows: list = []
    side_cnt = {k: SA.side_counts(sd, np.where(m, sd.b_s, np.nan)) for k, m in grp.items()}

    def tot(k, kind):
        cnt = bcount[k] if kind == "boundary" else ccount[k]
        return cnt, SF.pct2(int(cnt.sum()), gn[k])

    def ann(kind, k, stretches=True, sides=True):
        cnt, _ = tot(k, kind)
        n = gn[k]

        def f(ax, fig, f8):
            if kind == "boundary":
                r = SA.annotate_boundary(
                    ax, fig, sd, cnt, side_cnt[k], n, font=f8, stretches=stretches, side_labels=sides
                )
            else:
                r = (
                    SA.annotate_coast(
                        ax, fig, sd, cnt, n, font=f8, land_ll=dr.land_union, star_lonlat=sd.anchor
                    )
                    if stretches
                    else {"stretches": []}
                )
            for rank, s in enumerate(r["stretches"], start=1):
                stretch_rows.append([sd.site_id, kind, k, rank, s["pct"], s["count"], len(s["segments"]),
                                     f"{s['start_lonlat'][0]:.4f}", f"{s['start_lonlat'][1]:.4f}",
                                     f"{s['end_lonlat'][0]:.4f}", f"{s['end_lonlat'][1]:.4f}", s["segments"][0], s["segments"][-1]])  # fmt: skip
            return r

        return f

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
                total_text = f"{'穿越比例' if kind == 'boundary' else '海岸來源比例'} {t}%"
                kw = {}
                if sd.site_id == "lienchiang" and kind == "coast" and lienchiang_inset:
                    kw = {"inset_window": INSET_LC}
                recs[f"{kind}_all"] = SF.render_single(
                    sd,
                    dr,
                    kind,
                    cnt / gn["all"] * 100.0,
                    total_text,
                    out / f"{kind}_all.png",
                    font,
                    annotate=ann(kind, "all"),
                    **kw,
                )
                if kw:
                    recs[f"{kind}_all_noinset"] = SF.render_single(
                        sd,
                        dr,
                        kind,
                        cnt / gn["all"] * 100.0,
                        total_text,
                        out / f"{kind}_all_noinset.png",
                        font,
                        annotate=ann(kind, "all"),
                    )
                continue
            m = mode[kind][name]
            lay = layout
            panels = []
            for k in keys:
                cnt, t = tot(k, kind)
                title = f"{label[k]}\n{KIND_ZH[kind]} {t}%"
                annotate = None
                if kind == "boundary" and m == "title":
                    parts = SA.side_title_string(sd, side_cnt[k], gn[k])
                    if name == "class":
                        title += "\n" + "｜".join(parts[:2]) + "\n" + "｜".join(parts[2:])
                    else:
                        title += "\n" + "｜".join(parts)
                elif m == "annot":
                    annotate = ann(kind, k)
                panels.append(SF.Panel(title, cnt / gn[k] * 100.0, annotate))
            if kind == "boundary" and m == "title":
                lay = replace(lay, title_h=0.58 if name == "class" else 0.45)
            fname = f"{kind}_{name}.png"
            recs[f"{kind}_{name}"] = SF.render_panels(sd, dr, kind, panels, lay, out / fname, font)
    return recs, stretch_rows
