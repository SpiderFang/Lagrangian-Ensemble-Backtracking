# -*- coding: utf-8 -*-
"""合併圖試作：邊界穿越與海岸來源分段畫在同一張地圖（同一色階）。
沿用同資料夾之 fp_lib／fp_lib2／fp_lib4（版面、數值框放置、碰撞檢查），以 monkeypatch 加上「combined」種類。
v37：多面板（十類、季節、大小潮）改用與全年圖相同之註記（LABELMODE=True）：邊界括號旁直接標 xx.xx%、海岸以引線標 xx.xx%，不再用圈號與數值框；
「邊界穿越」一律改稱「域外來源」（TERM_BOUNDARY）。LABELMODE=False 即回到 v36 之圈號＋數值框版與「邊界穿越」用語（用於逐位元重現舊圖）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, box as sbox
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 同資料夾之 fp_lib／fp_lib2／fp_lib4
import fp_lib as L0  # noqa: E402
import fp_lib2 as L2  # noqa: E402
import fp_lib4 as L4  # noqa: E402

rs, SF, ma, sp, SA = L0.rs, L0.SF, L0.ma, L0.sp, L0.SA
W_IN, RATIO = L0.W_IN, L0.RATIO

ARR = "A"  # A：數值框含穿越／海岸來源總計；B：總計寫在面板標題第二行，數值框只列圈號項與其餘
BOXMODE = "list"  # list：數值框每項一行；compact：同類圈號項併成一行
HOOK = {"lines": [], "shapes": None}
BONLY = False  # True：數值框只列穿越組，海岸不編號
NOBOX = False  # True：只留標題兩個合計與括號，不放數值框、不編圈號
INSET_PT = 7.0  # 放大小圖的刻度與百分比字級
BOX_PT = 7.0  # 數值框字級
LABELMODE = True  # True（v37）：多面板用括號旁 xx.xx%＋海岸引線 xx.xx%，無圈號、無數值框；False：v36 圈號＋數值框
TERM_BOUNDARY = "域外來源"  # 圖面用語（v36 為「邊界穿越」）


def term():
    """標題與圖例用語：LABELMODE（v37）用「域外來源」，否則沿用 v36 之「邊界穿越」。"""
    return TERM_BOUNDARY if LABELMODE else "邊界穿越"
COASTM = "ⓐⓑⓒⓓⓔⓕⓖ"  # 海岸段標記（與邊界 ①②③ 區別）
SF.LW_COAST = 1.7  # 海岸線較邊界色帶（3.2）細



# ------------------------------------------------------------------ 候選位置放寬（只擴大搜尋範圍，不改判定規則）
import inspect as _insp  # noqa: E402

_sb = _insp.getsource(SA.annotate_boundary)
assert "shifts = [0, 6, -6, 12, -12, 20, -20, 30, -30]" in _sb
_sb = _sb.replace("shifts = [0, 6, -6, 12, -12, 20, -20, 30, -30]", "shifts = [0, 6, -6, 12, -12, 20, -20, 30, -30, 45, -45, 60, -60, 80, -80, 100, -100]")


def _sbrep(a, b):
    global _sb
    assert _sb.count(a) == 1, a[:60]
    _sb = _sb.replace(a, b)


_sbrep('room = not scene.violations(rb, None, False)', '_rt_ = scene.violations(rb, None, False) - {"land"}\n                    room = not _rt_')
_sbrep('"hit_col": hit_col, "room": room,', '"hit_col": hit_col, "room": room, "roomA": _roomA, "rt": sorted(_rt_),')
_sbrep('if mode == "label":  # 括號外（開放海面側）要放得下置中的 % 文字', 'if True:  # 括號外要放得下標籤（單張圖為 % 文字，多面板為圈號）')
_sbrep('w_, h_ = (tw_pt, 8.0) if horiz_ else (8.0, tw_pt)', 'w_, h_ = (tw_pt, 8.0) if horiz_ else (8.0, tw_pt)\n                    if mode != "label":\n                        w_, h_ = 8.4, 8.4')
_sbrep('if need_room and mode != "label":\n                continue', 'if False:\n                continue')
_sbrep('pick = None\n        far_lv = max(levels_pt) * 1.4\n', """pick = None
        far_lv = max(levels_pt) * 1.4
        for need_a_, far_ in ((True, False), (True, True), (False, False), (False, True)):  # 先一律放邊界外側；先找上下側放得下者，再找端點旁放得下者
            for e in evals:
                if e["outward"] and (e["lv"] == far_lv) == far_ and e["room"] and (e["roomA"] or not need_a_) and not (e["hit_br"] or e["hit_frame"] or e["hit_col"]):
                    pick = e
                    break
            if pick is not None:
                break
""")
_sbrep('for need_room, far in ((True, False), (True, True), (False, False), (False, True)):', 'for need_room, far in (() if pick is not None else ((True, False), (True, True), (False, False), (False, True))):')
_sbrep('islet = False\n        if pick is None:', 'islet = bool(pick is not None and pick["outward"] and (pick["land_pref"] > 0 or pick["land_min"] > 0))\n        if pick is None:')
_sbrep('a, bb, tags = _place(scene, cands, make, None, leader=False, bbfun=bbf)', """a = None
            if PLABEL:  # v37 多面板：先要求標籤不壓到陸地與島嶼（含極小島）；全部候選都做不到才放寬為可蓋在陸地上
                a, bb, tags = _place(scene, cands, make, None, leader=False, bbfun=bbf)
                if tags:
                    a.remove()
                    a = None
            if a is None:
                a, bb, tags = _place(scene, cands, make, None, leader=False, bbfun=bbf, ignore={"land"})
            _inner = False
            if (mode == "marker" or PLABEL) and "axes" in tags:  # 例外：外圈放不下而落到圖框外或貼圖框者，改放流場範圍邊界內側（括號同一側之反側）
                _lk = _lastc[0]
                _bc = [("B", sh_, q_) for q_ in (5.0, 6.2, 7.6, 9.5, 12.0) for sh_ in shifts]
                _a2, _bb2, _t2 = _place(scene, _bc, make, None, leader=False, bbfun=bbf, ignore={"land"})
                if "axes" not in _t2 and sum(_W[t_] for t_ in _t2) <= sum(_W[t_] for t_ in tags):
                    a.remove()
                    a, bb, tags, _inner = _a2, _bb2, _t2, True
                else:
                    _a2.remove()
                    _lastc[0] = _lk""")
_sbrep('ignore={"axes"}', 'ignore={"axes", "land"}')
assert _sb.count("path_effects=_halo()") >= 1
_sb = _sb.replace("path_effects=_halo()", "path_effects=None")  # 邊界括號標籤不加白色光暈
_sbrep('halo = _halo_bracket(br_lw) if c["islet"] else None', 'halo = None')  # 括號線不加光暈
_sbrep('        chosen[j] = pick\n', "        chosen[j] = pick\n        rec.setdefault('side', []).append({'pct': SF.pct2(s.count, n_total), 'gap': j >= n_main, 'outward': bool(pick['outward']), 'islet': bool(islet), 'lv': round(float(pick['lv']), 1), 'n_out_eval': [(round(float(e['lv']), 1), bool(e['room']), bool(e['hit_br']), bool(e['hit_frame']), bool(e['hit_col']), e['rt']) for e in evals if e['outward']]})\n")
# ------------------------------------------------------------------ proto4：一律外圈（括號與標籤都在邊界外側）
def _end_geom(sd, scene, off, which, mpp, lonlat, full=None):
    L = off.length
    d = min(0.5 * L, 8.0 * mpp)
    pe = off.interpolate(0.0 if which == 0 else L)
    pi = off.interpolate(d if which == 0 else L - d)

    def disp(p):
        lon, lat = lonlat(sd, [p.x], [p.y])
        return np.array(scene.tr.transform((float(lon[0]), float(lat[0]))))

    de, di = disp(pe), disp(pi)
    t = de - di
    t = t / (np.hypot(*t) or 1.0)
    lon, lat = lonlat(sd, [pe.x], [pe.y])
    n = np.array([-t[1], t[0]])
    if full is not None:
        pf = full.interpolate(full.project(pe))
        v = de - disp(pf)
        if np.hypot(*v) > 1e-6:
            n = v - np.dot(v, t) * t
            n = n / (np.hypot(*n) or 1.0)
    return float(lon[0]), float(lat[0]), de, t, n


SA._end_geom = _end_geom
EGAPS = (2.5, 3.0, 4.5)  # 括號端點與標籤的間距（pt），先試 2 pt
SA.EG0 = EGAPS[0]
SA.EGAPS = EGAPS
ECS = (0.0, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5)  # 垂直方向（向外）微移，先試 0（與括號置中）
SA.ECS = ECS

_sbrep('for outward in (True, False):\n            for lv_pt in lv:', 'for outward in (True,):\n            for lv_pt in lv:')
_sbrep('''            for outward, key in (
                (True, "land_pref"),
                (True, "land_min"),
                (False, "land_pref"),
                (False, "land_min"),
            ):''', '''            for outward, key in (
                (True, "land_pref"),
                (True, "land_min"),
            ):''')
_sbrep('''                    _rt_ = scene.violations(rb, None, False) - {"land"}
                    room = not _rt_''', '''                    _rt_ = scene.violations(rb, None, False) - {"land"}
                    room = not _rt_
                    _roomA = room
                    if not room:
                        _mw = (7.8 * MARKER_FS / 8.0, 7.8 * MARKER_FS / 8.0) if mode != "label" else (tw_pt, 8.0)
                        for _wh, _cs in [(w0, c0) for c0 in ECS for w0 in (0, 1)]:
                            _lo, _la, _de, _t, _nn = _end_geom(sd, scene, off, _wh, mpp, _lonlat, full)
                            _hz = abs(_t[0]) >= abs(_t[1])
                            _ww, _hh = ((_mw[0], 8.0) if _hz else (8.0, _mw[0])) if mode == "label" else _mw
                            _al = _ww if _hz else _hh
                            _ce = _de + _t * (EG0 + _al / 2) * scene.k + _nn * _cs * scene.k
                            _rbe = SimpleNamespace(x0=_ce[0] - _ww / 2 * scene.k, x1=_ce[0] + _ww / 2 * scene.k,
                                                   y0=_ce[1] - _hh / 2 * scene.k, y1=_ce[1] + _hh / 2 * scene.k)
                            _re = scene.violations(_rbe, None, False) - {"land"}
                            if not _re:
                                room = True
                                _rt_ = set()
                                break''')
_sbrep('w_, h_ = 8.4, 8.4', 'w_, h_ = 7.8 * MARKER_FS / 8.0, 7.8 * MARKER_FS / 8.0')
# v37：多面板「括號旁直接標數值」（PLABEL）：字級＝該圖面板字級（十類 7 pt、其餘 8 pt；PLABEL_FS），其餘規則與全年圖相同
_sbrep('tw_pt = text_width_pt(font, label_txt[j], 7.0 if j >= n_main else 8.0) + 1.0', 'tw_pt = text_width_pt(font, label_txt[j], PLABEL_FS if PLABEL else (7.0 if j >= n_main else 8.0)) + 1.0')
_sbrep('fontproperties=font, fontsize=7 if is_gap else 8, zorder=13, rotation=0 if horiz else 90,', 'fontproperties=font, fontsize=PLABEL_FS if PLABEL else (7 if is_gap else 8), zorder=13, rotation=0 if horiz else 90,')
_sbrep('abs(lv_pt - pb["lv"]) < 2.0 * (', 'abs(lv_pt - pb["lv"]) < 1.3 * (')
_sbrep('cx_, cy_ = pm_d.ravel() + dd_ * (2.5 + 4.0) * scene.k', 'cx_, cy_ = pm_d.ravel() + dd_ * ((2.5 + 4.0) if mode == "label" else 7.6) * scene.k')
_sbrep('            def make(cd, txt=txt, d=d, horiz=horiz, tang=tang):\n                if cd[0] == "C":', '''            cands = [c_ for c_ in cands if c_[0] != "B"] + [("E", w_, g_, cs_) for cs_ in ECS for g_ in EGAPS for w_ in (0, 1)]
            if PLABEL:  # v37 多面板：括號中段外側（沿線 ±12 pt）→ 括號中段正對面的流場範圍內側 → 沿線較遠的外側 → 沿線較遠的內側 → 括號兩端旁
                _An = [c_ for c_ in cands if c_[0] == "A" and abs(c_[1]) <= 12]
                _Af = [c_ for c_ in cands if c_[0] == "A" and abs(c_[1]) > 12]
                _Bn = [("B", sh_, q_) for q_ in (3.0, 4.5, 6.0, 8.0) for sh_ in (0, 6, -6, 12, -12)]
                _Bf = [("B", sh_, q_) for q_ in (3.0, 4.5, 6.0, 8.0) for sh_ in (20, -20, 30, -30, 45, -45)]
                cands = _An + _Bn + _Af + _Bf + [c_ for c_ in cands if c_[0] == "E"]
            _eg = [_end_geom(sd, scene, off, w_, mpp, _lonlat, sea) for w_ in (0, 1)]
            _fs_ = PLABEL_FS if PLABEL else (MARKER_FS if mode == "marker" else (7.0 if is_gap else 8.0))
            _tw_ = text_width_pt(font, txt, _fs_) + 1.0
            _lastc = [None]

            def make(cd, txt=txt, d=d, horiz=horiz, tang=tang):
                _lastc[0] = cd
                if cd[0] == "E":
                    _lo, _la, _de, _t, _nn = _eg[cd[1]]
                    _hz = abs(_t[0]) >= abs(_t[1])
                    if mode == "marker":
                        _w, _h = 7.8 * MARKER_FS / 8.0, 7.8 * MARKER_FS / 8.0
                    else:
                        _w, _h = (_tw_, 8.0) if _hz else (8.0, _tw_)
                    _al = _w if _hz else _h
                    _o = _t * (cd[2] + _al / 2) + _nn * cd[3]
                    return ax.annotate(
                        txt, xy=(_lo, _la), xytext=(float(_o[0]), float(_o[1])), textcoords="offset points",
                        ha="center", va="center", fontproperties=font, fontsize=_fs_,
                        rotation=0 if (mode == "marker" or _hz) else 90, zorder=13, path_effects=None,
                        color=GAP_TEXT_COLOR if (is_gap and mode == "label") else "black",
                    )  # fmt: skip
                if cd[0] == "C":''')
_sbrep('where = "beside"\n            if tags and mode == "marker":', """_lc0 = _lastc[0]
            if (tags or __import__("os").environ.get("CPDBG") == "2") and __import__("os").environ.get("CPDBG"):
                for _c in cands:
                    _aa = make(_c)
                    _bb = bbf(_aa, scene.r)
                    print("DBG", txt, _c, sorted(scene.violations(_bb, None, False) - {"land"}), flush=True)
                    _aa.remove()
            _lastc[0] = _lc0
            where = "inner_side" if _lastc[0][0] == "B" else ("beside" if _lastc[0][0] != "E" else ("end" if _lastc[0][3] == 0 else "end_shift%.1f" % _lastc[0][3]))
            where = "inner_axes" if _inner else where
            if False and tags and mode == "marker":""")

assert _sb.count("fontproperties=font, fontsize=8, zorder=13, path_effects=None,") == 2
_sb = _sb.replace("fontproperties=font, fontsize=8, zorder=13, path_effects=None,", "fontproperties=font, fontsize=MARKER_FS, zorder=13, path_effects=None,")
SA.MARKER_FS = 8.0
SA.PLABEL, SA.PLABEL_FS = False, 8.0  # 須在 exec 之前設定（編譯後的函式讀 SA 模組全域名稱）
_mbb = _insp.getsource(SA._marker_bb)
assert _mbb.count("h = 3.9 * renderer.dpi / 72.0") == 1
exec(compile(_mbb.replace("h = 3.9 * renderer.dpi / 72.0", "h = 3.9 * MARKER_FS / 8.0 * renderer.dpi / 72.0"), "sa_marker_bb", "exec"), SA.__dict__)  # noqa: S102
exec(compile(_sb, "sa_annotate_boundary", "exec"), SA.__dict__)  # noqa: S102
_sc = _insp.getsource(SA.annotate_coast_labels)
assert "dists = (13, 19, 26, 34, 44, 56, 70)" in _sc
_sc = _sc.replace("dists = (13, 19, 26, 34, 44, 56, 70)", "dists = (13, 19, 26, 34, 44, 56, 70, 85, 100)").replace("lw_pt=SF.LW_COAST,", "lw_pt=1.7,")
assert _sc.count("fontsize=8,") == 1
_sc = _sc.replace("fontsize=8,", "fontsize=COAST_FS,")
SA.COAST_FS = 8.0
exec(compile(_sc, "sa_annotate_coast_labels", "exec"), SA.__dict__)  # noqa: S102

PICK = {"fracs": None, "i": 0}
_pa0 = SA._pick_anchor


def _pick_anchor_x(sd, line, scene):
    if PICK["fracs"] is None:
        return _pa0(sd, line, scene)
    frac = PICK["fracs"][PICK["i"]]
    PICK["i"] += 1
    if frac is None:  # None＝該集中段用程式原本的錨點規則
        return _pa0(sd, line, scene)
    return SA._sea_dir(sd, line, frac)


SA._pick_anchor = _pick_anchor_x

_FR_CACHE: dict = {}


def coast_labels_retry(ax, fig, sd, dr, ccnt, st_c, n, **kw):
    """多面板海岸引線：先照原規則標；若某集中段的標籤有違規（壓陸地、壓上色線…），只對該段改試其他引線落點
    （集中段上 0.5、0.4、0.6…0.9 比例處），取違規最輕者；其餘集中段不動。判定規則（哪些段要標）不變。"""
    FR = (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9)

    def run(fracs):
        n0 = len(ax.texts)
        PICK["fracs"], PICK["i"] = list(fracs), 0
        try:
            r = SA.annotate_coast_labels(ax, fig, sd, dr, ccnt, st_c, n, **kw)
        finally:
            PICK["fracs"] = None
        return r, n0

    def sev(r):
        return sum(SA._W.get(t, 0) for s_ in r["stretches"] for t in s_.get("label_violations", []))

    def undo(n0):
        for t_ in list(ax.texts)[n0:]:
            t_.remove()

    key = (sd.site_id, hash(np.asarray(ccnt).tobytes()), len(st_c))
    fracs, exhausted = _FR_CACHE.get(key, ([None] * len(st_c), False))  # 版面收斂會反覆重畫：沿用上次選定的落點，已試遍者不再重試
    r, n0 = run(fracs)
    best = (sev(r), list(fracs), r["stretches"])
    tried = 0
    for j in range(len(st_c)):
        if exhausted:
            break
        if not best[2][j].get("label_violations"):
            continue
        for fr in FR:
            undo(n0)
            f2 = list(best[1])
            f2[j] = fr
            r2, _ = run(f2)
            tried += 1
            if sev(r2) < best[0]:
                best = (sev(r2), f2, r2["stretches"])
            if not r2["stretches"][j].get("label_violations") and sev(r2) <= best[0]:
                break
        undo(n0)
        r, _ = run(best[1])
    undo(n0)
    r, _ = run(best[1])
    _FR_CACHE[key] = (best[1], True)
    r["retry"] = {"tried": tried, "fracs": best[1]}
    return r


# ------------------------------------------------------------------ Scene 擴充：額外障礙（另一類上色線、已畫之括號與文字）
_Scene0 = SA.Scene


class SceneX(_Scene0):
    def __init__(self, fig, ax, *, lines_ll, **kw):
        super().__init__(fig, ax, lines_ll=list(lines_ll) + list(HOOK["lines"]), **kw)
        fn = HOOK["shapes"]
        if fn is not None:
            shp = fn(ax, self.r, self.k)
            if shp:
                u = unary_union(shp)
                self.avoid = u if self.avoid is None else unary_union([self.avoid, u])

    def violations(self, bb, anchor=None, leader=True):
        tags = super().violations(bb, anchor, leader)
        if SA.PLABEL and leader and anchor is not None and self.land is not None and "land" not in tags:
            # v37 多面板：引線（扣掉離錨點 4 pt 內）不可橫越陸地（小面板裡引線若越過陸地會拉得很長、貼著海岸線，難以辨識）
            c = ((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2)
            ls = LineString([c, anchor]).difference(sbox(bb.x0, bb.y0, bb.x1, bb.y1))
            if not ls.is_empty:
                far = ls.difference(Point(anchor).buffer(4.0 * self.k))
                for g_ in getattr(far, "geoms", [far]):
                    if g_.is_empty or g_.geom_type != "LineString":
                        continue
                    if self.land.intersects(LineString(self.inv.transform(np.asarray(g_.coords)))):
                        tags.add("land")
                        break
        return tags


SA.Scene = SceneX


def shapes_from_axes(ax, r, k):
    from matplotlib.text import Annotation

    out = []
    for t in ax.texts:
        if not t.get_text():
            continue
        if len(t.get_text()) == 1 and t.get_text() in SA.CIRCLED + COASTM:
            bb = SA._marker_bb(t, r)
        elif isinstance(t, Annotation):
            bb = SA._text_bb(t, r)
        else:
            bb = t.get_window_extent(r)
        out.append(sbox(bb.x0, bb.y0, bb.x1, bb.y1).buffer(1.0 * k))
    for ln in ax.lines:
        if ln.get_marker() not in (None, "", "None", " "):
            continue
        xy = ln.get_xydata()
        if xy is None or len(xy) < 2:
            continue
        d = ax.transData.transform(xy)
        out.append(LineString(d).buffer(1.2 * k))
    return out


def _pos_lines(lines, cnt):
    return [a for i, parts in enumerate(lines) if cnt[i] > 0 for a in parts]


# ------------------------------------------------------------------ 繪製
def draw_map_c(ax, sd, dr, bpct, cpct, *, window, star=True, lw_scale=1.0, star_size=None):
    from matplotlib.collections import LineCollection
    from shapely.plotting import plot_polygon

    _cm, _nm, cols = SF.colormap_and_norm()
    for poly in dr.land_polys:
        plot_polygon(poly, ax=ax, add_points=False, facecolor=SF.LAND_FACE, edgecolor=SF.LAND_EDGE,
                     linewidth=SF.LAND_EDGE_LW * lw_scale, zorder=1)
    if dr.b_land:
        ax.add_collection(LineCollection(dr.b_land, colors=SF.LAND_LINE, linewidths=SF.LW_LAND_LINE * lw_scale,
                                         zorder=3, capstyle="butt"))
    bcls, ccls = sp.bin_index(bpct, SF.EDGES), sp.bin_index(cpct, SF.EDGES)
    for k in range(len(cols)):
        for lines, cls, lw in ((dr.b_sea, bcls, SF.LW_BOUNDARY * lw_scale), (dr.c_lines, ccls, SF.LW_COAST * lw_scale)):
            segs = [a for i, parts in enumerate(lines) if cls[i] == k for a in parts]
            if segs:
                ax.add_collection(LineCollection(segs, colors=cols[k], linewidths=lw, zorder=4 + k * 0.01, capstyle="butt"))
    if star:
        st = dict(SF.STAR)
        if star_size is not None:
            st["markersize"] = star_size
        ax.plot([sd.anchor[0]], [sd.anchor[1]], zorder=9, **st)
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])
    ax.set_aspect(SF.SPEC.aspect, adjustable="box")


_draw0 = SF.draw_map


def _draw_dispatch(ax, sd, dr, kind, pct, *, window, star=True, lw_scale=1.0, star_size=None):
    if kind == "combined":
        return draw_map_c(ax, sd, dr, pct[0], pct[1], window=window, star=star, lw_scale=lw_scale, star_size=star_size)
    return _draw0(ax, sd, dr, kind, pct, window=window, star=star, lw_scale=lw_scale, star_size=star_size)


SF.draw_map = _draw_dispatch


def legend_handles_c(has_land, totals=None):
    from matplotlib.lines import Line2D

    mid = SF.COLORS[2]
    h = [Line2D([], [], label="研究地點錨點", **SF.STAR),
         ]
    if has_land:
        h.append(Line2D([], [], color=SF.LAND_LINE, linewidth=SF.LW_LAND_LINE * 1.6, label="陸上邊界（無法穿越）"))
    elif totals:
        h.append(Line2D([], [], linestyle="none", label=""))
    if totals:
        h += [Line2D([], [], linestyle="none", label=totals[0]), Line2D([], [], linestyle="none", label=totals[1])]
    return h


_lh0 = SF._legend_handles


def _lh_dispatch(kind, total_text, has_land):
    if kind == "combined":
        return legend_handles_c(has_land)
    return _lh0(kind, total_text, has_land)


SF._legend_handles = _lh_dispatch


# ------------------------------------------------------------------ 數值框位置：combined 同時避開兩類上色線
_seg0, _static0 = L4.seg_dist, L4.static_occ


def seg_dist_c(S, kind, cnts, w, h, tag):
    if kind != "combined":
        return _seg0(S, kind, cnts, w, h, tag)
    key = ("segc", S.sd.site_id, tag, round(w, 3))
    if key in L4._RC:
        return L4._RC[key]
    dr = S.dr
    win, sx, sy = L4._tr(S, w, h)
    if "#" in tag:
        base_, idx_ = tag.split("#")
        keys = [L4._keys(S, base_)[int(idx_)]]
    else:
        keys = L4._keys(S, tag)
    lines = []
    for src, cm in ((dr.b_sea, S.bcount), (dr.c_lines, S.ccount)):
        pos = np.zeros(len(src), dtype=bool)
        for k in keys:
            pos |= np.asarray(cm[k]) > 0
        for i, parts in enumerate(src):
            if pos[i]:
                for a in parts:
                    a = np.asarray(a)
                    if len(a) >= 2:
                        lines.append(np.column_stack([(a[:, 0] - win[0]) * sx, (a[:, 1] - win[2]) * sy]))
    d = L4._dist(L4._raster_lines(lines, w, h))
    L4._RC[key] = d
    return d


def static_occ_c(S, kind, w, h, land_min_area=None, ring=True):
    if kind == "combined":
        return _static0(S, "boundary", w, h, land_min_area, ring)
    return _static0(S, kind, w, h, land_min_area, ring)


L4.seg_dist = seg_dist_c
L4.static_occ = static_occ_c


# ------------------------------------------------------------------ 海岸圈號（接續邊界編號）
def annotate_coast_markers_off(ax, fig, sd, dr, seg_counts, stretches, n_total, *, font, star_lonlat, star_size, lw_pt, first):
    rec = {"stretches": [], "collisions": []}
    scene = SA._coast_scene(ax, fig, dr, seg_counts, lw_pt=lw_pt, star_lonlat=star_lonlat, star_size=star_size,
                            avoid=(), land_ll=dr.land_union)
    dists = (9, 13, 18, 24, 32, 42, 54)
    angs = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)
    for i, s in enumerate(stretches, start=first):
        line = SA._coast_line(sd, s)
        mid, sdir = SA._pick_anchor(sd, line, scene)
        lon, lat = SA._lonlat(sd, [mid.x], [mid.y])
        anchor = tuple(scene.tr.transform((lon[0], lat[0])))
        cands = sorted(((d, a) for d in dists for a in angs), key=lambda c: (c[0], abs(c[1])))
        glyph = COASTM[i - first]

        def make(c, glyph=glyph, lon=lon, lat=lat, sdir=sdir):
            d = SA._rot(sdir, c[1])
            return ax.annotate(glyph, xy=(lon[0], lat[0]), xytext=(d[0] * c[0], d[1] * c[0]), textcoords="offset points",
                               ha="center", va="center", fontproperties=font, fontsize=8, zorder=14, path_effects=SA._halo(),
                               arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#222222", "shrinkA": 0, "shrinkB": 0})

        a, bb, tags = SA._place(scene, cands, make, anchor)
        scene.commit(bb, anchor)
        rec["stretches"].append(SA._stretch_record(sd, s, n_total, {"marker": glyph, "violations": sorted(tags)}))
        if tags:
            rec["collisions"].append({"label": glyph, "violations": sorted(tags)})
    return rec


# ------------------------------------------------------------------ 多面板
def make_panels_c(S, kind, name, w, *, class_brackets=False):
    sd = S.sd
    keys = L4._keys(S, name)
    lw_scale, star_size = L0.lw_for(w), L0.star_for(w)
    panels, boxes, log = [], [], []
    for k in keys:
        bcnt, ccnt, n = S.bcount[k], S.ccount[k], S.gn[k]
        tb, tc = SF.pct2(int(bcnt.sum()), n), SF.pct2(int(ccnt.sum()), n)
        st_b = SA.boundary_stretches(sd, bcnt, n)
        nb = min(len(st_b), SA.STRETCH_BOUNDARY_MAX_NUMBERED)
        st_c = SA.coast_stretches(sd, ccnt, n) if (name in ("season", "tide") and not NOBOX and not BONLY) else []
        if LABELMODE:  # v37：不編圈號、不放數值框；數值直接標在括號旁（邊界）與引線旁（海岸），數字仍為各自 ROUND_HALF_UP 之 pct2
            st_c = SA.coast_stretches(sd, ccnt, n) if name in ("season", "tide") else []
            log.append([S.label[k], len(st_b), len(st_c)])

            def annotate(ax, fig, f8_, bcnt=bcnt, ccnt=ccnt, n=n, st_c=st_c):
                fs = SA.MARKER_FS  # 十類 7 pt、其餘 8 pt（與 v36 圈號同字級）
                SA.PLABEL, SA.PLABEL_FS, SA.COAST_FS = True, fs, fs
                try:
                    HOOK["lines"] = _pos_lines(S.dr.c_lines, ccnt)
                    HOOK["shapes"] = None
                    rb = SA.annotate_boundary(ax, fig, sd, S.dr, bcnt, n, font=f8_, star_size=star_size,
                                              lw_pt=SF.LW_BOUNDARY * lw_scale, mode="label", levels_pt=SA.BRACKET_LEVELS_PANEL_PT,
                                              br_lw=max(0.4, 0.8 * lw_scale), land_pad_pt=1.0, gaps=False)
                    rc = {"stretches": [], "collisions": []}
                    if st_c:
                        HOOK["lines"] = _pos_lines(S.dr.b_sea, np.ones(len(S.dr.b_sea))) + list(S.dr.b_land)
                        HOOK["shapes"] = shapes_from_axes
                        rc = coast_labels_retry(ax, fig, sd, S.dr, ccnt, st_c, n, font=f8_, star_lonlat=sd.anchor,
                                                star_size=star_size, lw_pt=SF.LW_COAST * lw_scale)
                finally:
                    HOOK["lines"], HOOK["shapes"] = [], None
                    SA.PLABEL, SA.PLABEL_FS, SA.COAST_FS = False, 8.0, 8.0
                return {"stretches": rb["stretches"] + rc["stretches"], "collisions": rb["collisions"] + rc["collisions"],
                        "n_boundary": len(rb["stretches"]), "n_coast": len(rc["stretches"]), "side": rb.get("side")}

            panels.append(SF.Panel(f"{S.label[k]}\n{term()} {tb}%　海岸來源 {tc}%", (bcnt / n * 100.0, ccnt / n * 100.0), annotate, ()))
            boxes.append(["　"])
            continue
        lines = []
        if nb:
            lines += ["邊界穿越"] + [f"{SA.circled(k)} {SF.pct2(st.count, n)}%" for k, st in enumerate(st_b[:nb], start=1)]
        if st_c:
            lines += ["海岸來源"] + [f"{COASTM[i]} {SF.pct2(st.count, n)}%" for i, st in enumerate(st_c)]
        if NOBOX:
            lines = ["　"]
        if not lines:
            lines = ["無集中段"]
        log.append([S.label[k], len(st_b), len(st_c)])

        def annotate(ax, fig, f8_, bcnt=bcnt, ccnt=ccnt, n=n, st_c=st_c, nb=nb):
            HOOK["lines"] = _pos_lines(S.dr.c_lines, ccnt)
            HOOK["shapes"] = None
            rb = SA.annotate_boundary(ax, fig, sd, S.dr, bcnt, n, font=f8_, star_size=star_size,
                                      lw_pt=SF.LW_BOUNDARY * lw_scale, mode="marker", levels_pt=SA.BRACKET_LEVELS_PANEL_PT,
                                      br_lw=max(0.4, 0.8 * lw_scale), land_pad_pt=1.0)
            rc = {"stretches": [], "collisions": []}
            if st_c:
                HOOK["lines"] = _pos_lines(S.dr.b_sea, np.ones(len(S.dr.b_sea))) + list(S.dr.b_land)
                HOOK["shapes"] = shapes_from_axes
                rc = annotate_coast_markers_off(ax, fig, sd, S.dr, ccnt, st_c, n, font=f8_, star_lonlat=sd.anchor,
                                                star_size=star_size, lw_pt=SF.LW_COAST * lw_scale, first=1)
            HOOK["lines"], HOOK["shapes"] = [], None
            return {"stretches": rb["stretches"] + rc["stretches"], "collisions": rb["collisions"] + rc["collisions"],
                    "n_boundary": len(rb["stretches"]), "n_coast": len(rc["stretches"]), "side": rb.get("side")}

        title = S.label[k] if ARR == "A" else f"{S.label[k]}\n{term()} {tb}%　海岸來源 {tc}%"
        panels.append(SF.Panel(title, (bcnt / n * 100.0, ccnt / n * 100.0), annotate, ()))
        boxes.append(lines)
    return panels, boxes, log


L4.make_panels3 = make_panels_c  # build3 以全域名稱呼叫


# ------------------------------------------------------------------ 全年單張
def _densify(a, step=0.01):
    a = np.asarray(a)
    out = [a[:1]]
    for p0, p1 in zip(a[:-1], a[1:], strict=True):
        n = int(np.ceil(np.hypot(*(p1 - p0)) / step)) + 1
        out.append(np.linspace(p0, p1, n)[1:])
    return np.vstack(out)


def free_inset_box_c(sd, dr, bpct, cpct, window, w_frac, h_frac, right_max=0.995):
    lon_span, lat_span = window[1] - window[0], window[3] - window[2]
    pts = [np.vstack([np.asarray(a) for a in parts]) for i, parts in enumerate(dr.c_lines) if cpct[i] > 0]
    bl = [_densify(a) for parts in dr.b_sea for a in parts if len(a) >= 2] + [_densify(a) for a in dr.b_land if len(a) >= 2]
    pts = np.vstack(pts + bl)
    fx = (pts[:, 0] - window[0]) / lon_span
    fy = (pts[:, 1] - window[2]) / lat_span
    star = ((sd.anchor[0] - window[0]) / lon_span, (sd.anchor[1] - window[2]) / lat_span)
    for x0 in np.arange(right_max - w_frac, 0.25, -0.005):
        for y0 in np.arange(0.10, 0.80, 0.01):
            rx0, rx1, ry0, ry1 = x0 - 0.095, x0 + w_frac + 0.004, y0 - 0.09, y0 + h_frac + 0.004
            if y0 + h_frac > 0.97:
                continue
            if rx0 <= star[0] <= rx1 and ry0 <= star[1] <= ry1:
                continue
            if not ((fx >= rx0) & (fx <= rx1) & (fy >= ry0) & (fy <= ry1)).any():
                return (float(x0), float(y0), w_frac, h_frac)
    return None


def render_single_c(S, path, *, dpi=600, inset_window=None, inset_min_mag=4.0, inset_max_mag=5.5, inset_right_max=0.995):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Bbox
    sd, dr, font = S.sd, S.dr, S.font
    n = S.gn["all"]
    bcnt, ccnt = S.bcount["all"], S.ccount["all"]
    bpct, cpct = bcnt / n * 100.0, ccnt / n * 100.0
    tb, tc = SF.pct2(int(bcnt.sum()), n), SF.pct2(int(ccnt.sum()), n)
    f8 = rs.text_font(font)
    with rs.report_rc(font.get_name()):
        rs.require_report_rc(font)
        legend_rows = 2
        fig = plt.figure(figsize=rs.canvas_in(legend_rows), dpi=150)
        fw, fh = fig.get_size_inches()
        bx, by, bw, bh = rs.frame_box_in(SF.SPEC.frame_width_over_height, legend_rows)
        ax = fig.add_axes((bx / fw, by / fh, bw / fw, bh / fh))
        win = SF.WINDOWS[sd.site_id]
        draw_map_c(ax, sd, dr, bpct, cpct, window=win)
        SF._format_axes(ax, win, SF.SPEC.step_deg, SF.SPEC.step_deg, xlabels=True, ylabels=True, f8=f8)
        iax, inset_rec, avoid = None, None, []
        st_c = SA.coast_stretches(sd, ccnt, n)
        st_in, st_main = [], st_c
        if inset_window is not None:
            iw = inset_window
            from wp75_make import _in_window  # noqa
            st_in = [s for s in st_c if _in_window(SA.stretch_mid_lonlat(sd, s), iw)]
            st_main = [s for s in st_c if s not in st_in]
            lon_span, lat_span = win[1] - win[0], win[3] - win[2]
            h_over_w = (lon_span / lat_span) * ((iw[3] - iw[2]) / (iw[1] - iw[0]))
            pos, mag = None, inset_max_mag
            while mag >= inset_min_mag - 1e-9 and pos is None:
                w_frac = mag * (iw[1] - iw[0]) / lon_span
                pos = free_inset_box_c(sd, dr, bpct, cpct, win, w_frac, w_frac * h_over_w, inset_right_max)
                if pos is None:
                    mag -= 0.1
            if pos is None:
                raise RuntimeError("找不到不蓋到上色線之放大小圖位置")
            iax = ax.inset_axes(pos)
            draw_map_c(iax, sd, dr, bpct, cpct, window=iw, star=True, lw_scale=1.0)
            iax.set_xticks([iw[0], iw[1]])
            iax.set_yticks([iw[2], iw[3]])
            iax.xaxis.set_major_formatter(FuncFormatter(SF.inset_format))
            iax.yaxis.set_major_formatter(FuncFormatter(SF.inset_format))
            rs.style_map_axes(iax, f8)
            for lab in list(iax.get_xticklabels()) + list(iax.get_yticklabels()):
                lab.set_fontsize(INSET_PT)
            for j, lab in enumerate(iax.get_xticklabels()):
                lab.set_ha("left" if j == 0 else "right")
            for sp_ in iax.spines.values():
                sp_.set_linewidth(0.8)
            iax.set_zorder(30)
            ax.add_patch(Rectangle((iw[0], iw[2]), iw[1] - iw[0], iw[3] - iw[2], fill=False, edgecolor="#222222", linewidth=0.6, zorder=12))
            fig.canvas.draw()
            r_ = fig.canvas.get_renderer()
            avoid.append(iax.get_tightbbox(r_))
            (rx0, ry0), (rx1, ry1) = ax.transData.transform((iw[0], iw[2])), ax.transData.transform((iw[1], iw[3]))
            avoid.append(Bbox([[rx0 - 2, ry0 - 2], [rx1 + 2, ry1 + 2]]))
            inset_rec = {"window_lon_lat": list(iw), "axes_fraction_box": list(pos), "magnification": float(pos[2] * lon_span / (iw[1] - iw[0]))}
        # 邊界括號與百分比（先）
        HOOK["lines"] = _pos_lines(dr.c_lines, ccnt)
        HOOK["shapes"] = None
        rb = SA.annotate_boundary(ax, fig, sd, dr, bcnt, n, font=f8, avoid=avoid, mode="label", gaps=True)
        # 海岸引線與百分比（後）：避開邊界括號、邊界文字與整圈流場邊界
        HOOK["lines"] = [a for parts in dr.b_sea for a in parts] + list(dr.b_land)
        HOOK["shapes"] = shapes_from_axes
        rc = SA.annotate_coast_labels(ax, fig, sd, dr, ccnt, st_main, n, font=f8, star_lonlat=sd.anchor, avoid=avoid)
        ric = None
        if iax is not None and st_in:
            HOOK["lines"] = []
            import itertools
            fr = (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9)
            best = None
            for combo in itertools.product(fr, repeat=len(st_in)):
                PICK["fracs"], PICK["i"] = list(combo), 0
                n0 = len(iax.texts)
                SA.COAST_FS = INSET_PT
                r_ = SA.annotate_coast_labels(iax, fig, sd, dr, ccnt, st_in, n, font=f8, star_lonlat=sd.anchor, avoid=[])
                nc = sum(len(c["violations"]) for c in r_["collisions"])
                if best is None or nc < best[0]:
                    best = (nc, combo)
                for t_ in list(iax.texts)[n0:]:
                    t_.remove()
                if nc == 0:
                    break
            PICK["fracs"], PICK["i"] = list(best[1]), 0
            SA.COAST_FS = INSET_PT
            ric = SA.annotate_coast_labels(iax, fig, sd, dr, ccnt, st_in, n, font=f8, star_lonlat=sd.anchor, avoid=[])
            PICK["fracs"] = None
            SA.COAST_FS = 8.0
        HOOK["lines"], HOOK["shapes"] = [], None
        SF._discrete_colorbar(fig, (bx + bw + rs.CBAR_PAD_IN, by, rs.CBAR_W_IN, bh), f8)
        has_land = dr.has_land_boundary
        handles = legend_handles_c(has_land, (f"{term()}比例 {tb}%", f"海岸來源比例 {tc}%"))
        leg = rs.framed_legend(fig, ax, handles, f8, rows=legend_rows)
        rec = {"kind": "combined", "site": sd.site_id, "canvas_in": [float(fw), float(fh)], "window_lon_lat": list(win),
               "legend": rs.legend_geometry(fig, leg, ax), "legend_labels": [t.get_text() for t in leg.get_texts()],
               "annotations": {"boundary": rb, "coast": rc, "coast_inset": ric}, "inset": inset_rec,
               "totals": {"crossing": tb, "coast": tc}}
        return SF._finish(fig, Path(path), rec, dpi)


# ------------------------------------------------------------------ 數值框位置：可選之第三階（蓋住陸地）
ALLOW_LAND = False
LAST_TIER = {"t": 0}


def place_box_c(S, kind, cnts, w, h, bw, bh, corner=None, margins=L4.MARGINS, extra=None, tag="", corners=None):
    """同 fp_lib4.place_box；ALLOW_LAND 為 True 時，前兩階（避開陸地、只蓋極小島嶼）都放不下才允許第三階（框可蓋住陸地，
    仍不蓋上色線段、括號、圈號、星號）。"""
    nx, ny = L4._dims(w, h)
    xs, ys = (np.arange(nx) + 0.5) * L4.GRID, (np.arange(ny) + 0.5) * L4.GRID
    X, Y = np.meshgrid(xs, ys)
    frame = (X < 0.04) | (X > w - 0.04) | (Y < 0.04) | (Y > h - 0.04)
    dsg = L4.seg_dist(S, kind, cnts, w, h, tag)
    want = list(corners) if corners else ([corner] if corner else list(L4.CORNERS))
    found = {}
    tiers = [(None, True), (L4.LAND_MIN_AREA, True)] + ([(1e9, True)] if ALLOW_LAND else [])
    for tier, (lma, ring_) in enumerate(tiers):
        base = L4.static_occ(S, kind, w, h, lma, ring_) | frame
        if extra is not None:
            base = base | extra
        for m in tuple(margins) + L4.MARGINS_TIGHT:
            occ = base | (dsg < m)
            ok, bwc, bhc = L4.free_positions(occ, bw, bh)
            jj, ii = np.nonzero(ok)
            if len(jj) == 0:
                continue
            x0, y0 = ii * L4.GRID, jj * L4.GRID
            for c in want:
                if c in found:
                    continue
                xr, yt = x0 + bwc * L4.GRID, y0 + bhc * L4.GRID
                ap = {"UR": (xr, yt, w, h), "UL": (x0, yt, 0, h), "LR": (xr, y0, w, 0), "LL": (x0, y0, 0, 0)}[c]
                d = np.hypot(ap[0] - ap[2], ap[1] - ap[3])
                k = int(np.argmin(d))
                bx, by = float(x0[k]), float(y0[k])
                bw_, bh_ = bwc * L4.GRID, bhc * L4.GRID
                anchor = {"UR": (bx + bw_, by + bh_), "UL": (bx, by + bh_), "LR": (bx + bw_, by), "LL": (bx, by)}[c]
                found[c] = dict(corner=c, margin=m, x=anchor[0], y=anchor[1], ha="right" if c in ("UR", "LR") else "left",
                                va="top" if c in ("UR", "UL") else "bottom", dist=float(d[k]), tier=tier)
            if all(c in found for c in want):
                break
        if all(c in found for c in want):
            break
    if corners is not None:
        return found
    res = found.get(want[0]) if corner else (min(found.values(), key=lambda r_: r_["dist"]) if found else None)
    if res is not None:
        LAST_TIER["t"] = res["tier"]
    return res


_mb = _insp.getsource(L4._measure_boxes)
assert _mb.count("fontsize=rs.TEXT_PT") == 1
_mb = _mb.replace("fontsize=rs.TEXT_PT", "fontsize=BOX_PT")
L4.BOX_PT = BOX_PT
exec(compile(_mb, "l4_measure_boxes", "exec"), L4.__dict__)  # noqa: S102
L4.place_box = place_box_c
_bva0 = L4.box_vs_artists


def _bva(ax, box_text, rend, fig, land_ok_area=None):
    if LAST_TIER["t"] >= 2:
        land_ok_area = 1e9
    return _bva0(ax, box_text, rend, fig, land_ok_area=land_ok_area)


L4.box_vs_artists = _bva


# ------------------------------------------------------------------ build3 改版：數值框可每面板各自找位置
import inspect  # noqa: E402

PER_PANEL = False
_src = inspect.getsource(L4.build3)
_a = _src.index("    # ---- 數值框：本圖最大框尺寸 → 固定錨點")
_b = _src.index("    fig.canvas.draw()\n    rend = fig.canvas.get_renderer()\n    d = fig.dpi")
_new = '''    # ---- 數值框（cp_lib 改版）：C_PER_PANEL 時每面板以自己的上色線與註記找位置，否則全面板共用固定位置
    if LABELMODE_BOXES():  # v37：多面板不放數值框（數值直接標在括號旁與引線旁）
        sizes = [(0.0, 0.0)] * len(boxlines)
        bw = bh = 0.0
        fig.canvas.draw()
        box_texts, pls = [], []
        pl = dict(corner="none", margin=0.0, x=0.0, y=0.0, tier=0, soft_fail=False, each=[])
        import cp_lib as _C
        _C.LAST_TIER["t"] = 0
    else:
        sizes = _measure_boxes_cached(S, boxlines)
        bw = max(s[0] for s in sizes)
        bh = max(s[1] for s in sizes)
        cnts = [S.bcount[k] if kind == "boundary" else S.ccount[k] for k in _keys(S, name)]
        fig.canvas.draw()
        pad_in = BOX_PAD * BOX_PT * PT
        box_texts, pls = [], []
        if C_PER_PANEL:
            for i, (ax_i, ls_i) in enumerate(zip(axes, boxlines)):
                xo_i = artist_occ([ax_i], fig, g["w"], g["h"])
                pls.append(place_box(S, kind, cnts, g["w"], g["h"], sizes[i][0], sizes[i][1], corner=corner, extra=xo_i, tag=f"{name}#{i}"))
        else:
            xo = artist_occ(axes, fig, g["w"], g["h"])
            p0 = place_box(S, kind, cnts, g["w"], g["h"], bw, bh, corner=corner, extra=xo, tag=name)
            pls = [p0] * len(axes)
        for i, p_ in enumerate(pls):
            if p_ is None:
                c_ = corner or "UR"
                pls[i] = dict(corner=c_, margin=0.0, x=(g["w"] - 0.05) if c_ in ("UR", "LR") else 0.05, y=(g["h"] - 0.05) if c_ in ("UR", "UL") else 0.05,
                              ha="right" if c_ in ("UR", "LR") else "left", va="top" if c_ in ("UR", "UL") else "bottom", dist=0.0, tier=0, soft_fail=True)
        for ax, ls, pl_ in zip(axes, boxlines, pls):
            sgn_x = -1 if pl_["ha"] == "right" else 1
            sgn_y = -1 if pl_["va"] == "top" else 1
            t = ax.text((pl_["x"] + sgn_x * pad_in) / g["w"], (pl_["y"] + sgn_y * pad_in) / g["h"], "\\n".join(ls),
                        transform=ax.transAxes, ha=pl_["ha"], va=pl_["va"], multialignment="left", fontproperties=f8,
                        fontsize=BOX_PT, linespacing=1.0, zorder=12,
                        bbox=dict(boxstyle=f"square,pad={BOX_PAD}", fc=BOX_FC, ec=BOX_EC, lw=BOX_LW))
            box_texts.append(t)
        pl = dict(corner=("per-panel:" + ",".join(p_["corner"] for p_ in pls)) if C_PER_PANEL else pls[0]["corner"],
                  margin=min(p_["margin"] for p_ in pls), x=pls[0]["x"], y=pls[0]["y"], tier=max(p_["tier"] for p_ in pls),
                  soft_fail=any(p_.get("soft_fail") for p_ in pls), each=[{k: v for k, v in p_.items()} for p_ in pls])
        import cp_lib as _C
        _C.LAST_TIER["t"] = pl["tier"]
'''
_src = _src[:_a] + _new + _src[_b:]
_t_old = "        ax.set_title(p.title, fontproperties=f8, fontsize=rs.TEXT_PT, pad=2.0, linespacing=1.0)\n"
assert _t_old in _src
_t_new = """        if "\\n" in p.title:
            _t1, _t2 = p.title.split("\\n", 1)
            _rend = fig.canvas.get_renderer()
            _sub = ax.annotate(_t2, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 2.0), textcoords="offset points",
                               ha="center", va="bottom", fontproperties=f8, fontsize=SUB_PT, annotation_clip=False)
            SUB_LIST.append(_sub)
            _hs = _sub.get_window_extent(_rend).height / fig.dpi * 72.0
            if LABELMODE_BOXES():  # v37：字形不同（域／邊）會使行高差約 0.1 pt；版面高度仍以 v36 用語（「邊界穿越」）量測，使圖尺寸與舊版逐像素相同
                _ref = ax.annotate(_t2.replace("域外來源", "邊界穿越"), xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 2.0),
                                   textcoords="offset points", ha="center", va="bottom", fontproperties=f8, fontsize=SUB_PT,
                                   annotation_clip=False)
                _hs = _ref.get_window_extent(_rend).height / fig.dpi * 72.0
                _ref.remove()
            _target = 2.0 + _hs + 1.4
            if "pad" not in _PADC or _PADC.get("hs") != round(_hs, 3):
                ax.set_title(_t1, fontproperties=f8, fontsize=rs.TEXT_PT, pad=_target, linespacing=1.0)
                fig.canvas.draw()
                _gap = (ax.title.get_window_extent(_rend).y0 - ax.get_window_extent(_rend).y1) / fig.dpi * 72.0
                _PADC["pad"] = _target + (_target - _gap)
                _PADC["hs"] = round(_hs, 3)
            ax.set_title(_t1, fontproperties=f8, fontsize=rs.TEXT_PT, pad=_PADC["pad"], linespacing=1.0)
            L4_SUBX[0] = (_hs + 1.4) / 72.0
        else:
            ax.set_title(p.title, fontproperties=f8, fontsize=rs.TEXT_PT, pad=2.0, linespacing=1.0)
"""
_src = _src.replace(_t_old, _t_new)
_c_old = "    for i, p in enumerate(panels):\n        r, c = divmod(i, ncol)\n"
assert _src.count(_c_old) == 1
_src = _src.replace(_c_old, "    SUB_LIST.clear()\n    L4_SUBX[0] = 0.0\n" + _c_old)
_f_old = "    ftexts, legends, caxs = [], [], []"
assert _src.count(_f_old) == 1
_src = _src.replace(_f_old, "    ftexts, legends, caxs = list(SUB_LIST), [], []")
_m_old = '    tit = [b[3] - b[1] for nme, b in texts if nme.endswith(".title")]'
assert _src.count(_m_old) == 1
_src = _src.replace(_m_old, '    tit = [b[3] - b[1] + L4_SUBX[0] for nme, b in texts if nme.endswith(".title")]')
L4.SUB_LIST, L4.L4_SUBX, L4.SUB_PT, L4._PADC = [], [0.0], 7.0, {}
SUB_LIST = L4.SUB_LIST
L4.C_PER_PANEL = False
L4.LABELMODE_BOXES = lambda: LABELMODE  # build3 內以全域名稱呼叫
exec(compile(_src, "cp_build3", "exec"), L4.__dict__)  # noqa: S102  重新定義 L4.build3


def set_per_panel(v: bool):
    L4.C_PER_PANEL = bool(v)
