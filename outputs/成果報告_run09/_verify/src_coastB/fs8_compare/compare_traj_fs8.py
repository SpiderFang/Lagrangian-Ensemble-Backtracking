"""水平軌跡總覽新舊版之軌跡比對（資料座標系）：先由 SVG 內刻度標記的位置與刻度文字，求出 SVG 座標轉經度、緯度的線性關係，
再把每條軌跡（stroke-opacity 0.85 之折線）轉成經緯度；舊版（-coastB）與報告版（-coastB-fs8）的同一條軌跡，其端點相同，且每個頂點到另一版折線之最短距離
（度）極小（SVG 輸出會依畫面解析度簡化共線頂點，所以頂點數可不同）。
用法：python compare_traj.py OLD_DIR NEW_DIR"""
import re, sys, json
import numpy as np
import xml.etree.ElementTree as ET
NS = "{http://www.w3.org/2000/svg}"

def tick_fit(txt, axis):
    # 每個刻度群組：<g id="xtick_N"> ... <use ... x="..." y="..."/> ... <!-- 文字 -->
    pat = re.compile(r'<g id="%stick_\d+">(.*?)</g>\s*</g>\s*</g>' % axis, re.S)
    vals, pos = [], []
    for m in pat.finditer(txt):
        blk = m.group(1)
        u = re.search(r'<use [^>]*? %s="([-\d.]+)"' % ('x' if axis == 'x' else 'y'), blk)
        lab = re.search(r'<!-- ([\d.]+)(?:°[EN])? -->', blk)
        if u and lab:
            pos.append(float(u.group(1))); vals.append(float(lab.group(1)))
    A = np.vstack([pos, np.ones(len(pos))]).T
    (a, b), res, *_ = np.linalg.lstsq(A, np.array(vals), rcond=None)
    err = float(np.abs(A @ np.array([a, b]) - vals).max())
    return a, b, err, len(vals)

def traj_paths(root):
    out = []
    for el in root.iter(NS + "path"):
        st = dict(kv.split(":", 1) for kv in (el.get("style") or "").replace(" ", "").split(";") if ":" in kv)
        if st.get("fill") == "none" and st.get("stroke-opacity") == "0.85":
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", el.get("d"))
            out.append(np.array([float(x) for x in nums]).reshape(-1, 2))
    return out

def to_lonlat(paths, fx, fy):
    return [np.column_stack([fx[0] * p[:, 0] + fx[1], fy[0] * p[:, 1] + fy[1]]) for p in paths]

def dist_to_poly(P, Q):
    # P 的每個點到折線 Q 的最短距離（度；經緯度當平面座標，僅作極小量比較）
    A, B = Q[:-1], Q[1:]
    d = B - A; L2 = (d ** 2).sum(1); L2[L2 == 0] = 1e-30
    best = np.full(len(P), np.inf)
    for i in range(0, len(P), 500):
        p = P[i:i + 500, None, :]
        t = np.clip(((p - A) * d).sum(2) / L2, 0, 1)
        proj = A + t[..., None] * d
        best[i:i + 500] = np.sqrt(((p - proj) ** 2).sum(2)).min(1)
    return best

def main():
    old, new = sys.argv[1:3]
    rep = {}
    for site in ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang"):
        res = {}
        data = {}
        for tag, d in (("old", old), ("new", new)):
            txt = open(f"{d}/{site}/horizontal_overview.svg", encoding="utf-8").read()
            fx = tick_fit(txt, "x"); fy = tick_fit(txt, "y")
            ps = to_lonlat(traj_paths(ET.fromstring(txt.encode("utf-8"))), fx, fy)
            data[tag] = ps; res[f"tick_fit_max_err_deg_{tag}"] = max(fx[2], fy[2]); res[f"ticks_{tag}"] = (fx[3], fy[3])
        o, n = data["old"], data["new"]
        res["paths"] = (len(o), len(n))
        if len(o) == len(n):
            res["max_endpoint_diff_deg"] = max(float(max(np.abs(a[0] - b[0]).max(), np.abs(a[-1] - b[-1]).max())) for a, b in zip(o, n))
            res["max_vertex_to_other_polyline_deg"] = max(float(max(dist_to_poly(a, b).max(), dist_to_poly(b, a).max())) for a, b in zip(o, n))
        rep[site] = res
    print(json.dumps(rep, ensure_ascii=False, indent=1))
main()
