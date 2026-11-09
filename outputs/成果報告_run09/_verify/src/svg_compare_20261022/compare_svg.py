"""新舊 SVG 之折線形狀比對：把新舊版同一條折線的頂點座標做「橫向、縱向各自一個比例與位移」的最小平方擬合（即座標軸改版所造成的唯一變化），
殘差（SVG 座標單位，1 單位 = 1/72 吋）極小且頂點數相同，即表示折線在資料座標系中的形狀與位置未變。
同一圖（多格圖則每一格）內的全部折線共用同一組比例與位移擬合。
用法：python compare_svg.py traj|polar OLD_DIR NEW_DIR  （目錄內含 <地點>/ 檔）。"""
import re, sys, json
import numpy as np
import xml.etree.ElementTree as ET

NS = "{http://www.w3.org/2000/svg}"

def style_of(el):
    d = {}
    for kv in (el.get("style") or "").split(";"):
        if ":" in kv:
            k, v = kv.split(":", 1); d[k.strip()] = v.strip()
    return d

def pts(dstr):
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", dstr)
    a = np.array([float(x) for x in nums]).reshape(-1, 2)
    return a

def collect(path, mode):
    root = ET.parse(path).getroot()
    out = []
    for el in root.iter(NS + "path"):
        d = el.get("d")
        if not d or "C" in d or "Q" in d:
            continue
        st = style_of(el)
        if st.get("fill", "none") != "none":
            continue
        sw = float(st.get("stroke-width", "0").replace("px", "") or 0)
        if mode == "traj":
            if st.get("stroke-opacity") == "0.85" and "stroke-width" not in st:
                out.append(("traj", st["stroke"], pts(d)))
        else:
            a = pts(d)
            # 資料折線：36 個扇區 + 閉合 = 37 個頂點；含白色外描邊（同形狀、不同顏色），兩者都收
            if len(a) == 37 and st.get("stroke", "") not in ("#222222", "#555555", "#000000", "#7f7f7f"):
                out.append(("polar", st["stroke"], a))
    if mode == "traj":
        # 起點與終止標記：<use> 元素的位置
        pass
    return out

def fit(a, b):
    res = []
    for k in (0, 1):
        A = np.vstack([a[:, k], np.ones(len(a))]).T
        coef, *_ = np.linalg.lstsq(A, b[:, k], rcond=None)
        res.append(float(np.abs(A @ coef - b[:, k]).max()))
    return max(res)

def main():
    mode, old, new = sys.argv[1:4]
    import glob, os
    report = {}
    for site in ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang"):
        if mode == "traj":
            names = ["horizontal_overview"]
        else:
            names = ["polar_source_v2_by_class", "polar_source_v2_by_season", "polar_source_v2_by_tide", "polar_source_v2_velocity_panels"]
        for n in names:
            o = collect(f"{old}/{site}/{n}.svg", mode); w = collect(f"{new}/{site}/{n}.svg", mode)
            ok = len(o) == len(w) and all(x[1] == y[1] and len(x[2]) == len(y[2]) for x, y in zip(o, w))
            g = 20 if n.endswith("velocity_panels") else max(len(o), 1)   # 同一組共用一個橫向、一個縱向的比例與位移
            mx = None
            if ok:
                mx = max(fit(np.vstack([x[2] for x in o[i:i + g]]), np.vstack([y[2] for y in w[i:i + g]])) for i in range(0, len(o), g)) if o else float("nan")
            report[f"{site}/{n}"] = {"paths_old": len(o), "paths_new": len(w), "same_count_colors_vertices": ok, "max_fit_residual_svg_units": mx}
    print(json.dumps(report, ensure_ascii=False, indent=1))
main()
