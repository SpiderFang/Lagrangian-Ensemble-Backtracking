"""WP-74：3-4-3 圖 1–7 重畫用之共用核對函式（資料指紋、文字稽核、SVG 文字字級）。

資料指紋只含「資料」：座標軸範圍、刻度值與刻度文字、軸標題、線條與標記之資料座標、多邊形頂點、散點位置、
圖例與註解文字；不含字級、線寬、面板位置，故版面或字級改變時指紋不變，資料或用語改變時指紋改變。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


def _arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes()).hexdigest()


def fingerprint(fig) -> dict:
    from matplotlib.collections import Collection

    fig.canvas.draw()
    parts, n_lines, n_patches, n_coll = [], 0, 0, 0
    for i, ax in enumerate(fig.get_axes()):
        if not ax.get_visible():
            continue
        rec = {
            "i": i,
            "xlim": [round(float(v), 9) for v in ax.get_xlim()],
            "ylim": [round(float(v), 9) for v in ax.get_ylim()],
            "xticks": [round(float(v), 9) for v in ax.get_xticks()],
            "yticks": [round(float(v), 9) for v in ax.get_yticks()],
            "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
            "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "title": ax.get_title(),
            "lines": [(_arr(ln.get_xdata(orig=False)), _arr(ln.get_ydata(orig=False)), str(ln.get_marker())) for ln in ax.lines],
            "patches": [
                _arr(p.get_patch_transform().transform(p.get_path().vertices)) for p in ax.patches
            ],
            "collections": [
                (
                    _arr(np.concatenate([q.vertices.reshape(-1, 2) for q in c.get_paths()])) if c.get_paths() else "",
                    _arr(c.get_offsets()) if hasattr(c, "get_offsets") and len(np.asarray(c.get_offsets())) else "",
                )
                for c in ax.collections
                if isinstance(c, Collection)
            ],
            "texts": [(t.get_text(), [round(float(v), 9) for v in getattr(t, "xy", ())]) for t in ax.texts],
            "legend": [t.get_text() for t in (ax.get_legend().get_texts() if ax.get_legend() else [])],
        }
        n_lines += len(rec["lines"])
        n_patches += len(rec["patches"])
        n_coll += len(rec["collections"])
        parts.append(rec)
    figrec = {
        "fig_texts": [t.get_text() for t in fig.texts],
        "fig_legends": [[t.get_text() for t in lg.get_texts()] for lg in fig.legends],
        "fig_legend_titles": [lg.get_title().get_text() for lg in fig.legends],
    }
    blob = json.dumps([parts, figrec], ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "sha256": hashlib.sha256(blob).hexdigest(),
        "axes": len(parts),
        "lines": n_lines,
        "patches": n_patches,
        "collections": n_coll,
        "legend_labels": [x for lg in figrec["fig_legends"] for x in lg] + [x for p in parts for x in p["legend"]],
        "axis_labels": [(p["xlabel"], p["ylabel"], p["title"]) for p in parts],
    }


def text_audit(fig) -> list[dict]:
    import matplotlib.font_manager as fm
    from matplotlib.text import Text

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    out = []
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        try:
            bb = t.get_window_extent(rend)
        except Exception:
            continue
        if bb.x1 < 0 or bb.y1 < 0 or bb.x0 > W or bb.y0 > H or bb.width <= 0:
            continue
        path = fm.findfont(t.get_fontproperties())
        out.append(
            {
                "text": s,
                "size_pt": float(t.get_fontsize()),
                "font_file": Path(path).name,
                "family": fm.get_font(path).family_name,
                "mathtext": "$" in s,
            }
        )
    return out


def text_summary(audit: list[dict]) -> dict:
    return {
        "n_texts": len(audit),
        "sizes_pt": sorted({a["size_pt"] for a in audit}),
        "min_size_pt": min((a["size_pt"] for a in audit), default=None),
        "font_families": sorted({a["family"] for a in audit}),
        "font_files": sorted({a["font_file"] for a in audit}),
        "mathtext_texts": [a["text"] for a in audit if a["mathtext"]],
    }


def clearance_in(fig) -> dict:
    """圖上全部可見文字、刻度、圖例之外框離畫布四邊的距離（英寸，須為正）。"""

    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())  # inch
    w, h = fig.get_size_inches()
    return {
        "left": round(float(tb.x0), 4),
        "top": round(float(h - tb.y1), 4),
        "right": round(float(w - tb.x1), 4),
        "bottom": round(float(tb.y0), 4),
    }


def axes_boxes_in(fig) -> list[list[float]]:
    w, h = fig.get_size_inches()
    out = []
    for ax in fig.get_axes():
        b = ax.get_position()
        out.append([round(b.x0 * w, 4), round(b.y0 * h, 4), round(b.width * w, 4), round(b.height * h, 4)])
    return out


def svg_text_audit(svg_path: Path) -> dict:
    """SVG（svg.fonttype=none 者）內 <text> 之字級與字型；路徑字型（fonttype=path）者無文字元素，回傳 n=0。"""

    s = Path(svg_path).read_text(encoding="utf-8")
    sizes, fams, n = set(), set(), 0
    for m in re.finditer(r"<text\b[^>]*style=\"([^\"]*)\"", s):
        n += 1
        mm = re.search(r"font:\s*(?:[a-z]+\s+)*([0-9.]+)px\s+'?([^';]+)'?", m.group(1)) or re.search(
            r"font-size:\s*([0-9.]+)px", m.group(1)
        )
        if mm:
            sizes.add(float(mm.group(1)))
            if mm.lastindex and mm.lastindex >= 2:
                fams.add(mm.group(2).strip())
    return {"n_text_elements": n, "sizes_px_eq_pt": sorted(sizes), "families": sorted(fams)}
