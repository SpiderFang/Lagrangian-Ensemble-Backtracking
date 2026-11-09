# ruff: noqa: E501, E402
"""WP-74：3-4-3 圖 3–7（到達時間驗證圖，全年總覽）以 8 pt／TRAPs 規格重畫，並逐站核對「只差字級與字型」。

做法：每個地點各畫兩次——前版程式（git 指定 commit 之 scripts/wp61/arrival_figs.py，即現行 3-4-3 嵌入圖之程式）畫到暫存資料夾作對照，
現行程式畫到輸出資料夾；兩次都在存檔前以 fig_audit 取資料指紋與文字稽核。manifest 記錄兩者之指紋是否相同、字級與字型、
畫布像素、面板位置、文字外框離畫布邊距，以及前版輸出與現行 3-4-3 嵌入圖（--control-dir）之 sha256 是否逐位元相同。

用法（環境同 ctl_run.sh／run_dev.sh：LBT_CODE_SRC、LBT_ARRIVAL_POP、LBT_ARRIVAL_DATA、LBT_ARRIVAL_PAIRED=0）::

    python scripts/wp74_343/render_arrival_fs8.py gongliao,guishan,hsinchu,lienchiang,nanwan <輸出資料夾> \\
        --prev-rev b756ef7 --control-dir <figures-zh-v5>
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "wp61"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("LBT_CODE_SRC", str(REPO / "src")))

import arrival_check as ac
import arrival_figs as af
import fig_audit as fa
from PIL import Image


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_prev(rev: str):
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:scripts/wp61/arrival_figs.py"], check=True, capture_output=True, text=True).stdout
    tmp = Path(tempfile.mkdtemp(prefix="arrival_figs_prev_")) / "arrival_figs_prev.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("arrival_figs_prev", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, hashlib.sha256(src.encode("utf-8")).hexdigest()


def _render(mod, site, S, R, fonts, plan, out: Path):
    rec = {}
    orig = mod._save

    def hook(fig, path):
        rec["fingerprint"] = fa.fingerprint(fig)
        audit = fa.text_audit(fig)
        rec["text_audit"] = audit
        rec["text_summary"] = fa.text_summary(audit)
        rec["clearance_in"] = fa.clearance_in(fig)
        rec["canvas_in"] = [round(float(v), 4) for v in fig.get_size_inches()]
        rec["axes_boxes_in_left_bottom_w_h"] = fa.axes_boxes_in(fig)
        return orig(fig, path)

    mod._save = hook
    try:
        mod.render_overview(site, S[site], R[site], fonts, plan, "S", False, out)
    finally:
        mod._save = orig
    im = Image.open(out)
    rec["png_px"] = list(im.size)
    rec["png_sha256"] = _sha(out)
    rec["svg_text"] = fa.svg_text_audit(out.with_suffix(".svg"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sites")
    ap.add_argument("outroot")
    ap.add_argument("--prev-rev", required=True)
    ap.add_argument("--control-dir", required=True)
    a = ap.parse_args()
    sites = a.sites.split(",")
    outroot = Path(a.outroot)
    prev, prev_sha = _load_prev(a.prev_rev)
    fonts, fonts_prev = af.Fonts(), prev.Fonts()
    recs = ac.load_manifest_records()
    S = {s: ac.load_series(s) for s in ac.SITES}
    R = {s: ac.analyze_site(s, recs[s], S[s], None) for s in ac.SITES}
    stats = {s: af.site_stats(S[s], R[s]["rows"], R[s]["idx_of"], None) for s in ac.SITES}
    man = {
        "prev_rev": a.prev_rev,
        "prev_arrival_figs_sha256": prev_sha,
        "new_arrival_figs_sha256": _sha(Path(af.__file__)),
        "font": fonts.record,
        "variant": "S",
        "figures": {},
    }
    tmp = Path(tempfile.mkdtemp(prefix="arrival_prev_out_"))
    for site in sites:
        plan = af.axis_plan(stats, "S", site)
        assert plan == prev.axis_plan(stats, "S", site)
        (outroot / site).mkdir(parents=True, exist_ok=True)
        (tmp / site).mkdir(parents=True, exist_ok=True)
        name = f"wp61_{site}_overview_S_notext.png"
        p_prev = _render(prev, site, S, R, fonts_prev, plan, tmp / site / name)
        out = outroot / site / name
        for p in (out, out.with_suffix(".svg")):
            if p.exists():
                raise SystemExit(f"{p} 已存在，不覆寫")
        p_new = _render(af, site, S, R, fonts, plan, out)
        ctl = Path(a.control_dir) / site / name
        man["figures"][f"{site}/{name}"] = {
            "axis_plan": {k: list(v) for k, v in plan.items()},
            "prev_png_sha256": p_prev["png_sha256"],
            "control_png_sha256": _sha(ctl),
            "prev_equals_control": p_prev["png_sha256"] == _sha(ctl),
            "data_fingerprint_equal": p_prev["fingerprint"] == p_new["fingerprint"],
            "fingerprint_new": p_new["fingerprint"],
            "fingerprint_prev": p_prev["fingerprint"],
            "png_px_prev": p_prev["png_px"],
            "png_px_new": p_new["png_px"],
            "png_sha256": p_new["png_sha256"],
            "canvas_in_new": p_new["canvas_in"],
            "canvas_in_prev": p_prev["canvas_in"],
            "axes_boxes_new": p_new["axes_boxes_in_left_bottom_w_h"],
            "axes_boxes_prev": p_prev["axes_boxes_in_left_bottom_w_h"],
            "clearance_in_new": p_new["clearance_in"],
            "text_summary_new": p_new["text_summary"],
            "text_summary_prev": p_prev["text_summary"],
            "svg_text_new": p_new["svg_text"],
            "svg_text_prev": p_prev["svg_text"],
            "text_audit_new": p_new["text_audit"],
        }
        print(site, "指紋相同" if p_prev["fingerprint"] == p_new["fingerprint"] else "指紋不同！", "前版＝對照" if p_prev["png_sha256"] == _sha(ctl) else "前版≠對照！", p_new["text_summary"]["sizes_pt"], p_new["clearance_in"], flush=True)
    (outroot / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
