# ruff: noqa: E501, E402
"""WP-74：3-4-3 圖 1–2 重畫（v5：8 pt、TRAPs 框線與刻度）與前版（v4，現行 3-4-3 嵌入圖之程式）逐項比對。

前版與新版各以相同輸入畫一次；前版輸出須與現行 3-4-3 嵌入圖（--control-pngs）逐位元相同（可重現），
新舊兩版存檔前之資料指紋（座標範圍、刻度、軸標題、線條與多邊形與散點之資料座標、圖例與註解文字）須相同。
用法::

    PYTHONPATH=<src> python compare_fig12.py --prev-script make_343_figs_run09_v4.py --new-script make_343_figs_run09.py \\
        --out-dir <新輸出> --control-png1 <現行圖1.png> --control-png2 <現行圖2.png> -- <make_343_figs 之參數（不含 --out-dir）>
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_audit as fa


def _load(path: str, name: str):
    p = Path(path).resolve()
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev-script", required=True)
    ap.add_argument("--new-script", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--control-png1", required=True)
    ap.add_argument("--control-png2", required=True)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = [x for x in a.rest if x != "--"]
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"{out} 已存在，不覆寫")
    prev, new = _load(a.prev_script, "make_343_prev"), _load(a.new_script, "make_343_new")
    fp_prev = {}
    orig = prev._save

    def hook(fig, stem, dpi):
        fp_prev[stem.name] = fa.fingerprint(fig)
        return orig(fig, stem, dpi)

    prev._save = hook
    tmp = Path(tempfile.mkdtemp(prefix="fig12_prev_"))
    prev.main([*rest, "--out-dir", str(tmp)])
    new.main([*rest, "--out-dir", str(out)])
    prefix = rest[rest.index("--out-prefix") + 1]
    res = {}
    for k, ctl in (("fig1_circle", a.control_png1), ("fig2_circle", a.control_png2)):
        stem = f"{prefix}_{k}"
        fp_new = new.AUDIT[stem]["data_fingerprint"]
        res[k] = {
            "prev_png_sha256": _sha(tmp / f"{stem}.png"),
            "control_png_sha256": _sha(Path(ctl)),
            "prev_equals_control": _sha(tmp / f"{stem}.png") == _sha(Path(ctl)),
            "data_fingerprint_equal": fp_prev[stem] == fp_new,
            "fingerprint_prev": fp_prev[stem],
            "fingerprint_new": fp_new,
            "new_png_sha256": _sha(out / f"{stem}.png"),
        }
        print(k, "前版＝現行嵌入圖" if res[k]["prev_equals_control"] else "前版≠現行嵌入圖！", "指紋相同" if res[k]["data_fingerprint_equal"] else "指紋不同！", new.AUDIT[stem]["text_summary"]["sizes_pt"], new.AUDIT[stem]["clearance_pt_left_top_right_bottom"], flush=True)
    (out / "compare_fig12.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
