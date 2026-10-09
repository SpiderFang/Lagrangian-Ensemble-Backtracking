#!/usr/bin/env python3
"""WP-53：把四個 domain 各自建好的 patch（reconstruction_parts/<domain>/<domain>）合成單一 patch 根。

做法：把每個 ``<parts>/<domain>/<domain>`` 目錄 rename 到 ``<output>/<domain>``（同一檔案系統，不複製資料），
再用 ``write_root_index`` 寫根索引，最後逐域 ``validate_reconstruction_patch``。輸出根必須不存在。
"""
import argparse, json, os, sys
from pathlib import Path
from lagrangian_backtracking.ocm_reconstruction import validate_reconstruction_patch
from lagrangian_backtracking.ocm_reconstruction_v2 import write_root_index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    parts, out = Path(a.parts), Path(a.output)
    if out.exists():
        raise SystemExit(f"輸出已存在，拒絕覆寫：{out}")
    out.mkdir(parents=True)
    manifests = []
    for p in sorted(parts.iterdir()):
        dom = p.name
        src = p / dom
        if not (src / "reconstruction-manifest.json").is_file():
            raise SystemExit(f"{src} 缺少 domain manifest（該域尚未建好？）")
        os.rename(src, out / dom)
        sha = (out / dom / "reconstruction-manifest.json.sha256").read_text().strip()
        manifests.append({"flow_id": dom, "manifest_sha256": sha})
    write_root_index(out, manifests)
    for m in manifests:
        validate_reconstruction_patch(out / m["flow_id"])
        print("validate ok", m["flow_id"], flush=True)
    print(json.dumps(manifests, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
