"""獨立先導預覽命令列；只讀已完成 pilot，不接入正式 report-build 或修改來源。

操作端明示 run、config 與新的 output 目錄，必要時提供原 checkpoint root。所有路徑
只用於本次 I/O；成功摘要不輸出私有路徑，失敗回傳非零狀態。MPLCONFIGDIR 須預先設定，
字型可選；無中文字型時只將圖面改用英文，不下載地圖或字型。只有明示已驗證的
``--storage-gate-evidence`` 才會啟用 NFS completion-marker 發布；同時必須提供
base commit、Git tree object、dirty file 清單與 diff hash，讓完成標記不會掩蓋程式碼漂移。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from lagrangian_backtracking.pilot_preview import PilotPreviewError, build_pilot_preview


def main(argv: Sequence[str] | None = None) -> int:
    """解析明示路徑與容量上限，回傳 0 或 2，不自行建立快取或放寬來源驗證。"""

    parser = argparse.ArgumentParser(description="建立 pilot_exact 或受限 full 單站的獨立工程預覽")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--max-particles", type=int, default=2000)
    parser.add_argument("--max-observations", type=int, default=250_000)
    parser.add_argument("--max-curves-per-vertical", type=int, default=20)
    parser.add_argument(
        "--storage-gate-evidence",
        type=Path,
        help=(
            "已驗證且含九個 NFS roots、write_probe 與 same_host_flock_probe PASS 的 "
            "storage gate snapshot；明示後才啟用 NFS marker"
        ),
    )
    parser.add_argument("--base-commit", help="完成標記綁定的基準 Git commit（40 位 hex）")
    parser.add_argument("--base-tree", help="完成標記綁定的基準 Git tree object（40 位 hex）")
    parser.add_argument(
        "--dirty-file",
        action="append",
        default=[],
        help="repository-relative dirty file；可重複指定，會排序後寫入完成標記",
    )
    parser.add_argument("--diff-sha256", help="目前允許 dirty files 的 binary diff SHA-256")
    args = parser.parse_args(argv)
    release_provenance = None
    provenance_values = (args.base_commit, args.base_tree, args.diff_sha256)
    if any(value is not None for value in provenance_values) or args.dirty_file:
        release_provenance = {
            "base_commit": args.base_commit,
            "base_tree": args.base_tree,
            # CLI 可重複指定同一檔案；在進入共用驗證器前排序並去重，讓完成標記
            # 的 provenance bytes 穩定且不因參數順序產生不同發布指紋。
            "dirty_files": sorted(set(args.dirty_file)),
            "diff_sha256": args.diff_sha256,
        }
    try:
        manifest = build_pilot_preview(
            args.run,
            config_path=args.config,
            output=args.output,
            checkpoint_root=args.checkpoint_root,
            font_path=args.font_path,
            max_particles=args.max_particles,
            max_observations=args.max_observations,
            max_curves_per_vertical=args.max_curves_per_vertical,
            storage_gate_evidence=args.storage_gate_evidence,
            nfs_marker_protocol=args.storage_gate_evidence is not None,
            release_provenance=release_provenance,
        )
    except (PilotPreviewError, FileExistsError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "valid": True,
                "run_id": manifest["run_id"],
                "artifact_kind": manifest["artifact_kind"],
                "output_count": len(manifest["files"]) + 1,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
