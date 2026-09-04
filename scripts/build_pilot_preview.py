"""獨立先導預覽命令列；只讀已完成 pilot，不接入正式 report-build 或修改來源。

操作端明示 run、config 與新的 output 目錄，必要時提供原 checkpoint root。所有路徑
只用於本次 I/O；成功摘要不輸出私有路徑，失敗回傳非零狀態。MPLCONFIGDIR 須預先設定，
字型可選；無中文字型時只將圖面改用英文，不下載地圖或字型。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from lagrangian_backtracking.pilot_preview import PilotPreviewError, build_pilot_preview


def main(argv: Sequence[str] | None = None) -> int:
    """解析明示路徑與容量上限，回傳 0 或 2，不自行建立快取或放寬來源驗證。"""

    parser = argparse.ArgumentParser(description="建立完整 pilot_exact 的獨立工程預覽")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--max-particles", type=int, default=2000)
    parser.add_argument("--max-observations", type=int, default=250_000)
    parser.add_argument("--max-curves-per-vertical", type=int, default=20)
    args = parser.parse_args(argv)
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
