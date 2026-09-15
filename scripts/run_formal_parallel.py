"""在匯入科學套件前設定 Numba cache 的正式平行 CLI 啟動器。

此 wrapper 讓 SERVER 可用追蹤版腳本啟動 run-formal-parallel，並可在匯入套件／Numba
前讀取操作者明示的 ``--numba-cache-dir``。實際路徑、儲存 gate、run plan 與 provenance
仍由共用 CLI coordinator 嚴格驗證；本檔不建立 workspace、不修改科學設定或啟動 SERVER。
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path


def _early_numba_cache(argv: Sequence[str]) -> None:
    """只為了讓 Numba 首次匯入前讀到 CLI 指定值，不在驗證前建立目錄。"""

    for index, value in enumerate(argv):
        if value == "--numba-cache-dir" and index + 1 < len(argv):
            os.environ["NUMBA_CACHE_DIR"] = argv[index + 1]
            return
        if value.startswith("--numba-cache-dir="):
            os.environ["NUMBA_CACHE_DIR"] = value.split("=", 1)[1]
            return


def main(argv: Sequence[str] | None = None) -> int:
    """將平行 runner 選項轉交正式 CLI，並在此前準備本機 source import path。"""

    arguments = list(sys.argv[1:] if argv is None else argv)
    _early_numba_cache(arguments)
    if arguments and arguments[0] == "run-formal-parallel":
        arguments = arguments[1:]
    source_root = Path(__file__).resolve(strict=True).parent.parent / "src"
    sys.path.insert(0, str(source_root))

    # 延遲到 cache 環境值設定完成後才匯入 CLI，避免頂層 science import 先讀取預設 HOME cache。
    from lagrangian_backtracking.cli import main as cli_main

    return int(cli_main(["run-formal-parallel", *arguments]))


if __name__ == "__main__":
    raise SystemExit(main())
