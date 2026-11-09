#!/bin/bash
# WP-76（v37）：水平軌跡總覽（報告圖 8 型）重畫——圖例用語改為「域外來源」「海岸來源」「回溯起點（受體位置）」，數據、座標軸與版面不變。
#   用法：run_traj_v37.sh <輸出目錄>
#   例：  run_traj_v37.sh ~/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/figures-notext-traj-v37
#   五個地點一次畫完（<輸出目錄>/<地點>/horizontal_overview.png／.svg）；輸出目錄已存在者拒絕執行（不覆蓋）。
#   資料與參數同 scripts/wp73_deploy/render_fs8.py traj（postprocess-coastB、24 條代表軌跡、dpi 600、無圖內說明文字）。不啟動任何粒子追蹤運算。
set -uo pipefail
CODE=$(readlink -f "$(dirname "$0")/../..")
PY=${LBT_PY:-$HOME/Workspace/lbt-repo/.venv/bin/python}
OUT=${1:?用法：run_traj_v37.sh <輸出目錄>}
[ -e "$OUT" ] && { echo "中止：$OUT 已存在（不覆蓋）" >&2; exit 2; }
PYTHONPATH=$CODE/src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg exec "$PY" "$CODE/scripts/wp73_deploy/render_fs8.py" traj "$OUT"
