#!/bin/bash
# WP-75 v37：來源分段機率圖（全年、十類、季節、大小潮；邊界＋海岸合併版，報告用）一條指令重出。
#   用法：run_combined.sh <輸出根目錄> <地點|all> [圖種…]
#     地點：guishan gongliao hsinchu nanwan lienchiang，或 all（依報告站序逐地點執行）
#     圖種：all season tide class（預設全部；all＝全年圖）
#   例：  run_combined.sh ~/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/figures-segprob-v37 all
#         run_combined.sh ~/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/figures-segprob-v37 guishan
#   產出：<輸出根目錄>/<地點>/combined_all.png、combined_season_B.png、combined_tide_B.png、combined_class_B.png（及 mockup、manifest、summary）。
#   地點資料夾已存在者不覆蓋：指定單一地點時中止，all 時略過該地點。--legacy（置於最前）重現 v36（圈號＋數值框、「邊界穿越」用語），僅供比對。
#   不啟動任何粒子追蹤運算；只讀 run09 的 postprocess-coastB 結果。
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
WT=$(readlink -f "$HERE/../../..")
PY=${LBT_PY:-$HOME/Workspace/lbt-repo/.venv/bin/python}
LEGACY=""
[ "${1:-}" = "--legacy" ] && { LEGACY="--legacy"; shift; }
OUT=${1:?用法：run_combined.sh <輸出根目錄> <地點|all> [圖種…]}
SITE=${2:?缺地點}
shift 2
FIGS=("$@")
[ ${#FIGS[@]} -eq 0 ] && FIGS=(all season tide class)
if [ "$SITE" = all ]; then SITES=(guishan gongliao hsinchu nanwan lienchiang); else SITES=("$SITE"); fi
export PYTHONPATH=$WT/src:$WT/scripts/wp75_segprob MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
for s in "${SITES[@]}"; do
  if [ -e "$OUT/$s" ]; then
    # 指定單一地點時拒絕覆蓋；all 時略過已存在的地點（可分批執行，例如先出龜山島樣張、確認後再補其餘四地點）
    [ "$SITE" = all ] && { echo "略過 $s：$OUT/$s 已存在（不覆蓋）"; continue; }
    echo "中止：$OUT/$s 已存在（不覆蓋）" >&2; exit 2
  fi
  echo "== $s"
  "$PY" "$HERE/cp_run.py" $LEGACY "$OUT" "$s" B "${FIGS[@]}" || { echo "失敗：$s" >&2; exit 1; }
done
