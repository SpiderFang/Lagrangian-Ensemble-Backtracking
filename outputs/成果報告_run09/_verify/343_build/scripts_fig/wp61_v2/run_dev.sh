#!/bin/bash
# 開發用：讀既有序列快取，只出無文字版總覽圖
source ~/Workspace/lbt-repo-wp63/scripts/wp63_deploy/v8_env.sh
S=$(latest_population)
export MPLCONFIGDIR=$W/lbt-runs/mpl-cache
export LBT_WORKSPACE=$W LBT_CODE_SRC=$CODE/src LBT_ARRIVAL_POP=$S LBT_ARRIVAL_DATA=$B/arrival-check/data LBT_ARRIVAL_PAIRED=0
export LBT_FIGS_NOTEXT_ONLY=1 LBT_FIGS_KINDS=overview
OUT=${1:?outdir}; SITES=${2:-gongliao,guishan,hsinchu,lienchiang,nanwan}
cd ~/Workspace/lbt-wp64-343/dev && $PY run_figs.py $SITES $OUT
