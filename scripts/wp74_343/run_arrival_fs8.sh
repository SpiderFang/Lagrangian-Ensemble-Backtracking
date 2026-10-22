#!/bin/bash
# WP-74：3-4-3 圖 3–7 以 8 pt 重畫（環境同 wp61 的 ctl_run.sh；前版＝commit b756ef7 之 scripts/wp61/arrival_figs.py，對照＝現行 3-4-3 嵌入圖之來源 figures-zh-v5）
# 用法：run_arrival_fs8.sh <輸出資料夾（須不存在或無同名檔）>
source ~/Workspace/lbt-repo-wp63/scripts/wp63_deploy/v8_env.sh
S=$(latest_population)
export MPLCONFIGDIR=$W/lbt-runs/mpl-cache
export LBT_WORKSPACE=$W LBT_CODE_SRC=$CODE/src LBT_ARRIVAL_POP=$S LBT_ARRIVAL_DATA=$B/arrival-check/data LBT_ARRIVAL_PAIRED=0
export LBT_FIGS_NOTEXT_ONLY=1 LBT_FIGS_KINDS=overview LBT_FIGS_VARIANTS=S
cd $CODE && $PY scripts/wp74_343/render_arrival_fs8.py gongliao,guishan,hsinchu,lienchiang,nanwan "$1" --prev-rev b756ef7 --control-dir $B/arrival-check/figures-zh-v5
