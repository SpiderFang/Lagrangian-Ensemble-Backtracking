#!/bin/bash
# WP-38c：以既有 postprocess.wp38 為輸入、帶正式釋放設定（--config）重新出圖，輸出到新目錄 *.wp38c，
# 不覆蓋 run04 的任何既有輸出。用法：run_figures_wp38c.sh <no_stokes|finite_depth_stokes> <full|none>
set -uo pipefail
CASE=${1:?案例}; MODE=${2:?full|none}
W=$HOME/Workspace
CODE=$W/lbt-repo-wp38
B=$W/lbt-runs/run04-h90-m10-v3
R=$B/$CASE
S=$(ls -d $W/lbt-inputs/formal/v3-newsettling/*_final-sites_h30-h60-h90_common_m10_v3 | tail -1)
CFG=$S/release-configs/release-90d-full_horizon_from_deposition.yaml
COAST=$W/lbt-inputs/coastline-taiwan.geojson
ISO=$B/isobaths.wp38.geojson
export PYTHONPATH=$CODE/src PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=$W/lbt-runs/mpl-cache
NAME=figures.wp38c; LOG=figures.wp38c.log
if [ $MODE = none ]; then NAME=figures-notext.wp38c; LOG=figures-notext.wp38c.log; fi
[ -e $R/$NAME ] && { echo "$R/$NAME 已存在，中止"; exit 2; }
echo "[$(date -u '+%F %T')Z] $CASE $MODE 開始；程式碼 $(git -C $CODE rev-parse --short HEAD)；config sha256 $(sha256sum $CFG | cut -d' ' -f1)"
/usr/bin/time -v $W/lbt-repo/.venv/bin/lbt native-figures --postprocess-dir $R/postprocess.wp38 \
  --coastline $COAST --bathymetry $ISO --config $CFG --output-dir $R/$NAME --figure-text $MODE > $R/logs/$LOG 2>&1
echo "[$(date -u '+%F %T')Z] $CASE $MODE 離開碼 $?"
