#!/bin/bash
# WP-38：單一案例的後處理（含足跡產品）與兩種圖面文字版本出圖，全量正式重算用。
# 與 run_post_figures_wp37.sh 的差異：程式碼改用 WP-38 的乾淨工作樹（lbt-repo-wp38，需為 git clean）、
# 後處理加 --with-footprint／--bathymetry-grid-root／--region-flow-domain-map、
# 出圖加 --bathymetry；輸出一律寫到新目錄 *.wp38，不覆蓋任何既有輸出（目錄已存在就中止）。
set -uo pipefail
CASE=${1:?用法：run_post_figures_wp38.sh <no_stokes|finite_depth_stokes>}
W=$HOME/Workspace
CODE=$W/lbt-repo-wp38
cd $CODE
B=$W/lbt-runs/run04-h90-m10-v3
S=$(ls -d $W/lbt-inputs/formal/v3-newsettling/*_final-sites_h30-h60-h90_common_m10_v3 | tail -1)
D=$W/lbt-local-data
CFG=$S/release-configs/release-90d-full_horizon_from_deposition.yaml
COAST=$W/lbt-inputs/coastline-taiwan.geojson
PY=$W/lbt-repo/.venv/bin/python
LBT=$W/lbt-repo/.venv/bin/lbt
export PYTHONPATH=$CODE/src
export LBT_NATIVE_CONFIG=$CFG
export OCM_NATIVE_ROOT=${D}/ocm_native NWW_ANALYSIS_ROOT=${D}/nww3_analysis OCM_RECONSTRUCTION_ROOT=${D}/reconstruction
export LBT_CHECKSUM_CACHE_DIR=$W/lbt-runs/checksum-cache LBT_BAYTRACE_ROOT=$W/lbt-runs/baytrace-v4.7.2
export NUMBA_CACHE_DIR=$W/lbt-runs/numba-cache PYTHONDONTWRITEBYTECODE=1
export LBT_NATIVE_IDENTITY_TABLE=$B/inputs/identity_table.npz
export LBT_NATIVE_SMAGORINSKY_IMPL=numba
export LBT_NATIVE_BACKTRACK_MODE=full_horizon_from_deposition
export LBT_NATIVE_FORMAL=1
export LBT_NATIVE_EXPERIMENT_CASE=$CASE
R=$B/$CASE
export LBT_NATIVE_RESULTS_DIR=$R/results LBT_NATIVE_CHECKPOINT_ROOT=$R/checkpoints
export LBT_NATIVE_PLAN=$R/native-plan.json
mkdir -p $R/logs
log() { echo "[$(date -u "+%F %T")Z] $*"; }
for d in postprocess.wp38 figures.wp38 figures-notext.wp38; do
  if [ -e $R/$d ]; then log "$R/$d 已存在，為避免覆蓋而中止"; exit 2; fi
done
log "程式碼來源：$CODE（$(git -C $CODE rev-parse --short HEAD)；git status：$(git -C $CODE status --porcelain | wc -l) 項變更）"

ISO=$B/isobaths.wp38.geojson
if [ ! -e $ISO ]; then
  log "匯出等深線（跨案例共用，已存在則沿用）"
  $PY $CODE/scripts/export_isobaths.py --identity-table $B/inputs/identity_table.npz \
    --region-flow-domain-map $B/inputs/region_to_flow_domain.json --grid-root $D/ocm_native \
    --output $ISO > $B/logs-isobaths.wp38.log 2>&1 || { log "等深線匯出失敗"; exit 3; }
fi

log "案例 $CASE：後處理（含足跡產品）"
OMP_NUM_THREADS=4 /usr/bin/time -v $LBT native-postprocess \
  --identity-table $B/inputs/identity_table.npz --results-dir $R/results \
  --output $R/postprocess.wp38 --config "$CFG" --with-boundary-segments --formal \
  --plan $R/native-plan.json --project-root $CODE \
  --with-footprint --footprint-land-geojson $COAST --bathymetry-grid-root $D/ocm_native \
  --region-flow-domain-map $B/inputs/region_to_flow_domain.json > $R/logs/postprocess.wp38.log 2>&1
RC=$?
log "案例 $CASE：後處理結束，離開碼 $RC"
if [ $RC -ne 0 ]; then log "後處理失敗，停止"; exit $RC; fi

for MODE in full none; do
  NAME=figures.wp38; LOG=figures.wp38.log
  if [ $MODE = none ]; then NAME=figures-notext.wp38; LOG=figures-notext.wp38.log; fi
  log "案例 $CASE：出圖（$MODE）"
  /usr/bin/time -v $LBT native-figures --postprocess-dir $R/postprocess.wp38 --coastline $COAST \
    --bathymetry $ISO --output-dir $R/$NAME --figure-text $MODE > $R/logs/$LOG 2>&1
  log "案例 $CASE：出圖（$MODE）離開碼 $?"
done
log "案例 $CASE：完成；figures.wp38 $(ls $R/figures.wp38 2>/dev/null | wc -l) 項、figures-notext.wp38 $(ls $R/figures-notext.wp38 2>/dev/null | wc -l) 項"
