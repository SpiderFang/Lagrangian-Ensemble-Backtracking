#!/bin/bash
# WP-38 試產：run04 no_stokes 的子集（每站 2 個到達時刻 x 10 個沉降速度 = 20 個工作單元），
# 完整跑「後處理（含足跡產品）→ 等深線匯出 → 出圖（full／none）」，量測耗時與尖峰記憶體。
# 只寫 ~/Workspace/lbt-runs/checks/wp38/ 之下；不動 run04 的任何檔案、不動 lbt-repo。
# 用法：wp38_trial.sh <isobaths|post|figures|all>
set -uo pipefail
STEP=${1:-all}
W=$HOME/Workspace
CODE=$W/lbt-repo-wp38
B=$W/lbt-runs/run04-h90-m10-v3
R=$B/no_stokes
OUT=$W/lbt-runs/checks/wp38
S=$(ls -d $W/lbt-inputs/formal/v3-newsettling/*_final-sites_h30-h60-h90_common_m10_v3 | tail -1)
D=$W/lbt-local-data
CFG=$S/release-configs/release-90d-full_horizon_from_deposition.yaml
COAST=$W/lbt-inputs/coastline-taiwan.geojson
PY=$W/lbt-repo/.venv/bin/python
LBT=$W/lbt-repo/.venv/bin/lbt
export PYTHONPATH=$CODE/src
export OCM_NATIVE_ROOT=${D}/ocm_native NWW_ANALYSIS_ROOT=${D}/nww3_analysis OCM_RECONSTRUCTION_ROOT=${D}/reconstruction
export LBT_CHECKSUM_CACHE_DIR=$W/lbt-runs/checksum-cache LBT_BAYTRACE_ROOT=$W/lbt-runs/baytrace-v4.7.2
export NUMBA_CACHE_DIR=$W/lbt-runs/numba-cache PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=$W/lbt-runs/mpl-cache TMPDIR=$OUT/tmp
mkdir -p $OUT/tmp $OUT/logs
log() { echo "[$(date -u "+%F %T")Z] $*"; }
log "程式碼：$CODE（$(git -C $CODE rev-parse --short HEAD)）；載入路徑：$($PY -c 'import lagrangian_backtracking as m, os; print(os.path.dirname(m.__file__))')"

if [ "$STEP" = isobaths ] || [ "$STEP" = all ]; then
  log "匯出等深線"
  rm -f $OUT/isobaths.geojson
  /usr/bin/time -v $PY $CODE/scripts/export_isobaths.py --identity-table $B/inputs/identity_table.npz \
    --region-flow-domain-map $B/inputs/region_to_flow_domain.json --grid-root $D/ocm_native \
    --output $OUT/isobaths.geojson > $OUT/logs/isobaths.log 2>&1
  log "等深線離開碼 $?"
fi

if [ "$STEP" = post ] || [ "$STEP" = all ]; then
  log "後處理（含足跡產品）"
  mv $OUT/postprocess $OUT/postprocess.prev-$(date +%s) 2>/dev/null
  OMP_NUM_THREADS=4 /usr/bin/time -v $LBT native-postprocess \
    --identity-table $B/inputs/identity_table.npz --results-dir $OUT/subset \
    --trial-subset-from-results \
    --output $OUT/postprocess --config "$CFG" --with-boundary-segments \
    --plan $R/native-plan.json \
    --with-footprint --footprint-land-geojson $COAST --bathymetry-grid-root $D/ocm_native \
    --region-flow-domain-map $B/inputs/region_to_flow_domain.json > $OUT/logs/postprocess.log 2>&1
  log "後處理離開碼 $?"
fi

if [ "$STEP" = figures ] || [ "$STEP" = all ]; then
  for MODE in full none; do
    NAME=figures; [ $MODE = none ] && NAME=figures-notext
    log "出圖 $MODE"
    mv $OUT/$NAME $OUT/$NAME.prev-$(date +%s) 2>/dev/null
    /usr/bin/time -v $LBT native-figures --postprocess-dir $OUT/postprocess \
      --coastline $COAST --bathymetry $OUT/isobaths.geojson --output-dir $OUT/$NAME \
      --figure-text $MODE > $OUT/logs/$NAME.log 2>&1
    log "出圖 $MODE 離開碼 $?"
  done
fi
log "完成"
