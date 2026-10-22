#!/bin/bash
# WP-70：run09（含斯托克斯漂流）停留時數密度改用 1 km × 1 km 規則公尺網格的重算。
#   用法：run_fp1km.sh post      後處理（含足跡產品）輸出到 postprocess-fp1km/
#         run_fp1km.sh figures   用 postprocess-fp1km/ 出圖：figures-fp1km/（full）、figures-fp1km-notext/（none）
#         run_fp1km.sh regress   回歸檢查：用既有的 postprocess/（0.02° 網格）以目前程式碼出 none 版圖，
#                                輸出到 lbt-runs/scratch/wp70-regress/，供與 figures-notext-r3 逐位元比對
#   後處理與出圖的指令、參數、環境變數逐項沿用 run_formal_v8.sh，只多 --footprint-grid-km 1.0。
#   不覆蓋、不刪除任何既有輸出；輸出目錄已存在時中止。
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/../wp63_deploy/v8_env.sh"
STEP=${1:?用法：run_fp1km.sh post|figures|regress}
GRID_KM=${LBT_FP_GRID_KM:-1.0}

S=$(latest_population)
[ -n "$S" ] || die "找不到 v8 母體"
CFG=$S/release-configs/release-${HORIZON_DAYS}d-full_horizon_from_deposition.yaml
R=$B/$CASE
ISO=$B/isobaths.geojson
for f in "$CFG" "$B/inputs/identity_table.npz" "$B/inputs/region_to_flow_domain.json" "$R/native-plan.json" "$COAST" "$ISO"; do
  [ -e "$f" ] || die "缺少 $f"
done
export LBT_NATIVE_CONFIG=$CFG
export LBT_NATIVE_IDENTITY_TABLE=$B/inputs/identity_table.npz
export LBT_NATIVE_CHECKPOINT_EVERY_HOURS=24
export LBT_NATIVE_PLAN=$R/native-plan.json
export LBT_NATIVE_RESULTS_DIR=$R/results LBT_NATIVE_CHECKPOINT_ROOT=$R/checkpoints
export LBT_NATIVE_EXPERIMENT_CASE=$CASE
PP=$R/postprocess-fp1km
mkdir -p "$R/logs"

case "$STEP" in
  post)
    [ -e "$PP" ] && die "$PP 已存在，為避免覆蓋而中止"
    log "git commit：$(git -C "$CODE" rev-parse HEAD)（變更項數 $(git -C "$CODE" status --porcelain | wc -l)）"
    log "後處理（--footprint-grid-km $GRID_KM）→ $PP"
    OMP_NUM_THREADS=4 "$LBT" native-postprocess \
      --identity-table "$B/inputs/identity_table.npz" --results-dir "$R/results" \
      --output "$PP" --config "$CFG" --with-boundary-segments --formal \
      --plan "$R/native-plan.json" --project-root "$CODE" \
      --with-footprint --footprint-land-geojson "$COAST" --bathymetry-grid-root "$D/ocm_native" \
      --footprint-grid-km "$GRID_KM" \
      --region-flow-domain-map "$B/inputs/region_to_flow_domain.json" > "$R/logs/postprocess-fp1km.log" 2>&1
    RC=$?
    log "後處理結束，離開碼 $RC"
    exit $RC
    ;;
  figures)
    [ -d "$PP" ] || die "找不到 $PP，請先執行 post"
    RC=0
    for MODE in full none; do
      NAME=figures-fp1km; LOGN=figures-fp1km.log
      [ $MODE = none ] && { NAME=figures-fp1km-notext; LOGN=figures-fp1km-notext.log; }
      [ -e "$R/$NAME" ] && die "$R/$NAME 已存在，為避免覆蓋而中止"
      log "出圖（$MODE）→ $R/$NAME"
      "$LBT" native-figures --postprocess-dir "$PP" --coastline "$COAST" \
        --bathymetry "$ISO" --config "$CFG" --output-dir "$R/$NAME" --figure-text $MODE > "$R/logs/$LOGN" 2>&1 &
    done
    wait
    log "出圖結束"
    ;;
  regress)
    OUT=$W/lbt-runs/scratch/wp70-regress
    [ -e "$OUT" ] && die "$OUT 已存在，為避免覆蓋而中止"
    mkdir -p "$W/lbt-runs/scratch"
    log "回歸檢查：既有 postprocess/（0.02° 網格）以目前程式碼出 none 版圖 → $OUT"
    "$LBT" native-figures --postprocess-dir "$R/postprocess" --coastline "$COAST" \
      --bathymetry "$ISO" --config "$CFG" --output-dir "$OUT" --figure-text none > "$W/lbt-runs/scratch/wp70-regress.log" 2>&1
    log "回歸出圖結束，離開碼 $?"
    ;;
  *) die "未知步驟 $STEP" ;;
esac
