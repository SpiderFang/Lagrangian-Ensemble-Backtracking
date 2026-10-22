#!/bin/bash
# WP-72：run09（含斯托克斯漂流）方法 B（回溯途中首次抵達海岸即視為找到來源）的後處理與出圖。
#   用法：run_coastB.sh post         後處理（--coast-arrival，1 km 網格足跡）→ postprocess-coastB/
#         run_coastB.sh smoke        冒煙測試：取前 N 個結果檔（LBT_SMOKE_N，預設 60）的符號連結，輸出到 lbt-runs/scratch/wp72_smoke/
#   後處理指令與參數逐項沿用 run_fp1km.sh post，只多 --coast-arrival。不覆蓋、不刪除任何既有輸出；輸出目錄已存在時中止。
set -uo pipefail
export LBT_CODE=${LBT_CODE:-$HOME/Workspace/lbt-repo-wp72}
source "$(dirname "$(readlink -f "$0")")/../wp63_deploy/v8_env.sh"
STEP=${1:?用法：run_coastB.sh post|smoke}
S=$(latest_population); [ -n "$S" ] || die "找不到 v8 母體"
CFG=$S/release-configs/release-${HORIZON_DAYS}d-full_horizon_from_deposition.yaml
R=$B/$CASE
for f in "$CFG" "$B/inputs/identity_table.npz" "$B/inputs/region_to_flow_domain.json" "$R/native-plan.json" "$COAST"; do [ -e "$f" ] || die "缺少 $f"; done
export LBT_NATIVE_CONFIG=$CFG LBT_NATIVE_IDENTITY_TABLE=$B/inputs/identity_table.npz LBT_NATIVE_CHECKPOINT_EVERY_HOURS=24
export LBT_NATIVE_PLAN=$R/native-plan.json LBT_NATIVE_RESULTS_DIR=$R/results LBT_NATIVE_CHECKPOINT_ROOT=$R/checkpoints LBT_NATIVE_EXPERIMENT_CASE=$CASE
mkdir -p "$R/logs"
common_args() {
  echo --identity-table "$B/inputs/identity_table.npz" --config "$CFG" --with-boundary-segments --formal \
    --plan "$R/native-plan.json" --project-root "$CODE" --with-footprint --footprint-land-geojson "$COAST" \
    --bathymetry-grid-root "$D/ocm_native" --footprint-grid-km 1.0 --region-flow-domain-map "$B/inputs/region_to_flow_domain.json" --coast-arrival
}
case "$STEP" in
  post)
    PP=$R/postprocess-coastB
    [ -e "$PP" ] && die "$PP 已存在，為避免覆蓋而中止"
    log "git commit：$(git -C "$CODE" rev-parse HEAD)（變更項數 $(git -C "$CODE" status --porcelain | wc -l)）"
    OMP_NUM_THREADS=4 "$LBT" native-postprocess $(common_args) --results-dir "$R/results" --output "$PP" > "$R/logs/postprocess-coastB.log" 2>&1
    log "後處理結束，離開碼 $?"
    ;;
  smoke)
    N=${LBT_SMOKE_N:-60}
    T=$W/lbt-runs/scratch/wp72_smoke; [ -e "$T" ] && die "$T 已存在"
    mkdir -p "$T/results"
    for f in $(ls "$R/results" | grep '^D__lienchiang\|^A__gongliao' | awk -v n="$N" 'NR%80==1' | head -n "$N"); do ln -s "$R/results/$f" "$T/results/$f"; done
    log "冒煙測試：$(ls "$T/results" | wc -l) 個結果檔"
    OMP_NUM_THREADS=4 "$LBT" native-postprocess $(common_args) --trial-subset-from-results --results-dir "$T/results" --output "$T/pp" > "$T/smoke.log" 2>&1
    log "冒煙測試結束，離開碼 $?"
    ;;
  *) die "未知步驟 $STEP" ;;
esac
