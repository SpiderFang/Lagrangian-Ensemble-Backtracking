#!/bin/bash
# WP-51 整合：v4（逐成員沉底年齡）正式重算。仿 run_formal_v3.sh：
#   H90、M=10、full_horizon_from_deposition，no_stokes 先、finite_depth_stokes 後依序執行；
#   每案 native-run -> native-postprocess（含 WP-38 足跡產品）-> native-figures（full 與 none 兩版），
#   任一步失敗即停止（不繼續後續步驟或案例）。
#
#   用法：run_formal_v4.sh --check   只檢查前置條件並印出擴散組態與將使用的路徑，不建立任何檔案
#         run_formal_v4.sh           正式開跑（需先執行 prep_v4.sh）
#   環境變數：LBT_WORKERS（預設 42）、LBT_CODE（程式碼 clone，見 v4_env.sh）
#
# 本腳本「準備但尚未執行」：不得在主代理核准前啟動。
# 續跑：native-run 讀同一個 checkpoint 目錄接續；擴散組態與 checkpoint 簽章不符時會拒絕續跑。
# 後處理與出圖的輸出目錄已存在時中止（不覆蓋）；要重做請先把舊目錄改名。
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/v4_env.sh"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

require_clean_code
S=$(latest_population)
[ -n "$S" ] || die "找不到 v4 母體（$V4ROOT/*_final-sites_h30-h60-h90_common_m10_v4），請先執行 prep_v4.sh"
CFG=$S/release-configs/release-90d-full_horizon_from_deposition.yaml
[ -f "$CFG" ] || die "找不到 release config：$CFG"
for f in "$B/inputs/identity_table.npz" "$B/inputs/region_to_flow_domain.json" \
         "$B/no_stokes/native-plan.json" "$B/finite_depth_stokes/native-plan.json" "$COAST"; do
  [ -f "$f" ] || die "缺少 $f（prep_v4.sh 尚未完成？）"
done
export LBT_NATIVE_CONFIG=$CFG
export LBT_NATIVE_IDENTITY_TABLE=$B/inputs/identity_table.npz
export LBT_NATIVE_CHECKPOINT_EVERY_HOURS=24
SHA=$(sha256sum "$(readlink -f "${CFG}")" | cut -d" " -f1)
WORKERS=${LBT_WORKERS:-42}
ISO=$B/isobaths.geojson

DIFF_JSON=$(resolve_and_check_diffusion) || die "擴散組態與預期不符（見上方訊息）"
if [ $CHECK_ONLY = 1 ]; then
  log "檢查模式：程式碼 $CODE（$(git -C "$CODE" rev-parse --short HEAD)）、母體 $S、輸出 $B、worker $WORKERS"
  log "擴散組態：$DIFF_JSON"
  for CASE in no_stokes finite_depth_stokes; do
    for d in results checkpoints postprocess figures figures-notext; do
      [ -e "$B/$CASE/$d" ] && log "注意：$B/$CASE/$d 已存在（native-run 會續跑；後處理與出圖會因輸出已存在而中止）"
    done
  done
  exit 0
fi

mkdir -p "$B/logs"
record_provenance "$B/logs/run_formal_v4.provenance.txt"
cd "$CODE"
log "母體：$S"
log "程式碼：$CODE（$(git rev-parse --short HEAD)）；擴散組態：$DIFF_JSON"

if [ ! -e "$ISO" ]; then
  log "匯出等深線（跨案例共用）"
  "$PY" "$CODE/scripts/export_isobaths.py" --identity-table "$B/inputs/identity_table.npz" \
    --region-flow-domain-map "$B/inputs/region_to_flow_domain.json" --grid-root "$D/ocm_native" \
    --output "$ISO" > "$B/logs/isobaths.log" 2>&1 || { log "等深線匯出失敗，停止"; exit 3; }
fi

for CASE in no_stokes finite_depth_stokes; do
  R=$B/$CASE
  mkdir -p "$R/logs"
  for d in postprocess figures figures-notext; do
    if [ -e "$R/$d" ]; then log "$R/$d 已存在，為避免覆蓋而中止"; exit 2; fi
  done
  export LBT_NATIVE_PLAN=$R/native-plan.json
  export LBT_NATIVE_RESULTS_DIR=$R/results LBT_NATIVE_CHECKPOINT_ROOT=$R/checkpoints
  export LBT_NATIVE_EXPERIMENT_CASE=$CASE

  log "=== 案例 $CASE：native-run 開始（${WORKERS} worker）==="
  time "$LBT" native-run "$R/native-plan.json" \
    --output "$R/unit-records" \
    --resource-provider lagrangian_backtracking.native_runner.hour_major_provider:make_hour_major_resource_provider \
    --numba-cache-dir "${NUMBA_CACHE_DIR}" --baytrace-root "${LBT_BAYTRACE_ROOT}" \
    --project-root . --config-sha256 "${SHA}" --mp-context spawn --formal \
    --worker-count "${WORKERS}" > "$R/logs/native-run.log" 2>&1
  RC=$?
  log "案例 $CASE：native-run 結束，離開碼 $RC"
  if [ $RC -ne 0 ]; then log "native-run 失敗，停止後續步驟與案例"; exit $RC; fi

  log "案例 $CASE：後處理（含 WP-38 足跡產品與沉底時刻標籤分組）"
  OMP_NUM_THREADS=4 "$LBT" native-postprocess \
    --identity-table "$B/inputs/identity_table.npz" --results-dir "$R/results" \
    --output "$R/postprocess" --config "$CFG" --with-boundary-segments --formal \
    --plan "$R/native-plan.json" --project-root "$CODE" \
    --with-footprint --footprint-land-geojson "$COAST" --bathymetry-grid-root "$D/ocm_native" \
    --region-flow-domain-map "$B/inputs/region_to_flow_domain.json" > "$R/logs/postprocess.log" 2>&1
  RC=$?
  log "案例 $CASE：後處理結束，離開碼 $RC"
  if [ $RC -ne 0 ]; then log "後處理失敗，停止"; exit $RC; fi

  for MODE in full none; do
    NAME=figures; LOG=figures.log
    if [ $MODE = none ]; then NAME=figures-notext; LOG=figures-notext.log; fi
    log "案例 $CASE：出圖（$MODE）"
    "$LBT" native-figures --postprocess-dir "$R/postprocess" --coastline "$COAST" \
      --bathymetry "$ISO" --config "$CFG" --output-dir "$R/$NAME" --figure-text $MODE > "$R/logs/$LOG" 2>&1
    RC=$?
    log "案例 $CASE：出圖（$MODE）離開碼 $RC"
    if [ $RC -ne 0 ]; then log "出圖失敗，停止"; exit $RC; fi
  done
  log "案例 $CASE：完成；checkpoints $(du -sh "$R/checkpoints" 2>/dev/null | cut -f1)"
done
log "=== 兩案例全部完成 ==="
df -h /home | tail -1
