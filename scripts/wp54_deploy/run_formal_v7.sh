#!/bin/bash
# WP-54：v7（十類沉降速度每類 10 個候選、逐粒子均衡抽樣）正式重算。仿 run_formal_v6.sh：
#   H90、M=10、full_horizon_from_deposition，no_stokes 與 finite_depth_stokes 兩案同時執行（LBT_PARALLEL_CASES=1）或依序執行；
#   每案 native-run -> native-postprocess（含 WP-38 足跡產品）-> native-figures（full 與 none 兩版），
#   任一步失敗即停止（不繼續後續步驟或案例）。
#
#   用法：run_formal_v7.sh --check   只檢查前置條件（程式碼、資料根、磁碟、設定、擴散組態、將使用的路徑），不建立任何檔案
#         run_formal_v7.sh           正式開跑（需先執行 prep_v7.sh）
#   環境變數：
#     LBT_WORKERS（預設 42，序列模式用）、LBT_CODE（程式碼 clone，見 v7_env.sh）；
#     LBT_PARALLEL_CASES（預設 1）兩案同時跑，設 0 改依序（LBT_WORKERS_NO_STOKES 預設 20、LBT_WORKERS_STOKES 預設 26）；
#     LBT_PURGE_CHECKPOINTS（預設 0）設 1 時，某案 native-run、後處理與兩版出圖「全部成功」後，刪除該案自己的
#       $B/<案例>/checkpoints（只刪 run08 自己的；絕不碰 run04～run07）。搭配 LBT_PARALLEL_CASES=0 可把峰值磁碟用量
#       從約 70 GB 壓到約 45 GB；空間足夠時建議保持 0，以便日後驗證或續跑。
#     LBT_MIN_FREE_GB（預設 100）磁碟可用空間下限。
#
# 起動時記錄 git commit、擴散組態、設定檔與範本 SHA-256 到 $B/logs/run_formal_v7.provenance.txt。
# 續跑：native-run 讀同一個 checkpoint 目錄接續；擴散組態、逐粒子沉降速度指紋與 checkpoint 簽章不符時會拒絕續跑。
# 後處理與出圖的輸出目錄已存在時中止（不覆蓋）；要重做請先把舊目錄改名。
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/v7_env.sh"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

require_clean_code
require_v6_data_root
S=$(latest_population)
[ -n "$S" ] || die "找不到 v7 母體（$V7ROOT/*_final-sites_h30-h60-h90_common_m10_v7），請先執行 prep_v7.sh"
CFG=$S/release-configs/release-90d-full_horizon_from_deposition.yaml
[ -f "$CFG" ] || die "找不到 release config：$CFG"
for f in "$B/inputs/identity_table.npz" "$B/inputs/region_to_flow_domain.json" \
         "$B/no_stokes/native-plan.json" "$B/finite_depth_stokes/native-plan.json" "$COAST"; do
  [ -f "$f" ] || die "缺少 $f（prep_v7.sh 尚未完成？）"
done
export LBT_NATIVE_CONFIG=$CFG
export LBT_NATIVE_IDENTITY_TABLE=$B/inputs/identity_table.npz
export LBT_NATIVE_CHECKPOINT_EVERY_HOURS=24
require_v7_config "$CFG"
"$PY" "$CODE/scripts/wp54_deploy/check_identity_table_v7.py" "$B/inputs/identity_table.npz" --expect-per-candidate 1000 \
  || die "身分表的逐粒子沉降速度核對失敗"
SHA=$(sha256sum "$(readlink -f "${CFG}")" | cut -d" " -f1)
WORKERS=${LBT_WORKERS:-42}
ISO=$B/isobaths.geojson

DIFF_JSON=$(resolve_and_check_diffusion) || die "擴散組態與預期不符（見上方訊息）"
NCASE=2
[ "${LBT_PARALLEL_CASES:-1}" = 1 ] || { [ "${LBT_PURGE_CHECKPOINTS:-0}" = 1 ] && NCASE=1; }
# 續跑時已用掉的空間不必再算；這裡以「目前可用空間」對「本次仍需空間」的保守估計檢查。
require_disk "$NCASE"
if [ $CHECK_ONLY = 1 ]; then
  log "檢查模式：程式碼 $CODE（$(git -C "$CODE" rev-parse --short HEAD)）、母體 $S、輸出 $B、worker $WORKERS、並行案例 ${LBT_PARALLEL_CASES:-1}、刪 checkpoint ${LBT_PURGE_CHECKPOINTS:-0}"
  log "擴散組態：$DIFF_JSON"
  for CASE in no_stokes finite_depth_stokes; do
    for d in results checkpoints postprocess figures figures-notext; do
      [ -e "$B/$CASE/$d" ] && log "注意：$B/$CASE/$d 已存在（native-run 會續跑；後處理與出圖會因輸出已存在而中止）"
    done
  done
  exit 0
fi

mkdir -p "$B/logs"
record_provenance "$B/logs/run_formal_v7.provenance.txt"
cd "$CODE"
log "母體：$S"
log "程式碼：$CODE（$(git rev-parse --short HEAD)）；擴散組態：$DIFF_JSON"

if [ ! -e "$ISO" ]; then
  log "匯出等深線（跨案例共用）"
  "$PY" "$CODE/scripts/export_isobaths.py" --identity-table "$B/inputs/identity_table.npz" \
    --region-flow-domain-map "$B/inputs/region_to_flow_domain.json" --grid-root "$D/ocm_native" \
    --output "$ISO" > "$B/logs/isobaths.log" 2>&1 || { log "等深線匯出失敗，停止"; exit 3; }
fi

# 提交順序成本：沿用 run07 實測的分段成本（unit-costs.json，只影響動態佇列的提交順序，不改任何物理結果）。
# 分段 id 形如 hourmajor::<流場>__seg<K>of<N>__<內容雜湊>，v7 身分表不同、雜湊部分會變，所以以去掉雜湊的前綴
# 對應；前綴集合必須與 v7 計畫完全相同才沿用，否則沿用計畫檔內建估計。
reuse_unit_costs() {
  local CASE=$1
  local SRC=$RUN07/$CASE/unit-costs.json DST=$B/$CASE/unit-costs.json
  [ -f "$SRC" ] && [ ! -e "$DST" ] || return 0
  "$PY" - "$SRC" "$B/$CASE/native-plan.json" "$DST" <<'PYEOF'
import json, sys
src, plan, dst = sys.argv[1:4]
old = json.load(open(src))
plan_keys = list(json.load(open(plan))["hour_major_members"])
strip = lambda k: k.rsplit("__", 1)[0]
old_by_prefix = {strip(k): v for k, v in old.items()}
new_prefixes = {strip(k): k for k in plan_keys}
if len(old_by_prefix) == len(old) and len(new_prefixes) == len(plan_keys) and set(old_by_prefix) == set(new_prefixes):
    json.dump({new_prefixes[p]: old_by_prefix[p] for p in new_prefixes}, open(dst, "w"), indent=1)
    print("沿用 run07 的 unit-costs.json（%d 段，依去雜湊前綴對應）" % len(plan_keys))
else:
    print("run07 的 unit-costs 分段前綴與 v7 計畫不同（%d 對 %d），不沿用" % (len(old_by_prefix), len(new_prefixes)))
PYEOF
}

# 刪除某案自己的 checkpoints（僅限 $B 之下，且僅在該案全流程成功後）。
purge_case_checkpoints() {
  local R=$1
  [ "${LBT_PURGE_CHECKPOINTS:-0}" = 1 ] || return 0
  case "$R/checkpoints" in
    "$B"/no_stokes/checkpoints|"$B"/finite_depth_stokes/checkpoints) ;;
    *) log "拒絕刪除不在 $B 之下的路徑：$R/checkpoints"; return 1 ;;
  esac
  [ -d "$R/checkpoints" ] || return 0
  log "案例完成，刪除自己的 checkpoints：$R/checkpoints（$(du -sh "$R/checkpoints" 2>/dev/null | cut -f1)）"
  rm -rf -- "$R/checkpoints"
}

# 單一案例的完整流程（native-run -> 後處理 -> 出圖 full／none）。以子 shell 執行，環境變數彼此隔離，
# 因此兩案可以同時跑。成功回傳 0；失敗回傳該步驟的離開碼（不繼續該案後續步驟）。
run_case() {
  local CASE=$1 CASE_WORKERS=$2
  local R=$B/$CASE RC
  mkdir -p "$R/logs"
  local d
  for d in postprocess figures figures-notext; do
    if [ -e "$R/$d" ]; then log "$R/$d 已存在，為避免覆蓋而中止"; return 2; fi
  done
  export LBT_NATIVE_PLAN=$R/native-plan.json
  export LBT_NATIVE_RESULTS_DIR=$R/results LBT_NATIVE_CHECKPOINT_ROOT=$R/checkpoints
  export LBT_NATIVE_EXPERIMENT_CASE=$CASE

  reuse_unit_costs "$CASE"
  local UC_ARGS=()
  if [ -f "$R/unit-costs.json" ]; then UC_ARGS=(--unit-costs "$R/unit-costs.json"); log "案例 $CASE：使用實測成本排序 $R/unit-costs.json"; fi

  log "=== 案例 $CASE：native-run 開始（${CASE_WORKERS} worker）==="
  time "$LBT" native-run "$R/native-plan.json" \
    --output "$R/unit-records" \
    --resource-provider lagrangian_backtracking.native_runner.hour_major_provider:make_hour_major_resource_provider \
    --numba-cache-dir "${NUMBA_CACHE_DIR}" --baytrace-root "${LBT_BAYTRACE_ROOT}" \
    --project-root . --config-sha256 "${SHA}" --mp-context spawn --formal \
    --worker-count "${CASE_WORKERS}" "${UC_ARGS[@]}" > "$R/logs/native-run.log" 2>&1
  RC=$?
  log "案例 $CASE：native-run 結束，離開碼 $RC"
  if [ $RC -ne 0 ]; then log "案例 $CASE：native-run 失敗，停止後續步驟"; return $RC; fi

  log "案例 $CASE：後處理（含 WP-38 足跡產品、沉底時刻標籤分組與逐粒子沉降速度欄位）"
  OMP_NUM_THREADS=4 "$LBT" native-postprocess \
    --identity-table "$B/inputs/identity_table.npz" --results-dir "$R/results" \
    --output "$R/postprocess" --config "$CFG" --with-boundary-segments --formal \
    --plan "$R/native-plan.json" --project-root "$CODE" \
    --with-footprint --footprint-land-geojson "$COAST" --bathymetry-grid-root "$D/ocm_native" \
    --region-flow-domain-map "$B/inputs/region_to_flow_domain.json" > "$R/logs/postprocess.log" 2>&1
  RC=$?
  log "案例 $CASE：後處理結束，離開碼 $RC"
  if [ $RC -ne 0 ]; then log "案例 $CASE：後處理失敗，停止"; return $RC; fi

  local MODE NAME LOG
  for MODE in full none; do
    NAME=figures; LOG=figures.log
    if [ $MODE = none ]; then NAME=figures-notext; LOG=figures-notext.log; fi
    log "案例 $CASE：出圖（$MODE）"
    "$LBT" native-figures --postprocess-dir "$R/postprocess" --coastline "$COAST" \
      --bathymetry "$ISO" --config "$CFG" --output-dir "$R/$NAME" --figure-text $MODE > "$R/logs/$LOG" 2>&1
    RC=$?
    log "案例 $CASE：出圖（$MODE）離開碼 $RC"
    if [ $RC -ne 0 ]; then log "案例 $CASE：出圖失敗，停止"; return $RC; fi
  done
  log "案例 $CASE：完成；checkpoints $(du -sh "$R/checkpoints" 2>/dev/null | cut -f1)"
  purge_case_checkpoints "$R" || return 1
  return 0
}

if [ "${LBT_PARALLEL_CASES:-1}" = 1 ]; then
  # 兩案同時跑（縮短牆鐘）：worker 總數 = 兩案相加，預設 20 + 26 = 46；含 Stokes 案總機時較多，所以多分幾個 worker。
  # A 區每 worker 約 2.4 GB，最壞情況全是 A 區約 110 GB（主機 125 GB）。
  log "兩案同時執行：no_stokes ${LBT_WORKERS_NO_STOKES:-20} worker、finite_depth_stokes ${LBT_WORKERS_STOKES:-26} worker"
  ( run_case no_stokes "${LBT_WORKERS_NO_STOKES:-20}" ) & P1=$!
  ( run_case finite_depth_stokes "${LBT_WORKERS_STOKES:-26}" ) & P2=$!
  wait $P1; R1=$?
  wait $P2; R2=$?
  log "兩案結束：no_stokes 離開碼 $R1、finite_depth_stokes 離開碼 $R2"
  if [ $R1 -ne 0 ] || [ $R2 -ne 0 ]; then exit 1; fi
else
  for CASE in no_stokes finite_depth_stokes; do
    ( run_case "$CASE" "$WORKERS" ) || { RC=$?; log "案例 $CASE 失敗（離開碼 $RC），停止後續案例"; exit $RC; }
  done
fi
log "=== 兩案例全部完成 ==="
df -h /home | tail -1
