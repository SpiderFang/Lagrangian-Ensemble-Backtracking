#!/bin/bash
# WP-63：等待鏈。條件滿足後自動執行 prep_v8.sh plan（含母體核對）與 run_formal_v8.sh；任一條件不成立或任一步失敗就停止並記錄原因。
#   條件 1：~/Workspace/lbt-runs/logs/build_v7_full_part2.log 出現 FINALIZE_DONE；
#   條件 2：lbt-local-data-v7-full/verify_result.json 的 errors 為 0（整數 0 或空清單）；
#   條件 3：母體已建好（prep_v8.sh population 完成），且 plan 階段的母體核對全數通過；
#   條件 4：run_formal_v8.sh 的前置檢查（資料根實檔、涵蓋月份、擴散組態、磁碟）通過。
# 用法（背景）：nohup bash wait_and_run_v8.sh > ~/Workspace/lbt-runs/run09-h180-m10-v8/logs/wait_and_run_v8.log 2>&1 < /dev/null &
# 環境變數：LBT_WAIT_POLL_S（預設 60）、LBT_WAIT_MAX_H（預設 48，逾時放棄並記錄）。
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
source "$HERE/v8_env.sh"
BUILD_LOG=$W/lbt-runs/logs/build_v7_full_part2.log
VERIFY=$D7/verify_result.json
BLOCK=$B/RUN_BLOCKED.txt
mkdir -p "$B/logs"
block() { log "不開跑：$*"; { echo "# run09 未開跑"; echo "- 時間（UTC）：$(date -u '+%F %T')Z"; echo "- 原因：$*"; } > "$BLOCK"; exit 1; }
[ -e "$B/finite_depth_stokes/unit-records" ] && block "$B/finite_depth_stokes/unit-records 已存在（run09 已開跑過？）"
rm -f "$BLOCK"
START=$(date +%s)
MAXS=$(( ${LBT_WAIT_MAX_H:-48} * 3600 ))
log "等待 FINALIZE_DONE（$BUILD_LOG）與 $VERIFY"
while true; do
  if [ -f "$BUILD_LOG" ] && grep -q FINALIZE_DONE "$BUILD_LOG" && [ -f "$VERIFY" ]; then break; fi
  [ $(( $(date +%s) - START )) -gt $MAXS ] && block "等待逾時（${LBT_WAIT_MAX_H:-48} 小時）仍未看到 FINALIZE_DONE 與 verify_result.json"
  sleep "${LBT_WAIT_POLL_S:-60}"
done
log "FINALIZE_DONE 已出現；檢查 verify_result.json"
ERRS=$("$PY" - "$VERIFY" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
e = d.get("errors", None)
if e is None: print("missing")
elif isinstance(e, bool): print(int(e))
elif isinstance(e, int): print(e)
else: print(len(e))
PYEOF
)
if [ "$ERRS" != 0 ]; then
  # 預設一律停止。只有啟動者明確用 LBT_VERIFY_ERRORS_OK=<錯誤數> 表示「已逐項審視並接受這個數量的錯誤」才放行（並寫入日誌）。
  [ -n "${LBT_VERIFY_ERRORS_OK:-}" ] && [ "$ERRS" = "$LBT_VERIFY_ERRORS_OK" ] \
    || block "verify_result.json 的 errors 不是 0（$ERRS）；若已審視並接受，請以 LBT_VERIFY_ERRORS_OK=$ERRS 重新啟動等待鏈"
  log "注意：verify_result.json 有 $ERRS 個錯誤，已由 LBT_VERIFY_ERRORS_OK 明示接受"
fi
S=$(latest_population)
[ -n "$S" ] || block "找不到 v8 母體"
log "資料驗證通過；開始 plan 階段（含母體核對）"
bash "$HERE/prep_v8.sh" plan > "$B/logs/prep_v8_plan.log" 2>&1 || block "prep_v8.sh plan 失敗（母體核對或資料根檢查未通過），見 $B/logs/prep_v8_plan.log"
bash "$HERE/run_formal_v8.sh" --check > "$B/logs/run_formal_v8_check.log" 2>&1 || block "run_formal_v8.sh --check 失敗，見 $B/logs/run_formal_v8_check.log"
log "全部條件滿足，開跑 run09"
nohup bash "$HERE/run_formal_v8.sh" > "$B/logs/run_formal_v8.log" 2>&1 < /dev/null &
log "run_formal_v8.sh 已在背景啟動（PID $!）；日誌 $B/logs/run_formal_v8.log"
