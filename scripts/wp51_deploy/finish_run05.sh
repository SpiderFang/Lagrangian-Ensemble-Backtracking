#!/bin/bash
# WP-51 run05 收尾自動化 watcher（NE seg41 卡住事件，修正 commit 3312a0a）。
#
# 背景：東北區 2025-11-19/20 重建缺口 hvel 發散，兩案 NE seg41 的正式 worker 實質卡死。修正後的副本行程
# （PID 497698 no_stokes、498718 finite_depth_stokes，從 144 小時 checkpoint 續跑）把該段跑完，最終 checkpoint
# 在 ~/Workspace/lbt-wp51/debug/ckpt-<案>FIX/。本腳本每 INTERVAL 秒檢查一次，條件全部成立才動作：
#   (a) 兩個副本都已正常結束（日誌有 "DONE in"、無 Traceback）且各自 FIX checkpoint 顯示全部單元 finished；
#   (b) 兩案正式 unit-records 都只剩 NE seg41 未完成（167/168）。
# 動作（任一步失敗就停下，不繼續）：
#   備份原 NE seg41 checkpoint 與 provenance／native-run 日誌 → 凍結 worker 並確認所有 checkpoint 無寫到一半
#   → 只以 PID 結束 run_formal_v4.sh 與兩個 native-run 及其全部子孫 → 以 FIX checkpoint 覆蓋兩案正式 NE seg41
#   → lbt-repo-wp51-int fast-forward 到 3312a0a（在殺程序之後）→ 以原環境重啟 run_formal_v4.sh（附加輸出）
#   → 監看到重啟後的流程（含後處理與兩版出圖）結束並記錄結果。
# 副本異常結束時：寫明原因、停下，不動任何正式程序。
#
# 程序偵測範圍（2026-10-09 加註）：本腳本所有「程序」判斷都限定 run05——行程樹以 PID（MAIN_PID/新主程序）展開，
# native-run 計數只算新主程序子孫且命令列含 $RUN/ 者，孤兒清掃只清 cwd 為 $CODE（lbt-repo-wp51-int）者，
# 單元／checkpoint 狀態只讀 $RUN；與 run06（lbt-repo-wp52、run06-h90-m10-v5）互不影響。
#
# 用法：finish_run05.sh --dry-run   只印出條件狀態與將執行的動作（寫入 finish_run05.dryrun.log），不改任何東西
#       setsid nohup finish_run05.sh > /dev/null 2>&1 &     正式背景啟動（日誌 finish_run05.log）
set -uo pipefail

DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
W=$HOME/Workspace
RUN=$W/lbt-runs/run05-h90-m10-v4
CODE=$W/lbt-repo-wp51-int
DBG=$W/lbt-wp51/debug
BACKUP=$DBG/ckpt-orig-backup
BUNDLE=$W/lbt-runs/tools/fix-3312a0a.bundle
TARGET=3312a0aca61cefedc0e1438a2a4a55661761359e
UNIT="hourmajor::northeast_taiwan_common_cache_v3__seg41of42__b9daf000c53c"
MAIN_PID=${MAIN_PID:-483627}
REPLICA_NO=${REPLICA_NO:-497698}; REPLICA_FD=${REPLICA_FD:-498718}
CASES="no_stokes finite_depth_stokes"
INTERVAL=${FINISH_INTERVAL:-300}
OUT=$W/lbt-runs/logs/run05-formal_v4.out
if [ $DRY = 1 ]; then LOG=$W/lbt-runs/logs/finish_run05.dryrun.log; else LOG=$W/lbt-runs/logs/finish_run05.log; fi
TS=$(date -u +%Y%m%dT%H%M%SZ)

log() { echo "[$(date '+%F %T %z')]${DRYTAG:-} $*" | tee -a "$LOG"; }
die() { log "停止：$*（未再繼續後續步驟）"; exit 1; }
[ $DRY = 1 ] && DRYTAG=" [DRY-RUN]"

if [ $DRY = 0 ]; then
  exec 9>"$W/lbt-runs/tools/.finish_run05.lock"
  flock -n 9 || die "已有另一個 finish_run05.sh 在執行"
fi

replica_pid() { [ "$1" = no_stokes ] && echo $REPLICA_NO || echo $REPLICA_FD; }
fixdir() { echo "$DBG/ckpt-$1FIX/$UNIT"; }
fixlog() { echo "$DBG/dbg_fix_$1.log"; }
alive() { kill -0 "$1" 2>/dev/null; }
desc() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do echo "$c"; desc "$c"; done; }

# 正式 unit-records 狀態：輸出 "已完成數 失敗數 預期數 未完成清單"
unit_state() {
python3 - "$RUN/$1" <<'PYEOF'
import json, glob, sys
d = sys.argv[1]
exp = set(json.load(open(d + '/unit-costs.json')).keys())
comp, fail = set(), set()
for f in glob.glob(d + '/unit-records/*.done.json'):
    try: m = json.load(open(f))
    except Exception: continue
    (comp if m.get('status') == 'completed' else fail).add(m['unit_id'])
rem = sorted(exp - comp)
print(len(comp), len(fail), len(exp), ';'.join(rem) if len(rem) <= 3 else 'many:%d' % len(rem))
PYEOF
}
# FIX checkpoint：輸出 "未完成單元數 已走小時 current_utc_ns"
meta_state() {
python3 - "$1" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1] + '/meta.json'))
n = sum(1 for g in m['groups'] for u in g['units'] if not u['finished'])
print(n, m['master_hours_walked'], m['current_utc_ns'])
PYEOF
}
progress_of() { grep -o 'hour_end=[0-9T:-]*' "$(fixlog $1)" 2>/dev/null | tail -1 | cut -d= -f2; }

# 副本狀態：回傳 0=正常完成、1=執行中、2=異常；原因寫到 REASON
check_replica() {
  local c=$1 pid; pid=$(replica_pid $c); REASON=""
  if alive "$pid"; then REASON="執行中，目前回溯到 $(progress_of $c)"; return 1; fi
  local lg; lg=$(fixlog $c)
  [ -f "$lg" ] || { REASON="找不到副本日誌 $lg"; return 2; }
  grep -q '^Traceback' "$lg" && { REASON="副本日誌含 Traceback（異常結束）：$(grep -m1 -A0 -B0 'Error' "$lg" | cut -c1-200)"; return 2; }
  grep -q '^DONE in' "$lg" || { REASON="副本行程已不存在但日誌沒有 'DONE in'（被中斷或異常結束）"; return 2; }
  [ -f "$(fixdir $c)/meta.json" ] || { REASON="FIX checkpoint 不存在：$(fixdir $c)"; return 2; }
  local ms; ms=$(meta_state "$(fixdir $c)")
  [ "${ms%% *}" = 0 ] || { REASON="FIX checkpoint 仍有未完成單元（$ms）"; return 2; }
  REASON="已正常完成（checkpoint $ms）"; return 0
}

# 條件 (b)：回傳 0=兩案皆只剩 NE seg41、1=尚未、2=異常（失敗標記或已 168）
check_units() {
  local c st comp fail exp rem rc=0; UNITMSG=""
  for c in $CASES; do
    st=$(unit_state $c); read -r comp fail exp rem <<<"$st"
    UNITMSG+=" $c:完成$comp/$exp,失敗$fail,未完成[$rem]"
    [ "$exp" = 168 ] || { UNITMSG+=" (預期單元數不是 168!)"; rc=2; }
    [ "$fail" = 0 ] || { UNITMSG+=" (有 failed 標記!)"; rc=2; }
    [ "$comp" = 168 ] && { UNITMSG+=" (警告：已 168/168，不應發生)"; rc=2; }
    if [ "$rem" != "$UNIT" ] && [ $rc != 2 ]; then rc=1; fi
  done
  return $rc
}

preflight() {
  [ -f "$BUNDLE" ] || die "找不到 bundle $BUNDLE"
  git -C "$CODE" bundle verify "$BUNDLE" > /dev/null 2>&1 || die "bundle 驗證失敗"
  [ -z "$(git -C "$CODE" status --porcelain)" ] || die "$CODE 工作樹不乾淨"
  for c in $CASES; do
    [ -d "$RUN/$c/checkpoints/$UNIT" ] || die "找不到正式 checkpoint $RUN/$c/checkpoints/$UNIT"
    [ -e "$BACKUP/$c" ] && die "備份目錄 $BACKUP/$c 已存在（避免覆蓋原始備份，請人工確認）"
  done
  grep -q run_formal_v4.sh "/proc/$MAIN_PID/cmdline" 2>/dev/null || die "PID $MAIN_PID 不是 run_formal_v4.sh（或已不存在）"
  [ "$(cd /proc/$MAIN_PID/cwd 2>/dev/null && pwd -P)" = "$(cd $CODE && pwd -P)" ] || die "PID $MAIN_PID 的 cwd 不是 $CODE"
}

ckpt_consistent() {
  local c d u bad=0
  for c in $CASES; do
    d=$RUN/$c/checkpoints
    [ -n "$(find "$d" -name '*.tmp' 2>/dev/null | head -1)" ] && { log "  有 .tmp 暫存檔：$d"; bad=1; }
    for u in "$d"/*/; do
      if [ -f "$u/meta.json" ]; then
        [ -n "$(find "$u" -name 'group_*.npz' -newer "$u/meta.json" 2>/dev/null | head -1)" ] && { log "  checkpoint 寫到一半：$u"; bad=1; }
      elif [ -n "$(find "$u" -name 'group_*.npz' 2>/dev/null | head -1)" ]; then
        log "  checkpoint 首次寫入未完成：$u"; bad=1
      fi
    done
  done
  return $bad
}

step() { # step "描述" 命令...
  local d=$1; shift
  if [ $DRY = 1 ]; then log "將執行：$d"; return 0; fi
  log "開始：$d"
  if "$@" >> "$LOG" 2>&1; then log "完成：$d"; else die "失敗：$d"; fi
}

do_backup() {
  local c
  for c in $CASES; do
    mkdir -p "$BACKUP/$c" && cp -a "$RUN/$c/checkpoints/$UNIT" "$BACKUP/$c/" || return 1
    cp -a "$RUN/$c/logs/native-run.log" "$RUN/$c/logs/native-run.log.pre-restart-$TS" || return 1
  done
  cp -a "$RUN/logs/run_formal_v4.provenance.txt" "$RUN/logs/run_formal_v4.provenance.txt.pre-restart-$TS" || return 1
  cp -a "$OUT" "$OUT.pre-restart-$TS" || return 1
}

freeze_and_kill() {
  local pids i
  grep -q run_formal_v4.sh "/proc/$MAIN_PID/cmdline" 2>/dev/null || { echo "主程序已不在"; return 1; }
  pids=$(desc $MAIN_PID | sort -un | tr '\n' ' ')
  echo "目標 PID：主程序 $MAIN_PID；子孫 $(echo $pids | wc -w) 個"
  for i in $(seq 1 15); do
    kill -STOP $MAIN_PID $pids 2>/dev/null
    if ckpt_consistent; then
      echo "checkpoint 一致，結束行程"
      kill -KILL $MAIN_PID 2>/dev/null
      kill -KILL $pids 2>/dev/null
      break
    fi
    echo "第 $i 次檢查 checkpoint 不一致，解除凍結等 20 秒再試"
    kill -CONT $MAIN_PID $pids 2>/dev/null; sleep 20
    pids=$(desc $MAIN_PID | sort -un | tr '\n' ' ')
    [ $i = 15 ] && { echo "checkpoint 15 次都不一致，放棄（行程已解除凍結）"; return 1; }
  done
  for i in $(seq 1 30); do
    local left=0 p
    for p in $MAIN_PID $pids; do alive $p && left=$((left+1)); done
    [ $left = 0 ] && break; sleep 2
  done
  [ $left = 0 ] || { echo "仍有 $left 個行程未結束"; return 1; }
  # 掃描遺留孤兒：父行程為 1、cwd 是 $CODE 的 spawn worker／resource_tracker（只以 PID 結束）
  local orphans
  orphans=$(ps -eo pid=,ppid=,args= | awk '$2==1 && /spawn_main|resource_tracker/ {print $1}')
  for p in $orphans; do
    if [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$(cd $CODE && pwd -P)" ]; then echo "清除孤兒 PID $p"; kill -KILL $p 2>/dev/null; fi
  done
  sleep 2
  echo "全部結束"
}

swap_checkpoints() {
  local c dst
  for c in $CASES; do
    dst=$RUN/$c/checkpoints/$UNIT
    rm -rf "$dst.new" && cp -a "$(fixdir $c)" "$dst.new" || return 1
    diff -rq "$(fixdir $c)" "$dst.new" || return 1
    rm -rf "$dst" && mv "$dst.new" "$dst" || return 1
    echo "$c：已以 FIX checkpoint 覆蓋（$(meta_state $dst)）"
  done
}

ff_code() {
  git -C "$CODE" fetch "$BUNDLE" HEAD || return 1
  [ "$(git -C "$CODE" rev-parse FETCH_HEAD)" = "$TARGET" ] || { echo "bundle HEAD 不是 $TARGET"; return 1; }
  git -C "$CODE" merge --ff-only "$TARGET" || return 1
  [ "$(git -C "$CODE" rev-parse HEAD)" = "$TARGET" ] || return 1
  [ -z "$(git -C "$CODE" status --porcelain)" ] || { echo "ff 後工作樹不乾淨"; return 1; }
}

restart_run() {
  cd "$CODE" || return 1
  LBT_PARALLEL_CASES=1 setsid nohup bash scripts/wp51_deploy/run_formal_v4.sh >> "$OUT" 2>&1 < /dev/null &
  NEW_MAIN=$!
  echo "$NEW_MAIN" > "$W/lbt-runs/tools/finish_run05.newmain.pid"
  echo "已啟動 run_formal_v4.sh，PID $NEW_MAIN"
  sleep 90
  alive $NEW_MAIN || { echo "新主程序 90 秒內就結束了，見 $OUT"; return 1; }
  # 只計算「新主程序的子孫」中、命令列含 run05 輸出目錄路徑的 native-run；run06 等其他批次的 native-run
  # （不同 PID 樹、不同輸出目錄）不會被計入，也不會被誤判為 run05 已重啟。
  local n=0 p
  for p in $(desc $NEW_MAIN | sort -un); do
    if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "lbt native-run $RUN/"; then n=$((n+1)); fi
  done
  echo "run05 的 native-run 行程數（新主程序 $NEW_MAIN 的子孫、路徑含 $RUN/）：$n"
  [ "$n" -ge 2 ] || return 1
}

# ───────────────────────── 主流程 ─────────────────────────
log "=== finish_run05 啟動（dry-run=$DRY；PID $$；間隔 ${INTERVAL}s；目標 commit ${TARGET:0:7}）==="
preflight
[ $DRY = 1 ] || log "前置檢查通過"

while true; do
  check_replica no_stokes; ra=$?; RA="$REASON"
  check_replica finite_depth_stokes; rb=$?; RB="$REASON"
  check_units; rc=$?
  log "副本 no_stokes：$RA；副本 finite_depth_stokes：$RB；正式單元：$UNITMSG"
  if [ $ra = 2 ] || [ $rb = 2 ]; then
    [ $DRY = 1 ] && { log "[條件 a 異常] 副本異常；真正執行時腳本會停下並不動正式程序"; break; }
    die "副本異常結束（no_stokes：$RA；finite_depth_stokes：$RB），未動任何正式程序"
  fi
  if [ $rc = 2 ]; then
    [ $DRY = 1 ] && { log "[條件 b 異常] 真正執行時腳本會停下"; break; }
    die "正式 unit-records 異常（$UNITMSG）"
  fi
  if [ $DRY = 0 ]; then
    alive $MAIN_PID || die "正式主程序 $MAIN_PID 已不存在，條件尚未成立"
  fi
  if [ $ra = 0 ] && [ $rb = 0 ] && [ $rc = 0 ]; then log "條件 (a)(b) 全部成立"; break; fi
  [ $DRY = 1 ] && { log "條件尚未成立（a：no_stokes=$ra finite=$rb [0=完成 1=執行中]；b=$rc [0=成立 1=尚未]）；真正執行時會每 ${INTERVAL}s 再檢查"; break; }
  sleep "$INTERVAL"
done

if [ $DRY = 1 ]; then
  log "---- 條件成立後將依序執行的動作（以下用目前偵測到的資訊展開）----"
  log "目標行程樹：主程序 $MAIN_PID，子孫 PID $(desc $MAIN_PID | sort -un | tr '\n' ' ')"
fi
step "備份原 NE seg41 checkpoint 到 $BACKUP/<案>/，並備份 provenance、兩案 native-run.log、$OUT（後綴 .pre-restart-$TS）" do_backup
step "凍結全部 worker、確認所有 checkpoint 無寫到一半，然後只以 PID 結束 run_formal_v4.sh($MAIN_PID)、兩個 native-run 及全部子孫" freeze_and_kill
step "以 FIX checkpoint 覆蓋兩案正式 NE seg41 checkpoint（diff 驗證）" swap_checkpoints
step "lbt-repo-wp51-int fast-forward 到 ${TARGET:0:7}（git fetch bundle、merge --ff-only、確認工作樹乾淨）" ff_code
step "以原環境重啟：cd $CODE && LBT_PARALLEL_CASES=1 setsid nohup bash scripts/wp51_deploy/run_formal_v4.sh >> $OUT；確認 90 秒後仍在跑且有 2 個 native-run" restart_run
if [ $DRY = 1 ]; then
  log "之後：監看新主程序結束，檢查 '兩案例全部完成'、postprocess／figures／figures-notext 目錄，並統計 NE seg41 的 FORCING_MISSING 粒子數，寫入日誌"
  log "=== dry-run 結束（未改動任何東西）==="; exit 0
fi

# 階段 2：監看重啟後的流程直到結束
NEW_MAIN=$(cat "$W/lbt-runs/tools/finish_run05.newmain.pid")
log "階段 2：監看新主程序 $NEW_MAIN（native-run、後處理、出圖）"
while alive "$NEW_MAIN"; do sleep "$INTERVAL"; done
log "新主程序已結束；日誌尾端："; tail -n 8 "$OUT" | tee -a "$LOG"
ok=1
grep -q "兩案例全部完成" "$OUT" || ok=0
for c in $CASES; do for d in postprocess figures figures-notext; do
  [ -d "$RUN/$c/$d" ] && log "  $c/$d 存在" || { log "  $c/$d 不存在"; ok=0; }
done; done
python3 - <<'PYEOF' | tee -a "$LOG"
import glob, os, re, numpy as np
B = os.path.expanduser('~/Workspace/lbt-runs/run05-h90-m10-v4/')
LO = 1761220800000000000  # NE seg41 的最早到達時刻 2025-10-23T12:00Z；seg40 最晚到 10-23T08:00
for c in ('no_stokes', 'finite_depth_stokes'):
    n = fm = 0
    for f in glob.glob(B + c + '/results/A__*.driver_result.npz'):
        m = re.search(r'__arr(\d+)__', f)
        if not m or int(m.group(1)) < LO: continue
        fs = np.load(f)['final_status']; n += len(fs); fm += int(np.count_nonzero(fs == 5))
    print(c, 'NE seg41 粒子', n, '其中 FORCING_MISSING', fm)
PYEOF
[ $ok = 1 ] && log "=== 全部完成：native-run、後處理、出圖皆成功 ===" || log "=== 流程結束但有項目不成立，請人工檢查 $OUT 與各案 logs/ ==="
