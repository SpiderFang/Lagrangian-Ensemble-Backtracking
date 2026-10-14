#!/bin/bash
# WP-55：對 run08（v7）兩案產出「沿開放邊界之相對來源權重曲線（依類別、季節、大小潮分線）」的圖與量化表。
# 使用時機：run08 的 chain（scripts/wp54_deploy/run_formal_v7.sh）全部跑完、兩案的 postprocess 都已存在之後。
# 一行執行：
#   bash ~/Workspace/lbt-repo-wp55-fig/scripts/wp55_deploy/run_arc_by_group_run08.sh
#
# 本腳本只「讀」run08 兩案的 postprocess 目錄，並把結果寫到各案目錄下的新子目錄
#   <run08>/<案例>/figures-arc-by-group/
# 該子目錄必須尚不存在（指令本身也會拒絕覆寫）；不會碰 run08 自己的 figures、figures-notext 與任何其他檔案。
# 程式碼來自 lbt-repo-wp55-fig（WP-55 分支），與 run08 使用的 lbt-repo-wp54 是兩份獨立工作樹，互不修改。
# 可用環境變數覆寫：LBT_RUN08（run08 根目錄）、LBT_CODE55（本程式碼 clone）、LBT_ARC_DPI（預設 600）、
#   LBT_ARC_FORCE_WHILE_RUNNING=1（略過「chain 仍在執行」的檢查，不建議）。

set -u
W=$HOME/Workspace
CODE55=${LBT_CODE55:-$W/lbt-repo-wp55-fig}
B=${LBT_RUN08:-$W/lbt-runs/run08-h90-m10-v7}
PY=$W/lbt-repo/.venv/bin/python
LBT=$W/lbt-repo/.venv/bin/lbt
DPI=${LBT_ARC_DPI:-600}
OUTNAME=figures-arc-by-group

log() { echo "[$(date -u "+%F %T")Z] $*"; }
die() { log "中止：$*"; exit 2; }

[ -d "$CODE55/.git" ] || die "$CODE55 不是 git clone"
[ -z "$(git -C "$CODE55" status --porcelain)" ] || die "$CODE55 工作樹不乾淨"
grep -q "native-arc-by-group" "$CODE55/src/lagrangian_backtracking/cli.py" || die "$CODE55 不含 native-arc-by-group（不是 WP-55 版本）"
[ -x "$PY" ] && [ -x "$LBT" ] || die "找不到 $W/lbt-repo/.venv 的 python／lbt"

if [ "${LBT_ARC_FORCE_WHILE_RUNNING:-0}" != 1 ] && pgrep -f "[r]un_formal_v7.sh" > /dev/null; then
  die "run08 的 chain（run_formal_v7.sh）仍在執行；等它結束再跑（或設 LBT_ARC_FORCE_WHILE_RUNNING=1）"
fi

CASES=(no_stokes finite_depth_stokes)
ARGS=()
for CASE in "${CASES[@]}"; do
  PP=$B/$CASE/postprocess
  [ -f "$PP/departures.parquet" ] && [ -f "$PP/footprint_geometry_by_site.parquet" ] || die "$PP 缺少後處理產品（需含足跡產品）"
  [ ! -e "$B/$CASE/$OUTNAME" ] || die "$B/$CASE/$OUTNAME 已存在，不覆寫（請改名或移走後重跑）"
  ARGS+=(--case "$CASE" "$PP" "$B/$CASE/$OUTNAME")
done

export PYTHONPATH=$CODE55/src
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export MPLCONFIGDIR=${MPLCONFIGDIR:-$W/lbt-runs/mpl-cache}
mkdir -p "$MPLCONFIGDIR"

log "程式碼 $CODE55 @ $(git -C "$CODE55" rev-parse --short HEAD)；輸出 $B/<案例>/$OUTNAME；dpi $DPI"
mkdir -p "$B/logs"
LOG=$B/logs/arc-by-group.log
time "$LBT" native-arc-by-group "${ARGS[@]}" --dpi "$DPI" > "$LOG" 2>&1
RC=$?
log "native-arc-by-group 離開碼 $RC（日誌 $LOG）"
exit $RC
