#!/bin/bash
# WP-59：run08（v7）兩案以新版 native-figures 重新出圖（抽樣範圍由矩形虛線框改為以錨點為圓心、半徑 12.5 km 的測地線圓；龜山島圖面名稱改「龜山島」），只寫入各案的 figures.v5／figures-notext.v5，
# 不覆蓋既有的 figures、figures-notext、figures.v2／.v3／.v4、figures-notext.v2／.v3／.v4、figures-arc-by-group、postprocess（任何已存在的輸出目錄都會使本腳本中止）。
# 與 WP-58 的 run_figs_v4_run08.sh 相同的指令與參數，只換輸出後綴（預設 .v5）與程式碼分支（wp59-circle-range）。
# 程式碼來自本 clone（lbt-repo-wp55-fig，分支 wp59-circle-range）。
# 與 v4 相比只改：深藍色矩形虛線框 → 深藍色虛線圓（抽樣範圍，半徑 12.5 km 測地線圓），龜山島圖面名稱；
# 資料、座標範圍與版面不變（以資料指紋與去文字像素雜湊比對確認）。弧長圖（figures-arc-by-group）不受影響，不重出。
# 用法：bash scripts/wp59_deploy/run_figs_v5_run08.sh   （兩案同時跑；nice -n 19）
# 可用環境變數：LBT_FIG_SUFFIX（預設 .v5）。

set -u
export LBT_CODE=${LBT_CODE:-$HOME/Workspace/lbt-repo-wp55-fig}
source "$LBT_CODE/scripts/wp54_deploy/v7_env.sh"
export MPLCONFIGDIR=${MPLCONFIGDIR:-$W/lbt-runs/mpl-cache}
SUFFIX=${LBT_FIG_SUFFIX:-.v5}

[ -d "$CODE/.git" ] || die "$CODE 不是 git clone"
[ -z "$(git -C "$CODE" status --porcelain)" ] || die "$CODE 工作樹不乾淨，請先 commit"
[ -x "$LBT" ] || die "找不到 $LBT"
S=$(latest_population)
CFG=$S/release-configs/release-90d-full_horizon_from_deposition.yaml
ISO=$B/isobaths.geojson
[ -f "$CFG" ] && [ -f "$ISO" ] || die "找不到設定檔 $CFG 或等深線 $ISO"
log "程式碼 $CODE @ $(git -C "$CODE" rev-parse --short HEAD)；輸出後綴 $SUFFIX"

one() {
  local CASE=$1 MODE=$2 NAME=$3
  local R=$B/$CASE
  [ -e "$R/$NAME" ] && { log "$R/$NAME 已存在，中止"; return 2; }
  mkdir -p "$R/logs"
  log "start $CASE $MODE -> $NAME"
  nice -n 19 env OMP_NUM_THREADS=2 "$LBT" native-figures --postprocess-dir "$R/postprocess" --coastline "$COAST" \
    --bathymetry "$ISO" --config "$CFG" --output-dir "$R/$NAME" --figure-text "$MODE" > "$R/logs/$NAME.log" 2>&1
  local RC=$?
  log "done $CASE $MODE rc=$RC"
  return $RC
}
for CASE in no_stokes finite_depth_stokes; do
  for NAME in figures$SUFFIX figures-notext$SUFFIX; do
    [ ! -e "$B/$CASE/$NAME" ] || die "$B/$CASE/$NAME 已存在，不覆寫"
  done
done
for CASE in no_stokes finite_depth_stokes; do
  one "$CASE" full "figures$SUFFIX" &
  one "$CASE" none "figures-notext$SUFFIX" &
done
wait
log ALLDONE
