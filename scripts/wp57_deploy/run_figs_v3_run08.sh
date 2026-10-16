#!/bin/bash
# WP-57：run08（v7）兩案以新版 native-figures 重新出圖（圖內文字改依報告用語），只寫入各案的 figures.v3／figures-notext.v3，
# 不覆蓋既有的 figures、figures-notext、figures.v2、figures-notext.v2、figures-arc-by-group、postprocess（任何已存在的輸出目錄都會使本腳本中止）。
# 與 WP-56 的 run_figs_v2_run08.sh 相同的指令與參數，只換輸出後綴（預設 .v3）與程式碼分支（wp57-fig-terms）。
# 程式碼來自本 clone（lbt-repo-wp55-fig，分支 wp57-fig-terms）。
# 新版只改圖內文字：粒子→虛擬質點、站點→研究地點、單位改英文符號（m、h、d、°、m/s、km²、%/km）、英文縮寫 HDR 不再出現；
# 資料、座標範圍與版面不變（以資料指紋與去文字像素雜湊比對確認）。
# 用法：bash scripts/wp57_deploy/run_figs_v3_run08.sh   （兩案同時跑；nice -n 19）
# 可用環境變數：LBT_FIG_SUFFIX（預設 .v3）。

set -u
export LBT_CODE=${LBT_CODE:-$HOME/Workspace/lbt-repo-wp55-fig}
source "$LBT_CODE/scripts/wp54_deploy/v7_env.sh"
export MPLCONFIGDIR=${MPLCONFIGDIR:-$W/lbt-runs/mpl-cache}
SUFFIX=${LBT_FIG_SUFFIX:-.v3}

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
