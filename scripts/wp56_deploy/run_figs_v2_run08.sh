#!/bin/bash
# WP-56：run08（v7）兩案以新版 native-figures 重新出圖，只寫入各案的 figures.v2／figures-notext.v2，
# 不覆蓋既有的 figures、figures-notext、figures-arc-by-group、postprocess（任何已存在的輸出目錄都會使本腳本中止）。
# 與 run07 的 figures.v3 重出圖（lbt-wp53/figs-v2/run_figs_v3.sh）相同的指令與參數，只換輸出目錄與程式碼來源：
# 程式碼來自本 clone（lbt-repo-wp55-fig，含 WP-56 沉降速度細分圖改版），不是 run08 當初用的 lbt-repo-wp54。
# 新版只改兩件事：沉降速度細分圖縱軸改標「類別顯示名稱＋該類實際速度範圍」，以及該圖的圖說與 README 文字；
# 其餘圖的資料與版面不變（出圖後請以像素或資料比對確認）。
# 用法：bash scripts/wp56_deploy/run_figs_v2_run08.sh   （兩案同時跑；nice -n 19）
# 可用環境變數：LBT_FIG_SUFFIX（預設 .v2）。

set -u
export LBT_CODE=${LBT_CODE:-$HOME/Workspace/lbt-repo-wp55-fig}
source "$LBT_CODE/scripts/wp54_deploy/v7_env.sh"
export MPLCONFIGDIR=${MPLCONFIGDIR:-$W/lbt-runs/mpl-cache}
SUFFIX=${LBT_FIG_SUFFIX:-.v2}

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
  ( one "$CASE" full "figures$SUFFIX"; one "$CASE" none "figures-notext$SUFFIX" ) &
done
wait
log ALLDONE
