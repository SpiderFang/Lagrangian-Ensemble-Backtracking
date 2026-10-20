#!/bin/bash
# WP-63：run09 的補充出圖（在 native-postprocess 與 native-figures 之後執行；失敗不影響主要成果）。
#   1. 圓形來源方位圖（native-polar-source；含文字版 figures-polar 與無文字版 figures-polar-notext）
#   2. 「回溯時間內仍在流場區域內的比例」曲線（0～180 天，90 與 180 天標線；含文字版與無文字版）
#   3. WP-61 到達時刻驗證圖（run09 母體 manifest ＋ 修正後水位資料；接縫排除清單取自 manifest）
# 各步驟輸出目錄必須尚不存在（不覆寫）；每一步獨立，任一步失敗只記錄、繼續下一步，最後離開碼 = 失敗步驟數。
# 用法：bash post_run09.sh [polar|fraction|arrival ...]（不給參數＝全部）
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/v8_env.sh"
require_clean_code
S=$(latest_population); [ -n "$S" ] || die "找不到 v8 母體"
R=$B/$CASE
CFG=$S/release-configs/release-${HORIZON_DAYS}d-full_horizon_from_deposition.yaml
ISO=$B/isobaths.geojson
[ -f "$CFG" ] && [ -f "$R/postprocess/departures.parquet" ] && [ -f "$ISO" ] || die "缺少 release config、departures.parquet 或等深線（後處理尚未完成？）"
export MPLCONFIGDIR=${MPLCONFIGDIR:-$W/lbt-runs/mpl-cache}
mkdir -p "$MPLCONFIGDIR" "$B/logs"
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(polar fraction arrival)
FAILS=0

step_polar() {
  for d in figures-polar figures-polar-notext; do [ ! -e "$R/$d" ] || { log "$R/$d 已存在，略過 polar"; return 0; }; done
  "$LBT" native-polar-source --case "$CASE" "$R/postprocess" "$R/figures-polar" "$R/figures-polar-notext" \
    --release-config "$CFG" --coastline "$COAST" --bathymetry "$ISO" --dpi "${LBT_POLAR_DPI:-600}" \
    --workers "${LBT_POLAR_WORKERS:-20}" --envelope-auto
}

step_fraction() {
  for d in figures-domain-fraction figures-domain-fraction-notext; do [ ! -e "$R/$d" ] || { log "$R/$d 已存在，略過 fraction"; return 0; }; done
  "$PY" "$CODE/scripts/wp63_deploy/domain_fraction_figure.py" "$R/postprocess" \
    "$R/figures-domain-fraction" "$R/figures-domain-fraction-notext" \
    --max-days "$HORIZON_DAYS" --marks 90 "$HORIZON_DAYS" --dpi "${LBT_FRACTION_DPI:-300}"
}

step_arrival() {
  local A=$B/arrival-check
  [ ! -e "$A" ] || { log "$A 已存在，略過 arrival"; return 0; }
  mkdir -p "$A/data" "$A/figures"
  export LBT_WORKSPACE=$W LBT_CODE_SRC=$CODE/src LBT_ARRIVAL_POP=$S LBT_ARRIVAL_DATA=$A/data
  export LBT_STAGE1_SURFACE_ROOT=$APRIME_SURFACE LBT_STAGE1_OCM_ROOT=$APRIME_OCM LBT_STAGE1_NWW_ROOT=$APRIME_NWW
  export LBT_STAGE1_NWW_FIRST_MONTH=202401 LBT_STAGE1_SEAM_FROM_MANIFEST=1 LBT_ARRIVAL_PAIRED=0
  "$PY" "$CODE/scripts/wp61/stage1.py" || return 1
  "$PY" "$CODE/scripts/wp61/run_figs.py" "$(IFS=,; echo gongliao,guishan,hsinchu,lienchiang,nanwan)" "$A/figures"
}

for s in "${STEPS[@]}"; do
  log "=== 補充出圖：$s ==="
  "step_$s" >> "$B/logs/post_run09_$s.log" 2>&1
  rc=$?
  log "$s 離開碼 $rc（日誌 $B/logs/post_run09_$s.log）"
  [ $rc -eq 0 ] || FAILS=$((FAILS + 1))
done
exit $FAILS
