#!/bin/bash
# WP-72 方法 B：由 postprocess-coastB/ 產出結果表與全部圖（新資料夾，後綴 -coastB；不覆蓋、不刪除既有輸出）。
#   用法：run_coastB_figs.sh tables|facts|fraction|fp|traj|polar
set -uo pipefail
export LBT_CODE=${LBT_CODE:-$HOME/Workspace/lbt-repo-wp72}
source "$(dirname "$(readlink -f "$0")")/../wp63_deploy/v8_env.sh"
STEP=${1:?用法：run_coastB_figs.sh tables|facts|fraction|fp|traj|polar}
S=$(latest_population); [ -n "$S" ] || die "找不到 v8 母體"
CFG=$S/release-configs/release-${HORIZON_DAYS}d-full_horizon_from_deposition.yaml
R=$B/$CASE
ISO=$B/isobaths.geojson
PP=$R/postprocess-coastB
[ -f "$PP/departures.parquet" ] || die "找不到 $PP"
mkdir -p "$R/logs"
case "$STEP" in
  tables)
    OUT=$R/coastB-tables; [ -e "$OUT" ] && die "$OUT 已存在"
    "$PY" "$CODE/scripts/wp72_deploy/coastB_tables.py" "$PP" "$OUT" > "$R/logs/coastB-tables.log" 2>&1; log "tables 結束，離開碼 $?" ;;
  facts)
    OUT=$R/coastB-tables/facts_remote_run09B.json; [ -e "$OUT" ] && die "$OUT 已存在"
    "$PY" "$CODE/scripts/wp72_deploy/facts_remote_run09B.py" "$PP" > "$OUT" 2> "$R/logs/coastB-facts.log"; log "facts 結束，離開碼 $?" ;;
  fraction)
    for d in figures-domain-fraction-coastB figures-domain-fraction-notext-coastB; do [ -e "$R/$d" ] && die "$R/$d 已存在"; done
    "$PY" "$CODE/scripts/wp63_deploy/domain_fraction_figure.py" "$PP" "$R/figures-domain-fraction-coastB" "$R/figures-domain-fraction-notext-coastB" \
      --max-days "$HORIZON_DAYS" --marks 90 "$HORIZON_DAYS" --dpi 300 > "$R/logs/figures-domain-fraction-coastB.log" 2>&1; log "fraction 結束，離開碼 $?" ;;
  fp)
    NAME=figures-fp1km-b1-ax2-notext-coastB; [ -e "$R/$NAME" ] && die "$R/$NAME 已存在"
    "$LBT" native-figures --postprocess-dir "$PP" --coastline "$COAST" --bathymetry "$ISO" --config "$CFG" --output-dir "$R/$NAME" --figure-text none \
      --footprint-colorbar-range 1e-3 1e3 --footprint-colormap-scheme nonuniform --axes-unified > "$R/logs/$NAME.log" 2>&1; log "fp 結束，離開碼 $?" ;;
  traj)
    NAME=figures-notext-traj-v5-ax2-coastB; [ -e "$R/$NAME" ] && die "$R/$NAME 已存在"
    "$PY" "$CODE/scripts/wp72_deploy/render_horizontal_overview_coastB.py" "$R/$NAME" > "$R/logs/$NAME.log" 2>&1; log "traj 結束，離開碼 $?" ;;
  polar)
    for d in figures-polar-v3-ax2-coastB figures-polar-v3-ax2-notext-coastB; do [ -e "$R/$d" ] && die "$R/$d 已存在"; done
    "$LBT" native-polar-source --case "$CASE" "$PP" "$R/figures-polar-v3-ax2-coastB" "$R/figures-polar-v3-ax2-notext-coastB" \
      --release-config "$CFG" --coastline "$COAST" --bathymetry "$ISO" --dpi 600 --workers 20 --modes none --report-layout --report-axes-style one_decimal \
      > "$R/logs/figures-polar-v3-ax2-coastB.log" 2>&1; log "polar 結束，離開碼 $?" ;;
  *) die "未知步驟 $STEP" ;;
esac
