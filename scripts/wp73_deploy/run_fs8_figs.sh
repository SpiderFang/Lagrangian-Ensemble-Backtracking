#!/bin/bash
# WP-73：報告圖字級與版面與 TRAPs 一致（fs8）——由 postprocess-coastB/ 重畫報告圖（新資料夾，後綴 -coastB-fs8；不覆蓋、不刪除既有輸出）。
#   用法：run_fs8_figs.sh fraction|fp|traj|polar|cmp
#   fraction：仍在追蹤比例曲線；fp：條件式來源足跡（1 km 網格、色階 B1）；traj：水平軌跡總覽；
#   polar：圓形來源方位圖本來就是 6.27 吋畫布、8 pt、不縮放，不重畫，只複製成 -fs8 資料夾並逐檔核對 SHA-256；
#   cmp：以舊版面重畫軌跡與足跡圖（dpi 100，只取資料指紋，供「數值未變」比對），輸出到 scratch/wp73/cmp。
set -uo pipefail
export LBT_CODE=${LBT_CODE:-$HOME/Workspace/lbt-repo-wp63}
source "$(dirname "$(readlink -f "$0")")/../wp63_deploy/v8_env.sh"
STEP=${1:?用法：run_fs8_figs.sh fraction|fp|traj|polar|cmp}
R=$B/$CASE
PP=$R/postprocess-coastB
[ -f "$PP/departures.parquet" ] || die "找不到 $PP"
mkdir -p "$R/logs"
export PYTHONPATH=$CODE/src
case "$STEP" in
  fraction)
    for d in figures-domain-fraction-coastB-fs8 figures-domain-fraction-notext-coastB-fs8; do [ -e "$R/$d" ] && die "$R/$d 已存在"; done
    "$PY" "$CODE/scripts/wp63_deploy/domain_fraction_figure.py" "$PP" "$R/figures-domain-fraction-coastB-fs8" "$R/figures-domain-fraction-notext-coastB-fs8" \
      --max-days "$HORIZON_DAYS" --marks 90 "$HORIZON_DAYS" --dpi 600 --report-fs8 > "$R/logs/figures-domain-fraction-coastB-fs8.log" 2>&1; log "fraction 結束，離開碼 $?" ;;
  fp)
    NAME=figures-fp1km-b1-ax2-notext-coastB-fs8; [ -e "$R/$NAME" ] && die "$R/$NAME 已存在"
    "$PY" "$CODE/scripts/wp73_deploy/render_fs8.py" fp "$R/$NAME" > "$R/logs/$NAME.log" 2>&1; log "fp 結束，離開碼 $?" ;;
  traj)
    NAME=figures-notext-traj-v5-ax2-coastB-fs8; [ -e "$R/$NAME" ] && die "$R/$NAME 已存在"
    "$PY" "$CODE/scripts/wp73_deploy/render_fs8.py" traj "$R/$NAME" > "$R/logs/$NAME.log" 2>&1; log "traj 結束，離開碼 $?" ;;
  polar)
    SRC=$R/figures-polar-v3-ax2-notext-coastB; DST=$R/figures-polar-v3-ax2-notext-coastB-fs8
    [ -d "$SRC" ] || die "找不到 $SRC"; [ -e "$DST" ] && die "$DST 已存在"
    cp -a "$SRC" "$DST" && (cd "$SRC" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$R/logs/polar-fs8.src.sha"
    (cd "$DST" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$R/logs/polar-fs8.dst.sha"
    cmp "$R/logs/polar-fs8.src.sha" "$R/logs/polar-fs8.dst.sha" && log "polar：複製並逐檔 SHA-256 相同" ;;
  cmp)
    D=$HOME/Workspace/lbt-runs/scratch/wp73/cmp; mkdir -p "$D"
    for k in traj_old fp_old; do [ -e "$D/$k" ] && die "$D/$k 已存在"; done
    "$PY" "$CODE/scripts/wp73_deploy/render_fs8.py" traj_old "$D/traj_old" > "$D/traj_old.log" 2>&1 &
    "$PY" "$CODE/scripts/wp73_deploy/render_fs8.py" fp_old "$D/fp_old" > "$D/fp_old.log" 2>&1 &
    wait; log "cmp 結束" ;;
  *) die "未知步驟 $STEP" ;;
esac
