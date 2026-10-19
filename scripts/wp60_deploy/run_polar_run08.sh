#!/bin/bash
# WP-60：對 run08（v7）兩案產出「圓形來源方位圖」（每案每地點 13 張，含文字版與無文字版）與統計表、summary.md。
# 規則（2026-10-19 PI 定案）：未穿越開放邊界者一律以最終位置方位計入（海岸來源），每組 36 扇區合計 100%。
# 一行執行：bash ~/Workspace/lbt-repo-wp60/scripts/wp60_deploy/run_polar_run08.sh
# 只「讀」run08 兩案的 postprocess/departures.parquet 與釋放設定，輸出到各案目錄下的新子目錄
#   <run08>/<案例>/figures-polar（含文字版與表、summary.md）與 <run08>/<案例>/figures-polar-notext（無文字版）。
# 兩個子目錄必須尚不存在（指令本身也會拒絕覆寫）；不會碰 run08 既有的 figures、figures.v2~v5、figures-arc-by-group 等任何輸出。
# 程式碼來自 lbt-repo-wp60（分支 wp60-polar-source）。可用環境變數：LBT_RUN08、LBT_CODE60、LBT_POLAR_DPI（預設 600）、LBT_POLAR_WORKERS（預設 20）、LBT_POLAR_EXTRA_ARGS。
set -u
W=$HOME/Workspace
CODE=${LBT_CODE60:-$W/lbt-repo-wp60}
B=${LBT_RUN08:-$W/lbt-runs/run08-h90-m10-v7}
LBT=$W/lbt-repo/.venv/bin/lbt
DPI=${LBT_POLAR_DPI:-600}
WORKERS=${LBT_POLAR_WORKERS:-20}
EXTRA=${LBT_POLAR_EXTRA_ARGS:-}
CFG=$(ls -d "$W"/lbt-inputs/formal/v7-class-velocity/*_final-sites_h30-h60-h90_common_m10_v7 | tail -1)/release-configs/release-90d-full_horizon_from_deposition.yaml
log() { echo "[$(date -u "+%F %T")Z] $*"; }
die() { log "中止：$*"; exit 2; }
[ -e "$CODE/.git" ] || die "$CODE 不是 git clone 或 worktree"
[ -z "$(git -C "$CODE" status --porcelain)" ] || die "$CODE 工作樹不乾淨，請先 commit"
[ -x "$LBT" ] && [ -f "$CFG" ] && [ -f "$B/isobaths.geojson" ] || die "找不到 lbt、釋放設定 $CFG 或等深線"
ARGS=()
for CASE in no_stokes finite_depth_stokes; do
  [ -f "$B/$CASE/postprocess/departures.parquet" ] || die "$B/$CASE/postprocess 缺少 departures.parquet"
  for D in figures-polar figures-polar-notext; do [ ! -e "$B/$CASE/$D" ] || die "$B/$CASE/$D 已存在，不覆寫"; done
  ARGS+=(--case "$CASE" "$B/$CASE/postprocess" "$B/$CASE/figures-polar" "$B/$CASE/figures-polar-notext")
done
export PYTHONPATH=$CODE/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR=${MPLCONFIGDIR:-$W/lbt-runs/mpl-cache}
mkdir -p "$MPLCONFIGDIR" "$W/lbt-runs/logs"
log "程式碼 $CODE @ $(git -C "$CODE" rev-parse --short HEAD)；dpi $DPI；workers $WORKERS"
# shellcheck disable=SC2086
time "$LBT" native-polar-source "${ARGS[@]}" --release-config "$CFG" --coastline "$W/lbt-inputs/coastline-taiwan.geojson" \
  --bathymetry "$B/isobaths.geojson" --dpi "$DPI" --workers "$WORKERS" $EXTRA
log "native-polar-source 離開碼 $?"
