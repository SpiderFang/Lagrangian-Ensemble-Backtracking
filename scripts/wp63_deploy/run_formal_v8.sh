#!/bin/bash
# WP-63：run09（v8：H180、M=10、full_horizon_from_deposition、只跑 finite_depth_stokes）正式重算。
#   native-run -> native-postprocess（含 WP-38 足跡產品）-> native-figures（full 與 none 兩版），任一步失敗即停止。
#
#   用法：run_formal_v8.sh --check   只檢查前置條件，不建立任何檔案
#         run_formal_v8.sh           正式開跑（需先完成 prep_v8.sh population 與 plan）
#   環境變數：LBT_WORKERS（預設 52）、LBT_CODE、LBT_MIN_FREE_GB（預設 150）、LBT_PURGE_CHECKPOINTS（預設 0）。
#
# worker 數的決定（記憶體）：主機 56 核、125 GB。WP-34 實測每個 worker 的常駐記憶體 A 區（northeast_taiwan）約 2.4 GB，
# 其餘區約 1.8 GB。動態佇列最壞情況是同時跑最大的那批分段，所以以「計畫內最大的 W 個分段的記憶體總和 < 0.9 × MemTotal」
# 為條件，自動把 worker 數降到可行的最大值（不超過 LBT_WORKERS）。52 個 worker 留 4 核給後處理與系統。
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/v8_env.sh"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

require_clean_code
require_reconstruction_root
require_data_roots
S=$(latest_population)
[ -n "$S" ] || die "找不到 v8 母體，請先執行 prep_v8.sh"
CFG=$S/release-configs/release-${HORIZON_DAYS}d-full_horizon_from_deposition.yaml
[ -f "$CFG" ] || die "找不到 release config：$CFG"
R=$B/$CASE
for f in "$B/inputs/identity_table.npz" "$B/inputs/region_to_flow_domain.json" "$R/native-plan.json" "$COAST"; do
  [ -f "$f" ] || die "缺少 $f（prep_v8.sh plan 尚未完成？）"
done
"$PY" "$CODE/scripts/wp63_deploy/check_run09_data_roots.py" population --arrival "$S/common-input/arrival.json" \
  --horizon-days $HORIZON_DAYS --max-age-days $MAX_AGE_DAYS \
  --runtime-ocm "$OCM_NATIVE_ROOT" --runtime-nww "$NWW_ANALYSIS_ROOT" --aprime-ocm "$APRIME_OCM" || die "母體與資料根的涵蓋檢查失敗"
export LBT_NATIVE_CONFIG=$CFG
export LBT_NATIVE_IDENTITY_TABLE=$B/inputs/identity_table.npz
export LBT_NATIVE_CHECKPOINT_EVERY_HOURS=24
require_v8_config "$CFG"
"$PY" "$CODE/scripts/wp54_deploy/check_identity_table_v7.py" "$B/inputs/identity_table.npz" --expect-per-candidate 1000 \
  || die "身分表的逐粒子沉降速度核對失敗"
SHA=$(sha256sum "$(readlink -f "${CFG}")" | cut -d" " -f1)
ISO=$B/isobaths.geojson
DIFF_JSON=$(resolve_and_check_diffusion) || die "擴散組態與預期不符（見上方訊息）"
require_disk 1

# worker 數：記憶體條件（見檔頭）
read -r WORKERS SEGMENTS MEMNEED < <(R="$R" "$PY" - <<'PYEOF'
import json, os
want = int(os.environ.get("LBT_WORKERS", "52"))
mem_total = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) / 1048576.0
d = json.load(open(os.environ["R"] + "/native-plan.json"))
keys = list(d["hour_major_members"])
gb = sorted((2.4 if k.startswith("hourmajor::northeast_taiwan") else 1.8 for k in keys), reverse=True)
w = min(want, len(gb))
while w > 1 and sum(gb[:w]) > 0.9 * mem_total:
    w -= 1
print(w, len(keys), round(sum(gb[:w]), 1))
PYEOF
)
if [ $CHECK_ONLY = 1 ]; then
  log "檢查模式：程式碼 $CODE（$(git -C "$CODE" rev-parse --short HEAD)）、母體 $S、輸出 $R、worker $WORKERS（分段 $SEGMENTS，最壞記憶體 ${MEMNEED} GB）"
  log "擴散組態：$DIFF_JSON"
  for d in results checkpoints postprocess figures figures-notext; do
    [ -e "$R/$d" ] && log "注意：$R/$d 已存在（native-run 會續跑；後處理與出圖會因輸出已存在而中止）"
  done
  exit 0
fi

mkdir -p "$B/logs" "$R/logs"
record_provenance "$B/logs/run_formal_v8.provenance.txt"
cd "$CODE"
START_UTC=$(date -u '+%F %T')
cat > "$B/RUN_STATUS.md" <<EOF
# run09（H180、M=10、finite_depth_stokes）執行狀態
- 啟動時間（UTC）：${START_UTC}Z
- 主機：$(hostname)
- worker 數：$WORKERS（分段總數 $SEGMENTS；最壞情況記憶體估計 ${MEMNEED} GB）
- 程式碼：$CODE @ $(git -C "$CODE" rev-parse --short HEAD)
- 母體：$S
- 設定檔：$CFG
- 日誌：$R/logs/native-run.log（native-run）、$R/logs/postprocess.log、$R/logs/figures.log、$R/logs/figures-notext.log、$B/logs/run_formal_v8.log
- 進度查詢：bash $TOOLS/run09_progress.sh
EOF
log "母體：$S；worker $WORKERS；擴散組態：$DIFF_JSON"

if [ ! -e "$ISO" ]; then
  log "匯出等深線"
  "$PY" "$CODE/scripts/export_isobaths.py" --identity-table "$B/inputs/identity_table.npz" \
    --region-flow-domain-map "$B/inputs/region_to_flow_domain.json" --grid-root "$D/ocm_native" \
    --output "$ISO" > "$B/logs/isobaths.log" 2>&1 || { log "等深線匯出失敗，停止"; exit 3; }
fi

for d in postprocess figures figures-notext; do
  [ -e "$R/$d" ] && die "$R/$d 已存在，為避免覆蓋而中止"
done
export LBT_NATIVE_PLAN=$R/native-plan.json
export LBT_NATIVE_RESULTS_DIR=$R/results LBT_NATIVE_CHECKPOINT_ROOT=$R/checkpoints
export LBT_NATIVE_EXPERIMENT_CASE=$CASE

log "=== native-run 開始（${WORKERS} worker）==="
time "$LBT" native-run "$R/native-plan.json" \
  --output "$R/unit-records" \
  --resource-provider lagrangian_backtracking.native_runner.hour_major_provider:make_hour_major_resource_provider \
  --numba-cache-dir "${NUMBA_CACHE_DIR}" --baytrace-root "${LBT_BAYTRACE_ROOT}" \
  --project-root . --config-sha256 "${SHA}" --mp-context spawn --formal \
  --worker-count "${WORKERS}" > "$R/logs/native-run.log" 2>&1
RC=$?
log "native-run 結束，離開碼 $RC"
echo "- native-run 結束（UTC）：$(date -u '+%F %T')Z，離開碼 $RC" >> "$B/RUN_STATUS.md"
[ $RC -eq 0 ] || { log "native-run 失敗，停止後續步驟"; exit $RC; }

log "後處理（含 WP-38 足跡產品、沉底時刻標籤分組與逐粒子沉降速度欄位）"
OMP_NUM_THREADS=4 "$LBT" native-postprocess \
  --identity-table "$B/inputs/identity_table.npz" --results-dir "$R/results" \
  --output "$R/postprocess" --config "$CFG" --with-boundary-segments --formal \
  --plan "$R/native-plan.json" --project-root "$CODE" \
  --with-footprint --footprint-land-geojson "$COAST" --bathymetry-grid-root "$D/ocm_native" \
  --region-flow-domain-map "$B/inputs/region_to_flow_domain.json" > "$R/logs/postprocess.log" 2>&1
RC=$?
log "後處理結束，離開碼 $RC"
echo "- 後處理結束（UTC）：$(date -u '+%F %T')Z，離開碼 $RC" >> "$B/RUN_STATUS.md"
[ $RC -eq 0 ] || { log "後處理失敗，停止"; exit $RC; }

for MODE in full none; do
  NAME=figures; LOGN=figures.log
  [ $MODE = none ] && { NAME=figures-notext; LOGN=figures-notext.log; }
  log "出圖（$MODE）"
  "$LBT" native-figures --postprocess-dir "$R/postprocess" --coastline "$COAST" \
    --bathymetry "$ISO" --config "$CFG" --output-dir "$R/$NAME" --figure-text $MODE > "$R/logs/$LOGN" 2>&1
  RC=$?
  log "出圖（$MODE）離開碼 $RC"
  [ $RC -eq 0 ] || { log "出圖失敗，停止"; exit $RC; }
done
log "補充出圖（圓形來源方位圖、域內比例曲線、到達時刻驗證圖；失敗不影響主要成果）"
bash "$CODE/scripts/wp63_deploy/post_run09.sh" >> "$B/logs/post_run09.log" 2>&1
RC=$?
log "補充出圖結束，失敗步驟數 $RC"
echo "- 補充出圖結束（UTC）：$(date -u '+%F %T')Z，失敗步驟數 $RC（日誌 $B/logs/post_run09_*.log）" >> "$B/RUN_STATUS.md"
echo "- 全部完成（UTC）：$(date -u '+%F %T')Z" >> "$B/RUN_STATUS.md"
log "=== 完成 ==="
df -h /home | tail -1
