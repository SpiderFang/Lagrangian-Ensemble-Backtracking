#!/bin/bash
# WP-63：run09（v8，H180、只跑 finite_depth_stokes）輸入準備。分兩個階段：
#
#   prep_v8.sh --check        只檢查前置條件，不建立任何檔案或目錄
#   prep_v8.sh population     產生 v8 範本（若尚未存在）並建立母體（horizon-suite-create --backtrack-days 180）。
#                             只讀母體建置資料根（APRIME_OCM／APRIME_SURFACE／NAS NWW）；不需要執行期資料根的月份已搬完。
#   prep_v8.sh plan           （需要 v7-full 資料根搬完）建立 M=10 身分表與 finite_depth_stokes 的工作計畫，並核對母體。
#
# 母體建置完成後的核對由 verify_population_v8.py 做（plan 階段會呼叫）。
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/v8_env.sh"
HERE=$(dirname "$(readlink -f "$0")")
STAGE=${1:---check}
case "$STAGE" in --check|population|plan) ;; *) die "用法：prep_v8.sh --check|population|plan" ;; esac

require_clean_code
[ -f "$V7TEMPLATE" ] || die "找不到 v7 範本 $V7TEMPLATE"

if [ "$STAGE" = "--check" ] || [ "$STAGE" = population ]; then
  require_reconstruction_root
  require_data_roots --aprime-only
  require_disk 1
  [ -e "$B/inputs" ] && [ "$STAGE" = population ] && die "$B/inputs 已存在，為避免覆蓋而中止"
  [ -z "$(find "$APRIME_OCM" -type l | head -1)" ] || die "$APRIME_OCM 內有 symbolic link（母體建置會拒絕）"
  [ -z "$(find "$APRIME_SURFACE" -type l | head -1)" ] || die "$APRIME_SURFACE 內有 symbolic link（母體建置會拒絕）"
  if [ -f "$TEMPLATE" ]; then
    require_v8_config "$TEMPLATE"
  else
    "$PY" "$HERE/make_template_v8.py" --check --output "$TEMPLATE" || die "無法產生 v8 範本"
  fi
  if [ "$STAGE" = "--check" ]; then
    log "檢查模式：程式碼 $CODE（$(git -C "$CODE" rev-parse --short HEAD)）、範本 $TEMPLATE、輸出 $B 通過前置檢查"
    exit 0
  fi
  DEST=$V8ROOT/$(date -u +%Y%m%d)_final-sites_h180_common_m10_v8
  [ -e "$DEST" ] && die "母體目的地 $DEST 已存在，為避免覆蓋而中止"
  [ -f "$TEMPLATE" ] || "$PY" "$HERE/make_template_v8.py" --output "$TEMPLATE"
  require_v8_config "$TEMPLATE"
  LBT_NATIVE_CONFIG=$TEMPLATE resolve_and_check_diffusion > /dev/null || die "範本的擴散組態與預期不符"
  mkdir -p "$V8ROOT" "$B/inputs" "$B/logs"
  LBT_NATIVE_CONFIG=$TEMPLATE record_provenance "$B/logs/prep_v8.provenance.txt"
  cd "$CODE"
  log "[1/1] 建立 v8 母體與 release suite：$DEST（H$HORIZON_DAYS）"
  # 母體建置同時開啟四個區域的 NWW（各 120 檔）與 OCM 月檔，預設 1024 個檔案描述符不夠（實測在第 4 站建立工作行程時失敗）。
  ulimit -n 65536 || die "無法提高檔案描述符上限"
  export LBT_BAD_HOUR_CACHE_DIR=$V8ROOT/.bad-hour-cache
  export LBT_BAD_HOUR_CACHE_TAG=${LBT_BAD_HOUR_CACHE_TAG:-$(printf '%s\n' "$APRIME_OCM" "$APRIME_SURFACE" "$APRIME_NWW" "$(git -C "$CODE" rev-parse HEAD)" | sha256sum | cut -c1-16)}
  time env -u OCM_NATIVE_ROOT -u NWW_ANALYSIS_ROOT -u OCM_RECONSTRUCTION_ROOT -u OCM_SURFACE_ROOT \
    LBT_BAD_HOUR_WORKERS=${LBT_BAD_HOUR_WORKERS:-24} LBT_BAD_HOUR_CACHE_DIR="$LBT_BAD_HOUR_CACHE_DIR" LBT_BAD_HOUR_CACHE_TAG="$LBT_BAD_HOUR_CACHE_TAG" \
    "$LBT" horizon-suite-create \
    --config-template "$TEMPLATE" \
    --backtrack-days $HORIZON_DAYS \
    --destination "$DEST" \
    --ocm-native-root "$APRIME_OCM" \
    --ocm-surface-root "$APRIME_SURFACE" \
    --nww-analysis-root "$APRIME_NWW" \
    --formal
  ls "$DEST/release-configs"
  export LBT_NATIVE_CONFIG=$DEST/release-configs/release-${HORIZON_DAYS}d-full_horizon_from_deposition.yaml
  require_v8_config "$LBT_NATIVE_CONFIG"
  record_provenance "$B/logs/prep_v8.provenance.txt"
  echo "母體：$DEST" | tee -a "$B/logs/prep_v8.provenance.txt"
  log "母體建置完成：$DEST"
  exit 0
fi

# ---- plan 階段
S=$(latest_population)
[ -n "$S" ] || die "找不到 v8 母體（$V8ROOT/*_final-sites_h180_common_m10_v8），請先執行 prep_v8.sh population"
CFG=$S/release-configs/release-${HORIZON_DAYS}d-full_horizon_from_deposition.yaml
[ -f "$CFG" ] || die "找不到 release config $CFG"
require_reconstruction_root
require_data_roots
"$PY" "$CODE/scripts/wp63_deploy/check_run09_data_roots.py" population --arrival "$S/common-input/arrival.json" \
  --horizon-days $HORIZON_DAYS --max-age-days $MAX_AGE_DAYS \
  --runtime-ocm "$OCM_NATIVE_ROOT" --runtime-nww "$NWW_ANALYSIS_ROOT" --aprime-ocm "$APRIME_OCM" || die "母體與資料根的涵蓋檢查失敗"
export LBT_NATIVE_CONFIG=$CFG
require_v8_config "$CFG"
resolve_and_check_diffusion > /dev/null || die "release config 的擴散組態與預期不符"
require_disk 1
[ -e "$B/$CASE" ] && die "$B/$CASE 已存在，為避免覆蓋而中止"
cd "$CODE"
log "[1/3] 建立 M=10 身分表"
time "$PY" "$TOOLS/prep_m10_probe.py" "$B/inputs" 10
log "[2/3] 核對身分表的逐粒子沉降速度（每候選粒子數必須完全相等）與母體"
"$PY" "$CODE/scripts/wp54_deploy/check_identity_table_v7.py" "$B/inputs/identity_table.npz" --expect-per-candidate 1000 \
  | tee -a "$B/logs/prep_v8.provenance.txt" || die "身分表的逐粒子沉降速度核對失敗"
"$PY" "$HERE/verify_population_v8.py" --population "$S" --identity-table "$B/inputs/identity_table.npz" \
  --aprime-ocm "$APRIME_OCM" --runtime-ocm "$OCM_NATIVE_ROOT" --output "$B/logs/verify_population_v8.json" \
  | tee -a "$B/logs/prep_v8.provenance.txt" || die "母體核對失敗（見 $B/logs/verify_population_v8.json）"
SHA=$(sha256sum "$(readlink -f "${LBT_NATIVE_CONFIG}")" | cut -d" " -f1)
A=$S/common-input/arrival.json
log "[3/3] 建立工作計畫：$CASE（H$HORIZON_DAYS）"
mkdir -p "$B/$CASE"
time "$LBT" native-plan --identity-table "$B/inputs/identity_table.npz" \
  --ocm-native-root "${OCM_NATIVE_ROOT}" --config "${LBT_NATIVE_CONFIG}" --arrival-manifest "$A" \
  --region-to-flow-domain "$B/inputs/region_to_flow_domain.json" \
  --max-backtrack-days $HORIZON_DAYS --project-root . --formal --config-sha256 "${SHA}" \
  --experiment-case "$CASE" \
  --schedule hour-major-segmented --target-worker-count "${LBT_WORKERS:-52}" \
  --node-counts "$B/inputs/node_counts.json" --layer-counts "$B/inputs/layer_counts.json" \
  --seconds-per-node-layer-hour 4.2e-6 --worker-count-for-estimate "${LBT_WORKERS:-52}" \
  --output "$B/$CASE/native-plan.json"
B="$B" CASE="$CASE" "$PY" - <<'PYEOF'
import json, os
import numpy as np
B, case = os.environ["B"], os.environ["CASE"]
d = json.load(open(f"{B}/{case}/native-plan.json"))
s = d["summary"]
units = [m for seg in d["hour_major_members"].values() for m in seg]
hrs = np.array([u["resolved_max_backtrack_seconds"] / 3600.0 for u in units if u.get("resolved_max_backtrack_seconds") is not None])
pre = sum(1 for u in units if u.get("pre_window_deposition"))
print(case, "工作單元", s["unit_count"], "粒子", s["particle_count_total"], "分段", len(d["hour_major_members"]), "pre_window", pre)
if hrs.size:
    print("   有效回溯小時：min %.0f max %.0f 相異值 %d" % (hrs.min(), hrs.max(), len(set(hrs.round(3)))))
PYEOF
log "plan 階段完成；母體 $S；輸出目錄 $B"
