#!/bin/bash
# WP-54：v7（十類沉降速度每類 10 個候選、逐粒子均衡抽樣；其餘設定同 v6）正式輸入準備。
#
#   用法：prep_v7.sh --check   只檢查前置條件並印出擴散組態與將建立的路徑，不建立任何檔案或目錄
#         prep_v7.sh           產生 v7 範本（若尚未存在）、建立 v7 母體（horizon-suite-create）、M=10 身分表（含逐粒子速度）、兩案例的工作計畫
#
# 母體建置的 ocm_native／ocm_surface 來源與 v6 相同（$APRIME_OCM／$APRIME_SURFACE，唯讀）；nww3_analysis 仍讀 NAS。
# 母體建好後，執行期只使用 lbt-local-data-v6（唯讀共用，不複製）。
# 執行前提：
#   1. 乾淨的程式碼 clone（$CODE，見 v7_env.sh），HEAD 是要上線的版本；
#   2. 有 $V6TEMPLATE（v6 範本；v7 範本由 make_template_v7.py 從它產生，不覆蓋任何既有檔案）；
#   3. 目的地目錄都不存在（絕不覆蓋 run04～run07 或其母體）；磁碟可用空間足夠（require_disk）。
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/v7_env.sh"
HERE=$(dirname "$(readlink -f "$0")")
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

require_clean_code
require_v6_data_root
require_disk 2
[ -f "$V6TEMPLATE" ] || die "找不到 v6 範本 $V6TEMPLATE"
[ -e "$B/inputs" ] && die "$B/inputs 已存在，為避免覆蓋而中止"
DEST=$V7ROOT/$(date -u +%Y%m%d)_final-sites_h30-h60-h90_common_m10_v7
[ -d "$APRIME_SURFACE" ] || die "找不到母體建置用 ocm_surface 根 $APRIME_SURFACE"
[ -d "$APRIME_OCM" ] || die "找不到母體建置用 ocm_native 根 $APRIME_OCM"
[ -z "$(find "$APRIME_OCM" -type l | head -1)" ] || die "$APRIME_OCM 內有 symbolic link（母體建置會拒絕）"
[ -z "$(find "$APRIME_SURFACE" -type l | head -1)" ] || die "$APRIME_SURFACE 內有 symbolic link（母體建置會拒絕）"
[ -e "$DEST" ] && die "母體目的地 $DEST 已存在，為避免覆蓋而中止"

if [ -f "$TEMPLATE" ]; then
  grep -q "age_sampling_scope: per_member" "$TEMPLATE" || die "v7 範本沒有 age_sampling_scope: per_member"
  require_v7_config "$TEMPLATE"
  LBT_NATIVE_CONFIG=$TEMPLATE resolve_and_check_diffusion > /dev/null || die "範本的擴散組態與預期不符"
else
  # 範本尚不存在：先以 --check 模式驗證可由 v6 範本與 repo 設定檔產生。
  "$PY" "$HERE/make_template_v7.py" --check --output "$TEMPLATE" || die "無法產生 v7 範本"
fi
if [ $CHECK_ONLY = 1 ]; then
  log "檢查模式：程式碼 $CODE（$(git -C "$CODE" rev-parse --short HEAD)）、v7 範本 $TEMPLATE（$([ -f "$TEMPLATE" ] && echo 已存在 || echo 尚未產生，正式執行時產生)）、目的地 $DEST、輸出 $B 全部通過前置檢查"
  [ -f "$TEMPLATE" ] && log "擴散組態（範本）：$(LBT_NATIVE_CONFIG=$TEMPLATE resolve_and_check_diffusion)"
  exit 0
fi

[ -f "$TEMPLATE" ] || "$PY" "$HERE/make_template_v7.py" --output "$TEMPLATE"
grep -q "age_sampling_scope: per_member" "$TEMPLATE" || die "v7 範本沒有 age_sampling_scope: per_member"
require_v7_config "$TEMPLATE"
LBT_NATIVE_CONFIG=$TEMPLATE resolve_and_check_diffusion > /dev/null || die "範本的擴散組態與預期不符"

mkdir -p "$V7ROOT" "$B/inputs" "$B/logs"
record_provenance_pre() { LBT_NATIVE_CONFIG=$TEMPLATE record_provenance "$B/logs/prep_v7.provenance.txt"; }
record_provenance_pre
cd "$CODE"

log "[1/4] 建立 v7 母體與 release suite：$DEST"
# 與 prep_v6.sh 相同：母體建置只用三個 --*-root 參數，所以只對這一個指令取消設定資料根環境變數。
time env -u OCM_NATIVE_ROOT -u NWW_ANALYSIS_ROOT -u OCM_RECONSTRUCTION_ROOT -u OCM_SURFACE_ROOT \
  "$LBT" horizon-suite-create \
  --config-template "$TEMPLATE" \
  --backtrack-days 30 60 90 \
  --destination "$DEST" \
  --ocm-native-root "$APRIME_OCM" \
  --ocm-surface-root "$APRIME_SURFACE" \
  --nww-analysis-root /data/NWW-Preprocessed-Data/preprocessed/nww3_analysis \
  --formal
ls "$DEST/release-configs"

S=$DEST
export LBT_NATIVE_CONFIG=$S/release-configs/release-90d-full_horizon_from_deposition.yaml
require_v7_config "$LBT_NATIVE_CONFIG"
resolve_and_check_diffusion > /dev/null || die "release config 的擴散組態與預期不符"
record_provenance "$B/logs/prep_v7.provenance.txt"
echo "母體：$S" | tee -a "$B/logs/prep_v7.provenance.txt"

log "[2/4] 建立 M=10 身分表（per_member 沉底年齡；逐站位逐類別 10 個候選速度均衡抽樣，各 1/10）"
time "$PY" "$TOOLS/prep_m10_probe.py" "$B/inputs" 10

log "[3/4] 核對身分表的逐粒子沉降速度（每候選粒子數必須完全相等、速度等於候選值）"
"$PY" "$HERE/check_identity_table_v7.py" "$B/inputs/identity_table.npz" --expect-per-candidate 1000 \
  | tee -a "$B/logs/prep_v7.provenance.txt" || die "身分表的逐粒子沉降速度核對失敗"

SHA=$(sha256sum "$(readlink -f "${LBT_NATIVE_CONFIG}")" | cut -d" " -f1)
A=$S/common-input/arrival.json
for CASE in no_stokes finite_depth_stokes; do
  log "[4/4] 建立工作計畫：$CASE"
  mkdir -p "$B/$CASE"
  time "$LBT" native-plan --identity-table "$B/inputs/identity_table.npz" \
    --ocm-native-root "${OCM_NATIVE_ROOT}" --config "${LBT_NATIVE_CONFIG}" --arrival-manifest "$A" \
    --region-to-flow-domain "$B/inputs/region_to_flow_domain.json" \
    --max-backtrack-days 90 --project-root . --formal --config-sha256 "${SHA}" \
    --experiment-case "$CASE" \
    --schedule hour-major-segmented --target-worker-count 42 \
    --node-counts "$B/inputs/node_counts.json" --layer-counts "$B/inputs/layer_counts.json" \
    --seconds-per-node-layer-hour 4.2e-6 --worker-count-for-estimate 42 \
    --output "$B/$CASE/native-plan.json"
done

B="$B" "$PY" - <<'PYEOF'
import json, os
import numpy as np

B = os.environ["B"]
for case in ("no_stokes", "finite_depth_stokes"):
    d = json.load(open(f"{B}/{case}/native-plan.json"))
    s = d["summary"]
    units = [m for seg in d["hour_major_members"].values() for m in seg]
    hrs = np.array([u["resolved_max_backtrack_seconds"] / 3600.0 for u in units if u.get("resolved_max_backtrack_seconds") is not None])
    pre = sum(1 for u in units if u.get("pre_window_deposition"))
    print(case, "工作單元", s["unit_count"], "粒子", s["particle_count_total"], "分段", len(d["hour_major_members"]), "pre_window", pre)
    if hrs.size:
        print("   有效回溯小時：min %.0f max %.0f 相異值 %d" % (hrs.min(), hrs.max(), len(set(hrs.round(3)))))
    print("   計畫檔擴散組態：", json.dumps(d.get("physics_case", {}).get("diffusion_scheme"), sort_keys=True, ensure_ascii=False))
PYEOF
log "準備完成；母體 $S；輸出目錄 $B"
