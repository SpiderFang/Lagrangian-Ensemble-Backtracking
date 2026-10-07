#!/bin/bash
# WP-51 整合：v4（逐成員沉底年齡 age_sampling_scope=per_member）正式輸入準備。
#
#   用法：prep_v4.sh --check   只檢查前置條件並印出擴散組態，不建立任何檔案或目錄
#         prep_v4.sh           建立 v4 母體（horizon-suite-create）、M=10 身分表、兩案例的工作計畫
#
# 本腳本「準備但尚未執行」：寫出時主代理尚未決定開跑。母體建置需讀 NAS（/data），耗時較長。
# 執行前提：
#   1. 已用 git bundle 建好乾淨的程式碼 clone（$CODE，見 v4_env.sh），HEAD 是要上線的版本；
#   2. 已產生 $TEMPLATE（scripts/make_formal_template_v4.py 由 v3 範本產生）；
#   3. 目的地目錄都不存在（絕不覆蓋 run04 或 v3 母體）。
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/v4_env.sh"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

require_clean_code
[ -f "$TEMPLATE" ] || die "找不到 v4 範本 $TEMPLATE（先執行 scripts/make_formal_template_v4.py）"
grep -q "age_sampling_scope: per_member" "$TEMPLATE" || die "v4 範本沒有 age_sampling_scope: per_member"
[ -e "$B/inputs" ] && die "$B/inputs 已存在，為避免覆蓋而中止"
DEST=$V4ROOT/$(date -u +%Y%m%d)_final-sites_h30-h60-h90_common_m10_v4
[ -e "$DEST" ] && die "母體目的地 $DEST 已存在，為避免覆蓋而中止"

# 以範本直接核對擴散組態（release config 尚未產生時先驗範本；母體建好後會再以 release config 驗一次）。
LBT_NATIVE_CONFIG=$TEMPLATE resolve_and_check_diffusion > /dev/null || die "範本的擴散組態與預期不符"
if [ $CHECK_ONLY = 1 ]; then
  log "檢查模式：程式碼 $CODE（$(git -C "$CODE" rev-parse --short HEAD)）、範本 $TEMPLATE、目的地 $DEST、輸出 $B 全部通過前置檢查"
  log "擴散組態（範本）：$(LBT_NATIVE_CONFIG=$TEMPLATE resolve_and_check_diffusion)"
  exit 0
fi

mkdir -p "$V4ROOT" "$B/inputs" "$B/logs"
record_provenance_pre() { LBT_NATIVE_CONFIG=$TEMPLATE record_provenance "$B/logs/prep_v4.provenance.txt"; }
record_provenance_pre
cd "$CODE"

log "[1/3] 建立 v4 母體與 release suite：$DEST"
# release config 建立時 validator 會讀「環境變數指到的資料根目錄」；v4_env.sh 為執行階段設的是
# lbt-local-data（只暫存 2024-07 起的月份，且含符號連結），inventory 卻涵蓋 2024-01 起，會讓
# validator 回報 source_unreadable:ocm_native:FileNotFoundError（run05b 的實際失敗）。母體建置本來就
# 只用下面三個 --*-root 參數指向 /data（與 v3 的 build_v3.sh 一致、不設這些環境變數），所以這裡只對
# 這一個指令取消設定；後面 native-plan 仍用 v4_env.sh 的 local-data。
time env -u OCM_NATIVE_ROOT -u NWW_ANALYSIS_ROOT -u OCM_RECONSTRUCTION_ROOT -u OCM_SURFACE_ROOT \
  "$LBT" horizon-suite-create \
  --config-template "$TEMPLATE" \
  --backtrack-days 30 60 90 \
  --destination "$DEST" \
  --ocm-native-root /data/OCM-Preprocessed-Data/preprocessed/ocm_native \
  --ocm-surface-root /data/OCM-Preprocessed-Data/preprocessed/ocm_surface \
  --nww-analysis-root /data/NWW-Preprocessed-Data/preprocessed/nww3_analysis \
  --formal
ls "$DEST/release-configs"

S=$DEST
export LBT_NATIVE_CONFIG=$S/release-configs/release-90d-full_horizon_from_deposition.yaml
resolve_and_check_diffusion > /dev/null || die "release config 的擴散組態與預期不符"
record_provenance "$B/logs/prep_v4.provenance.txt"
echo "母體：$S" | tee -a "$B/logs/prep_v4.provenance.txt"

log "[2/3] 建立 M=10 身分表（per_member：50 觀測 × 10 成員，各成員自己的沉底時刻）"
time "$PY" "$TOOLS/prep_m10_probe.py" "$B/inputs" 10

SHA=$(sha256sum "$(readlink -f "${LBT_NATIVE_CONFIG}")" | cut -d" " -f1)
A=$S/common-input/arrival.json
for CASE in no_stokes finite_depth_stokes; do
  log "[3/3] 建立工作計畫：$CASE"
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
    # full_horizon 模式下工作單元的 resolved_max_backtrack_seconds 為 None（由全域 90 天上限決定；v3 計畫亦同）
    hrs = np.array([u["resolved_max_backtrack_seconds"] / 3600.0 for u in units if u.get("resolved_max_backtrack_seconds") is not None])
    pre = sum(1 for u in units if u.get("pre_window_deposition"))
    print(case, "工作單元", s["unit_count"], "粒子", s["particle_count_total"], "分段", len(d["hour_major_members"]), "pre_window", pre)
    if hrs.size:
        print("   有效回溯小時：min %.0f max %.0f 相異值 %d" % (hrs.min(), hrs.max(), len(set(hrs.round(3)))))
    print("   計畫檔擴散組態：", json.dumps(d.get("physics_case", {}).get("diffusion_scheme"), sort_keys=True, ensure_ascii=False))
PYEOF
log "準備完成；母體 $S；輸出目錄 $B"
