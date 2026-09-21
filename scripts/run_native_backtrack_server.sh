#!/usr/bin/env bash
# WP-11：在伺服器（例如 140.117.88.203）上執行完整 native 逆向批次流程的入口。
#
# 五個步驟依序是：preflight（唯讀資料契約檢查）→ gap-fill（已核准缺口重建，
# 逐流場區域執行）→ plan（工作單元切段規劃＋到達時刻缺口安全檢查）→ run（背景
# 平行執行＋續跑，`nohup` 落地到帶時間戳記的 log 檔）→ postprocess（合併全部
# 工作單元成果，輸出離開紀錄、通量與條件式來源足跡）。每一步都可以單獨呼叫；
# `all` 動作依序執行全部五步，已經完成的步驟（見下方各步驟小節的「略過條件」）
# 預設會被跳過，除非設定 `LBT_NATIVE_FORCE_RERUN=1`。
#
# 這支腳本本身不建立任何新的排程、切段或後處理邏輯——全部邏輯都在
# `src/lagrangian_backtracking/native_runner/`，經由既有 `lbt native-preflight`／
# `lbt native-gap-fill`／`lbt native-plan`／`lbt native-run`／`lbt native-status`／
# `lbt native-postprocess` 子命令呼叫（見
# docs/operations/19_native_backward_batch_orchestrator.md）。`run` 步驟固定使用
# `native_runner.hour_major_provider:make_hour_major_resource_provider` 當
# `--resource-provider`（WP-11 的小時主軸切段平行路徑），不支援切換回逐工作單元
# 路徑的 resource provider；若需要逐工作單元路徑，請直接呼叫 `lbt native-run`。
#
# 本檔不含任何密碼或伺服器私有資訊；所有路徑、憑證與伺服器層級限制一律由
# operator 透過環境變數在啟動當下注入，腳本本身不寫死。
#
# ---------------------------------------------------------------------------
# 使用方式
# ---------------------------------------------------------------------------
#
#   bash scripts/run_native_backtrack_server.sh preflight
#       唯讀資料契約檢查（目錄、節點數、缺口、到達窗）；回傳碼反映
#       `PreflightReport.is_run_ready`。不寫入任何檔案（除非指定
#       `LBT_NATIVE_PREFLIGHT_OUTPUT`），秒到分鐘級完成。
#
#   bash scripts/run_native_backtrack_server.sh gap-fill
#       逐流場區域呼叫已核准的 OCM／波浪缺口重建管線（`native_runner.
#       gap_fill`，由另一個工作包提供；若該模組尚未交付，這步會清楚失敗，不
#       假裝已完成重建）。略過條件：`${LBT_OUTPUT_ROOT}/native-run/gap-fill.done`
#       已存在且 `LBT_NATIVE_FORCE_RERUN` 不是 `1`。
#
#   bash scripts/run_native_backtrack_server.sh plan
#       規劃（區域, 段）工作單元＋到達時刻缺口安全檢查，寫出計畫 JSON。略過
#       條件：計畫檔已存在且 `LBT_NATIVE_FORCE_RERUN` 不是 `1`。
#
#   bash scripts/run_native_backtrack_server.sh run
#       以 `nohup` 把 `lbt native-run` 丟到背景執行，stdout／stderr 導向帶
#       時間戳記的 log 檔。`native-run` 本身的續跑機制（`*.done.json` 完成
#       證據）已經是冪等的：全部工作單元都完成時再次呼叫不會啟動任何 worker、
#       立即回傳，因此這步不需要另外的「略過」標記檔——重複呼叫本身就是安全
#       的（見 `orchestrator.run_native_plan` docstring「續跑行為」）。
#
#   bash scripts/run_native_backtrack_server.sh status
#       呼叫 `lbt native-status` 讀取目前的進度快照與完成證據統計；可在
#       `run` 之後隨時重複執行，不影響正在跑的背景程序。
#
#   bash scripts/run_native_backtrack_server.sh postprocess
#       合併全部工作單元成果，輸出離開紀錄、通量與條件式來源足跡。略過條件：
#       `${LBT_OUTPUT_ROOT}/native-run/postprocess/manifest.json` 已存在且
#       `LBT_NATIVE_FORCE_RERUN` 不是 `1`。
#
#   bash scripts/run_native_backtrack_server.sh all
#       依序執行 preflight → gap-fill → plan → run → postprocess；`run` 之後
#       不會自動等待背景程序完成才繼續 `postprocess`（背景執行可能長達數天，
#       見 WP-11 效能外推），所以 `all` 在啟動背景 `run` 後會停止、印出如何用
#       `status` 追蹤進度、以及完成後應該再手動呼叫 `postprocess`（或再次
#       `bash scripts/run_native_backtrack_server.sh all`，此時 preflight／
#       gap-fill／plan 會因為略過條件成立而快速跳過，`run` 因為已全部完成而
#       立即回傳，實際只會真的執行 `postprocess`）。
#
# 必要環境變數（全部只放路徑／模組規格，不放密碼）：
#
#   OCM_NATIVE_ROOT               ocm_native schema 3 根目錄
#   NWW_ANALYSIS_ROOT             nww3_analysis schema 1 根目錄
#   LBT_BAYTRACE_ROOT             BayTrace 原版原始碼根目錄
#   LBT_OUTPUT_ROOT               本次執行的計畫、完成證據、log、進度快照與
#                                 後處理輸出根目錄
#   LBT_SCRATCH_ROOT              NUMBA_CACHE_DIR 與 checkpoint 的預設落點
#                                 （本機磁碟，不是 `/data` NFS——見環境備註
#                                 「Numba 快取與檢查點放本機磁碟」）
#   LBT_NATIVE_CONFIG             正式設定 YAML 路徑
#   LBT_NATIVE_IDENTITY_TABLE     WP-4 粒子身分表 npz 路徑
#
# 只有 `plan`（未加 `LBT_NATIVE_SKIP_GAP_CHECK=1` 時）才需要：
#
#   LBT_NATIVE_REGION_TO_FLOW_DOMAIN
#       `{analysis_region_id: flow_domain_id}` JSON 檔路徑；`native-plan` 的
#       到達時刻缺口安全檢查用它把 `analysis_region_id` 換成真正的 OCM 產品
#       目錄名稱（見 `cli._discover_native_month_dirs`）——省略這個環境變數會
#       讓缺口檢查改用 `analysis_region_id` 當目錄名稱去找月份目錄，在正式
#       四區產品樹（目錄名稱是 `northeast_taiwan_common_cache_v3` 之類的
#       flow_domain_id，不是 `A`／`B`／`C`／`D`）下必然找不到目錄而失敗；正式
#       執行務必提供。（`hour_major_provider` 執行 `run` 步驟本身不再依賴這
#       個對照表——見該模組「已知限制」第 5 點的修正紀錄——但 `plan` 步驟的
#       缺口檢查仍然需要。）
#
# 可選環境變數（均有安全預設值，可省略）：
#
#   OCM_RECONSTRUCTION_ROOT       缺口重建 patch 根目錄；預設
#                                 `${LBT_OUTPUT_ROOT}/native-run/ocm-reconstruction`
#                                 （`gap-fill` 的輸出目錄，`run` 步驟的 worker
#                                 子程序透過同名環境變數讀取，見
#                                 `production_wiring` 模組文件）
#   LBT_NATIVE_SKIP_GAP_FILL      設為 `1` 略過 `gap-fill` 這步（例如已知目前
#                                 資料完全沒有缺口，或缺口重建已經在別處執行）
#   LBT_NATIVE_GAP_FILL_START_UTC／LBT_NATIVE_GAP_FILL_END_UTC
#                                 缺口重建區間，預設涵蓋 README 登錄的正式期間
#                                 `2024-01-01T00:00:00+00:00`～
#                                 `2025-12-31T23:00:00+00:00`
#   LBT_NATIVE_GAP_FILL_INCLUDE_WAVE
#                                 設為 `1` 時 `gap-fill` 同時重建波浪缺口（需要
#                                 `NWW_ANALYSIS_ROOT`）
#   LBT_NATIVE_MAX_BACKTRACK_DAYS 預設 30
#   LBT_NATIVE_SEGMENTS           `native-plan --segments`；單一整數（套用到全部
#                                 流場區域）或 `{flow_domain_id: K}` JSON 檔路徑；
#                                 省略時改用 `LBT_NATIVE_TARGET_WORKER_COUNT`
#   LBT_NATIVE_TARGET_WORKER_COUNT
#                                 `native-plan --target-worker-count`；兩者都省略
#                                 時等同 `K=1`（不切段）
#   LBT_NATIVE_NODE_COUNTS／LBT_NATIVE_LAYER_COUNTS
#                                 WP-13：兩者都提供時（各自一個
#                                 `{flow_domain_id: 數量}` JSON 檔路徑），
#                                 `native-plan` 改用正確成本模型（流場小時數×
#                                 節點數×層數）自動選段數，並在報表印出
#                                 `cost_model`（預估牆鐘時間、重複成本比例、
#                                 各段成本）；只提供其中一個不會生效，見
#                                 docs/operations/19_native_backward_batch_
#                                 orchestrator.md 第 9.10.2／9.10.3 節
#   LBT_NATIVE_SECONDS_PER_NODE_LAYER_HOUR
#                                 WP-13 成本模型的校準常數，省略時用模組內建
#                                 預設值（在小規模沙箱量測，正式伺服器應該
#                                 重新量測後覆寫，見同上文件節次）
#   LBT_NATIVE_WORKER_COUNT_FOR_ESTIMATE
#                                 WP-13：只影響 native-plan 報表估算的「預估
#                                 總牆鐘時間」，不影響實際切段；省略時退回
#                                 `LBT_NATIVE_TARGET_WORKER_COUNT`
#   LBT_NATIVE_WAIT_FOR_STAGED    WP-13：資料逐區域到位時，指到
#                                 `native-run --wait-for-staged` 用的標記檔
#                                 目錄；省略時（預設）行為與 WP-13 之前完全
#                                 相同，見同上文件第 9.10.4 節
#   LBT_NATIVE_SKIP_GAP_CHECK     設為 `1` 讓 `native-plan` 加 `--skip-gap-check`
#                                 （不建議正式使用；僅供合成資料試跑）
#   LBT_NATIVE_ALLOW_UNSAFE_ARRIVALS
#                                 設為 `1` 才允許在缺口不安全時仍啟動 `run`
#                                 （預設 fail-closed，拒絕啟動）
#   LBT_NATIVE_WORKER_COUNT       手動指定 `native-run` worker 數
#   LBT_NATIVE_AUTO_WORKER_COUNT  設為 `1` 改用測量式自動估算（見
#                                 `orchestrator.estimate_worker_count`）
#   LBT_NATIVE_PER_WORKER_MEMORY_BYTES  搭配上者使用，單一 worker 預估用量
#   LBT_NATIVE_RESERVED_MEMORY_BYTES    保留給其他程序的記憶體位元組數
#   LBT_NATIVE_MAX_WORKERS         worker 數上限
#   LBT_NATIVE_MP_CONTEXT          spawn／forkserver／fork（預設 spawn）
#   LBT_NATIVE_FORMAL              設為 `1`（預設）表示本次是正式（非工程）
#                                  執行；`0` 用 pilot 模式
#   LBT_NATIVE_CHECKPOINT_EVERY_HOURS
#                                  段內 `hour_major` checkpoint 間隔小時數，
#                                  預設 24
#   LBT_NATIVE_FORCE_RERUN         `all` 動作忽略既有完成標記，強制每步重跑
#   NUMBA_CACHE_DIR                預設 `${LBT_SCRATCH_ROOT}/numba-cache`
#   LBT_CHECKSUM_CACHE_DIR         預設 `${LBT_SCRATCH_ROOT}/checksum-cache`（重建 patch 已驗證雜湊快取）
#   LBT_DECLARED_GIT_COMMIT        寫入 manifest 的宣告 commit
#   LBT_NATIVE_CONFIG_SHA256       寫入 manifest 的設定雜湊
#
# 範例（先規劃再視報告決定是否啟動；正式執行請提供
# `LBT_NATIVE_REGION_TO_FLOW_DOMAIN`）：
#
#   export OCM_NATIVE_ROOT=/data/OCM-Preprocessed-Data/preprocessed/ocm_native
#   export NWW_ANALYSIS_ROOT=/data/NWW3-Preprocessed-Data/preprocessed/nww3_analysis
#   export LBT_BAYTRACE_ROOT=/data/LBT/baytrace-root
#   export LBT_OUTPUT_ROOT=/data/LBT/native-run/2026-09-21
#   export LBT_SCRATCH_ROOT=/scratch/lbt   # 本機磁碟，不是 /data
#   export LBT_NATIVE_CONFIG=/data/LBT/configs/formal.yaml
#   export LBT_NATIVE_IDENTITY_TABLE=/data/LBT/inputs/particle_identity_table.npz
#   export LBT_NATIVE_REGION_TO_FLOW_DOMAIN=/data/LBT/configs/region_to_flow_domain.json
#   export LBT_NATIVE_TARGET_WORKER_COUNT=56
#   bash scripts/run_native_backtrack_server.sh preflight
#   bash scripts/run_native_backtrack_server.sh gap-fill
#   bash scripts/run_native_backtrack_server.sh plan
#   # 檢查印出的 is_gap_safe_ready 與 segment_plan 之後：
#   bash scripts/run_native_backtrack_server.sh run
#   bash scripts/run_native_backtrack_server.sh status
#   # run 全部完成後：
#   bash scripts/run_native_backtrack_server.sh postprocess
#
# 已知限制（誠實揭露，非隱藏）：
#   - `all` 動作在啟動背景 `run` 後不會等待其完成，見上方 `all` 小節；操作者
#     必須自行用 `status` 追蹤，並在確認完成後再呼叫 `postprocess`。
#   - `gap-fill` 這步的重建邏輯由 `native_runner.gap_fill`（另一個並行開發的
#     工作包）提供，本腳本只負責呼叫與略過條件；若該模組品質或範圍有問題，
#     不在本腳本的驗證範圍內。
#   - `run` 固定使用 `hour_major_provider` 小時主軸切段平行路徑；不支援本腳本
#     切換回 WP-7 逐工作單元路徑的 resource provider。

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

readonly SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LBT_PROJECT_DIRECTORY="$(cd "${SCRIPT_DIRECTORY}/.." && pwd)"

fail() {
  printf '失敗：%s\n' "${1}" >&2
  exit 2
}

readonly ACTION="${1:-}"
case "${ACTION}" in
  preflight|gap-fill|plan|run|status|postprocess|all) ;;
  *) fail "用法：$0 preflight|gap-fill|plan|run|status|postprocess|all（見檔頭使用方式）" ;;
esac

# --- 必要環境變數與路徑存在性檢查 -------------------------------------------

readonly REQUIRED_ENVIRONMENTS=(
  OCM_NATIVE_ROOT
  NWW_ANALYSIS_ROOT
  LBT_BAYTRACE_ROOT
  LBT_OUTPUT_ROOT
  LBT_SCRATCH_ROOT
  LBT_NATIVE_CONFIG
  LBT_NATIVE_IDENTITY_TABLE
)
for variable_name in "${REQUIRED_ENVIRONMENTS[@]}"; do
  [[ -n "${!variable_name:-}" ]] || fail "缺少環境變數 ${variable_name}"
done

for variable_name in OCM_NATIVE_ROOT NWW_ANALYSIS_ROOT LBT_BAYTRACE_ROOT \
  LBT_OUTPUT_ROOT LBT_SCRATCH_ROOT LBT_NATIVE_CONFIG LBT_NATIVE_IDENTITY_TABLE; do
  case "${!variable_name}" in
    /*) ;;
    *) fail "${variable_name} 必須是絕對路徑" ;;
  esac
done
[[ -d "${OCM_NATIVE_ROOT}" ]] || fail "OCM_NATIVE_ROOT 指向的目錄不存在：${OCM_NATIVE_ROOT}"
[[ -d "${NWW_ANALYSIS_ROOT}" ]] || fail "NWW_ANALYSIS_ROOT 指向的目錄不存在：${NWW_ANALYSIS_ROOT}"
[[ -d "${LBT_BAYTRACE_ROOT}" ]] || fail "LBT_BAYTRACE_ROOT 指向的目錄不存在：${LBT_BAYTRACE_ROOT}"
[[ -f "${LBT_NATIVE_CONFIG}" ]] || fail "LBT_NATIVE_CONFIG 指向的檔案不存在：${LBT_NATIVE_CONFIG}"
[[ -f "${LBT_NATIVE_IDENTITY_TABLE}" ]] || fail "LBT_NATIVE_IDENTITY_TABLE 指向的檔案不存在：${LBT_NATIVE_IDENTITY_TABLE}"

readonly OCM_RECONSTRUCTION_ROOT="${OCM_RECONSTRUCTION_ROOT:-${LBT_OUTPUT_ROOT}/native-run/ocm-reconstruction}"
case "${OCM_RECONSTRUCTION_ROOT}" in
  /*) ;;
  *) fail "OCM_RECONSTRUCTION_ROOT 必須是絕對路徑" ;;
esac
export OCM_RECONSTRUCTION_ROOT

readonly NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${LBT_SCRATCH_ROOT}/numba-cache}"
case "${NUMBA_CACHE_DIR}" in
  /*) ;;
  *) fail "NUMBA_CACHE_DIR 必須是絕對路徑" ;;
esac
mkdir -p "${NUMBA_CACHE_DIR}"
export NUMBA_CACHE_DIR
export PYTHONDONTWRITEBYTECODE=1
# 每個工作程序只用一個數學函式庫執行緒：平行度由 native-run 的多程序提供，
# 若讓 NumPy／BLAS 各自再開多執行緒，56 個程序會嚴重過度訂閱核心。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"

# 重建 patch 的已驗證雜湊快取：同一個未變動的 patch 檔只完整計算一次 SHA-256，
# 避免每個工作程序啟動時都把數十 GB 的 patch 重讀一遍（見 forcing_window._sha256_file）。
readonly LBT_CHECKSUM_CACHE_DIR="${LBT_CHECKSUM_CACHE_DIR:-${LBT_SCRATCH_ROOT}/checksum-cache}"
mkdir -p "${LBT_CHECKSUM_CACHE_DIR}"
export LBT_CHECKSUM_CACHE_DIR

readonly FORCE_RERUN="${LBT_NATIVE_FORCE_RERUN:-0}"
readonly MAX_BACKTRACK_DAYS="${LBT_NATIVE_MAX_BACKTRACK_DAYS:-30}"
readonly FORMAL="${LBT_NATIVE_FORMAL:-1}"

readonly NATIVE_RUN_DIRECTORY="${LBT_OUTPUT_ROOT}/native-run"
readonly PLAN_JSON="${NATIVE_RUN_DIRECTORY}/native-plan.json"
readonly RESULTS_DIRECTORY="${NATIVE_RUN_DIRECTORY}/results"
readonly POSTPROCESS_DIRECTORY="${NATIVE_RUN_DIRECTORY}/postprocess"
readonly GAP_FILL_MARKER="${NATIVE_RUN_DIRECTORY}/gap-fill.done"
readonly LOG_DIRECTORY="${LBT_SCRATCH_ROOT}/native-run-logs"
readonly CHECKPOINT_ROOT="${LBT_SCRATCH_ROOT}/native-checkpoints"
mkdir -p "${NATIVE_RUN_DIRECTORY}" "${LOG_DIRECTORY}" "${CHECKPOINT_ROOT}"

# --- Python／numba 可用性檢查 ------------------------------------------------

command -v uv >/dev/null 2>&1 || fail "找不到 uv；請先於伺服器安裝並執行 uv sync --frozen"
(
  cd "${LBT_PROJECT_DIRECTORY}"
  uv run python3 -c "import sys; assert sys.version_info >= (3, 11), sys.version" \
    || fail "Python 版本不符（需要 3.11+）"
  uv run python3 -c "import numba" || fail "numba 未安裝或無法載入"
  uv run python3 -c "import lagrangian_backtracking.native_runner.hour_major_provider" \
    || fail "hour_major_provider 模組無法載入；請先確認 uv sync --frozen 已完成"
)

run_lbt() {
  (cd "${LBT_PROJECT_DIRECTORY}" && uv run lbt "$@")
}

# --- preflight：唯讀資料契約檢查 ---------------------------------------------

run_preflight() {
  local preflight_args=(native-preflight --config "${LBT_NATIVE_CONFIG}" --max-backtrack-days "${MAX_BACKTRACK_DAYS}")
  if [[ "${FORMAL}" == "1" ]]; then
    preflight_args+=(--formal)
  else
    preflight_args+=(--pilot)
  fi
  if [[ -n "${LBT_NATIVE_PREFLIGHT_OUTPUT:-}" ]]; then
    preflight_args+=(--output "${LBT_NATIVE_PREFLIGHT_OUTPUT}")
  fi
  run_lbt "${preflight_args[@]}"
}

# --- gap-fill：已核准缺口重建 -------------------------------------------------

run_gap_fill() {
  if [[ "${FORCE_RERUN}" != "1" && -f "${GAP_FILL_MARKER}" ]]; then
    printf 'gap-fill 已完成（見 %s），略過；如需強制重跑請設定 LBT_NATIVE_FORCE_RERUN=1\n' \
      "${GAP_FILL_MARKER}"
    return 0
  fi
  local start_utc="${LBT_NATIVE_GAP_FILL_START_UTC:-2024-01-01T00:00:00+00:00}"
  local end_utc="${LBT_NATIVE_GAP_FILL_END_UTC:-2025-12-31T23:00:00+00:00}"
  local gap_fill_args=(
    native-gap-fill
    --ocm-native-domain-root "${OCM_NATIVE_ROOT}"
    --output-root "${OCM_RECONSTRUCTION_ROOT}"
    --start-utc "${start_utc}"
    --end-utc "${end_utc}"
  )
  if [[ "${LBT_NATIVE_GAP_FILL_INCLUDE_WAVE:-0}" == "1" ]]; then
    gap_fill_args+=(--include-wave --nww-native-domain-root "${NWW_ANALYSIS_ROOT}")
  fi
  # 用 if 包住呼叫，避免 `set -e` 在非零結束碼時直接中止腳本、跳過下面「只有
  # 成功才寫完成標記檔」的判斷（標記檔只代表「gap-fill 呼叫本身成功」，不是
  # 「這批資料完全沒有缺口」——沒有缺口與有缺口但已重建，都會回傳 0）。
  if run_lbt "${gap_fill_args[@]}"; then
    mkdir -p "${NATIVE_RUN_DIRECTORY}"
    date -u +%Y-%m-%dT%H:%M:%SZ >"${GAP_FILL_MARKER}"
    return 0
  fi
  return 1
}

# --- plan：工作單元切段規劃＋缺口安全檢查 ------------------------------------

run_plan() {
  if [[ "${FORCE_RERUN}" != "1" && -f "${PLAN_JSON}" ]]; then
    printf 'native-plan 計畫檔已存在（%s），略過；如需強制重跑請設定 LBT_NATIVE_FORCE_RERUN=1\n' \
      "${PLAN_JSON}"
    return 0
  fi
  local plan_args=(
    native-plan
    --identity-table "${LBT_NATIVE_IDENTITY_TABLE}"
    --ocm-native-root "${OCM_NATIVE_ROOT}"
    --max-backtrack-days "${MAX_BACKTRACK_DAYS}"
    --project-root "${LBT_PROJECT_DIRECTORY}"
    --config-sha256 "${LBT_NATIVE_CONFIG_SHA256:-not_provided}"
    --output "${PLAN_JSON}"
    --schedule hour-major-segmented
    --config "${LBT_NATIVE_CONFIG}"
  )
  if [[ -n "${LBT_NATIVE_REGION_TO_FLOW_DOMAIN:-}" ]]; then
    plan_args+=(--region-to-flow-domain "${LBT_NATIVE_REGION_TO_FLOW_DOMAIN}")
  elif [[ "${LBT_NATIVE_SKIP_GAP_CHECK:-0}" != "1" ]]; then
    fail "plan 需要 LBT_NATIVE_REGION_TO_FLOW_DOMAIN（除非明確設定 LBT_NATIVE_SKIP_GAP_CHECK=1，不建議正式使用）"
  fi
  if [[ -n "${LBT_NATIVE_SEGMENTS:-}" ]]; then
    plan_args+=(--segments "${LBT_NATIVE_SEGMENTS}")
  elif [[ -n "${LBT_NATIVE_TARGET_WORKER_COUNT:-}" ]]; then
    plan_args+=(--target-worker-count "${LBT_NATIVE_TARGET_WORKER_COUNT}")
  fi
  # WP-13：--node-counts／--layer-counts 兩者都提供時才會啟用正確成本模型
  # 自動選段數＋native-plan 報表的 cost_model 欄位（見 docs/operations/
  # 19_native_backward_batch_orchestrator.md 第 9.10.2／9.10.3 節）；只提供
  # 其中一個沒有意義，這裡刻意要求兩者同時設定才加上旗標，避免呼叫端誤以為
  # 只設定一個就夠。
  if [[ -n "${LBT_NATIVE_NODE_COUNTS:-}" && -n "${LBT_NATIVE_LAYER_COUNTS:-}" ]]; then
    plan_args+=(--node-counts "${LBT_NATIVE_NODE_COUNTS}" --layer-counts "${LBT_NATIVE_LAYER_COUNTS}")
    if [[ -n "${LBT_NATIVE_SECONDS_PER_NODE_LAYER_HOUR:-}" ]]; then
      plan_args+=(--seconds-per-node-layer-hour "${LBT_NATIVE_SECONDS_PER_NODE_LAYER_HOUR}")
    fi
    if [[ -n "${LBT_NATIVE_WORKER_COUNT_FOR_ESTIMATE:-}" ]]; then
      plan_args+=(--worker-count-for-estimate "${LBT_NATIVE_WORKER_COUNT_FOR_ESTIMATE}")
    fi
  fi
  if [[ "${LBT_NATIVE_SKIP_GAP_CHECK:-0}" == "1" ]]; then
    plan_args+=(--skip-gap-check)
  fi
  if [[ -n "${LBT_DECLARED_GIT_COMMIT:-}" ]]; then
    plan_args+=(--declared-git-commit "${LBT_DECLARED_GIT_COMMIT}")
  fi
  if [[ "${FORMAL}" == "1" ]]; then
    plan_args+=(--formal)
  fi
  run_lbt "${plan_args[@]}"
}

# --- run：背景平行執行＋續跑 --------------------------------------------------

readonly HOUR_MAJOR_PROVIDER="lagrangian_backtracking.native_runner.hour_major_provider:make_hour_major_resource_provider"

run_status() {
  run_lbt native-status --output "${NATIVE_RUN_DIRECTORY}"
}

start_native_run_in_background() {
  [[ -f "${PLAN_JSON}" ]] || fail "找不到計畫檔 ${PLAN_JSON}；請先執行 plan 步驟"
  mkdir -p "${RESULTS_DIRECTORY}"

  local run_args=(
    native-run "${PLAN_JSON}"
    --output "${NATIVE_RUN_DIRECTORY}"
    --resource-provider "${HOUR_MAJOR_PROVIDER}"
    --numba-cache-dir "${NUMBA_CACHE_DIR}"
    --baytrace-root "${LBT_BAYTRACE_ROOT}"
    --project-root "${LBT_PROJECT_DIRECTORY}"
    --config-sha256 "${LBT_NATIVE_CONFIG_SHA256:-not_provided}"
    --mp-context "${LBT_NATIVE_MP_CONTEXT:-spawn}"
  )
  if [[ -n "${LBT_DECLARED_GIT_COMMIT:-}" ]]; then
    run_args+=(--declared-git-commit "${LBT_DECLARED_GIT_COMMIT}")
  fi
  if [[ "${FORMAL}" == "1" ]]; then
    run_args+=(--formal)
  fi
  if [[ "${LBT_NATIVE_ALLOW_UNSAFE_ARRIVALS:-0}" == "1" ]]; then
    run_args+=(--allow-unsafe-arrivals)
  fi
  if [[ -n "${LBT_NATIVE_WORKER_COUNT:-}" ]]; then
    run_args+=(--worker-count "${LBT_NATIVE_WORKER_COUNT}")
  elif [[ "${LBT_NATIVE_AUTO_WORKER_COUNT:-0}" == "1" ]]; then
    run_args+=(--auto-worker-count)
    [[ -n "${LBT_NATIVE_PER_WORKER_MEMORY_BYTES:-}" ]] || \
      fail "LBT_NATIVE_AUTO_WORKER_COUNT=1 需要 LBT_NATIVE_PER_WORKER_MEMORY_BYTES"
    run_args+=(--per-worker-memory-bytes "${LBT_NATIVE_PER_WORKER_MEMORY_BYTES}")
  else
    fail "run 需要 LBT_NATIVE_WORKER_COUNT 或 LBT_NATIVE_AUTO_WORKER_COUNT=1"
  fi
  if [[ -n "${LBT_NATIVE_RESERVED_MEMORY_BYTES:-}" ]]; then
    run_args+=(--reserved-memory-bytes "${LBT_NATIVE_RESERVED_MEMORY_BYTES}")
  fi
  if [[ -n "${LBT_NATIVE_MAX_WORKERS:-}" ]]; then
    run_args+=(--max-workers "${LBT_NATIVE_MAX_WORKERS}")
  fi
  # WP-13：資料依區域逐批到位（工作書記載的到站順序：連江 → 後灣 → 新竹 →
  # 東北）時，設定這個變數指到 --staged-root 用的同一個目錄，讓 native-run
  # 在早到的區域資料就緒後立即開始算，不必等全部區域到齊；見 docs/operations/
  # 19_native_backward_batch_orchestrator.md 第 9.10.4 節。省略時（預設）
  # 行為與 WP-13 之前完全相同——必須全部區域都已經在計畫檔的工作單元裡才會
  # 啟動（本腳本一律等 plan 完成後才呼叫 run，本來就隱含這個前提）。
  if [[ -n "${LBT_NATIVE_WAIT_FOR_STAGED:-}" ]]; then
    run_args+=(--wait-for-staged "${LBT_NATIVE_WAIT_FOR_STAGED}")
  fi

  local timestamp
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local log_path="${LOG_DIRECTORY}/native-run-${timestamp}.log"

  (
    cd "${LBT_PROJECT_DIRECTORY}"
    export LBT_NATIVE_CONFIG="${LBT_NATIVE_CONFIG}"
    export LBT_NATIVE_PLAN="${PLAN_JSON}"
    export LBT_NATIVE_IDENTITY_TABLE="${LBT_NATIVE_IDENTITY_TABLE}"
    export LBT_NATIVE_RESULTS_DIR="${RESULTS_DIRECTORY}"
    export LBT_NATIVE_FORMAL="${FORMAL}"
    export LBT_NATIVE_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}"
    export LBT_NATIVE_CHECKPOINT_EVERY_HOURS="${LBT_NATIVE_CHECKPOINT_EVERY_HOURS:-24}"
    export OCM_RECONSTRUCTION_ROOT="${OCM_RECONSTRUCTION_ROOT}"
    # shellcheck disable=SC2024
    nohup uv run lbt "${run_args[@]}" >"${log_path}" 2>&1 &
    local background_pid=$!
    disown "${background_pid}" 2>/dev/null || true
    printf '已背景啟動 native-run：pid=%s log=%s output=%s results=%s\n' \
      "${background_pid}" "${log_path}" "${NATIVE_RUN_DIRECTORY}" "${RESULTS_DIRECTORY}"
  )
}

# --- postprocess：合併全部工作單元成果 ---------------------------------------

run_postprocess() {
  local marker="${POSTPROCESS_DIRECTORY}/manifest.json"
  if [[ "${FORCE_RERUN}" != "1" && -f "${marker}" ]]; then
    printf 'postprocess 已完成（見 %s），略過；如需強制重跑請設定 LBT_NATIVE_FORCE_RERUN=1\n' "${marker}"
    return 0
  fi
  if [[ ! -d "${RESULTS_DIRECTORY}" ]] || [[ -z "$(ls -A "${RESULTS_DIRECTORY}" 2>/dev/null)" ]]; then
    fail "找不到任何 native-run 結果（${RESULTS_DIRECTORY} 是空的）；請先確認 run 步驟已完成"
  fi
  mkdir -p "${POSTPROCESS_DIRECTORY}"
  local postprocess_args=(
    native-postprocess
    --identity-table "${LBT_NATIVE_IDENTITY_TABLE}"
    --results-dir "${RESULTS_DIRECTORY}"
    --output "${POSTPROCESS_DIRECTORY}"
    --config "${LBT_NATIVE_CONFIG}"
  )
  run_lbt "${postprocess_args[@]}"
}

# --- 主流程 -------------------------------------------------------------------

main() {
  case "${ACTION}" in
    preflight)
      run_preflight
      ;;
    gap-fill)
      run_gap_fill
      ;;
    plan)
      run_plan
      ;;
    run)
      start_native_run_in_background
      ;;
    status)
      run_status
      ;;
    postprocess)
      run_postprocess
      ;;
    all)
      run_preflight || fail "preflight 未通過（is_run_ready=false 或執行失敗），已停止；請先排除問題"
      run_gap_fill
      run_plan
      start_native_run_in_background
      cat <<AFTER_RUN

已啟動背景 native-run；本動作（all）不會等待其完成（可能長達數天，見 WP-11
效能外推報告）。請用下列指令追蹤進度，全部完成後再呼叫 postprocess（或再次
執行 "bash scripts/run_native_backtrack_server.sh all"，屆時 preflight／
gap-fill／plan 會因為略過條件成立而快速跳過，run 因為已全部完成而立即
回傳，實際只會真的執行 postprocess）：

  bash scripts/run_native_backtrack_server.sh status
  bash scripts/run_native_backtrack_server.sh postprocess

AFTER_RUN
      ;;
  esac
}

main "$@"
