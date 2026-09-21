#!/usr/bin/env bash
# WP-7：在伺服器（例如 .203）上執行 native 逆向批次 orchestrator 的入口。
#
# 這個腳本只負責「檢查前置條件、跑一次唯讀的 native-plan 缺口安全檢查、
# 通過後才以背景程序啟動 native-run」三件事；它不建立任何新的排程或續跑邏輯——
# 真正的工作單元切分、worker 排程、記憶體估算、續跑與缺口安全重選都在
# `src/lagrangian_backtracking/native_runner/orchestrator.py`，經由既有
# `lbt native-plan`／`lbt native-run`／`lbt native-status` 子命令呼叫（見
# docs/operations/19_native_backward_batch_orchestrator.md）。
#
# 本檔不含任何密碼或伺服器私有資訊；所有路徑、憑證與伺服器層級限制一律由
# operator 透過環境變數在啟動當下注入，腳本本身不寫死。
#
# ---------------------------------------------------------------------------
# 使用方式
# ---------------------------------------------------------------------------
#
#   bash scripts/run_native_backtrack_server.sh plan
#       只執行 native-plan（工作單元切分＋到達時刻缺口安全檢查），把報告印到
#       stdout 並寫出計畫 JSON；不啟動任何背景執行、不呼叫 BayTrace 核心。
#       核准者應先看這一步的 `is_gap_safe_ready`／`unsafe_arrival_count`。
#
#   bash scripts/run_native_backtrack_server.sh start
#       先重跑一次 plan（與上面相同的唯讀檢查），只有 `is_gap_safe_ready=true`
#       （或明確設定 LBT_NATIVE_ALLOW_UNSAFE_ARRIVALS=1）才會用 `nohup` 把
#       `lbt native-run` 丟到背景執行；stdout／stderr 導向一個帶時間戳記的
#       log 檔，並把該檔路徑與背景 PID 印在最後一行，方便後續 `tail -f`
#       或寫進 operator 自己的排程紀錄。
#
#   bash scripts/run_native_backtrack_server.sh status
#       呼叫 `lbt native-status` 讀取目前的進度快照與完成證據統計；可在
#       `start` 之後隨時重複執行，不影響正在跑的背景程序。
#
# 必要環境變數（全部只放路徑／模組規格，不放密碼）：
#
#   OCM_NATIVE_ROOT           ocm_native schema 3 根目錄（缺口安全檢查用）
#   NWW_ANALYSIS_ROOT         nww3_analysis schema 1 根目錄（存在性檢查；
#                             wave bridge 尚未整合進本 orchestrator，見下方
#                             「已知限制」，此處只先驗證路徑就緒）
#   LBT_BAYTRACE_ROOT         BayTrace 原版原始碼根目錄（native-run 用於記錄
#                             manifest 內的來源雜湊；實際核心呼叫由
#                             --resource-provider 的模組自行載入）
#   LBT_OUTPUT_ROOT           本次執行的計畫、完成證據、log 與進度快照根目錄
#   LBT_SCRATCH_ROOT          NUMBA_CACHE_DIR 與其他暫存檔的預設落點
#   LBT_NATIVE_IDENTITY_TABLE WP-4 `build_particle_identity_table` 輸出的
#                             粒子身分表 npz 路徑
#
# 只有 `start`／`run` 才需要（`plan`／`status` 不需要）：
#
#   LBT_NATIVE_RESOURCE_PROVIDER
#       '模組路徑:函式名稱'；該函式須回傳 (build_worker_resources, run_unit)
#       兩個模組層級可 pickle 的 callable，實際載入網格、BayTrace 與 forcing
#       橋接。本腳本刻意不內建任何預設值——production 網格／interest region／
#       波浪橋接的接線屬於其他工作包，尚未在本專案落地前，不應該讓伺服器腳本
#       假造一個可能不正確的預設接線。
#
# 可選環境變數（均有安全預設值，可省略）：
#
#   LBT_NATIVE_REGION_TO_FLOW_DOMAIN   region→flow_domain_id JSON 檔路徑
#   LBT_NATIVE_MAX_BACKTRACK_DAYS      預設 30
#   LBT_NATIVE_ALLOW_UNSAFE_ARRIVALS   設為 1 才允許在缺口不安全時仍啟動
#                                      （預設 fail-closed，拒絕啟動）
#   LBT_NATIVE_WORKER_COUNT            手動指定 worker 數
#   LBT_NATIVE_AUTO_WORKER_COUNT       設為 1 改用測量式自動估算
#   LBT_NATIVE_PER_WORKER_MEMORY_BYTES 搭配上者使用，單一 worker 預估用量
#   LBT_NATIVE_RESERVED_MEMORY_BYTES   保留給其他程序的記憶體位元組數
#   LBT_NATIVE_MAX_WORKERS             worker 數上限
#   LBT_NATIVE_MP_CONTEXT              spawn／forkserver／fork（預設 spawn）
#   NUMBA_CACHE_DIR                    預設 `$LBT_SCRATCH_ROOT/numba-cache`
#   LBT_DECLARED_GIT_COMMIT            寫入 manifest 的宣告 commit
#   LBT_NATIVE_FORMAL                  設為 1 表示本次是正式（非工程）執行
#   LBT_NATIVE_CONFIG_SHA256           寫入 manifest 的設定雜湊
#
# 範例（先規劃再視報告決定是否啟動）：
#
#   export OCM_NATIVE_ROOT=/data/LBT/ocm_native
#   export NWW_ANALYSIS_ROOT=/data/LBT/nww3_analysis
#   export LBT_BAYTRACE_ROOT=/data/LBT/baytrace-root
#   export LBT_OUTPUT_ROOT=/data/LBT/native-run/2026-09-21
#   export LBT_SCRATCH_ROOT=/data/LBT/scratch
#   export LBT_NATIVE_IDENTITY_TABLE=/data/LBT/inputs/particle_identity_table.npz
#   bash scripts/run_native_backtrack_server.sh plan
#   # 檢查印出的 is_gap_safe_ready 與 unsafe_arrivals 之後：
#   export LBT_NATIVE_RESOURCE_PROVIDER=my_site_package.native_resources:make_resources
#   bash scripts/run_native_backtrack_server.sh start
#   bash scripts/run_native_backtrack_server.sh status
#
# 已知限制（誠實揭露，非隱藏）：
#   - 本腳本與 orchestrator 只涵蓋工作單元規劃、排程、平行執行、續跑與到達
#     時刻缺口安全檢查；`wave_bridge`／`postprocess` 尚未整合，`start` 不會
#     自動產生波浪或後處理輸出。
#   - `--resource-provider` 的實際網格／interest region／forcing 接線由
#     operator／其他工作包提供；本腳本不驗證其科學正確性，只負責啟動與
#     記錄 manifest。

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

readonly SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LBT_PROJECT_DIRECTORY="$(cd "$SCRIPT_DIRECTORY/.." && pwd)"

fail() {
  printf '失敗：%s\n' "$1" >&2
  exit 2
}

readonly ACTION="${1:-}"
if [[ "$ACTION" != "plan" && "$ACTION" != "start" && "$ACTION" != "status" ]]; then
  fail "用法：$0 plan|start|status（見檔頭註解的使用方式與環境變數說明）"
fi

# --- 必要環境變數與路徑存在性檢查 -----------------------------------------

readonly REQUIRED_ENVIRONMENTS=(
  OCM_NATIVE_ROOT
  NWW_ANALYSIS_ROOT
  LBT_BAYTRACE_ROOT
  LBT_OUTPUT_ROOT
  LBT_SCRATCH_ROOT
  LBT_NATIVE_IDENTITY_TABLE
)
for variable_name in "${REQUIRED_ENVIRONMENTS[@]}"; do
  [[ -n "${!variable_name:-}" ]] || fail "缺少環境變數 $variable_name"
done

if [[ "$ACTION" == "start" ]]; then
  [[ -n "${LBT_NATIVE_RESOURCE_PROVIDER:-}" ]] || \
    fail "start 需要 LBT_NATIVE_RESOURCE_PROVIDER（'模組路徑:函式名稱'）"
fi

for variable_name in OCM_NATIVE_ROOT NWW_ANALYSIS_ROOT LBT_BAYTRACE_ROOT \
  LBT_OUTPUT_ROOT LBT_SCRATCH_ROOT; do
  [[ "${!variable_name}" == /* ]] || fail "$variable_name 必須是絕對路徑"
  [[ -d "${!variable_name}" ]] || fail "$variable_name 指向的目錄不存在：${!variable_name}"
done

[[ "$LBT_NATIVE_IDENTITY_TABLE" == /* ]] || fail "LBT_NATIVE_IDENTITY_TABLE 必須是絕對路徑"
[[ -f "$LBT_NATIVE_IDENTITY_TABLE" ]] || \
  fail "LBT_NATIVE_IDENTITY_TABLE 指向的檔案不存在：$LBT_NATIVE_IDENTITY_TABLE"

readonly NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$LBT_SCRATCH_ROOT/numba-cache}"
[[ "$NUMBA_CACHE_DIR" == /* ]] || fail "NUMBA_CACHE_DIR 必須是絕對路徑"
mkdir -p "$NUMBA_CACHE_DIR"
export NUMBA_CACHE_DIR
export PYTHONDONTWRITEBYTECODE=1

# --- Python／numba 可用性檢查 ------------------------------------------------

command -v uv >/dev/null 2>&1 || fail "找不到 uv；請先於伺服器安裝並執行 uv sync --frozen"
(
  cd "$LBT_PROJECT_DIRECTORY"
  uv run python3 -c "import sys; assert sys.version_info >= (3, 11), sys.version" \
    || fail "Python 版本不符（需要 3.11+）"
  uv run python3 -c "import numba" || fail "numba 未安裝或無法載入"
  uv run python3 -c "import lagrangian_backtracking.native_runner.orchestrator" \
    || fail "orchestrator 模組無法載入；請先確認 uv sync --frozen 已完成"
)

readonly NATIVE_RUN_DIRECTORY="$LBT_OUTPUT_ROOT/native-run"
readonly PLAN_JSON="$NATIVE_RUN_DIRECTORY/native-plan.json"
readonly LOG_DIRECTORY="$LBT_SCRATCH_ROOT/native-run-logs"
mkdir -p "$NATIVE_RUN_DIRECTORY" "$LOG_DIRECTORY"

# --- native-plan：唯讀規劃＋缺口安全檢查 ------------------------------------

run_plan() {
  local plan_args=(
    native-plan
    --identity-table "$LBT_NATIVE_IDENTITY_TABLE"
    --ocm-native-root "$OCM_NATIVE_ROOT"
    --max-backtrack-days "${LBT_NATIVE_MAX_BACKTRACK_DAYS:-30}"
    --project-root "$LBT_PROJECT_DIRECTORY"
    --config-sha256 "${LBT_NATIVE_CONFIG_SHA256:-not_provided}"
    --output "$PLAN_JSON"
  )
  if [[ -n "${LBT_NATIVE_REGION_TO_FLOW_DOMAIN:-}" ]]; then
    plan_args+=(--region-to-flow-domain "$LBT_NATIVE_REGION_TO_FLOW_DOMAIN")
  fi
  if [[ -n "${LBT_DECLARED_GIT_COMMIT:-}" ]]; then
    plan_args+=(--declared-git-commit "$LBT_DECLARED_GIT_COMMIT")
  fi
  if [[ "${LBT_NATIVE_FORMAL:-0}" == "1" ]]; then
    plan_args+=(--formal)
  fi
  (
    cd "$LBT_PROJECT_DIRECTORY"
    uv run lbt "${plan_args[@]}"
  )
}

run_status() {
  (
    cd "$LBT_PROJECT_DIRECTORY"
    uv run lbt native-status --output "$NATIVE_RUN_DIRECTORY"
  )
}

start_native_run_in_background() {
  local run_args=(
    native-run "$PLAN_JSON"
    --output "$NATIVE_RUN_DIRECTORY"
    --resource-provider "$LBT_NATIVE_RESOURCE_PROVIDER"
    --numba-cache-dir "$NUMBA_CACHE_DIR"
    --baytrace-root "$LBT_BAYTRACE_ROOT"
    --project-root "$LBT_PROJECT_DIRECTORY"
    --config-sha256 "${LBT_NATIVE_CONFIG_SHA256:-not_provided}"
    --mp-context "${LBT_NATIVE_MP_CONTEXT:-spawn}"
  )
  if [[ -n "${LBT_DECLARED_GIT_COMMIT:-}" ]]; then
    run_args+=(--declared-git-commit "$LBT_DECLARED_GIT_COMMIT")
  fi
  if [[ "${LBT_NATIVE_FORMAL:-0}" == "1" ]]; then
    run_args+=(--formal)
  fi
  if [[ "${LBT_NATIVE_ALLOW_UNSAFE_ARRIVALS:-0}" == "1" ]]; then
    run_args+=(--allow-unsafe-arrivals)
  fi
  if [[ -n "${LBT_NATIVE_WORKER_COUNT:-}" ]]; then
    run_args+=(--worker-count "$LBT_NATIVE_WORKER_COUNT")
  fi
  if [[ "${LBT_NATIVE_AUTO_WORKER_COUNT:-0}" == "1" ]]; then
    run_args+=(--auto-worker-count)
    [[ -n "${LBT_NATIVE_PER_WORKER_MEMORY_BYTES:-}" ]] || \
      fail "LBT_NATIVE_AUTO_WORKER_COUNT=1 需要 LBT_NATIVE_PER_WORKER_MEMORY_BYTES"
    run_args+=(--per-worker-memory-bytes "$LBT_NATIVE_PER_WORKER_MEMORY_BYTES")
  fi
  if [[ -n "${LBT_NATIVE_RESERVED_MEMORY_BYTES:-}" ]]; then
    run_args+=(--reserved-memory-bytes "$LBT_NATIVE_RESERVED_MEMORY_BYTES")
  fi
  if [[ -n "${LBT_NATIVE_MAX_WORKERS:-}" ]]; then
    run_args+=(--max-workers "$LBT_NATIVE_MAX_WORKERS")
  fi

  local timestamp
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local log_path="$LOG_DIRECTORY/native-run-$timestamp.log"

  (
    cd "$LBT_PROJECT_DIRECTORY"
    # shellcheck disable=SC2024
    nohup uv run lbt "${run_args[@]}" >"$log_path" 2>&1 &
    local background_pid=$!
    disown "$background_pid" 2>/dev/null || true
    printf '已背景啟動 native-run：pid=%s log=%s output=%s\n' \
      "$background_pid" "$log_path" "$NATIVE_RUN_DIRECTORY"
  )
}

main() {
  case "$ACTION" in
    plan)
      run_plan
      ;;
    start)
      local plan_report
      local plan_exit=0
      # `run_plan` 在缺口不安全時會以狀態 2 結束（見 cli.py 的
      # `run_native_plan_command`）；這裡刻意先攔下結束碼、印出完整 JSON
      # 報告，再由下面的 Python 明確讀 `is_gap_safe_ready` 欄位判斷是否可
      # 啟動，避免只靠結束碼臆測原因，也避免半途被 `set -e` 中止而看不到報告。
      plan_report="$(run_plan)" || plan_exit=$?
      printf '%s\n' "$plan_report"
      if [[ "$plan_exit" -ne 0 && "$plan_exit" -ne 2 ]]; then
        fail "native-plan 未能正常完成（結束碼 $plan_exit）；請先排除錯誤再重試"
      fi
      if [[ "${LBT_NATIVE_ALLOW_UNSAFE_ARRIVALS:-0}" != "1" ]]; then
        python3 - "$plan_report" <<'PY'
"""只讀取上一步 native-plan 印出的 JSON 報告，確認 is_gap_safe_ready；
不重新規劃、不重新檢查缺口，避免與 CLI 的 fail-closed 判斷邏輯分岔。
"""
import json
import sys

report = json.loads(sys.argv[1])
if not report.get("is_gap_safe_ready", False):
    print("native-plan 回報到達時刻缺口不安全，且未設定 LBT_NATIVE_ALLOW_UNSAFE_ARRIVALS=1；停止啟動。", file=sys.stderr)
    raise SystemExit(2)
PY
      fi
      start_native_run_in_background
      ;;
    status)
      run_status
      ;;
  esac
}

main "$@"
