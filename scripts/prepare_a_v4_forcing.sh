#!/usr/bin/env bash
# 以相鄰 OCM/NWW 前處理專案的正式 CLI 產製 LBT A 區 v4 forcing。
#
# 資料來源與輸出：
# - OCM raw：CWA-OCM SCHISM 日 NetCDF；由 OCM-Data-Preprocessing 產生 schema 3
#   ocm_native 與 ocm_surface。缺日月份以 standard_partial_month 原樣記錄，絕不補零。
# - NWW native：既有 0.025°、2024–2025 完整逐時月份快取；每月直接以 native
#   time_utc_ns 作目標時間，重採樣到 v4 OCM 靜態格網，因此不沿用 OCM 缺時。
# - 輸出一律寫入新的 v4 domain ID，不覆寫既有 v3。
#
# 使用方式：
#   bash scripts/prepare_a_v4_forcing.sh dry-run
#   bash scripts/prepare_a_v4_forcing.sh month 2025 1
#   bash scripts/prepare_a_v4_forcing.sh all
#
# `dry-run` 只盤點 24 個月 OCM raw 並列出預定輸出；`month` 先產製與驗證單月 OCM，
# 再建立同月 NWW full-hour analysis；`all` 依年月循序呼叫相同流程。大型輸出若已存在，
# 腳本先以唯讀 validator 驗收後跳過，避免未授權的覆寫或刪除。

set -euo pipefail

readonly A_V4_DOMAIN_ID="northeast_taiwan_common_cache_v4_lbt_south_expanded"
readonly SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LBT_PROJECT_DIRECTORY="$(cd "$SCRIPT_DIRECTORY/.." && pwd)"
readonly A_V4_CONFIG="$LBT_PROJECT_DIRECTORY/configs/upstream/ocm_flow_domains_lbt_a_v4.json"

# 所有路徑均可由 task-specific 環境變數覆寫；預設值對應既有 SERVER layout。
# 不使用 HOME 或模糊 glob 作寫入根目錄，避免誤把大型產品寫到非預期磁碟。
readonly OCM_PROJECT_DIRECTORY="${LBT_OCM_PROJECT_ROOT:-/home/mustlab/Workspace/OCM-Data-Preprocessing}"
readonly NWW_PROJECT_DIRECTORY="${LBT_NWW_PROJECT_ROOT:-/home/mustlab/Workspace/NWW-Data-Preprocessing}"
readonly OCM_RAW_DIRECTORY="${LBT_OCM_RAW_ROOT:-/CWA-OCM}"
readonly OCM_PREPROCESSED_DIRECTORY="${LBT_OCM_PREPROCESSED_ROOT:-/data/OCM-Preprocessed-Data/preprocessed}"
readonly NWW_PREPROCESSED_DIRECTORY="${LBT_NWW_PREPROCESSED_ROOT:-/data/NWW-Preprocessed-Data/preprocessed}"
readonly NWW_NATIVE_GRID_ID="${LBT_NWW_NATIVE_GRID_ID:-ww3_grd3_253x237}"

# uv cache 使用本工項專屬名稱，避免依賴登入帳號的預設 cache；Python bytecode 不寫回
# Git worktree，使 SERVER 部署目錄在 dry-run 與正式批次後仍可稽核 dirty state。
export UV_CACHE_DIR="${LBT_WORKSPACE_UV_CACHE_ROOT:-/private/tmp/lbt-a-v4-uv-cache}"
export PYTHONDONTWRITEBYTECODE=1

require_file() {
  # 在啟動大量 I/O 前確認固定入口存在；輸入是單一明確路徑，沒有自動搜尋或猜測。
  local required_path="$1"
  if [[ ! -f "$required_path" ]]; then
    printf '缺少必要檔案：%s\n' "$required_path" >&2
    return 1
  fi
}

validate_year_month() {
  # 年月只允許研究期間 2024–2025 與合法月份，避免手誤建立範圍外大型產品。
  local year="$1"
  local month="$2"
  if [[ "$year" != "2024" && "$year" != "2025" ]]; then
    printf 'year 必須是 2024 或 2025：%s\n' "$year" >&2
    return 1
  fi
  if [[ ! "$month" =~ ^([1-9]|1[0-2])$ ]]; then
    printf 'month 必須介於 1 到 12：%s\n' "$month" >&2
    return 1
  fi
}

run_ocm_dry_run() {
  # `--allow-partial-months` 表示使用全部可得日檔並保存 missing_days；dry-run 不寫大型陣列。
  (
    cd "$OCM_PROJECT_DIRECTORY"
    uv run python3 scripts/preprocess_ocm_flow_domains.py \
      --config "$A_V4_CONFIG" \
      --raw-root "$OCM_RAW_DIRECTORY" \
      --output-root "$OCM_PREPROCESSED_DIRECTORY" \
      --years 2024 2025 \
      --months 1 2 3 4 5 6 7 8 9 10 11 12 \
      --domains "$A_V4_DOMAIN_ID" \
      --allow-partial-months \
      --skip-missing-months \
      --dry-run
  )
}

run_month() {
  # 單月流程保持 OCM→OCM validator→NWW→NWW validator 的依賴順序；只有完整驗收後才
  # 進入下一月份，讓中斷重啟時能明確辨識已完成與尚未完成的產品。
  local year="$1"
  local month="$2"
  validate_year_month "$year" "$month"
  local year_month
  year_month="$(printf '%04d%02d' "$year" "$month")"

  local ocm_surface_month="$OCM_PREPROCESSED_DIRECTORY/ocm_surface/$A_V4_DOMAIN_ID/months/$year_month"
  local nww_native_month="$NWW_PREPROCESSED_DIRECTORY/nww3_native/$NWW_NATIVE_GRID_ID/months/$year_month"
  local nww_analysis_month="$NWW_PREPROCESSED_DIRECTORY/nww3_analysis/$A_V4_DOMAIN_ID/months/$year_month"

  if [[ -d "$ocm_surface_month" ]]; then
    (
      cd "$OCM_PROJECT_DIRECTORY"
      uv run python3 scripts/validate_ocm_flow_cache.py "$ocm_surface_month"
    )
  else
    (
      cd "$OCM_PROJECT_DIRECTORY"
      uv run python3 scripts/preprocess_ocm_flow_domains.py \
        --config "$A_V4_CONFIG" \
        --raw-root "$OCM_RAW_DIRECTORY" \
        --output-root "$OCM_PREPROCESSED_DIRECTORY" \
        --years "$year" \
        --months "$month" \
        --domains "$A_V4_DOMAIN_ID" \
        --allow-partial-months
      uv run python3 scripts/validate_ocm_flow_cache.py "$ocm_surface_month"
    )
  fi

  require_file "$nww_native_month/time_utc_ns.npy"
  if [[ -d "$nww_analysis_month" ]]; then
    (
      cd "$NWW_PROJECT_DIRECTORY"
      uv run python3 scripts/validate_nww3_analysis_grid.py "$nww_analysis_month"
    )
  else
    (
      cd "$NWW_PROJECT_DIRECTORY"
      uv run python3 scripts/preprocess_nww3_analysis_grid.py \
        --native-month-dir "$nww_native_month" \
        --target-grid-dir "$OCM_PREPROCESSED_DIRECTORY/ocm_surface/$A_V4_DOMAIN_ID/grid" \
        --target-time-npy "$nww_native_month/time_utc_ns.npy" \
        --flow-domain-id "$A_V4_DOMAIN_ID" \
        --output-root "$NWW_PREPROCESSED_DIRECTORY"
      uv run python3 scripts/validate_nww3_analysis_grid.py "$nww_analysis_month"
    )
  fi
}

main() {
  require_file "$A_V4_CONFIG"
  require_file "$OCM_PROJECT_DIRECTORY/scripts/preprocess_ocm_flow_domains.py"
  require_file "$NWW_PROJECT_DIRECTORY/scripts/preprocess_nww3_analysis_grid.py"

  local action="${1:-dry-run}"
  case "$action" in
    dry-run)
      run_ocm_dry_run
      ;;
    month)
      if [[ "$#" -ne 3 ]]; then
        printf '用法：%s month YEAR MONTH\n' "$0" >&2
        return 2
      fi
      run_month "$2" "$3"
      ;;
    all)
      local year
      local month
      for year in 2024 2025; do
        for month in {1..12}; do
          run_month "$year" "$month"
        done
      done
      ;;
    *)
      printf '未知 action：%s；可用 dry-run、month、all。\n' "$action" >&2
      return 2
      ;;
  esac
}

main "$@"
