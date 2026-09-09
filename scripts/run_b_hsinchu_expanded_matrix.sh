#!/usr/bin/env bash
# B 區 hsinchu 擴充 pilot 的 tracked SERVER 入口。
#
# 這個入口只負責 deployment boundary：project/code/.venv 可以在 /home，但
# package、成果、scratch、checkpoint、logs 與所有執行快取必須由 operator 明示
# 放在同一個 NFS 結果根下。通過 storage gate 後才把 /data package 交給既有
# orchestration；因此 package 內的 mkdir、uv 與 run-create 不可能繞過本檢查。

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

readonly MODE="${1:-PREPARE}"

fail() {
  printf '失敗：%s\n' "$1" >&2
  exit 2
}

if [[ "$MODE" != "PREPARE" && "$MODE" != "RUN" ]]; then
  fail '模式只能是 PREPARE（預設）或 RUN'
fi

# 這些值全部由 SERVER operator 提供；入口不猜測 /data 子目錄，也不把結果
# fallback 到 project root 或作業系統暫存區。詳細的目錄、symlink、mount、空間、
# write 與 flock 驗證交給同一個 machine-readable Python gate。
readonly REQUIRED_ENVIRONMENTS=(
  LBT_PROJECT_ROOT
  LBT_RESULT_NFS_ROOT
  LBT_EXECUTION_PACKAGE_ROOT
  LBT_OUTPUT_ROOT
  LBT_SCRATCH_ROOT
  LBT_CHECKPOINT_ROOT
  LBT_UV_CACHE_ROOT
  LBT_MPL_CACHE_ROOT
  LBT_XDG_CACHE_ROOT
  LBT_TMP_ROOT
  OCM_NATIVE_ROOT
  OCM_SURFACE_ROOT
  NWW_ANALYSIS_ROOT
  LBT_EXPECTED_GIT_COMMIT
  LBT_MIN_FREE_GB
)
for variable_name in "${REQUIRED_ENVIRONMENTS[@]}"; do
  [[ -n "${!variable_name:-}" ]] || fail "缺少環境變數 $variable_name"
done

for variable_name in \
  LBT_PROJECT_ROOT LBT_RESULT_NFS_ROOT LBT_EXECUTION_PACKAGE_ROOT \
  LBT_OUTPUT_ROOT LBT_SCRATCH_ROOT LBT_CHECKPOINT_ROOT LBT_UV_CACHE_ROOT \
  LBT_MPL_CACHE_ROOT LBT_XDG_CACHE_ROOT LBT_TMP_ROOT OCM_NATIVE_ROOT \
  OCM_SURFACE_ROOT NWW_ANALYSIS_ROOT; do
  [[ "${!variable_name}" == /* ]] || fail "$variable_name 必須是絕對路徑"
done

readonly PROJECT_ROOT="$LBT_PROJECT_ROOT"
readonly PROJECT_VENV="$PROJECT_ROOT/.venv"
readonly GATE_SCRIPT="$PROJECT_ROOT/scripts/validate_server_storage.py"
readonly GATE_SNAPSHOT="$LBT_SCRATCH_ROOT/server-storage-gate-$(date -u +%Y%m%dT%H%M%SZ)-$$.json"

[[ -f "$GATE_SCRIPT" && ! -L "$GATE_SCRIPT" ]] || fail '找不到 tracked storage gate'

# cache/environment 變數在 gate 前就固定為 operator 指定的 NFS 子目錄；絕不採用
# caller 既有值。UV_PROJECT_ENVIRONMENT 明確鎖在 project root 下既有 .venv，讓
# /home 僅承載程式與 runtime，而不承載任何執行結果或 cache。
export UV_CACHE_DIR="$LBT_UV_CACHE_ROOT"
export MPLCONFIGDIR="$LBT_MPL_CACHE_ROOT"
export XDG_CACHE_HOME="$LBT_XDG_CACHE_ROOT"
export TMPDIR="$LBT_TMP_ROOT"
export UV_PROJECT_ENVIRONMENT="$PROJECT_VENV"
export PYTHONDONTWRITEBYTECODE=1

# 這是本入口唯一的 mutation 前置 gate。Python 會將 snapshot 原子寫入既有 NFS
# scratch parent；失敗時以狀態 2 結束，下一行 package orchestration 不會執行。
python3 "$GATE_SCRIPT" \
  --project-root "$PROJECT_ROOT" \
  --project-venv "$PROJECT_VENV" \
  --result-nfs-root "$LBT_RESULT_NFS_ROOT" \
  --execution-package-root "$LBT_EXECUTION_PACKAGE_ROOT" \
  --output-root "$LBT_OUTPUT_ROOT" \
  --scratch-root "$LBT_SCRATCH_ROOT" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT" \
  --uv-cache-root "$LBT_UV_CACHE_ROOT" \
  --mpl-cache-root "$LBT_MPL_CACHE_ROOT" \
  --xdg-cache-root "$LBT_XDG_CACHE_ROOT" \
  --tmp-root "$LBT_TMP_ROOT" \
  --minimum-free-gib "$LBT_MIN_FREE_GB" \
  --snapshot-output "$GATE_SNAPSHOT"

[[ -d "$LBT_EXECUTION_PACKAGE_ROOT" && ! -L "$LBT_EXECUTION_PACKAGE_ROOT" ]] || \
  fail 'execution package root gate 後不存在'
[[ -f "$LBT_EXECUTION_PACKAGE_ROOT/run_server_matrix.sh" && \
   ! -L "$LBT_EXECUTION_PACKAGE_ROOT/run_server_matrix.sh" ]] || \
  fail 'NFS execution package 缺少 run_server_matrix.sh'

# package 由 operator 以 /data 路徑提供；既有 orchestration 仍負責 pilot matrix
# manifest、preflight、run-create、run-shard、reconcile 與 validate-run。
exec bash "$LBT_EXECUTION_PACKAGE_ROOT/run_server_matrix.sh" "$MODE"
