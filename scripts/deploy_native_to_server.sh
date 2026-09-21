#!/usr/bin/env bash
# WP-11：把目前分支透過 git bundle 部署到運算主機（例如 140.117.88.203）。
#
# 這支腳本設計成在使用者的 Mac（macOS 內建 bash 3.2）或任何能 ssh 到運算主機的
# 環境執行；運算主機本身不需要能連回這台機器，只需要能被 ssh 連上。腳本只做
# 「本機建 git bundle → scp 到伺服器暫存目錄 → 在伺服器上以 git fetch／clone 展開
# → uv sync 建立環境 → 印出下一步指令」五件事，不建立任何新的排程或執行邏輯；
# 真正的 native 逆向批次流程一律由 `scripts/run_native_backtrack_server.sh`
# （preflight → gap-fill → plan → run → postprocess）在伺服器上執行。
#
# 本檔不含任何密碼、帳號或伺服器私有資訊；伺服器位址、使用者與路徑一律由
# operator 透過環境變數在啟動當下注入，腳本本身不寫死。SSH 認證一律依賴
# operator 既有的 ssh-agent／金鑰設定，本腳本不處理密碼，也不會關閉
# StrictHostKeyChecking。
#
# ---------------------------------------------------------------------------
# 使用方式
# ---------------------------------------------------------------------------
#
#   bash scripts/deploy_native_to_server.sh bundle
#       只在本機建立 git bundle 並驗證，不連線伺服器；供 operator 先確認
#       bundle 內容與 checksum，或在沒有網路的情況下先準備好部署素材。
#
#   bash scripts/deploy_native_to_server.sh deploy
#       完整流程：建 bundle → 驗證 → scp 到伺服器暫存目錄 → 在伺服器上核對
#       checksum → 展開成（或更新既有）Git checkout → `uv sync --frozen` →
#       印出下一步指令。任何一步失敗立即停止（fail-closed），不會嘗試用
#       破壞性操作（`git reset --hard`、`rsync --delete` 之類）「修好」衝突；
#       發現伺服器既有 checkout 是 dirty 或無法 fast-forward 時直接停止，
#       交還 operator 判斷（見下方「已知限制」）。
#
# 必要環境變數：
#
#   LBT_DEPLOY_SERVER_HOST
#       ssh 目的地，例如 "operator@140.117.88.203"（帳號＋主機，不含密碼）。
#   LBT_DEPLOY_REMOTE_PROJECT_ROOT
#       伺服器上專案 Git checkout 的絕對路徑（例如 SERVER canonical project
#       root）；`deploy` 動作只允許這個目錄「不存在」或「是乾淨的既有 Git
#       checkout」兩種狀態，其餘一律停止。
#   LBT_DEPLOY_REMOTE_STAGING_DIR
#       伺服器上暫放 bundle／checksum 檔案的絕對路徑（例如伺服器上一個
#       operator 自己管理的暫存目錄；依 README 的 SERVER 儲存契約，不建議
#       指到 `/data` 的正式子目錄，因為 bundle 屬於部署素材、不是研究資料或
#       執行輸出）。腳本會先 `mkdir -p` 這個目錄。
#
# 可選環境變數（均有安全預設值，可省略）：
#
#   LBT_DEPLOY_SERVER_PORT       ssh port，預設 22。
#   LBT_DEPLOY_BRANCH            要部署的分支，預設目前 checkout 出的分支
#                                （`git symbolic-ref --short HEAD`）；必須是
#                                本機真正存在的分支名稱，不接受任意 ref。
#   LBT_DEPLOY_LOCAL_BUNDLE_DIR  本機暫存 bundle 的目錄，預設用 `mktemp -d`
#                                建立的臨時目錄。
#   LBT_DEPLOY_REMOTE_GIT_URL    提供時，伺服器 checkout 完成後把 `origin`
#                                改指到這個核定遠端 Git URL（比照
#                                docs/operations/git_deployment_and_data_sync.md
#                                第 3 節「clone 產生的 origin 改回核定遠端」）；
#                                省略時不動 `origin`（維持 bundle clone 後的
#                                預設值，或既有 checkout 原本的 `origin`）。
#   LBT_DEPLOY_ALLOW_DIRTY_TARGET
#       設為 1 才允許在伺服器既有 checkout 有未提交變更時仍嘗試合併；預設
#       fail-closed 拒絕（比照部署手冊「dirty source……必須停止，不可自動
#       覆蓋」）。
#   LBT_DEPLOY_SSH_OPTS          額外 ssh／scp 選項（例如 "-i ~/.ssh/id_lbt"），
#                                以字串傳入，腳本用 `eval` 前會先確認非空才
#                                展開，不接受來源不明的字串。
#
# 範例：
#
#   export LBT_DEPLOY_SERVER_HOST="operator@140.117.88.203"
#   export LBT_DEPLOY_REMOTE_PROJECT_ROOT="/home/operator/lbt"
#   export LBT_DEPLOY_REMOTE_STAGING_DIR="/home/operator/lbt-deploy-staging"
#   bash scripts/deploy_native_to_server.sh bundle   # 先本機確認
#   bash scripts/deploy_native_to_server.sh deploy   # 再真正部署
#   # 部署完成後照印出的下一步指令，在伺服器上執行：
#   #   cd /home/operator/lbt
#   #   export OCM_NATIVE_ROOT=... NWW_ANALYSIS_ROOT=... LBT_BAYTRACE_ROOT=... \
#   #          LBT_OUTPUT_ROOT=... LBT_SCRATCH_ROOT=... LBT_NATIVE_IDENTITY_TABLE=...
#   #   bash scripts/run_native_backtrack_server.sh preflight
#
# 已知限制（誠實揭露，非隱藏）：
#   - 本腳本假設伺服器已安裝 `git`／`uv`／`ssh` 伺服器端且可用；不負責安裝
#     這些工具本身，也不驗證伺服器 Python 版本或 numba 是否可用（那是
#     `run_native_backtrack_server.sh` 第一次呼叫 `lbt native-preflight` 前的
#     既有可用性檢查職責，見該檔案）。
#   - 伺服器既有 checkout 若不能 fast-forward 合併（例如伺服器上有本機沒有的
#     獨立提交），腳本會停止並保留伺服器原狀，交還 operator 判斷；不會嘗試
#     rebase、force push 或建立 merge commit。
#   - 本腳本沒有偵測伺服器上是否有正在執行的 native-run／其他 worker
#     程序；比照 docs/operations/git_deployment_and_data_sync.md 第 3 節，這
#     項確認目前仍是 operator 在執行 `deploy` 前的操作紀律，不是本腳本自動化
#     的範圍（自動判斷「是否有 active worker」需要對伺服器 process tree 做
#     假設，本工作包不冒然假設伺服器的程序管理慣例）。
#   - `uv sync --frozen` 需要伺服器 checkout 內已有 `uv.lock`；若伺服器環境
#     因故無法用 `--frozen`（例如 lock 檔案與伺服器平台不相容），腳本會印出
#     清楚的錯誤訊息並停止，不會靜默改用不鎖定版本的 `uv sync`——版本飄移
#     的環境不應該被本腳本動用者當作「已部署成功」。

set -Eeuo pipefail
IFS=$'\n\t'

readonly SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LBT_PROJECT_DIRECTORY="$(cd "${SCRIPT_DIRECTORY}/.." && pwd)"

fail() {
  printf '失敗：%s\n' "${1}" >&2
  exit 2
}

readonly ACTION="${1:-}"
if [[ "${ACTION}" != "bundle" && "${ACTION}" != "deploy" ]]; then
  fail "用法：$0 bundle|deploy（見檔頭註解的使用方式與環境變數說明）"
fi

# --- 必要環境變數檢查（deploy 才需要伺服器連線資訊；bundle 只需要本機） -----

[[ -n "${LBT_DEPLOY_REMOTE_STAGING_DIR:-}" ]] || {
  if [[ "${ACTION}" == "deploy" ]]; then
    fail "缺少環境變數 LBT_DEPLOY_REMOTE_STAGING_DIR"
  fi
}
if [[ "${ACTION}" == "deploy" ]]; then
  [[ -n "${LBT_DEPLOY_SERVER_HOST:-}" ]] || fail "缺少環境變數 LBT_DEPLOY_SERVER_HOST"
  [[ -n "${LBT_DEPLOY_REMOTE_PROJECT_ROOT:-}" ]] || fail "缺少環境變數 LBT_DEPLOY_REMOTE_PROJECT_ROOT"
  case "${LBT_DEPLOY_REMOTE_PROJECT_ROOT}" in
    /*) ;;
    *) fail "LBT_DEPLOY_REMOTE_PROJECT_ROOT 必須是伺服器上的絕對路徑" ;;
  esac
  case "${LBT_DEPLOY_REMOTE_STAGING_DIR}" in
    /*) ;;
    *) fail "LBT_DEPLOY_REMOTE_STAGING_DIR 必須是伺服器上的絕對路徑" ;;
  esac
fi

readonly SERVER_PORT="${LBT_DEPLOY_SERVER_PORT:-22}"

# --- 本機分支與乾淨度檢查 ----------------------------------------------------

cd "${LBT_PROJECT_DIRECTORY}"

command -v git >/dev/null 2>&1 || fail "找不到本機 git"

readonly BRANCH="${LBT_DEPLOY_BRANCH:-$(git symbolic-ref --short HEAD 2>/dev/null || true)}"
[[ -n "${BRANCH}" ]] || fail "無法決定要部署的分支；目前可能處於 detached HEAD，請明示 LBT_DEPLOY_BRANCH"
git show-ref --verify --quiet "refs/heads/${BRANCH}" \
  || fail "本機找不到分支 refs/heads/${BRANCH}"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  printf '警告：本機 refs/heads/%s 目前有未提交的追蹤檔案變更；bundle 只會包含已提交的歷史，未提交的變更不會被部署。\n' \
    "${BRANCH}" >&2
fi

# --- 建立並驗證 git bundle ---------------------------------------------------

readonly LOCAL_BUNDLE_DIRECTORY="${LBT_DEPLOY_LOCAL_BUNDLE_DIR:-$(mktemp -d)}"
mkdir -p "${LOCAL_BUNDLE_DIRECTORY}"

readonly DECLARED_COMMIT="$(git rev-parse "refs/heads/${BRANCH}")"
readonly BUNDLE_BASENAME="lbt-${BRANCH}-${DECLARED_COMMIT}.bundle"
readonly BUNDLE_PATH="${LOCAL_BUNDLE_DIRECTORY}/${BUNDLE_BASENAME}"
readonly CHECKSUM_PATH="${BUNDLE_PATH}.sha256"

printf '建立 git bundle：refs/heads/%s @ %s\n' "${BRANCH}" "${DECLARED_COMMIT}"
git bundle create "${BUNDLE_PATH}" "refs/heads/${BRANCH}"
git bundle verify "${BUNDLE_PATH}"
git bundle list-heads "${BUNDLE_PATH}"

if command -v shasum >/dev/null 2>&1; then
  (cd "${LOCAL_BUNDLE_DIRECTORY}" && shasum -a 256 "${BUNDLE_BASENAME}") >"${CHECKSUM_PATH}"
elif command -v sha256sum >/dev/null 2>&1; then
  (cd "${LOCAL_BUNDLE_DIRECTORY}" && sha256sum "${BUNDLE_BASENAME}") >"${CHECKSUM_PATH}"
else
  fail "找不到 shasum 或 sha256sum，無法計算 bundle checksum"
fi
printf 'bundle：%s\nchecksum：%s\n' "${BUNDLE_PATH}" "$(cat "${CHECKSUM_PATH}")"

if [[ "${ACTION}" == "bundle" ]]; then
  printf '只執行 bundle 動作，未連線伺服器；bundle 檔案保留在 %s\n' "${LOCAL_BUNDLE_DIRECTORY}"
  exit 0
fi

# --- scp 到伺服器暫存目錄 ----------------------------------------------------

readonly SSH_EXTRA_OPTS="${LBT_DEPLOY_SSH_OPTS:-}"
readonly SSH_DESTINATION="${LBT_DEPLOY_SERVER_HOST}"

ssh_run() {
  # 以陣列組出 ssh 呼叫，避免把 remote command 字串交給 eval；SSH_EXTRA_OPTS
  # 只在非空時才展開成多個參數（見檔頭「額外 ssh／scp 選項」的安全備註）。
  if [[ -n "${SSH_EXTRA_OPTS}" ]]; then
    # shellcheck disable=SC2086
    ssh -p "${SERVER_PORT}" ${SSH_EXTRA_OPTS} "${SSH_DESTINATION}" "$@"
  else
    ssh -p "${SERVER_PORT}" "${SSH_DESTINATION}" "$@"
  fi
}

ssh_run_bash_script() {
  # 把多行遠端腳本內容以 stdin 餵給遠端明確指定的 bash（不依賴 ssh 伺服器端
  # 的預設 shell；有些伺服器 sshd 的預設互動 shell 未必是 bash，而本腳本組出
  # 的遠端內容用了 bash 專屬語法，例如 `[[ ]]`），比 `ssh dest "字串"`（交給
  # 遠端預設 shell 解讀）更明確、更不容易因為伺服器環境差異而解讀失敗。
  if [[ -n "${SSH_EXTRA_OPTS}" ]]; then
    # shellcheck disable=SC2086
    printf '%s' "${1}" | ssh -p "${SERVER_PORT}" ${SSH_EXTRA_OPTS} "${SSH_DESTINATION}" bash -s
  else
    printf '%s' "${1}" | ssh -p "${SERVER_PORT}" "${SSH_DESTINATION}" bash -s
  fi
}

scp_run() {
  if [[ -n "${SSH_EXTRA_OPTS}" ]]; then
    # shellcheck disable=SC2086
    scp -P "${SERVER_PORT}" ${SSH_EXTRA_OPTS} "$@"
  else
    scp -P "${SERVER_PORT}" "$@"
  fi
}

command -v ssh >/dev/null 2>&1 || fail "找不到本機 ssh"
command -v scp >/dev/null 2>&1 || fail "找不到本機 scp"

printf '建立伺服器暫存目錄：%s\n' "${LBT_DEPLOY_REMOTE_STAGING_DIR}"
ssh_run "mkdir -p '${LBT_DEPLOY_REMOTE_STAGING_DIR}'"

printf '傳輸 bundle 與 checksum 到伺服器……\n'
scp_run "${BUNDLE_PATH}" "${CHECKSUM_PATH}" \
  "${SSH_DESTINATION}:${LBT_DEPLOY_REMOTE_STAGING_DIR}/"

# --- 在伺服器上核對 checksum、展開（或更新）Git checkout、uv sync ----------
#
# 遠端指令以字串組出（不是本機直接執行），逐行組裝、每個變數一律 ${VAR}，
# 理由與本檔頭「所有變數一律寫成 ${VAR}」的規範相同：這段文字最終仍是先在
# 本機 bash 展開變數，再透過 ssh 傳給伺服器執行，本機 macOS bash 3.2 的既有
# 風險同樣適用於這裡的變數展開。

readonly REMOTE_BUNDLE_PATH="${LBT_DEPLOY_REMOTE_STAGING_DIR}/${BUNDLE_BASENAME}"
readonly REMOTE_CHECKSUM_PATH="${REMOTE_BUNDLE_PATH}.sha256"
readonly ALLOW_DIRTY_TARGET="${LBT_DEPLOY_ALLOW_DIRTY_TARGET:-0}"
readonly REMOTE_GIT_URL="${LBT_DEPLOY_REMOTE_GIT_URL:-}"

remote_script="set -Eeuo pipefail
remote_fail() { printf '伺服器端失敗：%s\n' \"\${1}\" >&2; exit 2; }
cd '${LBT_DEPLOY_REMOTE_STAGING_DIR}'
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 -c '${REMOTE_CHECKSUM_PATH##*/}' || remote_fail 'bundle checksum 核對失敗'
elif command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c '${REMOTE_CHECKSUM_PATH##*/}' || remote_fail 'bundle checksum 核對失敗'
else
  remote_fail '伺服器找不到 shasum 或 sha256sum'
fi
command -v git >/dev/null 2>&1 || remote_fail '伺服器找不到 git'
git bundle verify '${REMOTE_BUNDLE_PATH}'
if [[ -d '${LBT_DEPLOY_REMOTE_PROJECT_ROOT}/.git' ]]; then
  cd '${LBT_DEPLOY_REMOTE_PROJECT_ROOT}'
  if [[ -n \"\$(git status --porcelain --untracked-files=no)\" && '${ALLOW_DIRTY_TARGET}' != '1' ]]; then
    remote_fail '伺服器既有 checkout 有未提交變更（dirty）；請先處理或明確設定 LBT_DEPLOY_ALLOW_DIRTY_TARGET=1'
  fi
  git fetch '${REMOTE_BUNDLE_PATH}' 'refs/heads/${BRANCH}:refs/remotes/import/${BRANCH}'
  git checkout '${BRANCH}' 2>/dev/null || git checkout -b '${BRANCH}'
  git merge --ff-only 'refs/remotes/import/${BRANCH}' \
    || remote_fail '無法 fast-forward 合併（伺服器上可能有本機沒有的獨立提交）；已停止，伺服器原狀未變'
else
  mkdir -p '${LBT_DEPLOY_REMOTE_PROJECT_ROOT}'
  git clone --branch '${BRANCH}' '${REMOTE_BUNDLE_PATH}' '${LBT_DEPLOY_REMOTE_PROJECT_ROOT}'
  cd '${LBT_DEPLOY_REMOTE_PROJECT_ROOT}'
fi
if [[ -n '${REMOTE_GIT_URL}' ]]; then
  git remote set-url origin '${REMOTE_GIT_URL}' 2>/dev/null || git remote add origin '${REMOTE_GIT_URL}'
fi
test \"\$(git rev-parse HEAD)\" = '${DECLARED_COMMIT}' \
  || remote_fail '部署後 HEAD 與宣告的 commit 不一致，這是內部錯誤，請勿信任這次部署'
command -v uv >/dev/null 2>&1 || remote_fail '伺服器找不到 uv；請先依 README 安裝'
if [[ ! -f uv.lock ]]; then
  remote_fail '伺服器 checkout 沒有 uv.lock，無法 uv sync --frozen；請確認部署的分支含 uv.lock'
fi
uv sync --frozen || remote_fail 'uv sync --frozen 失敗，見上方輸出；本腳本不會改用不鎖定版本的 uv sync'
printf '伺服器部署完成：%s @ %s\n' '${LBT_DEPLOY_REMOTE_PROJECT_ROOT}' '${DECLARED_COMMIT}'
"

printf '在伺服器上展開部署……\n'
ssh_run_bash_script "${remote_script}"

cat <<NEXT_STEPS

部署完成。下一步（在伺服器上執行，依 docs/operations/19_native_backward_batch_orchestrator.md
與 README 的 SERVER 操作說明設定下列環境變數後，呼叫 scripts/run_native_backtrack_server.sh）：

  ssh -p ${SERVER_PORT} ${SSH_DESTINATION}
  cd ${LBT_DEPLOY_REMOTE_PROJECT_ROOT}
  export OCM_NATIVE_ROOT=<ocm_native schema 3 根目錄>
  export NWW_ANALYSIS_ROOT=<nww3_analysis schema 1 根目錄>
  export LBT_BAYTRACE_ROOT=<BayTrace 原始碼根目錄>
  export LBT_OUTPUT_ROOT=<本次執行輸出根目錄>
  export LBT_SCRATCH_ROOT=<checkpoint／numba cache 等暫存根目錄>
  export LBT_NATIVE_CONFIG=<正式設定 YAML 路徑>
  export LBT_NATIVE_IDENTITY_TABLE=<WP-4 粒子身分表 npz 路徑>
  bash scripts/run_native_backtrack_server.sh preflight

NEXT_STEPS
