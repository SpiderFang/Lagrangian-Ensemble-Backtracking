# SERVER 執行手冊

這是一份給實際操作者使用的手冊。若目的是執行既有 **B 區工程試跑**，只需依序完成
第 2～6 節；不必先理解程式架構或科學方法。

> 本文件中的命令都在 SERVER 執行。`<...>` 代表必須由本次執行負責人提供的值，不能原樣貼上，
> 也不能沿用過去紀錄猜測。每次執行都要重新通過 `PREPARE`；歷史成功不代表目前 SERVER 已通過。

## 1. 先確認要執行哪一種工作

| 目的 | 現況 | 應使用的入口 |
|---|---|---|
| 既有 B 區 12 組工程試跑 | 可執行，但必須先通過本次檢查 | `scripts/run_b_hsinchu_expanded_matrix.sh` |
| 新增其他 pilot（小型工程試跑） | 命令已存在，但目前沒有通用 SERVER 執行入口 | 由執行包維護者依第 7 節建立入口 |
| 2024–2025 五站正式批次 | **目前不可執行** | 等待第 8 節列出的正式輸入與驗證完成 |
| 完整報告建置 | **目前不可執行** | 專案目前尚無 `report-build` 命令 |

既有 B 區入口會依序完成儲存檢查、輸入檢查、建立或恢復工作目錄、執行分片、整理狀態及
完整驗證。一般操作者不需要手動串接這些內部命令。

## 2. 執行前先取得四個核准值

向本次執行負責人取得下列資料；缺少任一項就停止，不要自行選路徑或版本。

| 名稱 | 代表什麼 |
|---|---|
| `LBT_PROJECT_ROOT` | 本次真正要執行的 SERVER 程式目錄絕對路徑 |
| `LBT_EXPECTED_GIT_COMMIT` | 已核准的 40 碼 Git 版本編號（commit） |
| `LBT_EXECUTION_PACKAGE_ROOT` | 已部署完成的 B 區執行包（execution package）絕對路徑 |
| `LBT_MIN_FREE_GB` | 本次核准的最低剩餘空間，單位 GiB，必須是正整數 |

執行包內必須已包含 `run_server_matrix.sh`、固定設定、12 組清單、輸入成果與校準成果。
本手冊不負責建立或修改執行包。

## 3. 設定環境並核對版本

在 SERVER shell 貼上以下區塊。只替換有 `<...>` 的四個值；若資料管理者另有核准的 OCM／NWW
路徑，則以其書面提供值取代下列資料根目錄。

```bash
export LBT_PROJECT_ROOT="<本次核准的SERVER程式目錄絕對路徑>"
export LBT_EXPECTED_GIT_COMMIT="<本次核准的40碼Git版本編號>"
export LBT_EXECUTION_PACKAGE_ROOT="<本次核准的B區執行包絕對路徑>"
export LBT_MIN_FREE_GB="<本次核准的最低剩餘GiB>"

export LBT_RESULT_NFS_ROOT="/data/LBT"
export LBT_OUTPUT_ROOT="/data/LBT/outputs"
export LBT_SCRATCH_ROOT="/data/LBT/scratch"
export LBT_CHECKPOINT_ROOT="/data/LBT/checkpoints"
export LBT_UV_CACHE_ROOT="/data/LBT/cache/uv"
export LBT_MPL_CACHE_ROOT="/data/LBT/cache/matplotlib"
export LBT_XDG_CACHE_ROOT="/data/LBT/cache/xdg"
export LBT_TMP_ROOT="/data/LBT/tmp"

export OCM_NATIVE_ROOT="/data/OCM-Preprocessed-Data/preprocessed/ocm_native"
export OCM_SURFACE_ROOT="/data/OCM-Preprocessed-Data/preprocessed/ocm_surface"
export NWW_ANALYSIS_ROOT="/data/NWW-Preprocessed-Data/preprocessed/nww3_analysis"
```

接著核對程式目錄、版本與執行包：

```bash
test "$(git -C "$LBT_PROJECT_ROOT" rev-parse --show-toplevel)" = "$LBT_PROJECT_ROOT" && \
test "$(git -C "$LBT_PROJECT_ROOT" rev-parse HEAD)" = "$LBT_EXPECTED_GIT_COMMIT" && \
test -z "$(git -C "$LBT_PROJECT_ROOT" status --porcelain --untracked-files=all)" && \
test -f "$LBT_EXECUTION_PACKAGE_ROOT/run_server_matrix.sh" && \
echo "部署版本與執行包：PASS"
```

**成功判定：** 最後只出現 `部署版本與執行包：PASS`。

**沒有出現 PASS：** 立即停止。不要切換分支、清除工作樹、改寫執行包或改用另一個同名程式目錄；
請執行負責人重新提供正確值。

## 4. 先執行 PREPARE

`PREPARE` 會建立本次檢查紀錄，但不會建立、啟動或恢復粒子運算。

```bash
cd "$LBT_PROJECT_ROOT"
bash scripts/run_b_hsinchu_expanded_matrix.sh PREPARE
```

**成功判定：** 命令退出狀態為 0，且最後出現：

```text
PREPARE 完成：未建立、啟動或恢復任何 run workspace。
```

檢查內容包括：所有結果與快取都在 `/data/LBT` 的網路檔案系統（NFS）、目錄不是符號連結、
空間足夠、可寫入、可安全完成改名、跨程序鎖可用、Git 版本正確，以及執行包／輸入／設定彼此相符。

寫入能力以每個執行資料根上的實際探針為準：建立唯一暫存檔、寫入並同步檔案、在同一目錄
原子改名、重新開啟／同步後清理。NFS 的存取控制清單（ACL）或伺服器匯出設定（export）
可能使系統呼叫 `access(2)` 對寫入權限旗標 `W_OK` 的預先查詢，與實際檔案開啟 `open`／
寫入結果不同，因此不單獨依賴該查詢判定可寫性；目錄仍須通過搜尋權限檢查，而實際探針
任一步失敗會回報 `write_probe_failed` 並停止執行。

**任何檢查失敗：** 不要執行 `RUN`。保留完整終端輸出；若畫面是 JSON，查看其中的
`issues` 或 `errors`，否則查看第一個失敗命令的錯誤訊息，再交由執行負責人處理。

## 5. 在 tmux 中執行 RUN

先建立 tmux session，避免 SSH 中斷時連帶終止運算：

```bash
tmux new-session -s lbt-b-pilot
```

進入 tmux 後，**重新貼上第 3 節的環境變數區塊**，再執行：

```bash
cd "$LBT_PROJECT_ROOT"
bash scripts/run_b_hsinchu_expanded_matrix.sh RUN
```

`RUN` 會再次執行全部前置檢查，因此不會只依賴稍早的 `PREPARE` 結果。每組試跑含 4 個平行分片；
任一分片失敗時，腳本會保留日誌、失敗紀錄及先前已成功發布的續跑檔（checkpoint），並停止
啟動後續組別。若錯誤發生在第一份續跑檔建立前，該分片可能沒有可恢復的續跑檔。

離開 tmux 但保持運算：按 `Ctrl-b`，放開後按 `d`。重新查看：

```bash
tmux attach-session -t lbt-b-pilot
```

不要以「tmux 還在」或單一分片退出 0 判定全部完成。

## 6. 判定完成、失敗與續跑

### 6.1 完成判定

B 區 12 組工程試跑必須同時符合三項：

1. `RUN` 的 shell 退出狀態是 0。
2. 最後出現 `矩陣結果：12 組均已 COMPLETE 或安全跳過。`。
3. 本次 `summary.tsv` 的 12 列皆為 `COMPLETE` 或 `SKIPPED_COMPLETE`，且 validation 欄皆為 `valid`。

列出最近一次摘要：

```bash
SUMMARY_PATH="$(find "$LBT_SCRATCH_ROOT/b-hsinchu-expanded-m2-h1/logs" \
  -mindepth 2 -maxdepth 2 -name summary.tsv -print | sort | tail -n 1)"
test -n "$SUMMARY_PATH" && column -t -s $'\t' "$SUMMARY_PATH"
```

上述完成只代表 **B 區固定工程 pilot 的程式與資料完整性通過**，不是五站正式研究結果、
科學驗證、收斂證明或來源機率。

### 6.2 失敗或 SSH 中斷

先確認舊程序是否仍在執行：

```bash
pgrep -af 'run_server_matrix.sh|lbt run-shard'
```

- 若仍有程序：回到原 tmux 查看，不要啟動第二份 `RUN`。
- 若已無程序：保留 outputs、checkpoint、scratch 與 logs，不要刪除或手動改狀態。若原
  `lbt-b-pilot` tmux session 仍存在，就重新連入該 session，不要再建立同名 session。
- 重新取得同一份核准值，重做第 3、4 節，再依第 5 節回到原 tmux 或建立新的 tmux。既有 B 區
  runner 會跳過已完整通過的分片；從未開始的 `PLANNED` 分片直接執行，`RUNNING`、`PAUSED`、
  `FAILED` 分片才會以原設定、原續跑檔根目錄與 `--resume` 恢復。
- 若再次失敗：將畫面錯誤及最近一次 `summary.tsv`、`total.log` 的路徑交給執行負責人；不要換 seed、
  改 config 或從頭覆寫原 run。

## 7. 進階：建立新的 pilot 執行入口

本節只供執行包維護者使用。一般操作者執行既有 B 區試跑時不需要閱讀。

新的 SERVER 工作目前沒有通用 tracked runner；必須先建立會呼叫
`scripts/validate_server_storage.py` 的受追蹤入口，不能直接用下列命令繞過第 4 節的儲存檢查。
維護者必須先準備：

- 已生成且已驗證的 pilot config；不可使用含未決值的 example config。
- 同一 config 產生的 preflight JSON；它不是 input artifact 目錄。
- 新的 `RUN_ID`、NFS 輸出上層目錄，以及整個 run 固定不變的續跑檔根目錄。

完成上述條件後，建立工作目錄：

```bash
export CONFIG="<已生成且已驗證的pilot config絕對路徑>"
export INVENTORY="<同一config產生的preflight JSON絕對路徑>"
export RUN_ID="<新的run ID>"
export RUN_PARENT="$LBT_OUTPUT_ROOT/runs"
export WORKSPACE="$RUN_PARENT/$RUN_ID"

uv run lbt run-create \
  --config "$CONFIG" \
  --input-inventory "$INVENTORY" \
  --destination "$RUN_PARENT" \
  --run-id "$RUN_ID" \
  --run-kind pilot \
  --experiment-case finite_depth_stokes \
  --pilot-scenarios-per-stratum 1 \
  --project-root "$LBT_PROJECT_ROOT"
```

成功時 stdout JSON 會有 `"valid": true`，但此時只建立 `PLANNED` 工作目錄，尚未完成運算。
從計畫檔列出真正的分片 ID，不可自行編號：

```bash
jq -r '.shards[].shard_id' "$WORKSPACE/run_plan.json"
```

將列出的每個 ID 各放入一個 `--shard-id`，再依同一順序執行：

```bash
uv run lbt run-worker "$WORKSPACE" \
  --config "$CONFIG" \
  --shard-id "<第一個shard ID>" \
  --shard-id "<第二個shard ID；其餘照樣增加>" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT"
```

若既有分片是 `RUNNING`、`PAUSED` 或 `FAILED`，同一命令才加 `--resume`；初次 `PLANNED` 不加。
`run-worker` 遇到安全暫停也可能退出 0，因此最後仍須執行完整驗證：

```bash
uv run lbt run-reconcile "$WORKSPACE" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT"

uv run lbt validate-run "$WORKSPACE" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT" \
  --require-complete

uv run lbt benchmark-report "$WORKSPACE" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT"
```

唯一完成判定是 `validate-run` 退出 0、`valid=true`、`summary.run_lifecycle=COMPLETE`，且
`summary.completed_shard_count` 等於 `summary.shard_count`。`benchmark-report` 只提供工程量測；
分片時間加總與分片記憶體最大值不能當成整台 SERVER 的平行效能。

## 8. 為什麼目前不能執行五站正式批次

現行 A 區政策保留 v3 forcing 範圍、貢寮與龜山島各 20 km local domain、各 12.5 km receptor
core，且不納入 35 km sensitivity。A 區不再等待共同兩格 forcing margin；正式設定改以
`runtime_stage_fail_closed_no_expansion_v1` 在每個 RK4 階段驗證實際 OCM native／NWW
analysis 支援，無支援就保存 `data_gap`／對應品質狀態並停止，不補值或切成 current-only。
repository 現已提供可交給 horizon-suite 的 concrete source template
`configs/lagrangian_backtracking.formal_h30_h60_h90_m10.yaml`；它仍不是 approved release，
不要把 `lagrangian_backtracking.example.yaml` 改名後執行 formal，也不要把 A 區 1 天／20
情境或 B 區 pilot 當成正式替代品。資料與正式 gate 全部解鎖後，H30/H60/H90 只建立一次共同
輸入並產生六份 release：

```bash
export FORMAL_CONFIG_TEMPLATE="$LBT_PROJECT_ROOT/configs/lagrangian_backtracking.formal_h30_h60_h90_m10.yaml"
uv run lbt horizon-suite-create \
  --config-template "$FORMAL_CONFIG_TEMPLATE" \
  --backtrack-days 30 60 90 \
  --destination "$LBT_OUTPUT_ROOT/horizon-suites/h30-h60-h90-m10-v1" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --formal-release
```

正式執行至少要先具備：已核准且可重建的完整輸入成果、正式設定、逐階段 forcing 支援契約、
完整情境與 seed 綁定、checkpoint／隨機數延續驗證、效能證據與科學驗證。缺一項就停止。
完整輸入條件見[輸入衍生與發布契約](14_input_derivation_and_release_contract.md)，目前完成狀態見
[實作狀態](../implementation_status.md)。

### 正式 gate 開放後的完整母體平行執行流程

`run-formal-parallel` 是正式母體的通用執行介面，不代表目前五站正式 gate 已通過。只有第 8 節
列出的資料、設定、邊界與科學驗證條件全部解鎖，且已由核准部署建立 immutable formal workspace，
才使用本流程。worker 數由本次執行負責人明示；run plan 原有的情境、seed、M、回溯期與
checkpoint cadence 一律不由此命令改寫。

#### 8.1 核對正式 workspace 與部署

先使用已通過 formal release 驗證的 config、input inventory、study matrix 與 experiment case。
部署 checkout 必須與 run plan 綁定的 commit 完全一致且 clean；不得在工作樹有未提交變更時啟動。
以下建置命令只在核准 workspace 尚不存在時執行一次；`RUN_ID` 必須是本次新 ID，不可覆蓋舊 run：

```bash
export FORMAL_CONFIG="<已驗收正式 config 絕對路徑>"
export FORMAL_INPUT_INVENTORY="<同一 config 對應且已驗證的 inventory JSON 絕對路徑>"
export FORMAL_EXPERIMENT_CASE="<核准的 experiment case registry 值>"
export RUN_ID="<本次唯一 formal run ID>"
export WORKSPACE="$LBT_OUTPUT_ROOT/runs/$RUN_ID"

test ! -e "$WORKSPACE" || { echo "workspace 已存在；停止，不覆寫"; exit 2; }
uv run lbt run-create \
  --config "$FORMAL_CONFIG" \
  --input-inventory "$FORMAL_INPUT_INVENTORY" \
  --destination "$LBT_OUTPUT_ROOT/runs" \
  --run-id "$RUN_ID" \
  --run-kind formal \
  --experiment-case "$FORMAL_EXPERIMENT_CASE" \
  --project-root "$LBT_PROJECT_ROOT"

uv run lbt validate-run "$WORKSPACE" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT"
```

建立後應先確認 `run-create` 及唯讀 `validate-run` 均為 `valid=true`，且 summary 的 `run_kind`
是 `formal`、shard 數等於 run plan，不要手工編輯 plan／progress。formal initializer 還須成功
通過該版本的輸入 manifest、完整 inventory、A 區逐階段封閉失敗空間契約、正式 provenance 與設定 gate；
僅有 JSON 存在或 `run-create` 退出 0 不等於資料已被科學驗收。

#### 8.2 重新執行 SERVER 儲存 gate

結果、checkpoint、scratch、UV／Matplotlib／XDG cache 與 temporary 都應在 `/data/LBT` 同一 NFS
mount。先把執行環境的套件與暫存路徑導到已核准根目錄，並產生本次儲存快照；不要把
`NUMBA_CACHE_DIR`、Python bytecode、工作 log 或 temporary 放進 `/home`：

```bash
export UV_CACHE_DIR="$LBT_UV_CACHE_ROOT"
export MPLCONFIGDIR="$LBT_MPL_CACHE_ROOT"
export XDG_CACHE_HOME="$LBT_XDG_CACHE_ROOT"
export TMPDIR="$LBT_TMP_ROOT"
export LBT_STORAGE_GATE_JSON="$LBT_SCRATCH_ROOT/formal-parallel-storage-gate-$RUN_ID.json"
export LBT_PARALLEL_LOG_ROOT="$LBT_SCRATCH_ROOT/formal-parallel/logs"
export LBT_PARALLEL_CONSOLE_LOG="$LBT_SCRATCH_ROOT/formal-parallel/console-$RUN_ID.log"

uv run python scripts/validate_server_storage.py \
  --project-root "$LBT_PROJECT_ROOT" \
  --project-venv "$LBT_PROJECT_ROOT/.venv" \
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
  --snapshot-output "$LBT_STORAGE_GATE_JSON"
```

只有命令退出 0 且 JSON `gate_status=PASS` 時才繼續。接著在已通過 gate 的 scratch 下建立
既有 log 根目錄；執行器每次都會在此目錄建新 session，絕不覆寫舊 session：

```bash
mkdir -p "$LBT_PARALLEL_LOG_ROOT"
```

#### 8.3 （僅使用 Numba backend 時）確認 JIT 可編譯

目前 Numba dispatcher 全部使用 `cache=False`，因此不會建立 `.nbc`／`.nbi` 磁碟快取，也不能宣稱
下一個 Python process 能沿用這次編譯。正式執行時，每個長壽命 worker 會在自身程序內呼叫一次
`accelerated.warmup_numba_backend()`，編譯結果留在該 worker 記憶體並供後續 `run-worker` 重用。
本節 `--warmup-only` 是可選的部署檢查：用獨立小程序確認 kernel 可編譯，但不啟動粒子、也不
替正式 worker 預熱。若要執行，必須再次通過相同 run／provenance／storage gate，並明示
`NUMBA_CACHE_DIR` 為 scratch 下路徑；目前雖不寫磁碟，仍不允許使用 `/home` 或未驗證 temporary。

```bash
export LBT_NUMBA_CACHE_DIR="$LBT_SCRATCH_ROOT/formal-parallel/numba/$RUN_ID"
export FORMAL_WORKER_COUNT="<本次核准的固定 worker 數，正整數且不大於 shard 數>"

uv run python scripts/run_formal_parallel.py \
  "$WORKSPACE" \
  --config "$FORMAL_CONFIG" \
  --project-root "$LBT_PROJECT_ROOT" \
  --worker-count "$FORMAL_WORKER_COUNT" \
  --scratch-root "$LBT_SCRATCH_ROOT" \
  --log-root "$LBT_PARALLEL_LOG_ROOT" \
  --storage-gate-evidence "$LBT_STORAGE_GATE_JSON" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --numba-cache-dir "$LBT_NUMBA_CACHE_DIR" \
  --cpu-affinity auto \
  --warmup-only
```

`WARMUP_CHECKED` 只證明這個獨立檢查程序成功，不代表任何 shard 或 run 完成，也不會保存可供
其他 process 重用的編譯快取。記錄它輸出的 `log_session` 與 summary；若檢查失敗，保留
`numba-warmup.log`。正式 worker 仍會在各自程序內再次預熱一次。

#### 8.4 執行固定 worker 的正式完整母體

沿用同一 workspace、設定、roots、worker count、storage gate 與（如適用）Numba cache。runner
會在啟動子程序前以 `findmnt` 核對本次 `scratch_root` 的 NFS source token 與 PASS gate 相同；
若掛載來源已改變即停止，不能沿用舊快照。`auto` 在
同一個最深 resolved target 若同時回報 autofs 與非-autofs，runner 只採用唯一的非-autofs
`(fstype, source)`；相同列可去重，不同非-autofs identity 或多個最深 target 直接停止。只有
autofs 時保留實際 `autofs` 結果，交由後續 NFS gate 拒絕，不能把包裝層誤報為 NFS。
Linux 可用時為每個長壽命 worker 固定分配允許的 CPU；在其他平台或 cpuset API 不可用時安全不綁定，
並在 worker summary 記錄 fallback。worker group 按 run plan scenario 順序保持連續，以 plan 中的
region／arrival month 與表格已有 site／flow-domain 欄位摘要 locality，不會猜欄位，也不改物理設定：

```bash
PARALLEL_CACHE_ARGS=()
if [[ -n "${LBT_NUMBA_CACHE_DIR:-}" ]]; then
  PARALLEL_CACHE_ARGS=(--numba-cache-dir "$LBT_NUMBA_CACHE_DIR")
fi
set -o pipefail
uv run python scripts/run_formal_parallel.py \
  "$WORKSPACE" \
  --config "$FORMAL_CONFIG" \
  --project-root "$LBT_PROJECT_ROOT" \
  --worker-count "$FORMAL_WORKER_COUNT" \
  --scratch-root "$LBT_SCRATCH_ROOT" \
  --log-root "$LBT_PARALLEL_LOG_ROOT" \
  --storage-gate-evidence "$LBT_STORAGE_GATE_JSON" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --cpu-affinity auto \
  "${PARALLEL_CACHE_ARGS[@]}" \
  2>&1 | tee "$LBT_PARALLEL_CONSOLE_LOG"
RUNNER_EXIT=${PIPESTATUS[0]}
test "$RUNNER_EXIT" -eq 0
```

若不用 Numba backend，應先 `unset LBT_NUMBA_CACHE_DIR`，上方條件展開就不會傳 cache 參數；若
`auto` 不符合核准的排程政策，可改成 `--cpu-affinity 0,1,2` 等實際 cpuset 內 ID，不能憑機器核心
總數猜測。需要恢復部分 run 時，僅在下次明確加 `--resume`；不能更換 run ID、seed、config、輸入或
checkpoint root。worker count 只影響執行分組，不會改 run identity。

#### 8.5 完成條件與失敗處置

取得 CLI 回報的 log session 並核對完整摘要與正式 validator：

```bash
PARALLEL_SESSION="$(jq -r '.log_session // empty' "$LBT_PARALLEL_CONSOLE_LOG")"
test -n "$PARALLEL_SESSION" || { echo "沒有執行摘要；保留 console log 並停止"; exit 2; }
PARALLEL_SUMMARY="$LBT_PARALLEL_LOG_ROOT/$PARALLEL_SESSION/summary.json"
test "$(jq -r '.status' "$PARALLEL_SUMMARY")" = COMPLETE
jq -e '.valid == true and (.worker_records | all(.[]; .exit_code == 0 and .status == "EXITED"))' \
  "$PARALLEL_SUMMARY"
uv run lbt validate-run "$WORKSPACE" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT" \
  --require-complete
```

只有 runner shell 退出碼 0、machine summary 的 `status=COMPLETE`、每個 worker exit code 為 0、
完整 shard lifecycle 驗證全數 `COMPLETE`，且獨立 validator 回傳 `valid=true`、
`run_lifecycle=COMPLETE`、completed shard count 等於 shard count，才可記為正式 run 完成。
summary 的整機 elapsed、分組、CPU affinity 與每 worker log 是操作／資源資訊，不是新舊版本倍率
比較，也不能替代輸入與科學驗證。

任何 preflight 失敗都不得啟動 child；依 JSON 錯誤修正核准路徑、clean provenance 或 gate 後，重新
產生本次 storage snapshot，再從 8.2 重做。若有 child exit 非零、`INCOMPLETE`、SIGINT／SIGTERM，
或 SSH／程序中斷：不要把部分成果標成完成，也不要立刻啟動第二批。先確認程序已停止並保存完整
log session、console log、summary、trajectory 與 checkpoint；接著執行 `run-reconcile` 和不帶
`--require-complete` 的唯讀 validator，交由 run owner 檢查後，再以同 run identity 明示 `--resume`
恢復。若 checkpoint、RNG continuation、input binding 或 provenance 不符，停止並升級處理，不得從 seed
靜默重算或自行重建不同 run。

## 9. 操作時不可跨越的界線

| 發現 | 處理方式 |
|---|---|
| 任一結果、續跑檔、暫存、日誌或快取要寫到 `/home` | 停止；所有執行資料都必須在 `/data/LBT` |
| 實際程式目錄、Git 版本或未提交變更狀態與核准值不同 | 停止；不得自動切換或清理 |
| 儲存檢查、PREPARE 或完整驗證未通過 | 停止；不得跳過或把警告視為完成 |
| 已有同一矩陣程序或鎖定檔 | 回到原程序，不啟動第二份 |
| 執行中斷或失敗 | 保留續跑檔與日誌，以同一設定明示恢復 |
| 想刪除舊資料 | 先另做程序、執行包、續跑檔與發布證據盤點；本手冊不授權刪除 |

需要查個別命令參數時看 [CLI 參考](cli_reference.md)；需要部署新 Git 版本或同步資料時看
[Git 部署與資料同步手冊](git_deployment_and_data_sync.md)。
