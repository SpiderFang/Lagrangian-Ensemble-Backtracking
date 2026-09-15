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
core，且不納入 35 km sensitivity。但目前程式仍明確阻擋 v3／20 km 共同 forcing 邊界驗證，
repository 也沒有可直接使用的正式 release config。因此不要把 example config 改名後執行 formal，
也不要把 B 區 pilot 結果當成正式替代品。

正式執行至少要先具備：已核准且可重建的輸入成果、正式設定、實際共同 forcing-margin 證據、
完整情境與 seed 綁定、checkpoint／隨機數延續驗證、效能證據與科學驗證。缺一項就停止。
完整輸入條件見[輸入衍生與發布契約](14_input_derivation_and_release_contract.md)，目前完成狀態見
[實作狀態](../implementation_status.md)。

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
