# 單站 30 天工程測速操作契約

本文件定義第一輪真 forcing 測速的可重建操作邊界。它只量測一個研究站點、單一到達
時刻、單一沉降代理及 `M=1` 的 5 個既有 near-bed 受體，目的是取得 prepare、單步成本、checkpoint
I/O 與可恢復性的證據，供五站正式工期估算使用。這個樣本不代表五站完整設計、成員收斂、
物性校準、條件式來源足跡或正式成果。

操作端必須先把程式 checkout、輸入、checkpoint、輸出與 log 固定在已通過 SERVER 儲存閘門
的 NFS 根目錄下；路徑以環境變數注入，文件不保存登入資訊或私人主機位址。`/home` 只放
乾淨程式與虛擬環境，測速的輸入、暫存、checkpoint、timings 與 log 放在 `$LBT_RESULT_ROOT`
的嚴格子目錄。

## 候選測速範圍

| 項目 | 設定 | 工程意義與限制 |
|---|---|---|
| 站點 | `guishan`（A 區） | 保留 A 區 v3、局部 20 km、受體核心 12.5 km；不包含 35 km 或南擴 |
| 到達時刻 | `2024-08-02T00:00:00Z` | 往前至 `2024-07-03T00:00:00Z`，inclusive 721 個逐時節點；仍須由 adapter 重驗實際產品 |
| 材質代理 | `oca_fishinggear_open_mesh_bundle` | `settling_velocity_mps=-0.002` 的既有暫定代理，不是漁具類別校準值 |
| 速度案例 | `finite_depth_stokes` | 沿用既有 pilot 的有限水深 Stokes 工程設定，不新增物理候選 |
| 基礎情境 | 5 個既有 near-bed receptors × 1 arrival × 1 material | 從既有 5 水平位置 × 4 垂向層位清單取 near-bed，合計 5 個；不是新正式 20 XY 受體設計 |
| 成員 | `M=1` | 只供工程耗時，不能解讀為 member convergence |
| 分片 | 1 片，共 5 情境 | 只啟動 run plan 產生的唯一 shard；本輪不做多 worker 或整機 capacity benchmark |

表中的到達時刻與回溯起點是本輪工程候選；若改用其他候選，必須同步更新候選設定、命令
與輸入 manifest，並由每次 SERVER 預檢重新驗證實際產品。預檢應確認 source 動態月份標記
與 arrival UTC 的月份一致；若不一致，應保留該次 preflight artifact 作拒絕證據並停止，
不得將其重用為另一候選的輸入。這個 engineering-only 窗口不改寫 formal arrival selector
或 50,000 情境契約，且不把候選日期本身視為資料支援或科學結果證明。

## 啟動前環境

下列變數由 operator 依當次已核對的部署填入；它們是命令範例的介面，不是預設值。

```bash
export LBT_PROJECT_ROOT=/path/to/clean/LBT-checkout
export LBT_SOURCE_PYTHON="$LBT_PROJECT_ROOT/.venv/bin/python"
export LBT_ADAPTER="$LBT_PROJECT_ROOT/scripts/run_engineering_window.py"
export LBT_RESULT_ROOT=/path/to/accepted/nfs/LBT-root
export LBT_PACKAGE_ROOT="$LBT_RESULT_ROOT/packages/five-site-h30-benchmark-v1"
export SOURCE_CONFIG=/path/to/existing/source-config.yaml
export SOURCE_INPUT_DIRECTORY=/path/to/accepted/derived-v3
export LBT_OCM_NATIVE_ROOT=/path/to/accepted/ocm_native
export LBT_NWW_ANALYSIS_ROOT=/path/to/accepted/nww3_analysis
export LBT_ENGINEERING_CONFIG="$LBT_PACKAGE_ROOT/guishan-h30-r1/engineering-window.yaml"
export LBT_ARTIFACT_ROOT="$LBT_PACKAGE_ROOT/guishan-h30-r1/artifact"
export LBT_RUN_DESTINATION="$LBT_RESULT_ROOT/outputs/five-site-h30-benchmark-v1"
export LBT_CHECKPOINT_ROOT="$LBT_RESULT_ROOT/checkpoints/five-site-h30-benchmark-v1/guishan-h30-r1"
export LBT_MONITOR="$LBT_PACKAGE_ROOT/monitor_single_station.py"
export LBT_LOG_ROOT="$LBT_PACKAGE_ROOT/guishan-h30-r1/logs"
```

`LBT_ENGINEERING_CONFIG` 是本輪部署與候選證據的現場記錄；adapter 的核心輸入分別由
`SOURCE_CONFIG` 與 `SOURCE_INPUT_DIRECTORY` 傳入，因此 source config／derived
input 可以位於既有 package，而 artifact、log、workspace 與 checkpoint 仍寫入本輪 NFS
package／結果目錄。

啟動前先唯讀確認 `$LBT_PROJECT_ROOT` 的 commit、tree、`dirty=false`、來源 execution
manifest 與 input artifact index 的 SHA-256，以及 `$LBT_RUN_DESTINATION` 和
`$LBT_CHECKPOINT_ROOT` 的 mount identity。確認輸入目錄內的 geometry、material、forcing
inventory、actual-z 初始條件及 NWW3 完整逐時 manifest 與設定檔中的 SHA 一致；H30 候選
證據的 OCM native／NWW3 兩份 721 節點紀錄也必須在預檢完成後存在且 `missing_nodes=0`；
預檢未完成時不得填寫通過值。任一指紋不符
即停止，不以新檔覆蓋舊來源。

先閱讀 adapter 的實際介面，因 engineering-only CLI 是本輪新增的薄層入口：

```bash
"$LBT_SOURCE_PYTHON" "$LBT_ADAPTER" --help
```

adapter 應從既有已驗收 static geometry／material／forcing binding 建立本輪窗口，並在
prepare 產出的 artifact 中保存 station、arrival UTC、window start、721 節點、來源 hash、
scenario／seed identity 及 `engineering_pilot` 標記。不得讀取 raw NetCDF、沿用舊 24 小時
arrival ID、把舊 7 日 gap-safe manifest 冒充 H30，或把工程子集標為 formal。

## 分階段啟動順序

### 1. Prepare

prepare 只做輸入窗口、5 個 near-bed 受體／actual-z、5 個 scenario identity 與來源綁定的
artifact 建立和驗證，不啟動粒子追蹤，也不建立 run workspace 或 `run_plan.json`。artifact
目錄必須是 NFS 上尚未存在的專用路徑；已存在的 artifact 或不一致的 manifest 應由 adapter
拒絕。`run_plan.json` 會在下一階段 `run` 初始化 workspace 時由既有 controller 建立。

用外部監測器包住 prepare。`--` 後的內容是 adapter 的原樣 argv；以下是本 CLI 與
核心 `prepare_engineering_window` 的實際參數：

```bash
mkdir -p "$LBT_LOG_ROOT"
"$LBT_SOURCE_PYTHON" "$LBT_MONITOR" \
  --run-root "$LBT_LOG_ROOT" \
  --run-id guishan-h30-r1 \
  --phase prepare \
  --timings-json "$LBT_LOG_ROOT/timings-prepare.json" \
  --run-log "$LBT_LOG_ROOT/prepare.log" \
  --mountstats /proc/self/mountstats \
  --budget-seconds 14400 \
  --cwd "$LBT_PROJECT_ROOT" \
  -- "$LBT_SOURCE_PYTHON" "$LBT_ADAPTER" prepare \
  --source-config "$SOURCE_CONFIG" \
  --source-input-directory "$SOURCE_INPUT_DIRECTORY" \
  --study-site-id guishan \
  --arrival-utc 2024-08-02T00:00:00Z \
  --backtrack-days 30 \
  --material-id oca_fishinggear_open_mesh_bundle \
  --destination "$LBT_ARTIFACT_ROOT" \
  --ocm-native-root "$LBT_OCM_NATIVE_ROOT" \
  --nww-analysis-root "$LBT_NWW_ANALYSIS_ROOT" \
  --project-root "$LBT_PROJECT_ROOT" \
  --vertical-id near_bed \
  --shard-scenario-count 5
```

prepare 成功只代表 engineering artifact 的輸入、時間支援與來源 binding 可供下一階段使用。
應先檢查 `engineering_manifest.json`、`config.yaml`、`input_inventory.json`、scenario
數量與 source hash；再讓 `run` 建立 workspace／run plan。prepare 失敗時不應建立可被誤認為
完成的 run。

### 2. 唯一分片的真 forcing 與 sweep budget

第一輪由 `run` 以 artifact 初始化 workspace／`run_plan.json`，再執行唯一宣告的 shard
（應含 5 個情境、5 個粒子）。因為本輪 plan 只有一片，第一次呼叫可省略 `--shard-id`，
由核心選取 plan 中全部的一片；不要在 plan 建立前手填 shard 名稱。以
`--sweep-budget 1000` 讓核心 controller 在 checkpoint 邊界停止或標記未完成。這是確認
30 日真 forcing、adaptive step、輸出與 checkpoint 單步成本的最小工作單位；不是把 1000
當成情境數，也不是執行 1000 個粒子。

```bash
"$LBT_SOURCE_PYTHON" "$LBT_MONITOR" \
  --run-root "$LBT_LOG_ROOT" \
  --run-id guishan-h30-r1 \
  --phase run-initial \
  --timings-json "$LBT_LOG_ROOT/timings-run-initial.json" \
  --run-log "$LBT_LOG_ROOT/run-initial.log" \
  --mountstats /proc/self/mountstats \
  --budget-seconds 14400 \
  --cwd "$LBT_PROJECT_ROOT" \
  -- "$LBT_SOURCE_PYTHON" "$LBT_ADAPTER" run \
  --artifact "$LBT_ARTIFACT_ROOT" \
  --destination "$LBT_RUN_DESTINATION" \
  --run-id guishan-h30-r1 \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT" \
  --ocm-native-root "$LBT_OCM_NATIVE_ROOT" \
  --nww-analysis-root "$LBT_NWW_ANALYSIS_ROOT" \
  --project-root "$LBT_PROJECT_ROOT" \
  --experiment-case-id finite_depth_stokes \
  --sweep-budget 1000
```

實際 adapter 參數須包含同一份 artifact、run workspace parent、run ID、OCM native root、
NWW3 analysis root 與 external checkpoint root；這些值由上述 CLI 傳入，不在監測器內另造。
`run` 會在 `$LBT_RUN_DESTINATION/guishan-h30-r1/` 建立 workspace；不要預先 `mkdir` 該
final workspace，避免破壞 initializer 的 exclusive 建立檢查。若現場已有其他使用者工作，
先記錄干擾快照，但不停止他們的程序；第一輪以單程序量測，不能宣稱整機 capacity benchmark。

### 3. 同片 resume 與後續分片

若唯一分片在 sweep budget 形成合法 checkpoint，必須以完全相同的 config、input binding、
seed、shard ID 與 checkpoint root resume。resume 另用唯一 phase／timings 檔，避免無鎖
追加造成 lost update：

```bash
export SHARD_ID="$(LBT_RUN_DESTINATION="$LBT_RUN_DESTINATION" "$LBT_SOURCE_PYTHON" -c 'import json, os; from pathlib import Path; p=Path(os.environ["LBT_RUN_DESTINATION"]) / "guishan-h30-r1" / "run_plan.json"; rows=json.loads(p.read_text())["shards"]; assert len(rows) == 1; print(rows[0]["shard_id"])')"
"$LBT_SOURCE_PYTHON" "$LBT_MONITOR" \
  --run-root "$LBT_LOG_ROOT" \
  --run-id guishan-h30-r1 \
  --phase run-resume \
  --timings-json "$LBT_LOG_ROOT/timings-run-resume.json" \
  --run-log "$LBT_LOG_ROOT/run-resume.log" \
  --mountstats /proc/self/mountstats \
  --budget-seconds 86400 \
  --cwd "$LBT_PROJECT_ROOT" \
  -- "$LBT_SOURCE_PYTHON" "$LBT_ADAPTER" run \
  --artifact "$LBT_ARTIFACT_ROOT" \
  --destination "$LBT_RUN_DESTINATION" \
  --run-id guishan-h30-r1 \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT" \
  --ocm-native-root "$LBT_OCM_NATIVE_ROOT" \
  --nww-analysis-root "$LBT_NWW_ANALYSIS_ROOT" \
  --project-root "$LBT_PROJECT_ROOT" \
  --experiment-case-id finite_depth_stokes \
  --shard-id "$SHARD_ID" --resume
```

上面的 `$SHARD_ID` 是在第一次 `run` 已建立 `run_plan.json` 後讀出的 immutable ID；若
plan 未建立或不是單片，命令應停止並另行依 plan 選擇，不能自行造 ID。確認唯一 shard
通過 controller 的 shard／checkpoint validator 後，再依需求另立
20 個受體的工程 scope。若未來同一 station 擴增多片，必須每片使用自己的 log 與 timings
JSON；平行執行的整批 elapsed 由 operator 另以外層 wall-clock 記錄，不能把 shard wall
time 相加。每片仍須留下自己的 process tree、I/O、checkpoint、exit code 與
`validate-run` 證據。

## 監測資料的解讀

`monitor_single_station.py` 位於 ignored SERVER package，接受任意外部 argv，並為每個
phase 產生一份不可部分更新的 timings JSON。它在自己的 process group 中啟動子程序，
不會送出終止訊號。時間長度用 `time.monotonic()` 計算；UTC timestamp 只作人工追溯。

每份 timings 至少包含：

- `elapsed_seconds_monotonic`、PID／PGID、exit code、budget reached 與 phase status。
- process tree 的目前／峰值 RSS。RSS 是每次取樣時程序樹的總和再取最大值，收尾不重複
  累加；程序在兩次取樣間退出時，峰值可能低估。
- CPU ticks／秒與 proc I/O 差值：`rchar`、`wchar`、`read_bytes`、`write_bytes` 分開
  保存。前三者不可混稱 NFS 實際 bytes；`read_bytes`／`write_bytes` 也不是 NFS
  server-side 網路 accounting。
- 可選 `/proc/self/mountstats` 的 raw counter delta。這只能作 kernel mountstats
  觀察；若不可用或掛載於量測期間改變，記為 unavailable，不補零。
- phase 開始／結束時的其他程序數、CPU／RSS 總量與前十名摘要，作為共享 SERVER 資源
  干擾紀錄；它不提供精確的 CPU 歸因。

由於 `/proc` 是定期取樣，monitor 的 CPU／I/O 累積可能漏掉尚未被觀察到就退出的子程序；
不能把它當成完整 CPU 精確總量。SERVER 同時用 `/usr/bin/time -v` 保存外部命令的
elapsed、user／system CPU 與最大 RSS 作為核對；兩份資料的量測語意必須分開記錄，不能
把 RSS 或平行 shard elapsed 再次相加。

到達 monitor 的外部預算時，腳本只寫通知並繼續等待；`status` 會標示
`completed_after_budget` 或 `failed_after_budget`。它不執行 `kill -9`，不代替核心
controller 建立 checkpoint，也不把未完成 run 標為成功。是否暫停、resume、重試與
`validate-run --require-complete` 由核心 controller／operator 依 run plan 判定。

## 完成與停止條件

本輪工程測速可報告「可用耗時證據」需同時具備：prepare 成功的窗口／source／scenario
manifest；唯一 run plan 與分片 identity；該片 checkpoint／trajectory 的 controller
狀態；`exit_code`、timings、`/usr/bin/time -v`、干擾與 NFS mountstats（若可用）；以及
`run-reconcile`、`validate-run` 的可讀輸出。單一 shard 退出 0 不足以宣稱完整正式站點
完成，單一 station 也不足以外推五站工期。

遇到來源 hash、721 節點、actual-z、NWW3／OCM 支援、schema、checkpoint identity、
數值失敗或 NFS 寫入錯誤時，停止該片並保留 log／partial 證據；不得以最近值、零值、舊
24 小時資料、未校準物性或終止粒子數替代缺失資料。若現場需要改變日期、M、dt、物性、
情境數或並行度，另立新的 candidate scope／run ID，保留本輪設定與結果的不可變身分。

本輪結果最後只用於比較 prepare 與 tracking 的實際 elapsed、CPU、RSS、proc I/O、
checkpoint bytes、forcing cache 狀態與可恢復性，作為正式 30 日回溯工期模型的工程輸入。
它不解除 formal accepted-input、M convergence、科學驗證、聚合、圖表或 10/30 交付的
其他 gate，也不產生正式條件式來源足跡。
