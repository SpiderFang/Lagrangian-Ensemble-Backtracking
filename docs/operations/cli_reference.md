# CLI 與執行介面參考

> **閱讀提示**
> - 文件類型：目前 CLI 命令、參數與執行生命週期參考。
> - 它回答：每個命令接受什麼輸入、產生什麼輸出，以及哪些安全限制不可繞過。
> - 建議先讀：[實作狀態](../implementation_status.md)，再讀[SERVER 手冊](06_server_runbook_plan.md)。

本文件只列目前 `src/lagrangian_backtracking/cli.py` 實際註冊的命令與必要參數。所有
`<...>` 都是操作端必須替換的 placeholder；文件不提供私有 SERVER 絕對路徑、憑證或
未經確認的 arrival。輸入與輸出目錄應使用不同的全新目的地；任何 immutable writer
拒絕覆寫既有 final 目錄。

## 介面總覽

整合入口是 `uv run lbt <command>`。目前註冊的命令如下；`pilot-calibration-*` 是較長的
可讀 alias，與對應的 `pilot-calibrate-*` 共用 parser。

| 類別 | 實際命令 |
|---|---|
| 設定／輸入 | `config-check`、`preflight`、`inputs-build`、`inputs-validate`、`release-config-create`、`release-config-validate` |
| pilot | `pilot-calibrate`、`pilot-calibrate-validate`、`pilot-calibration-build`、`pilot-calibration-validate`、`pilot-config-create`、`pilot-config-validate` |
| 工程驗證 | `behavior-manifest`、`synthetic-smoke`、`validate-shard`、`code-provenance`、`validate-run`、`benchmark-report`、`pilot-matrix-validate` |
| run lifecycle | `run-create`、`run-shard`、`run-reconcile` |
| 聚合／報告 | `aggregate-spec-create`、`aggregate-build`、`aggregate-validate`、`report-spec-create`、`report-validate`、`source-pathway-build`、`source-pathway-validate` |

`report-build` 尚未註冊；不要用不存在的命令替代目前的 report preflight、aggregate 或
release validator。

## 共通準備與安全邊界

```text
LBT_PROJECT_ROOT=/path/to/Lagrangian-Ensemble-Backtracking
LBT_RESULT_NFS_ROOT=/data/LBT
LBT_EXECUTION_PACKAGE_ROOT=/data/LBT/execution-packages/<package-id>
LBT_OUTPUT_ROOT=/path/to/new/output-parent
LBT_SCRATCH_ROOT=/path/to/existing/scratch-parent
LBT_CHECKPOINT_ROOT=/path/to/checkpoints
LBT_UV_CACHE_ROOT=/data/LBT/cache/uv
LBT_MPL_CACHE_ROOT=/data/LBT/cache/matplotlib
LBT_XDG_CACHE_ROOT=/data/LBT/cache/xdg
LBT_TMP_ROOT=/data/LBT/tmp
OCM_NATIVE_ROOT=/path/to/accepted/ocm_native
OCM_SURFACE_ROOT=/path/to/accepted/ocm_surface
NWW_ANALYSIS_ROOT=/path/to/accepted/nww3_analysis
```

正式 forcing 只能是已驗收的 OCM schema 3 `ocm_native`／`ocm_surface` 與 NWW3 schema 1
`nww3_analysis`。root 可由命令列明示；省略時只讀 config 指定的環境變數，不猜測目前
工作目錄或 SERVER 路徑。`LBT_SCRATCH_ROOT` 應已存在，以下 final child 必須尚不存在。

SERVER 的部署契約另要求以 `/data/LBT` 作為單一 `LBT_RESULT_NFS_ROOT`：執行 package、output、scratch、
checkpoint、UV／Matplotlib／XDG cache 與 temporary root 都必須是該 NFS mount 的既有嚴格
子目錄，且 output、scratch、checkpoint 不得互相包含。專案 checkout 與既有 `.venv` 可
留在 `/home`；正式執行必須由 tracked
`scripts/run_b_hsinchu_expanded_matrix.sh`（後續區域採同一 gate）先驗證路徑、NFS mount、
剩餘空間、寫入、原子改名與跨程序鎖，再進入 execution package。一般可攜 CLI 不會自行
假設 SERVER mount，因此不得繞過這個入口直接啟動 SERVER batch。

## 設定與輸入衍生

### 設定與月份 preflight

```bash
uv run lbt config-check \
  --config configs/lagrangian_backtracking.example.yaml

uv run lbt preflight \
  --config "$FORMAL_CONFIG" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --output "$LBT_SCRATCH_ROOT/formal-input-inventory.json" \
  --formal-release
```

`preflight` 只讀月份 metadata／時間軸與已登錄資料契約；正式模式遇到 error 以非零狀態
停止。它不建立 forcing 副本，也不以 `trial_ready`、partial month、最近值或零值補齊。

### inputs artifact 與 release config

```bash
uv run lbt inputs-build \
  --config "$FORMAL_CONFIG" \
  --destination "$LBT_SCRATCH_ROOT/input-release-2024-2025" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --formal-release

uv run lbt inputs-validate "$LBT_SCRATCH_ROOT/input-release-2024-2025" \
  --config "$FORMAL_CONFIG" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --formal-release

uv run lbt release-config-create \
  --config-template "$FORMAL_CONFIG" \
  --input-directory "$LBT_SCRATCH_ROOT/input-release-2024-2025" \
  --output "$LBT_SCRATCH_ROOT/release-2024-2025.yaml" \
  --formal-release

uv run lbt release-config-validate "$LBT_SCRATCH_ROOT/release-2024-2025.yaml" \
  --input-directory "$LBT_SCRATCH_ROOT/input-release-2024-2025" \
  --formal-release
```

`inputs-build` 即使沒有 `--formal-release` 也固定 strict、fail-closed，禁止 synthetic
constant-field fallback。`release-config-create`、`release-config-validate` 只處理
immutable input binding；它們不啟動粒子運算。

四區 24 小時工程試跑可在 `inputs-build` 以版本化、pilot-only 的明示 UTC 入口替換既有
arrival。registry 固定使用 `2024-01-02T01:00:00Z`，回溯 24 小時並包含 25 個逐時節點。
A 區必須同時明示貢寮與龜山島；B／C／D 各自明示單站。新建流程使用共同 policy，既有
B 區 r5 成果的舊 policy 僅供唯讀相容驗證：

```bash
uv run lbt inputs-build \
  --config "$PILOT_CONFIG_TEMPLATE" \
  --destination "$LBT_SCRATCH_ROOT/hsinchu-2024-01-01-24h-inputs-v1" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --pilot-arrival-utc hsinchu=2024-01-02T01:00:00Z
```

A 區的入口改為：

```bash
uv run lbt inputs-build \
  --config "$PILOT_CONFIG_TEMPLATE" \
  --destination "$LBT_SCRATCH_ROOT/a-2024-01-01-24h-inputs-v1" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --pilot-arrival-utc gongliao=2024-01-02T01:00:00Z \
  --pilot-arrival-utc guishan=2024-01-02T01:00:00Z
```

其餘區域只把 `--pilot-arrival-utc` 的站點鍵換成 `houwan` 或 `lienchiang`。builder 會逐筆驗證
`2024-01-01T01:00:00Z` 至 `2024-01-02T01:00:00Z` inclusive 的 25 個 exact-hour 節點
是否同時存在於 OCM native、OCM surface、NWW3 analysis，並重做 NWW metric location 四角
static／dynamic 支援與 OCM native gap-safe gate；缺任何一小時即 fail closed，不使用
00Z 最近值、零值或跨缺口內插。輸出仍維持 250 arrivals／5,000 dynamic pairs，且
arrival metadata、gap manifest、provenance 與 artifact index 會記錄 pilot scope、原／替換
identity 及 1 日 horizon。這個選項不可與 `--formal-release` 同時使用；formal validator
也會以非正式 `pilot_explicit_window` label 拒絕升格為 48+2 正式 arrival。

## Pilot 與 synthetic

`pilot-calibrate`／`pilot-config-create` 的完整參數與 candidate gate 見
[SERVER 執行手冊](06_server_runbook_plan.md)；這些命令只接受已驗證輸入，輸出是 engineering
evidence，不是正式參數或成果。`behavior-manifest` 可建立十種全負沉降的材質／形狀代理
manifest；它不替代正式 source manifest。

### ABCD pilot 共同設定檢核

`pilot-matrix-validate` 是唯讀的跨區設定比較器。它至少接受兩個 run root，分別讀取其中的
`run_plan.json` 與 `normalized_config.json`，不讀 trajectory、forcing 或 checkpoint 大檔。
它要求 `run_kind`、experiment、M／seed、selection、shard／chunk／checkpoint、完整積分與
邊界設定、Stokes 與無效波政策、選用材質／沉降、execution scalar snapshot 以及程式／依賴
provenance 完全一致；研究站點、flow domain、arrival／scenario identity、輸入／幾何 hash／path
及區域校準的 Kh／Kz／Smagorinsky cap 才是明示允許的差異。

```bash
uv run lbt pilot-matrix-validate \
  "$A_RUN_ROOT" "$B_RUN_ROOT" "$C_RUN_ROOT" "$D_RUN_ROOT" \
  > "$LBT_SCRATCH_ROOT/abcd-first-pilot-matrix.json"
```

成功回傳 shell status `0`，不通過或輸入缺欄位、symbolic link、格式錯誤與超過大小上限回傳
`2`；stdout 是可保存的 canonical JSON。通過只表示設定可比較，不表示任何 run 已完成，
也不表示 forcing、trajectory、輸入產品或科學 gate 已驗收。ABCD 第一次試跑的實測診斷與
目前 A／B／C／D 設定差異見[四區試跑稽核](../results/15_four_region_first_pilot_audit.md)。

不需真資料的端到端 smoke：

```bash
uv run lbt synthetic-smoke --output "$LBT_SCRATCH_ROOT/synthetic-smoke-v1"
uv run lbt validate-shard "$LBT_SCRATCH_ROOT/synthetic-smoke-v1"
```

`synthetic-smoke-v1` 必須是尚不存在的 child；不要把已由 `mktemp -d` 建立的根目錄直接
當成 `--output`。輸出 metadata 會標示 `synthetic_smoke_not_scientific_result`，只能驗證
CLI、engine、trajectory I/O、Parquet、manifest 與 checksum。

### NFS preview／figure 的完成標記

`build_pilot_preview.py` 與 `build_pilot_coastline_preview.py` 只有在 caller 明示通過的
storage gate evidence 下才啟用 NFS `nfs_completion_marker_v1`。流程使用同父目錄 cooperative
lock、完整 staging inventory、逐檔 durable move，最後建立 `.complete`，並把 marker 綁定到
manifest SHA-256、程式 commit／tree／dirty diff 與九個儲存根目錄的 gate snapshot。`.complete`
只代表 preview／figure artifact 可由 reader 安全讀取，不是 run 完成旗標；run 完成仍以
`run_progress.json` 加上 `run-reconcile` 與 `validate-run --require-complete` 判定。沒有
`.complete` 的逐檔發布目錄不能當成 marker artifact，舊的 exclusive-directory-rename 目錄
只在未啟用 NFS gate 時保留相容性。

## Run workspace、shard 與外置 checkpoint

### 建立 workspace

```bash
uv run lbt run-create \
  --config "$PILOT_OR_FORMAL_CONFIG" \
  --input-inventory "$PILOT_OR_FORMAL_INVENTORY" \
  --destination "$LBT_OUTPUT_ROOT/runs" \
  --run-id "$RUN_ID" \
  --run-kind pilot \
  --experiment-case no_stokes \
  --pilot-scenarios-per-stratum 1
```

`--run-kind` 只能是 `pilot` 或 `formal`。分層 selector 只供 pilot sanity／benchmark，formal
禁止；formal 必須使用完整且已通過 gate 的 source selection。單站精確 pilot 若使用
`--pilot-study-site-id`、`--pilot-arrival-id`、`--pilot-material-id`，三者必須同時且各一次。

### 執行、暫停與恢復

```bash
uv run lbt run-shard "$LBT_OUTPUT_ROOT/runs/$RUN_ID" \
  --config "$PILOT_OR_FORMAL_CONFIG" \
  --shard-id "$SHARD_ID" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT" \
  --sweep-budget 10

uv run lbt run-shard "$LBT_OUTPUT_ROOT/runs/$RUN_ID" \
  --config "$PILOT_OR_FORMAL_CONFIG" \
  --shard-id "$SHARD_ID" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT" \
  --resume \
  --sweep-budget 10
```

同一 shard 的 resume 必須沿用原 config、seed、input binding 與 checkpoint root。external
checkpoint root 只在命令執行時傳入；run plan／progress 保存相對 token，不保存 SERVER 絕對
路徑。缺 generation、binding、checksum、particle order 或 RNG 不符時，controller 在建立
物理 request 前停止，不從 seed 靜默重算。

### reconcile 與唯讀驗證

```bash
uv run lbt run-reconcile "$LBT_OUTPUT_ROOT/runs/$RUN_ID" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT"

uv run lbt validate-run "$LBT_OUTPUT_ROOT/runs/$RUN_ID" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT" \
  --require-complete

uv run lbt benchmark-report "$LBT_OUTPUT_ROOT/runs/$RUN_ID" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT"
```

`run-reconcile` 不載入 forcing、不建立物理 request；`validate-run` 與 `benchmark-report`
是唯讀介面。benchmark 只報工程資源量測，不是科學結果。`RUNNING`、`PAUSED`、`FAILED`
若要繼續都必須明示 `--resume`；`COMPLETE` 不重跑。

## Aggregate 與 report

目前 report 前置流程的實際命令是 `aggregate-spec-create`、`aggregate-build`、
`aggregate-validate`、`report-spec-create` 與 `report-validate`。各 spec 的公尺制格網、
秒制 age 軸、KDE 帶寬與 bootstrap 參數必須由 caller 明示，不能從 fixture 或路徑猜測。

```bash
uv run lbt aggregate-validate "$LBT_OUTPUT_ROOT/aggregate-release"
uv run lbt report-validate "$LBT_OUTPUT_ROOT/<run_id>.report-v1"
```

`report-validate` 只讀 caller 明示的既有 report-v1 release，不自動搜尋、不建立 output。
正式 trajectory gate 保留 v2 的既有位置／環境報告用途並支援 v3 速度欄位；v1 不得當正式
垂向 evidence，v2／v3 混用的 run 保守拒絕。主線 `report-build` 尚未存在，不能以
`report-validate` 的通過宣稱報告建置或正式科學發布完成。

### 向下沉降來源路徑圖成果包

`source-pathway-build` 是獨立於 F01–F12／T01–T06 registry 的 `source-pathway-v1` 產品
入口。它只接受已通過 `aggregate-validate` 的 aggregate release、同一
`AggregateSpec` canonical hash 的 `ReportSpec`，並在 build 前拒絕任何
`settling_velocity_mps >= 0` 的 scenario。它不讀取 trajectory shard，也不重算統計分母；
統計直接沿用 `build_report_statistics` 的訪格、首次通過年齡、停留時間、KDE/HDR、邊界與
停止／QC products。

```bash
mkdir -p "$LBT_SCRATCH_ROOT/mplconfig-source-pathway"

uv run lbt source-pathway-build \
  --aggregate-release "$LBT_OUTPUT_ROOT/<run_id>.aggregate-v1" \
  --report-spec "$LBT_OUTPUT_ROOT/<run_id>.report-spec.json" \
  --destination "$LBT_OUTPUT_ROOT/<run_id>.source-pathway-v1" \
  --mplconfigdir "$LBT_SCRATCH_ROOT/mplconfig-source-pathway"

uv run lbt source-pathway-validate \
  "$LBT_OUTPUT_ROOT/<run_id>.source-pathway-v1"
```

`--mplconfigdir` 必須是 caller 事先建立的絕對、可寫、非 symbolic link 目錄；destination
必須是尚不存在且以 `.source-pathway-v1` 結尾的新 final。每站輸出 `figure.png`、
`figure.svg`、`figure.pdf`、`grid.parquet`、`boundary.parquet`、`outcomes.parquet` 與
`caption.json`，根目錄另有 README 與 schema 1.0.0 manifest。manifest 保存材料／沉降速度、
vertical／arrival、scenario strata、pooled 分母語意、aggregate release manifest SHA-256、
所有輸出 size／SHA-256；validator 不洩漏絕對路徑。

圖面 A／B 成對呈現每粒子每格一次的訪格比例與首次通過年齡，D 分開呈現保留重複迴游的
每成員停留時數；沒有樣本的格留白，低樣本格以斜線標示。`bed_first_contact_count` 與
`bed_repeated_contact_count` 是逆向粒子的底床邊界接觸診斷，不能當沉積質量／濃度；
`BED_DEPOSITED`、`DATA_GAP`、`NUMERICAL_FAILURE`、outer／MAX_AGE 等停止結果在 F 面板與
outcomes sidecar 保留。pooled 結果按已執行成員數加權，條件於本次情境設計與有效成員，
不推論材料自然比例或絕對來源機率。
C 面板的逆向 `local_first_exit` 在條件式解讀下對應正向潛在移入入口，E 面板呈現潛在
移入邊界區段；兩者都是邊界事件診斷，不能直接稱為確定來源。

合成工程 round-trip、available／低樣本 KDE、小型 1×N 格網、零／正沉降防線與 tamper
hash 測試位於 `tests/test_source_pathway_release.py`；測試產生的 PNG 可另以 `view_image`
進行圖面 QA，但不代表正式 OCM／NWW3 科學成果。

## Python API 與相關 runbook

需要建立 request、讀取已驗證 scenario、重建 geometry、使用
`ForcingWindowManager`、直接呼叫 `ProductionBatch` 或處理 checkpoint 的 API，請先讀：

- [架構與資料契約](../foundation/02_architecture_and_data_contract.md)
- [程式碼導覽與 plan traceability](../development/11_source_code_guide_and_plan_traceability.md)
- [SERVER 執行手冊](06_server_runbook_plan.md)
- [單站 pilot 計畫](pilot_run_plan.md)

API 不繞過 config、manifest、QC、source binding 或 formal gate；Python synthetic fixture
只能驗證工程行為。部署與資料同步遵循[Git 部署與資料同步手冊](git_deployment_and_data_sync.md)，
本機 Git 與 SERVER data／output／checkpoint 分開管理。
