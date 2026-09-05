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
| 工程驗證 | `behavior-manifest`、`synthetic-smoke`、`validate-shard`、`code-provenance`、`validate-run`、`benchmark-report` |
| run lifecycle | `run-create`、`run-shard`、`run-reconcile` |
| 聚合／報告 | `aggregate-spec-create`、`aggregate-build`、`aggregate-validate`、`report-spec-create`、`report-validate` |

`report-build` 尚未註冊；不要用不存在的命令替代目前的 report preflight、aggregate 或
release validator。

## 共通準備與安全邊界

```text
LBT_PROJECT_ROOT=/path/to/Lagrangian-Ensemble-Backtracking
LBT_OUTPUT_ROOT=/path/to/new/output-parent
LBT_SCRATCH_ROOT=/path/to/existing/scratch-parent
OCM_NATIVE_ROOT=/path/to/accepted/ocm_native
OCM_SURFACE_ROOT=/path/to/accepted/ocm_surface
NWW_ANALYSIS_ROOT=/path/to/accepted/nww3_analysis
LBT_CHECKPOINT_ROOT=/path/to/checkpoints
```

正式 forcing 只能是已驗收的 OCM schema 3 `ocm_native`／`ocm_surface` 與 NWW3 schema 1
`nww3_analysis`。root 可由命令列明示；省略時只讀 config 指定的環境變數，不猜測目前
工作目錄或 SERVER 路徑。`LBT_SCRATCH_ROOT` 應已存在，以下 final child 必須尚不存在。

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

## Pilot 與 synthetic

`pilot-calibrate`／`pilot-config-create` 的完整參數與 candidate gate 見
[SERVER 執行手冊](06_server_runbook_plan.md)；這些命令只接受已驗證輸入，輸出是 engineering
evidence，不是正式參數或成果。`behavior-manifest` 可建立十種全負沉降的材質／形狀代理
manifest；它不替代正式 source manifest。

不需真資料的端到端 smoke：

```bash
uv run lbt synthetic-smoke --output "$LBT_SCRATCH_ROOT/synthetic-smoke-v1"
uv run lbt validate-shard "$LBT_SCRATCH_ROOT/synthetic-smoke-v1"
```

`synthetic-smoke-v1` 必須是尚不存在的 child；不要把已由 `mktemp -d` 建立的根目錄直接
當成 `--output`。輸出 metadata 會標示 `synthetic_smoke_not_scientific_result`，只能驗證
CLI、engine、trajectory I/O、Parquet、manifest 與 checksum。

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
