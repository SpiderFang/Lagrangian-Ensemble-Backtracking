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
| 設定／輸入 | `config-check`、`preflight`、`inputs-build`、`inputs-validate`、`release-config-create`、`release-config-validate`、`horizon-suite-create`、`horizon-suite-resume`、`horizon-suite-validate` |
| pilot | `pilot-calibrate`、`pilot-calibrate-validate`、`pilot-calibration-build`、`pilot-calibration-validate`、`pilot-config-create`、`pilot-config-validate` |
| 工程驗證 | `behavior-manifest`、`synthetic-smoke`、`validate-shard`、`code-provenance`、`validate-run`、`benchmark-report`、`pilot-matrix-validate` |
| run lifecycle | `run-create`、`run-shard`、`run-worker`、`run-reconcile` |
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

### horizon suite：一次建立共同母體與六份 mode release

正式範例啟用每站 50 個隨機沉底年齡，最大 90 日，以 `20260916` seed 在 50 個離散小時 strata
各抽一值；五站共用完全相同的年齡向量。H30/H60/H90 的共同母體需要 180 日 observation
selection envelope（H90 + 最大年齡 90），而沉底後的逐筆 runtime support 是 90 日。`inputs-build`
先用觀測 anchor selector 驗 180 日，再轉為 deposition UTC；receptor 與 OCM dynamic initial
condition 以 deposition UTC 建立。Gap artifact 會分開保存 `selection_support_days=180` 與
`runtime_support_days=90`，root/per-deposition runtime `max_backtrack_days` 為 90，不能錯誤要求等於
selection envelope。legacy bed arrival/gap 與 release binding 使用 schema `1.1.0`；明示
gap-censored policy 的 bed 母體使用 schema `1.2.0`，
並在 root／record 保存第一缺口、截尾與分母 evidence；legacy generic 保持 `1.0.0`，validator
拒絕跨模式載入，artifact index／bindings 的自身 schema 不變。

兩種 config 模式的意思不同：`fixed_calendar_window` 從固定日曆終點回溯 H 日，若隨機沉底早於該窗，
實際期間為 H 減沉底年齡；`full_horizon_from_deposition` 則以沉底 UTC 為起點，再向前完整回溯 H 日。
前者的 pre-window 成員會留下 `pre_window_deposition` outcome，不執行 forcing；仍在 total outcome
分母計數，但不進 valid denominator 或來源 numerator，也不當作資料／數值失敗。這些統計是條件式來源
足跡／相對來源權重，不是絕對來源機率。

例如 `--backtrack-days 30 60 90` 時，suite 取 Hmax=90，`common-config` 寫
`inputs.backtrack_support_days: 180`、`bed_residence_time.runtime_horizon_support_days: 90`、
`boundaries.max_backtrack_days: 90`，使用同一組 accepted roots **只執行一次** strict `inputs-build`。
接著產生 H30/H60/H90 × `fixed_calendar_window`／`full_horizon_from_deposition` 六份 release config。
每份 step budget 依該 H 計算 `ceil(H * 86400 / integration.dt_min_seconds) + 1`；mode、selection/runtime
support 證據都進 release config hash/binding。六個檔名、validation 路徑與 manifest record 帶有 mode；
共同 input artifacts 與五站母體 fingerprint 必須完全相同，validator 會重建 exact expected payload 並拒絕
mode 或 topology tamper。legacy suite 缺少必要 180/90 日逐時節點時 fail closed；明示
gap-censored suite 則完整列舉缺時，runtime 在第一個向後缺口產生 `data_gap` 並截尾，但仍拒絕
deposition 起點缺資料、manifest 漂移或不完整 policy；任何模式都不能補零、用最近值或未登錄外插繞過。

```bash
uv run lbt horizon-suite-create \
  --config-template "$FORMAL_CONFIG_TEMPLATE" \
  --backtrack-days 30 60 90 \
  --destination "$LBT_SCRATCH_ROOT/horizon-suite-2025-observation-h30-h60-h90-v1" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --formal-release
```

`--formal-release` 可改寫為 `--formal`；若只建立 pilot 供工程稽核，改用 `--pilot`。
formal 仍須通過既有 A 區 v3/local20 formal gate，suite 不得繞過；兩種模式都仍要求
accepted roots、strict input derivation、完整結構與 hash 驗證；pilot
可以保留 `generated` 狀態，不能當作 `approved` 或正式科學成果。suite 對 template 的
改寫只允許原範例文件化的
`scenarios.receptor_arrival_initial_condition_manifest` placeholder（值為
`manifests/receptor_arrival_initial_condition.json`），再寫入
`inputs.backtrack_support_days` 與本次共同來源；若 template 已綁定 release／pilot，或
其他欄位指向非預期 derived path，建置會拒絕，不會沿用或覆寫既有綁定。

目的地必須是不存在的新目錄。成功的 suite 目錄具有固定語意的下列內容：

```text
<suite>/
├── source-template.yaml       # 原始 template 保存
├── common-config.yaml         # 180 日 observation selection／90 日 runtime support
├── common-input/              # 唯一一次 inputs-build 的 immutable artifacts
├── release-configs/           # H30/H60/H90 × 兩種 mode，共六份 release YAML
├── validations/               # common input 與六份 mode-specific release validator JSON
├── horizon-suite-manifest.json        # suite 拓撲、模式、日數與所有來源／artifact hash
└── horizon-suite-manifest.json.sha256 # manifest bytes 的 SHA-256 binding
```

`horizon-suite-validate` 是唯讀入口；它固定讀取 suite 內的 `common-input/`，重新核對
suite 拓撲、source-template 與 common-config 的來源關係、common-input 的 accepted product
provenance、bed suite 的 1.1.0／legacy suite 的 1.0.0 source schema、180/90 gap-safe 證據、一次
build、六份 release 的精確步數／日數／mode 與共同 artifact hash。省略
三個 accepted roots 時只驗 suite 內的 artifact closure，不代表重新核對 accepted source
bytes 或 canonical UTC axis；正式或移機驗收必須明示三個 roots。不提供外部輸入目錄 override，
以免破壞 release YAML 的 `../common-input/*` exact path binding。可明示三個 accepted roots：

新版 suite manifest 同時保存 `forcing_years=[2024, 2025]`、`observation_years=[2025]`、
arrival policy、`replicates_per_stratum=2`、48 個核心 strata 與 2 個事件；loader 不接受
舊兩年份 observation population 與新版 policy 混用。

```bash
uv run lbt horizon-suite-validate \
  "$LBT_SCRATCH_ROOT/horizon-suite-2025-observation-h30-h60-h90-v1" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --formal-release
```

`--pilot` 只切換驗證的正式狀態閘門；它不放寬 schema、來源、gap-safe、路徑或 hash 檢查。
驗證失敗時回傳非零狀態，且 create 任一步失敗都不發布成功的 final suite；失敗時保留
`.partial-*` 現場供人工稽核，不自動遞迴刪除。共享同帳號 SERVER 上，操作員清理前必須先
確認 process、partial 的目錄擁有者、inode 與 final 狀態，不可採用先 `stat` 再 `unlink` 的
競態方式；`.partial-*` 不能當作成功。即使六份 config 與 manifest 均有效，也只保證共同設計
母體與執行設定可比較，不保證每粒子走滿指定 H；粒子仍可能因海岸、域外、資料缺口或數值
狀態停止。正式輸入限 OCM schema 3 `ocm_native`、OCM schema 3 `ocm_surface` 與 NWW3 schema
1 `nww3_analysis`，禁止 raw NetCDF、transfer archive、零值與最近值補齊。

**資料可行性限制：** 新版正式 arrival 母體只以 2025 作 observation anchor，forcing 仍保留
2024–2025。最早的 2025-01-01 observation 往前 180 日約落在 2024-07-05，因此不把 2024
早季當作 observation strata，也不需另補 2023 forcing。範例仍為 `design_pending` template，
180/90 是必須驗證的支援契約，不是 accepted forcing 已通過的聲明，也不是 approved release 或
SERVER/input-build 實測；若 2024-07-05 前實際 accepted product 有缺時，legacy strict preflight/build 仍會
拒絕；新版 gap-censored build 則由 forcing inventory 與 gap-censored evidence 列舉缺口、確認
起點 exact-hour 可用並固定第一缺口截尾。任何模式都不能以最近值、零值或未登錄外插補足。

### 同一套 legacy generic 輸入產生不同回溯長度（手動模式）

本節只適用於未啟用 `scenarios.bed_residence_time` 的 legacy generic config。random bed
residence release 必須保留 mode 與 180/90 evidence，請使用上一節 `horizon-suite-create`，不可
把兩者混用。在 legacy config 中明示 `inputs.backtrack_support_days`，例如 30，先以此支援窗
完成上面的 `inputs-build` 與 `inputs-validate`。接著可重用同一目錄產生不同回溯長度；
以下變數須指向已驗證的共同母體、同一研究版本的完整模板及已核定的步數預算：

```bash
uv run lbt release-config-create \
  --config-template "$COMMON_CONFIG" \
  --input-directory "$COMMON_INPUT_DIRECTORY" \
  --max-backtrack-days 7 \
  --maximum-step-count "$STEP_BUDGET_7D" \
  --output "$LBT_SCRATCH_ROOT/release-7d.yaml" \
  --formal-release

uv run lbt release-config-create \
  --config-template "$COMMON_CONFIG" \
  --input-directory "$COMMON_INPUT_DIRECTORY" \
  --max-backtrack-days 30 \
  --maximum-step-count "$STEP_BUDGET_30D" \
  --output "$LBT_SCRATCH_ROOT/release-30d.yaml" \
  --formal-release
```

`--max-backtrack-days` 只改新輸出 YAML 的 `boundaries.max_backtrack_days`，
`--maximum-step-count` 只改新輸出的步數預算；省略時沿用模板。指定較長回溯時不會
自動放大步長、降低成員數或改動母體檔案。已指定的步數至少須達
`ceil(回溯日數 × 86400 / dt_max_seconds)`；自適應縮短步長可能需要更多步，
此式只是必要下限，不能當成足夠的執行預算。缺少正式數值設定仍會留下發布阻擋。

日數不是固定選單，新的正整日支援上限可任意設定，但執行長度不得超出母體。
7 日與 30 日設定各自執行 `preflight`，保存不同的 runtime inventory；兩者可共用
大型輸入，不能混用與另一份設定雜湊綁定的執行清單。更長上限、共同日期及
不覆寫限制見[輸入契約](14_input_derivation_and_release_contract.md#31-通用回溯支援與共同比較母體)。

### 既有 24 小時工程試跑

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

### 工程案例的共同亂數配對

`scripts/run_engineering_window.py run` 可透過 `--random-stream-id` 將
`no_stokes` 與 `finite_depth_stokes` 的 seed 導出案例命名空間固定為同一個明示 ID；兩個
run 仍須分別使用自己的 `--experiment-case-id`，因此物理案例、`particle_id` 與輸出目錄不
會合併。此選項不可為空白，建立後會把 ID 寫入 run plan schema `2.2.0`、seed table、
checkpoint binding 與 trajectory metadata：

```bash
uv run python scripts/run_engineering_window.py run \
  --artifact "$ENGINEERING_ARTIFACT" \
  --destination "$LBT_OUTPUT_ROOT/runs" \
  --run-id "$RUN_ID_NO_STOKES" \
  --experiment-case-id no_stokes \
  --random-stream-id abcd-common-rng-v1 \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT" \
  --shard-id 0
```

另一個物理案例沿用相同 `--random-stream-id`，但必須使用不同的 `--run-id` 與
`--experiment-case-id finite_depth_stokes`。resume 時省略 stream ID 會沿用 immutable
plan；若明示的 ID 與 plan 不同，或 checkpoint binding 的 stream 不同，流程會在建立
request／恢復粒子前 fail-closed。省略此選項的 run 維持 schema `2.1.0`、五欄 seed table
與原本以 `experiment_case_id` 導出的 seed，不需遷移既有 workspace。

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
路徑。execution checkpoint writer 目前固定發布 schema `3.1.0`，以 deterministic gzip 的
compact state 與 immutable history segment 保存；loader 同時保留 schema `3.0.0` 未壓縮拓撲，
並沿每代 `checkpoint.json` SHA-256 chain 檢查 generation、cursor、binding、checksum、particle
order、RNG 與前代 provenance。缺 generation、binding、checksum、
particle order、RNG 或前代 manifest 不符時，controller 在建立物理 request 前停止，不從 seed
靜默重算。latest/progress 更新若在完整 generation 發布後失敗，resume/reconcile 會驗證並採認
該固定序號 orphan，重新建立 pointer 與容量 metrics。

### 同程序接續執行多個分片

```bash
uv run lbt run-worker "$LBT_OUTPUT_ROOT/runs/$RUN_ID" \
  --config "$PILOT_OR_FORMAL_CONFIG" \
  --shard-id "$FIRST_SHARD_ID" \
  --shard-id "$SECOND_SHARD_ID" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT"
```

兩個 ID 都必須來自同一份已驗證 run plan。`--shard-id` 可重複指定，但值不得重複；
全部 ID 先檢查完才開始第一片，依命令列順序執行。此入口只開啟一次 controller，讓相鄰
分片重用相同流場管理器、網格與有界月份資料窗；不跨 run 共用，不自動挑選其他工作。
排程端應把同流場及接近的到達時段排在一起，並對不同 worker 指派互不重疊的 ID。

`--resume` 與 `--checkpoint-root` 沿用單片契約；可加 `--sweep-budget N` 限制每片最多
執行 N 輪粒子步進。任一片回報 `PAUSED` 後即停止後續片，已寫 checkpoint 可供下一次
明示恢復。鎖衝突或其他例外亦停止，不跳過錯誤繼續運算。`COMPLETE` 仍須通過既有輸出
驗證才視為完成，已有完成成果不重算。此入口不解除 SERVER 儲存及正式發布檢查。

worker 的總計時包含本次 handler 的前置驗證與連續分片執行，不含 Python 啟動與模組
匯入；亦不是多個 worker 的整機經過時間。原分片內部計時定義維持不變。流場載入／
命中／未命中／淘汰的底層計數仍是管理器累計值，分片紀錄則扣除本次執行前基準，
保存本次增量；續跑會與該分片先前已保存的增量合併。報告只加總語意可確認的分片
計數，舊紀錄或量測不完整時明示限制。`manager_count` 與 `resident_bytes` 取已觀測
樣本最大值，不跨片相加；大型陣列的映射位元組數不是實際 RSS 或 NFS 傳輸量，也不代表
多程序同時使用量或連續量測峰值。完整基準規範見[效能改善工作線](16_performance_improvement_tracks.md)。

### 正式完整母體的固定 worker 平行執行

`run-formal-parallel` 僅接受 `run_kind=formal`，依 run plan 的全部 shard ID 建立一次
確定性 worker assignment。scenario index 範圍維持連續，並以 plan 已有的分析區域／UTC 月份
和 scenario table 已存在的站點／flow-domain 欄位產生 locality 摘要；缺少表格欄位時會在
assignment JSON 明示 `plan_order_only`，不從 config、路徑或名稱推測 flow domain。每個固定
worker 只啟動一次 Python，內部以一個既有 `run-worker` controller 依序處理其整組 shard，因而
可在同程序重用 forcing manager 與 JIT import。assignment 綁定 run ID 和 run-plan SHA-256，
不寫回 run plan，也不因 worker 完成順序改變 run identity。

先準備 caller 明示且已通過 `scripts/validate_server_storage.py` 的 PASS JSON；runner 會再用
`findmnt` 即時確認本次 `scratch_root` 仍為 NFS 且 mount source token 與該 PASS JSON 相同，
防止把其他掛載點或過期環境的證據誤用。scratch、log 與 Numba cache 都必須是該 scratch root
下的非 symbolic-link 嚴格子目錄。下例的 worker 數由
本次排程負責人明示，不能由 CLI 猜測或代替研究設定。Numba backend 還必須明示
`--numba-cache-dir`；純 NumPy backend 可省略。建議透過 repository wrapper 啟動，使 cache
環境值在本程序第一次匯入科學套件之前設定：

```bash
uv run python scripts/run_formal_parallel.py \
  "$LBT_OUTPUT_ROOT/runs/$RUN_ID" \
  --config "$FORMAL_CONFIG" \
  --project-root "$LBT_PROJECT_ROOT" \
  --worker-count "$FORMAL_WORKER_COUNT" \
  --scratch-root "$LBT_SCRATCH_ROOT" \
  --log-root "$LBT_SCRATCH_ROOT/formal-parallel/logs" \
  --storage-gate-evidence "$LBT_STORAGE_GATE_JSON" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --cpu-affinity auto
```

目前的 Numba dispatcher 明確使用 `cache=False`，因此 `NUMBA_CACHE_DIR` 只作為已驗證的安全
執行環境值，不會保存 `.nbc`／`.nbi`，不能宣稱不同 worker 或下次執行能重用磁碟編譯結果。
正式執行時每個長壽命 worker 都會在自己的 Python 程序內呼叫一次
`accelerated.warmup_numba_backend()`，編譯結果只留在該程序記憶體；它隨後執行該 worker 的
`run-worker`，因而同一程序內可重用 dispatcher。`--warmup-only` 是可選的獨立程序編譯檢查，
檢查 log／summary 保存在新 session，但不會取代正式 worker 內的暖機，也不會快取供後續程序：

```bash
uv run python scripts/run_formal_parallel.py \
  "$LBT_OUTPUT_ROOT/runs/$RUN_ID" \
  --config "$FORMAL_CONFIG" \
  --project-root "$LBT_PROJECT_ROOT" \
  --worker-count "$FORMAL_WORKER_COUNT" \
  --scratch-root "$LBT_SCRATCH_ROOT" \
  --log-root "$LBT_SCRATCH_ROOT/formal-parallel/logs" \
  --storage-gate-evidence "$LBT_STORAGE_GATE_JSON" \
  --checkpoint-root "$LBT_CHECKPOINT_ROOT" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --numba-cache-dir "$LBT_SCRATCH_ROOT/formal-parallel/numba/$RUN_ID" \
  --cpu-affinity auto \
  --warmup-only
```

正式 run 再以同一組 workspace、config、roots、worker count 與 cache dir 執行上方主命令，並加
`--numba-cache-dir`；不要再帶 `--warmup-only`。cache 目錄仍須通過 scratch／NFS gate，即使目前
dispatcher 不落磁碟也不得把其值指向 `/home` 或未驗證 temporary directory。Linux 可用時 `auto` 會在 worker import 前套用
固定 CPU affinity；其他平台或不可用 cpuset 會不綁定並記錄 fallback，不能宣稱已綁定。需要指定
CPU 時可傳 `--cpu-affinity 0,1,2`，ID 必須屬目前允許 cpuset。`--resume` 只在任何 shard 已有
執行狀態時明示；全新 run 不加。此入口不提供 sweep budget，因此 worker 必須處理每個分配 shard
直到完整生命周期狀態。

只有 CLI 退出 0、summary `status=COMPLETE`、所有 child exit code 為 0、每個 worker summary
涵蓋分配的全部 shard 且 lifecycle 均為 `COMPLETE`，並且最後的 `validate-run --require-complete`
回傳 `valid=true`，才算整批完成。summary 同時記錄 whole-machine elapsed、各 child exit、CPU
affinity 實際狀態、固定分組與獨立 log 名稱。任何 child 失敗即停止後續派發；SIGINT／SIGTERM
會轉送至執行器建立的 worker process group，停止後保留 log、已完成 trajectory 與 checkpoint，
並以非零狀態回報。失敗或中斷不會自動恢復：先保留現場、執行 `run-reconcile` 與唯讀 validator，
確認原 run ID／config／checkpoint root 後，才由負責人明示 `--resume` 重啟；不得建立同內容的新
run 來掩蓋部分完成。此命令是正式運算入口，不是舊版／新版倍率比較或 benchmark A/B 工具。

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

新分片統計以 `forcing_cache_stats_semantics: invocation_delta_v1` 標示增量語意。
`benchmark-report` 的 `forcing_cache_stats_precise` 只說明快取計數是否具備完整且可
加總的紀錄，與 run 完整性驗證的 `valid` 分開判讀。若含歷史未標記或不可用的量測，
`forcing_cache_stats` 保持空物件，並以 `forcing_cache_stats_legacy_by_shard` 或
`forcing_cache_stats_unavailable_shards` 列出限制，不把缺資料當作零次載入。

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

### `horizon-suite-resume`

若 `horizon-suite-create` 在發布前中斷並留下 exact preserved `.partial-*`，可用下列命令
在新的 destination recovery：

```bash
uv run lbt horizon-suite-resume \
  --partial "$LBT_SCRATCH_ROOT/.horizon-suite-2025-observation-h30-h60-h90-v1.partial-<id>" \
  --backtrack-days 30 60 90 \
  --destination "$LBT_SCRATCH_ROOT/horizon-suite-2025-observation-h30-h60-h90-v1-recovered" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --formal-release
```

核心會唯讀重驗 source-template、common-config、common-input closure、source config hash
與三個 accepted roots，再於新的 owned partial 重建 release／validation／manifest；不修改
或刪除原 partial，destination 已存在、symlink、tampering 或日數不一致都會拒絕。新的
manifest 固定保存 `input_build_count=1`、
`recovery_method=resume_reuse_validated_common_input_v1` 與不含絕對路徑的
`recovery_source_fingerprint`，此流程不再次呼叫 `inputs-build`。
