# SERVER 執行手冊

## 1. 適用範圍

本文件定義 SERVER 部署、preflight、pilot、正式 batch、checkpoint、QC 與發布程序。
程式歷史接入、canonical 根目錄衝突保留、Git bundle 與大型資料同步邊界，另見[Git 部署與資料同步手冊](git_deployment_and_data_sync.md)；本文件只保留科學 runtime 與發布流程。
`config-check`、`preflight`、`behavior-manifest`、`synthetic-smoke` 與 `validate-shard` 已可
執行；Phase 3B1/3B2a 另已提供 `code-provenance`、`validate-run` 與 `benchmark-report`，
並以 schema 2 ordering、固定 lock topology 與跨程序 progress 契約約束 restart。
Phase 3A2 已提供 `receptor_arrival_initial_condition_manifest` 的 strict loader：它驗證
100 個 receptor templates、250 個 arrival-time 與 5,000 個 dynamic pair initial conditions，
並保存 pair component hash；formal actual z 必須取自 pair record。runtime 已由
`initialize_run`／`initialize_formal_run`、`RuntimeRequestFactory` 與
`open_run_controller` 接通 pilot／formal CPU 管線；`lbt run-create`、`lbt run-shard` 與
`lbt run-reconcile` 可依 immutable run plan 執行兩種模式。formal 仍會對正式 config、
approved manifests、192 筆月份 inventory、8 筆時間軸及逐 arrival flow 的時間支援
fail-closed。aggregate/release 仍是後續 gate，不得以 synthetic fixture 冒充正式批次。

本輪僅同步程式與 runbook，未登入或執行 SERVER，沒有正式研究結果。

2026-08-19 的舊版 preflight 曾把 NWW `trial_ready`、OCM partial month 與已知缺時合計成
188 項正式 blocker；此解釋已被 2026-08-20 available-data 決策取代。現有 OCM/NWW 即本
研究可取得的完整 2024–2025 母體，供應者補件不是前置條件；NWW native 實測為 17,544 個
連續逐時 UTC，OCM canonical 軸為 17,124 個 UTC、總缺 420 時次。後續 preflight 把前兩項
記為 accepted provenance，把 OCM 缺口送入 reconstruction/gap-safe gate，不能再輸出「等待
上游補資料」的建議。細節見[全部可得資料決策](10_available_data_time_reconstruction_and_a_expansion.md)
與[更正後稽核](09_implementation_audit_2026-08-19.md)。每次 release 仍須重跑並保存
machine-readable evidence；不得把密碼、private key 或 token 寫入 repository、設定、命令
紀錄或報告。

## 2. 預定路徑變數

```bash
export LBT_PROJECT_ROOT=/home/mustlab/Workspace/Lagrangian-Ensemble-Backtracking
export OCM_NATIVE_ROOT=/data/OCM-Preprocessed-Data/preprocessed/ocm_native
export OCM_SURFACE_ROOT=/data/OCM-Preprocessed-Data/preprocessed/ocm_surface
export NWW_ANALYSIS_ROOT=/data/NWW-Preprocessed-Data/preprocessed/nww3_analysis
export NWW_NATIVE_ROOT=/data/NWW-Preprocessed-Data/preprocessed/nww3_native/ww3_grd3_253x237

# 正式值需在容量與檔案系統檢查後核定；不可直接假設專案目錄有足夠空間。
export LBT_OUTPUT_ROOT=/path/to/approved/lagrangian-results
export LBT_SCRATCH_ROOT=/path/to/local-fast-scratch/lagrangian
export LBT_UV_CACHE_ROOT=/path/to/local-fast-scratch/uv-cache/lagrangian
export LBT_MPL_CACHE_ROOT=/path/to/local-fast-scratch/matplotlib-cache/lagrangian
```

正式設定與程式碼只引用這些 task-specific 變數，不以 `/Users/...`、`$HOME` 或 raw data 絕對路徑硬編碼。

## 3. G0 唯讀資料盤點

### 3.1 路徑與月份

在已認證的 SERVER shell 執行：

```bash
for root_path in "$OCM_NATIVE_ROOT" "$OCM_SURFACE_ROOT" "$NWW_ANALYSIS_ROOT"; do
  test -d "$root_path" || { echo "MISSING $root_path"; continue; }
  find "$root_path" -mindepth 3 -maxdepth 3 -type d -name '20????' | sort
done
```

預期不是只看「共 24 個」；必須依每個 `flow_domain_id` 分別列 202401-202512，並辨識完全缺月、partial month 與重複版本。

### 3.2 metadata 與狀態

```bash
find "$OCM_NATIVE_ROOT" "$OCM_SURFACE_ROOT" "$NWW_ANALYSIS_ROOT" \
  -path '*/months/20????/metadata.json' -print0 \
  | xargs -0 jq -c '{path: input_filename, status, cache_kind, schema: (.cache_schema_version // .schema_version), month, flow_domain_id: (.flow_domain_id // .domain.domain_id), source_day_coverage}'
```

此輸出需保存為 G0 evidence，但它還不能取代 `time_utc_ns.npy` 的逐值檢查。正式 `lbt-preflight` 會另外驗證：

- 設定恰有 A-D 四個 `analysis_region_id`／`flow_domain_id` 與五個唯一 `study_site_id`；貢寮、龜山島均對應 A 區，但情境與輸出不可合併。
- 貢寮／龜山島以 anchor 產生 12.5 km receptor core、25 km local domain 及 20/35 km 敏感度 polygon，與固定 OCM ocean polygon 相交；兩個 local domains 重疊時須完整保留，不作 Voronoi 切割。圓周外海 arc 與岸線必須分段，只有前者可計入 local-entry KDE。
- 貢寮／龜山島引用同一 A 區 forcing-domain ID 與 outer-boundary geometry；每條軌跡只以 own local domain 產生主要 first-exit。foreign-local crossing 必須是非終止診斷事件，不得改變 `study_site_id`、scenario、seed 或主要入口分母。
- 現行 A v3 bbox 南界 `24.600844°N` 只允許 `development_and_pilot`。正式候選採
  `northeast_taiwan_common_cache_v4_lbt_south_expanded` 與 bbox
  `[121.306315,122.793685,24.480000,25.499156]`，並對 OCM native、OCM surface、NWW
  analysis 逐一證明 25/35 km local boundary 至 outer boundary 至少兩個共同有效格點。
- 月份內原始 UTC 可含重複或亂序；先 stable sort/prefer-last，再驗證 canonical UTC 嚴格
  遞增與唯一，並保存來源月份/local index。
- 實際 start/end、正常間距、缺口長度與跨月銜接。已知 gap 是 reconstruction inventory，
  不是要求原始供應者補件的 blocker。
- array shape/dtype 與 metadata 相符。
- OCM native/surface pair 的空間相容，以及 NWW full-hour analysis 對 OCM observed 與
  reconstructed UTC 的 superset 支撐；不得再要求 NWW 時間軸與舊 gappy OCM 軸逐值相等。
- OCM 必要欄位與 NWW Hs/fp/DP/mask/QC 的存在及有限率。

### 3.3 正式時間產品產製

正式 batch 前依固定順序產製兩個不可變 manifest：

1. 從 NWW native 24 個月份重採樣到每個正式 OCM 靜態格網，產生 17,544 個完整逐時
   analysis slices；DP 在空間與時間均先轉成單位向量作圓形內插，再轉回角度。
2. 對 OCM canonical 軸建立缺口遮罩，執行實際缺口形狀的 blocked cross-validation；通過
   才產生 `observed/reconstructed_short/reconstructed_state_space` patch 與 posterior members。
3. validation 未通過的長缺口不硬補；arrival selector 只選擇不跨缺口、且能支援該
   horizon 的分層時窗。此 fallback 必須仍覆蓋兩年、四季、大小潮與三潮位相位 strata。
4. preflight 驗證 manifest、checksum、方法版本、時間 origin 與 forcing member 完整後，
   才將已知缺口視為可積分支撐；任何 runtime 臨時補值均禁止。

此外，正式 LBT config 必須指向不可變的
`receptor_arrival_initial_condition_manifest`。它不是由本 run 臨時從 OCM 讀取產生的檔案；
上游完成產製後，LBT loader 只驗證每個 arrival UTC 的 `eta`、`bed`、`zcor`、濕元素與
來源索引，並核對五站 5,000 個完整 pair coverage。`Receptor.z_m_positive_up` 仍只是
模板候選值，不能取代 pair actual z；若 manifest 尚不存在，formal gate 應停止而不是猜測
深度或以模板值補齊。

A 區 v4 的 OCM/NWW 上游產製已封裝為可重啟入口；其 config hash 會進入 OCM metadata，
OCM partial month 保留 coverage，NWW 則直接使用每月 native UTC 建 full-hour analysis：

```bash
cd "$LBT_PROJECT_ROOT"
bash scripts/prepare_a_v4_forcing.sh dry-run
bash scripts/prepare_a_v4_forcing.sh month 2025 1
# 單月 OCM/NWW validator 與實際共同 margin 通過後才執行：
bash scripts/prepare_a_v4_forcing.sh all
```

入口不使用 `--overwrite`。若新 domain 已有月份目錄，先由上游 validator 驗收；驗收失敗
即停止並保留現場，不能自動刪除或重建。`dry-run` 不寫大型 forcing，可直接用於確認 raw
月份、partial coverage、輸出路徑與參數。

### 3.4 容量與檔案系統

```bash
df -hT "$OCM_NATIVE_ROOT" "$NWW_ANALYSIS_ROOT" "$LBT_OUTPUT_ROOT" "$LBT_SCRATCH_ROOT"
findmnt -T "$LBT_OUTPUT_ROOT"
findmnt -T "$LBT_SCRATCH_ROOT"
```

若 output 在 NFS/NAS，checkpoint 與 active shard 優先寫本機 scratch；完成 checksum 後以單一 publisher 傳到同一遠端檔案系統的 `.incoming/<run_id>/`，最後原子改名。不得讓下游看到半套正式 run。

## 4. 環境建立

```bash
cd "$LBT_PROJECT_ROOT"
export UV_CACHE_DIR="$LBT_UV_CACHE_ROOT"
export UV_PROJECT_ENVIRONMENT="$LBT_SCRATCH_ROOT/venv"
export MPLCONFIGDIR="$LBT_MPL_CACHE_ROOT"
export PYTHONDONTWRITEBYTECODE=1

uv sync --frozen
uv run pytest -q -p no:cacheprovider
```

實作時鎖定具體 Python patch 版本並保存於 run manifest。Numba、NumPy 與 SciPy 版本變更可能改變浮點/JIT 行為；正式 release 只使用 `uv.lock`，不在 batch 中臨時更新套件。

## 5. CLI 流程

下列 validator、provenance、pilot 與 formal runtime CLI 均已接通。formal 命令可建立及
執行 workspace，但只有正式 release config、approved manifests／products 與
`preflight --formal-release` inventory 完整通過時才會開始；example config 仍刻意被 gate
阻擋，不能當成 SERVER release 設定。

### 5.0 Code provenance 與 run workspace validator

在 SERVER 乾淨 checkout 或沒有 `.git` 的部署目錄，先保存不含絕對專案路徑的程式指紋。
沒有 `.git` 時 pilot 可省略宣告 commit，來源固定為 `no_git_pilot`；formal 若沒有 Git
則必須由 release record 提供 40 位小寫 commit，來源固定為 `declared_deployment`，且
`git_dirty=None` 必須保留為「無法判定」而不能誤當成 clean。若有 Git，來源必須是
`git_repository`，formal 只接受 `git_dirty=false`：

```bash
uv run lbt-code-provenance --project-root "$LBT_PROJECT_ROOT"
uv run lbt-validate-run "$LBT_SCRATCH_ROOT/runs/<run_id>" \
  --checkpoint-root "$LBT_SCRATCH_ROOT/checkpoints"
uv run lbt-benchmark-report "$LBT_SCRATCH_ROOT/runs/<run_id>" \
  --checkpoint-root "$LBT_SCRATCH_ROOT/checkpoints"
```

`validate-run` 是純讀檢查；它會驗證 run plan/progress、immutable input file checksum、
schema 2 ordering policy/group metadata、scenario/seed row order、128-bit seed 導出、shard range、checkpoint generation/latest
與 COMPLETE trajectory shard 的完整 particle/scenario/member/site/region/receptor identity。
若 checkpoint 使用 workspace 外路徑，兩個命令都必須傳入執行時相同的
`--checkpoint-root`；此絕對路徑不寫入 JSON。缺失或落後的合法 latest、RUNNING orphan 及
published-before-progress 會回報 recoverable 但 `valid=false`，只有 controller reconcile
可更新現場。`benchmark-report` 只彙總已驗證 progress 的 wall time、
process CPU、max RSS、particle steps、output/checkpoint bytes 與 forcing cache stats，並
固定標記 `engineering_measurement_not_scientific_result=true`；pilot 的完成比例不得
改寫五站 50,000 基礎情境契約。

每個 schema 2 workspace 必須有固定且恰好完整的 `locks/`：`run_gate.lock`、
`progress.lock` 與每一個 `<shard_id>.lock`；它們是預建零長度普通檔案，不列入四個
immutable input checksum。worker 依 run gate shared → shard exclusive → progress exclusive
取得 Unix `fcntl.flock`；不同 shard 可同時執行，同 shard contention 在 request factory 前
失敗，reconcile 則需 run gate exclusive 且遇到 active worker 立即停止。NFS/NAS 的 flock
語意必須在 SERVER preflight 實測。舊 schema 1 synthetic/pilot workspace 不支援 resume，
必須以目前 ordering policy 重建，不能猜測或偷偷兼容舊 shard 順序。

### 5.1 Preflight

```bash
uv run lbt-preflight \
  --config configs/lagrangian_backtracking.example.yaml \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --output "$LBT_SCRATCH_ROOT/preflight/input_inventory.json"
```

Preflight 必須唯讀，不建立 forcing 副本。輸出至少包含 path token、schema、月份、raw 與
canonical time inventory、duplicate source choice、gap shapes、reconstruction/NWW-full-hour
manifest、array bytes、coverage、unit/direction decision、CRS/mesh QC、domain role、own/foreign
local-domain topology、各必要 forcing 的共同 margin、預估 working set 與輸出空間。
`trial_ready`/partial month 依 available-data contract 記錄為 accepted info，不得再當成等待外部
補件的錯誤。若 A 區仍為現行 v3，輸出必須明示 `PILOT_ONLY`，config validator 不得只因
metadata `status=ready` 就允許正式龜山島 25 km baseline。

runtime experiment case 由 `EXPERIMENT_CASE_SPECS` 唯一登錄五個值：
`finite_depth_stokes` 是 formal baseline 候選，`no_stokes` 是常數擴散敏感度，
`smagorinsky_cs_010`、`smagorinsky_cs_015`、`smagorinsky_cs_020` 是已接通但尚未升格
為正式結果的 Smagorinsky 敏感度。Smagorinsky diffusion facade 只載入 OCM native
current／mesh；三個案例的 velocity 仍含有限水深 Stokes，故 run-shard 仍必須提供
NWW analysis root。example config 的 Smagorinsky floor/cap 為 `null`，在填入經核定的
有限非負值且通過 floor/cap、well-mixed、PDE、收斂與 pilot gates 前會 fail-closed。

### 5.2 OCM 真資料擴散／步長校準 evidence

在 release config 與 input artifact 已通過唯讀 gate 後，SERVER 可建立 OCM-only pilot
calibration evidence。命令必須明示 accepted OCM native root、input directory、config、
project root 與新的 destination；不使用 raw NetCDF、transfer archive、NWW3 或環境中
未登錄的資料路徑：

```bash
uv run lbt pilot-calibrate \
  --config "$PILOT_CONFIG" \
  --input-directory "$LBT_INPUT_RELEASE" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --project-root "$LBT_PROJECT_ROOT" \
  --destination "$LBT_SCRATCH_ROOT/pilot-calibration-ocm"

uv run lbt pilot-calibrate-validate \
  "$LBT_SCRATCH_ROOT/pilot-calibration-ocm" \
  --config "$PILOT_CONFIG" \
  --input-directory "$LBT_INPUT_RELEASE"
```

建置器先驗證 release config、actual-z dynamic pair 與公尺制 geometry，才建立 hidden
partial directory；成功時以 atomic rename 發布，既有 destination 或任何 partial failure
均不會覆寫既有 evidence。每筆 pair 使用自己的 arrival UTC／`z_m_positive_up`，並以同一
OCM-only forcing manager 取樣 current 與 Cs=0.10、0.15、0.20 的 P1 nodal Smagorinsky。
`pair_samples.parquet` 的 QC 欄位非空，失敗物理值為 null；`calibration_report.json`
保存 stable-hash 抽樣政策、per-site／month／domain QC、候選 quantiles、time-limit
候選、input／code provenance 與 OCM cache resource counters；`manifest.json` 保存固定
三檔 closure、大小、SHA-256 與 schema fingerprint。

其中唯一常數水平擴散候選 `constant_kh_m2ps` 固定取 Cs=0.15 有效 particle Kh 的
q50；`constant_kz_m2ps` 取有效 OCM sampled Kz 的 q50，floor 固定為 0，cap 取 Cs=0.20
raw current-triangle Kh 的 q99.5。builder 輸出的 calibration report schema `1.1.0`
將 diffusion time-limit 分軸保存：`horizontal_diffusion` 使用
`(0.25*horizontal_scale_m)²/(2*constant_kh_m2ps)`，`vertical_diffusion` 使用
`(0.25*vertical_scale_m)²/(2*constant_kz_m2ps)`；尺度或對應 K 無效時只在該軸記為
unlimited，不製造有限步長，也不使另一軸一併失效。這是由三軸 Brownian 方差分軸得到
的契約，禁止以 `min(horizontal_scale_m, vertical_scale_m)` 與
`max(constant_kh_m2ps, constant_kz_m2ps)` 交叉配對。validator／reader 依明示 schema
版本保留既有 `1.0.0` 的 combined formula 相容驗證；legacy 不會被靜默套用 1.1.0 雙軸
公式。

validator 通過只表示 artifact topology、checksum、schema、QC、provenance consistency
與報告可由 pair table 重算；它不表示 diffusion candidate 已通過 well-mixed、PDE barrier、
時步／網格／系集收斂或正式 trajectory。完整設計只有在 100 receptors、250 arrivals、
5,000 unique pair records 且未用每站少於 1,000 筆的 explicit limit 時標為 `complete`；
explicit `--pair-limit-per-site < 1000` 一律標為 `partial_engineering_sample`。本機不執行真資料命令，該步驟
由具權限的 SERVER 完成。

### 5.2.1 Calibration-bound pilot execution config

calibration evidence 通過 `pilot-calibrate-validate` 後，SERVER operator 可用下列命令
建立 candidate-bound 的 pilot runtime 設定。所有 engineering scalar 必須由 operator
明示；`--active-chunk-size none` 是有意義的 explicit value，不可省略。候選 Kh/Kz 與
Smagorinsky floor/cap 不在命令列重複輸入，而由完整 schema `1.1.0` report 綁定。

```bash
uv run lbt pilot-config-create \
  --source-config "$PILOT_SOURCE_RELEASE_CONFIG" \
  --input-directory "$LBT_INPUT_RELEASE" \
  --calibration "$LBT_SCRATCH_ROOT/pilot-calibration-ocm" \
  --output "$LBT_SCRATCH_ROOT/pilot-execution.yaml" \
  --dt-min-seconds 60 \
  --dt-max-seconds 300 \
  --output-interval-seconds 900 \
  --max-backtrack-days 7 \
  --maximum-step-count 2016 \
  --members-per-scenario 4 \
  --master-seed 123 \
  --shard-scenario-count 100 \
  --checkpoint-interval-sweeps 10 \
  --active-chunk-size none \
  --max-resident-forcing-months 2

uv run lbt pilot-config-validate \
  "$LBT_SCRATCH_ROOT/pilot-execution.yaml" \
  --input-directory "$LBT_INPUT_RELEASE" \
  --calibration "$LBT_SCRATCH_ROOT/pilot-calibration-ocm"
```

建立器先以 source config 的 release/input binding 與 gap-safe horizon gate 驗證，再從
calibration report 取得四個 finite non-negative candidate。target 以同一 parent 的 hidden
YAML、file fsync 與 atomic rename 發布；既有 destination 不覆寫。target 搬到另一個 parent
時，`ARTIFACT_FILENAMES` 所登錄的 component binding 與 config runtime references 會同步
重建為新的相對路徑；根層 `pilot_execution_binding` schema `1.0.0` 只保存 source semantic
config hash（由 calibration report 的 `input_binding.config_hash` 驗證）、input/calibration
hash、候選值、套用欄位與 scalar snapshot，禁止保存 path。

兩個命令成功只代表 calibration candidate 已封裝為 `config_status=generated`，狀態仍固定
為 `candidate_pending_dt_and_member_convergence`。這不是 `approved`、`passed` 或正式
scientific baseline；`pilot-config-create` 不啟動 runtime，`pilot-config-validate` 只讀取
target、accepted input 與 calibration，invalid 以 exit code `2` 回傳。完成後才可依本節
5.4 的代表性 pilot runbook 另行執行 convergence 與 trajectory gate。

### 5.3 合成端到端 smoke test

```bash
uv run lbt synthetic-smoke \
  --output "$LBT_SCRATCH_ROOT/trials/synthetic-constant-flow"
uv run lbt validate-shard \
  "$LBT_SCRATCH_ROOT/trials/synthetic-constant-flow"
```

此命令不讀 SERVER forcing，只驗證 CLI、signed-time RK4、巢狀事件、ragged arrays、
Parquet、manifest 與 checksum。metadata 固定標示 `synthetic_smoke_not_scientific_result`，
不得作為 2024–2025 科學成果。接通實值單月 reference pilot 是 G3 下一切片。

### 5.4 代表性 pilot

```bash
uv run lbt run-create \
  --config configs/lagrangian_backtracking.example.yaml \
  --input-inventory "$LBT_PROJECT_ROOT/work/input-inventory.json" \
  --destination "$LBT_SCRATCH_ROOT/pilots" \
  --run-id pilot-representative \
  --run-kind pilot \
  --experiment-case no_stokes \
  --pilot-scenarios-per-stratum 1

uv run lbt run-shard "$LBT_SCRATCH_ROOT/pilots/pilot-representative" \
  --config configs/lagrangian_backtracking.example.yaml \
  --shard-id <shard-id> \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --resume \
  --sweep-budget 10

uv run lbt run-reconcile "$LBT_SCRATCH_ROOT/pilots/pilot-representative" \
  --checkpoint-root "$LBT_SCRATCH_ROOT/checkpoints"
uv run lbt validate-run "$LBT_SCRATCH_ROOT/pilots/pilot-representative" \
  --checkpoint-root "$LBT_SCRATCH_ROOT/checkpoints"
uv run lbt benchmark-report "$LBT_SCRATCH_ROOT/pilots/pilot-representative" \
  --checkpoint-root "$LBT_SCRATCH_ROOT/checkpoints"
```

`--pilot-scenarios-per-stratum 1` 只供工程 sanity／benchmark；selector 會由完整且已驗證的
scenario/receptor manifests 依 `(study_site_id, receptor.vertical_id)` 重算，每層取一筆，
目前完整五站×四垂向資料預期為 `5×4=20` 筆。20 不代表正式研究結果，也不改變每站 10,000、
全案 50,000 的正式 coverage。run plan 的 selection binding 保存 source/selected count、
order-independent ID hash 與 strata，static loader 在 `run-shard` 前會重算並比對；formal
command 禁止此參數且永遠使用完整 50,000 情境。

Pilot 報告需以五站點各固定 10,000、A 區 20,000、全案 50,000 個基礎 scenarios，外推各候選 `M` 與 experiment case 數的 particle-step、wall time、CPU、RAM、read bytes、trajectory bytes、event bytes、checkpoint bytes 與 NFS publish time；並比較 7/14/30/60 日 horizon 及貢寮／龜山島 20/25/35 km local boundary。A 區 paired-UTC shards 應共用同一 staged forcing time window，報告須證明沒有為兩站各自重複跨 NFS 載入同一 OCM/NWW 月窗。benchmark 用於衍生最小收斂 `M`、horizon、shard、並行度與儲存策略，不得據此把任一站完整交叉改回 1,000 或把五站合併為 10,000。

### 5.4 正式 batch（runtime／CLI 已接，release 證據仍須備妥）

正式 run 只能使用 `status=approved` 的 config、behavior、local-domain、每站 20／全案 100 receptor templates、每站 50／全案 250 arrival-time、每站 1,000／全案 5,000 dynamic pair initial-condition records、每站 10,000／全案 50,000 情境 coverage 與 member-convergence manifests。每個 material 共用 pair actual z，不得以模板 z 取代；A 區貢寮與龜山島必須引用同一個已通過共同 forcing margin 的 expanded flow-domain ID；現行 `northeast_taiwan_common_cache_v3` 不得出現在正式 release config。

正式鏈路固定為 formal preflight、建立 workspace、逐 shard 執行／暫停／恢復、reconcile，
最後要求完整驗證。第一次 `run-shard` 可用 sweep budget 在 checkpoint 邊界產生 `PAUSED`；
後續必須使用 `--resume` 與同一 checkpoint root：

```bash
FORMAL_CONFIG="$LBT_PROJECT_ROOT/configs/releases/lagrangian_backtracking_2024_2025_v1.yaml"
FORMAL_RUN_ROOT="$LBT_OUTPUT_ROOT/runs"
FORMAL_RUN_ID="lbt-2024-2025-v1"
FORMAL_INVENTORY="$LBT_SCRATCH_ROOT/preflight/formal-input-inventory.json"

uv run lbt preflight \
  --config "$FORMAL_CONFIG" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --output "$FORMAL_INVENTORY" \
  --formal-release

uv run lbt run-create \
  --config "$FORMAL_CONFIG" \
  --input-inventory "$FORMAL_INVENTORY" \
  --destination "$FORMAL_RUN_ROOT" \
  --run-id "$FORMAL_RUN_ID" \
  --run-kind formal \
  --experiment-case finite_depth_stokes \
  --project-root "$LBT_PROJECT_ROOT"

uv run lbt run-shard "$FORMAL_RUN_ROOT/$FORMAL_RUN_ID" \
  --config "$FORMAL_CONFIG" \
  --shard-id <shard-id> \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --checkpoint-root "$LBT_SCRATCH_ROOT/checkpoints" \
  --sweep-budget 10

uv run lbt run-shard "$FORMAL_RUN_ROOT/$FORMAL_RUN_ID" \
  --config "$FORMAL_CONFIG" \
  --shard-id <shard-id> \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --checkpoint-root "$LBT_SCRATCH_ROOT/checkpoints" \
  --resume

uv run lbt run-reconcile "$FORMAL_RUN_ROOT/$FORMAL_RUN_ID" \
  --checkpoint-root "$LBT_SCRATCH_ROOT/checkpoints"
uv run lbt validate-run "$FORMAL_RUN_ROOT/$FORMAL_RUN_ID" \
  --checkpoint-root "$LBT_SCRATCH_ROOT/checkpoints" \
  --require-complete
```

formal inventory 不是布林旗標通行證：它必須逐 flow 完整涵蓋 config 年月與產品契約，NWW
必須全期無缺，OCM 則只能是 `missing=0` 的完整／已重建產品，或由已核准 gap-safe manifest
逐站點所屬 flow 證明每個 inclusive 回溯窗不碰缺口。任一 topology、計數、boundary gap、
manifest binding 或時間支援矛盾都會在 forcing I/O 前拒絕。

目前 repository 的 example config，以及 SERVER 實際 approved manifests、正式產品與核准
參數尚未作為 release artifacts 提供，因此上述命令在這些證據到位前應 fail-closed。本文件
只說明程式已可正式執行；本輪未登入 SERVER，也未宣稱已完成任何正式科學批次。

每一 shard 完成後執行 schema/checksum/QC；失敗 shard 保留 failure manifest，不用不同 seed 手動重跑。相同 run 續跑只允許處理尚未完成且 checkpoint 相容的 shard。

## 6. tmux 作業方式

```bash
tmux new-session -s lbt-2024-2025
cd "$LBT_PROJECT_ROOT"
```

進入 session 後設定第 2、4 節的變數並執行命令。離開但保持執行使用 `Ctrl-b d`，重新連線後：

```bash
tmux attach-session -t lbt-2024-2025
```

每 5-15 分鐘更新 machine-readable progress：completed/failed/pending shards、particle steps、wall time、ETA、read/write bytes、RSS、checkpoint age。不要只輸出無法稽核的 progress bar。

## 7. Checkpoint 與恢復

Checkpoint 至少綁定：

- normalized config SHA-256。
- OCM/NWW input inventory hash。
- Git commit、dirty flag、lock hash、Python/NumPy/Numba 版本。
- scenario range、particle/member IDs 與 seed table hash。
- 最後完整 output time、particle state、triangle ID、status 與 RNG state/counter。
- shard output row count、partial checksum 與 schema version。

恢復時必須傳入原執行使用的 checkpoint root。run plan 只保存 `checkpoints` token，不保存
SERVER 絕對路徑；progress 宣告 sequence/path 而該 root 找不到合法 generation 時，controller
會在 request factory 與 forcing I/O 前停止，不能退回 seed 從頭重算。`RUNNING`、`PAUSED`、
`FAILED` 均須明示 resume；`COMPLETE` 只重驗輸出。PLANNED 卻已有 generation，或 checkpoint
tree 含 unknown、partial、symlink、checksum、binding、粒子順序錯誤，一律 fail-closed 且不
刪除現場。

每次從上一個合法 checkpoint 恢復後，最多前進 plan 指定的 interval sweeps；若尚未 terminal
便先發布新 generation/latest，再立即原子更新 RUNNING progress，budget pause 才轉 PAUSED。
因此 off-boundary pause 不會漏掉後續 periodic checkpoint。KeyboardInterrupt 盡力建立
checkpoint、累計 checkpoint bytes；只有 batch 已建構且可序列化、checkpoint generation
成功發布時才標 PAUSED，不建立 failure artifact。若 request factory／ProductionBatch
建構尚未完成，或 checkpoint 發布失敗，則保留 RUNNING，不建立虛假的可恢復 checkpoint；
operator 必須以 resume=True 重試，並由合法 generation 恢復，或在沒有 generation 時重新
建構。一般 Exception 才建立不含絕對路徑與 secret 的 immutable failure JSON。恢復前仍須重新執行相容性檢查；
任一關鍵 hash 不符即拒絕舊 checkpoint，不得混用不同 input、method、geometry 或 seed policy。

run plan schema `2.1.0` 的情境列順序固定為
`analysis_region_arrival_utc_site_material_receptor_scenario_v1`，並先以
`analysis_region_id`／到達 UTC 奈秒分 execution group；group 小於 shard 上限也不能與下
一組合併。2.1 plan 另保存 full 或 pilot stratified selection binding；2.0 plan 沒有此欄位
時按 full 唯讀相容。這是 I/O locality 與可恢復邊界，不是物理排序假設。one-process-per-shard 由
`lbt run-shard` 依 immutable plan 的 pilot／formal 模式呼叫同一套 CPU/NumPy request
factory；formal 會先重驗 config、manifest bindings 與 strict inventory，不能由 plan 標籤
繞過 release gate，也不會以最近值、零值或其他 forcing fallback 繼續。

## 8. 驗證與發布

每個 run 依序完成：

1. `lbt-validate-run`：schema、ID、time、status、event、row count、checksum、NaN/QC、scenario coverage。
2. `lbt-aggregate`：raw counts、有效分母、KDE/HDR、sensitivity、foreign-local crossing 與 paired-UTC cross-site overlap，不修改 trajectory shards；跨站比例按原站有效 members 正規化。
3. `lbt-validate-aggregate`：質量、邊界弧長、raster sum、bandwidth、bootstrap、failure density。
4. 第二次 publish dry-run，確認 source/destination 清單一致。
5. 傳至 `.incoming/<run_id>`，在遠端重驗 manifest/checksum後原子發布。
6. 寫 `release_manifest.json` 與上游／下游 impact map。

本機 scratch 不自動刪除。只有在正式發布、備份與 checksum 均由使用者確認後，才依明確 `run_id` 另行執行可復原的清理流程。

## 9. 故障分類

| 類型 | 處理 |
|---|---|
| authentication/path | 不反覆猜密碼；由資料管理者提供已認證環境或 inventory |
| input schema/time | manifest 外 schema/checksum/UTC 改變時停止受影響 domain/month；已知缺口則回到 reconstruction 或 gap-safe selector，不要求供應者補資料，也不在 runtime 臨時外插 |
| disk quota | 停止啟動新 shard，保留完整 checkpoint；調整 output/scratch 後續跑 |
| NFS I/O wait | active write 移至本機 scratch，單一 publisher；不並行灌 NAS |
| numerical failure | 保存 particle/scenario/step/forcing/event 診斷，以相同 seed 最小化重現 |
| code/config change | 新 run ID；舊 checkpoint 不相容，不在原 run 上覆寫 |
