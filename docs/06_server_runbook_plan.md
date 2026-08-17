# SERVER 執行手冊規劃

## 1. 適用範圍

本文件定義未來程式完成後的 SERVER 部署、preflight、pilot、正式 batch、checkpoint、QC 與發布程序。標示為「預定 CLI」的命令在目前 `planning` 狀態尚不可執行；正式實作時需同步更新 README 與此文件。

本次規劃環境嘗試以唯讀 SSH 連線時，因沒有可用的 public key 或互動式密碼代理而未完成認證。應由使用者在既有已登入終端執行下列唯讀盤點；不得把密碼、private key 或 token 寫入 repository 或聊天紀錄。

## 2. 預定路徑變數

```bash
export LBT_PROJECT_ROOT=/home/mustlab/Workspace/Lagrangian-Ensemble-Backtracking
export OCM_NATIVE_ROOT=/home/mustlab/data/OCM-Preprocessed-Data/preprocessed/ocm_native
export OCM_SURFACE_ROOT=/home/mustlab/data/OCM-Preprocessed-Data/preprocessed/ocm_surface
export NWW_ANALYSIS_ROOT=/home/mustlab/data/NWW-Preprocessed-Data/preprocessed/available_samples_v1/nww3_analysis

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
- 嚴格遞增與唯一 UTC。
- 實際 start/end、間距、缺口與跨月銜接。
- array shape/dtype 與 metadata 相符。
- OCM native/surface pair 及 NWW target grid/time 相容。
- OCM 必要欄位與 NWW Hs/fp/DP/mask/QC 的存在及有限率。

### 3.3 容量與檔案系統

```bash
df -hT "$OCM_NATIVE_ROOT" "$NWW_ANALYSIS_ROOT" "$LBT_OUTPUT_ROOT" "$LBT_SCRATCH_ROOT"
findmnt -T "$LBT_OUTPUT_ROOT"
findmnt -T "$LBT_SCRATCH_ROOT"
```

若 output 在 NFS/NAS，checkpoint 與 active shard 優先寫本機 scratch；完成 checksum 後以單一 publisher 傳到同一遠端檔案系統的 `.incoming/<run_id>/`，最後原子改名。不得讓下游看到半套正式 run。

## 4. 環境建立（預定）

```bash
cd "$LBT_PROJECT_ROOT"
export UV_CACHE_DIR="$LBT_UV_CACHE_ROOT"
export UV_PROJECT_ENVIRONMENT="$LBT_SCRATCH_ROOT/venv"
export MPLCONFIGDIR="$LBT_MPL_CACHE_ROOT"
export PYTHONDONTWRITEBYTECODE=1

uv sync --frozen --python 3.12 --managed-python
uv run python3 -m pytest -q
```

實作時鎖定具體 Python patch 版本並保存於 run manifest。Numba、NumPy 與 SciPy 版本變更可能改變浮點/JIT 行為；正式 release 只使用 `uv.lock`，不在 batch 中臨時更新套件。

## 5. 預定 CLI 流程

以下名稱是實作目標，不代表目前已有命令。

### 5.1 Preflight

```bash
uv run lbt-preflight \
  --config configs/lagrangian_backtracking.server.yaml \
  --years 2024 2025 \
  --output "$LBT_SCRATCH_ROOT/preflight/input_inventory.json"
```

Preflight 必須唯讀，不建立 forcing 副本。輸出至少包含 path token、schema、月份、time gap、array bytes、coverage、unit/direction decision、CRS/mesh QC、預估 working set 與輸出空間。

### 5.2 單日 smoke test

```bash
uv run lbt-run \
  --config configs/lagrangian_backtracking.server.yaml \
  --domain houwan_nmmba_cache_v3 \
  --scenario-manifest configs/trials/single_receptor_single_day.json \
  --output-root "$LBT_SCRATCH_ROOT/trials" \
  --label TRIAL-single-day
```

試跑只能寫 `TRIAL` namespace，圖面與 metadata 必須含 trial 標記。單日結果只驗證 I/O、速度、事件與資源，不得作為 2024-2025 科學成果。

### 5.3 代表性 pilot

```bash
uv run lbt-run \
  --config configs/lagrangian_backtracking.server.yaml \
  --scenario-manifest configs/pilot/representative_7_14_day.json \
  --output-root "$LBT_SCRATCH_ROOT/pilots" \
  --checkpoint-root "$LBT_SCRATCH_ROOT/checkpoints"

uv run lbt-validate-run "$LBT_SCRATCH_ROOT/pilots/<run_id>"
uv run lbt-benchmark-report "$LBT_SCRATCH_ROOT/pilots/<run_id>"
```

Pilot 報告需以五站點各固定 10,000、A 區 20,000、全案 50,000 個基礎 scenarios，外推各候選 `M` 與 experiment case 數的 particle-step、wall time、CPU、RAM、read bytes、trajectory bytes、event bytes、checkpoint bytes 與 NFS publish time；並比較 7/14/30/60 日 horizon 及貢寮／龜山島 20/25/35 km local boundary。benchmark 用於衍生最小收斂 `M`、horizon、shard、並行度與儲存策略，不得據此把任一站完整交叉改回 1,000 或把五站合併為 10,000。

### 5.4 正式 batch

正式 run 只能使用 `status=approved` 的 config、behavior、local-domain、每站 20／全案 100 receptor、每站 arrival-time、每站 10,000／全案 50,000 情境 coverage 與 member-convergence manifests：

```bash
uv run lbt-run \
  --config configs/releases/lagrangian_backtracking_2024_2025_v1.yaml \
  --output-root "$LBT_SCRATCH_ROOT/runs" \
  --checkpoint-root "$LBT_SCRATCH_ROOT/checkpoints"
```

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

恢復前重新執行相容性檢查；任一關鍵 hash 不符即拒絕使用舊 checkpoint。不得為了省時間混用不同 input、method、geometry 或 seed policy。

## 8. 驗證與發布

每個 run 依序完成：

1. `lbt-validate-run`：schema、ID、time、status、event、row count、checksum、NaN/QC、scenario coverage。
2. `lbt-aggregate`：raw counts、有效分母、KDE/HDR、sensitivity，不修改 trajectory shards。
3. `lbt-validate-aggregate`：質量、邊界弧長、raster sum、bandwidth、bootstrap、failure density。
4. 第二次 publish dry-run，確認 source/destination 清單一致。
5. 傳至 `.incoming/<run_id>`，在遠端重驗 manifest/checksum後原子發布。
6. 寫 `release_manifest.json` 與上游／下游 impact map。

本機 scratch 不自動刪除。只有在正式發布、備份與 checksum 均由使用者確認後，才依明確 `run_id` 另行執行可復原的清理流程。

## 9. 故障分類

| 類型 | 處理 |
|---|---|
| authentication/path | 不反覆猜密碼；由資料管理者提供已認證環境或 inventory |
| input schema/time | 停止受影響 domain/month，建立差異報告，不跨缺口外插 |
| disk quota | 停止啟動新 shard，保留完整 checkpoint；調整 output/scratch 後續跑 |
| NFS I/O wait | active write 移至本機 scratch，單一 publisher；不並行灌 NAS |
| numerical failure | 保存 particle/scenario/step/forcing/event 診斷，以相同 seed 最小化重現 |
| code/config change | 新 run ID；舊 checkpoint 不相容，不在原 run 上覆寫 |
