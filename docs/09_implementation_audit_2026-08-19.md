# 實作與 SERVER 驗證稽核（2026-08-19）

> **2026-08-20 更正。** 本文件原先把 NWW `trial_ready`、OCM partial month 與既有
> analysis 時間缺口解讀成必須等上游／供應者修復的正式阻擋。研究團隊已裁決現有
> OCM/NWW 為「2024–2025 全部可得資料」的完整正式母體，且供應者與額外 metadata 均
> 不可能再取得；NWW 方向也已由兩個獨立颱風事件定案。以下第 3、4 節已依重新盤點的
> canonical 時間軸改寫，舊 preflight JSON 只作歷史證據，不代表現行正式判準。

## 1. 結論

本專案已由純規劃狀態進入「可執行 reference core、可進資料重建與 pilot」階段。本機與
SERVER 均可由 `uv.lock` 重建環境、執行測試、產生 constant-flow backward
synthetic shard，並獨立驗證 checksum、CSR、Parquet、時間方向與停止狀態。現階段不可
直接宣稱 2024–2025 baseline 已完成；原因不是缺資料或尚待使用者選擇，而是 OCM
reconstruction、NWW full-hour analysis、expanded A 區、資料衍生 manifests、數值收斂與
production backend 尚待由既定方法產製及驗證。

使用者不需再提供額外科學數據或任意指定 `M`、Kh/Kz、dt、回溯期與 shard 大小；這些
欄位依設計文件由實際資料 QC、代表性 pilot、dt/member/horizon convergence 與 benchmark
衍生。任何未通過 gate 的值不得以方便執行為由自行填入正式 release config。

## 2. 已完成的可執行範圍

| 層級 | 已完成內容 | 驗證狀態 |
|---|---|---|
| 設定與 preflight | Pydantic 跨欄位契約、canonical config hash、4 flow domains／5 sites／每站 10,000／全案 50,000 計數、正式 fail-closed gate、OCM/NWW metadata/time/schema/status inventory | 本機 fixture 與 SERVER 192 筆月份 inventory 已執行 |
| 幾何與受體 | AEQD 投影、densified bbox、anchor local domain、deterministic maximin、persistent-wet 5×4 受體選擇核心、polygon crossing | 合成 geometry/receptor 測試通過；SERVER 正式 manifests 待產生 |
| 原生 forcing | SCHISM tri/quad 可追溯切分、uniform-bin locator、barycentric weights、OCM x/y/z/t 保守內插、NWW mask-aware/circular 內插、跨月 provider | 線性解析場、乾 face、無外插、NumPy/Numba OCM 內插對照通過 |
| 物理與積分 | signed-time RK4、常數 Brownian split、adaptive dt、Smagorinsky 候選、有限/深水 Stokes、10 種浮沉行為 | 常流、四 stage、Brownian variance、dispersion residual、深水極限及方向測試通過 |
| 邊界與事件 | own-local first exit、foreign-local 非終止 enter/exit、共用 A outer stop、顯式 open-water/海岸分類、海面／海床政策、forcing/data/max-age/numerical stop | 步內 crossing、重合邊界、海岸、foreign endpoint 去重、RK stage 域外 terminal recovery 測試通過 |
| Scenario/member | 五站完整交叉、穩定 scenario/particle ID、SHA-256 128-bit seed、scenario shard、`scenario×M` reference executor | shard 大小、manifest 列順序與 worker-independent identity/seed 測試通過 |
| 輸出與恢復 | 不可變 CSR-like trajectory arrays、particle/event Parquet、原子發布、checksum、formal metadata gate、binding/checksum checkpoint | round-trip、破損拒絕、相容/不相容 checkpoint 測試通過 |
| 聚合 | 2D KDE、50/75/90% HDR、open-boundary arclength histogram、unique-particle pathway、秒數守恆 residence、停止比例、跨站條件比例 | 正規化、分母、跨格線時間分配與去重測試通過 |
| CLI | `config-check`、`preflight`、`behavior-manifest`、`synthetic-smoke`、`validate-shard` | 本機與 SERVER 端到端執行通過 |

實作過程另修正四個若只做理想常流測試容易遺漏的問題：polygon exit 原先未區分海岸與
開放水域；RK stage 可能先落到無效 native mesh 而漏記已發生的 coast/outer crossing；
fraction=0 終止會產生同時刻重複 observation；foreign-local 的步末／下一步步首同一交點
會產生假 exit。四者均已有針對性回歸測試。

## 3. SERVER 可重建證據

部署位置為 `/home/mustlab/Workspace/Lagrangian-Ensemble-Backtracking`，不包含本機 `.git`、
`.venv`、大型 `data/` 或任何認證資料。SERVER 安裝使用 uv 0.12.5、CPython 3.12.13 與
專案 `uv.lock`；當次測試結果為 `50 passed`，synthetic shard 的獨立 validator 回報
`valid=true`。唯讀全期報告保存在 SERVER：

```text
/home/mustlab/Workspace/Lagrangian-Ensemble-Backtracking/work/preflight-20260819.json
```

該報告只保存環境變數 path token 與相對路徑，不保存 SSH 密碼。舊報告涵蓋 4 domains ×
24 months × OCM/NWW 兩類產品，共 192 筆 inventory；其中 188 項 finding 的原始計數如下，
但現行解釋已更正：

| finding | 數量 | 解釋 |
|---|---:|---|
| `STATUS_NOT_READY` | 96 | 舊規則只接受 `ready`；現行 available-data contract 接受 `trial_ready`，並明示不宣稱 provider-confirmed best forecast cycle |
| `NWW_TIME_GAP_EXCEEDED` | 68 | 舊 analysis 沿用 gappy OCM target times；NWW native 實際完整，不是波浪原始資料缺時 |
| `CACHE_KIND_REJECTED` | 16 | 舊規則拒絕 partial month；現行視為全部可得資料的一部分，原標籤與 coverage 照實保存 |
| `CROSS_MONTH_TIME_GAP_EXCEEDED` | 8 | 舊月份邊界檢查未先建立全期 canonical 軸；缺口現改由精確 missing-step inventory 與 reconstruction gate 處理 |

2026-08-20 重新以跨月份 stable sort／`prefer_last` 計算後，四個 OCM domains 的 canonical
時間軸完全相同：raw 17,196 列、唯一 17,124 列、去除 72 個重複 UTC；相鄰最大間距為
50 小時，而不是舊月內報表所稱 72 小時。全期 17,544 個理論逐時時次共缺 420 個，其中
419 個位於內部、另 1 個是 2024-01-01 00:00 的左側邊界；內部缺口為 33 個單一缺時、
1 個 23-step、11 個 24-step、2 個 25-step 與 1 個 49-step block。coverage 為 97.606%。

NWW native 24 個月份則恰有 17,544 個唯一逐時 UTC，起訖為 2024-01-01 00:00 至
2025-12-31 23:00、最大間距 1 小時且無缺口。故波浪正式 analysis 可由既有 native 資料與
OCM 靜態格網直接重建完整逐時產品，不需統計補值。OCM 的 420 個缺時則依
`ocm_multivariate_eof_harmonic_state_space_smoother_v1` 做 blocked validation，或使用
gap-safe 分層 arrival windows；兩者都不需外部補件，也不會把所有軌跡在已知缺口停止。

## 4. 正式批次前的未完成閘門

### 4.1 必須由既有資料完成的輸入與幾何工作

1. 由完整 NWW native 產生四個 domain 各 17,544 UTC 的 full-hour analysis；上游
   `trial_ready` 原樣保留並由 available-data contract 接受，不需也不得冒稱 provider 已確認
   best forecast cycle。方向固定採已核定的 `nww3_dp_wnd_two_typhoon_adopted_v1`。
2. 產生 OCM canonical time manifest，對實際 1/23/24/25/49-step gap shapes 進行
   blocked validation；通過者輸出 immutable reconstruction patch 與 forcing members，未通過
   者由 gap-safe arrival/horizon selector 迴避。這是專案內運算工作，不等待上游補資料。
3. 產生不覆寫現行 v3 的 `northeast_taiwan_common_cache_v4_lbt_south_expanded`，候選 bbox
   為 `[121.306315,122.793685,24.480000,25.499156]`；25 km baseline 與 35 km sensitivity
   對 OCM native、OCM surface、NWW analysis 都須有至少兩個共同有效格點的 outer margin。
4. 由實際 native mesh 產生 domain、static ocean、local-domain、open-boundary 與 receptor
   manifests。貢寮、龜山島各保留 20 receptors，各自 10,000 基礎情境，僅共用 A forcing
   與 outer boundary；local domains 可重疊。
5. 由 observed/reconstructed forcing coverage 產生五站各 50 個 arrival UTC；若某種 gap
   reconstruction 未通過，只在 gap-safe 支撐區間內選擇，且 arrival 的完整回溯窗必須另行
   檢查。不得因使用 fallback 而刪除任何年份／季節／潮況 strata。

### 4.2 必須由 pilot 衍生的數值與工程值

1. 由 Kh/Kz 診斷與 well-mixed/敏感度測試核定 baseline diffusivity。
2. 由 dt-halving、boundary-recovery 診斷與主要統計穩定性核定 dt min/max 及輸出間隔。
3. 比較 7/14/30/60 日，選擇最小穩定回溯期與 maximum step count。
4. 依 exit ranking、HDR、median travel time、pathway density 與 bootstrap CI 收斂固定最小
   合格 `M`；軌跡數仍為每站 `10,000×M`、A 區 `20,000×M`、全案 `50,000×M`。
5. benchmark 後固定 scenario shard、checkpoint interval、RAM/I/O 與本機 scratch 配置。

### 4.3 尚待完成的 production/成果層

- 目前有逐粒子 NumPy reference batch 與 Numba OCM 內插核心，尚不是完整 chunked/vectorized
  Numba production engine；須補 active-particle compaction、forcing-window cache、mid-run RNG
  checkpoint、restart/merge 等價與吞吐/RAM benchmark。
- `lbt-run`、完整 run validator、實值 pilot builder、streaming aggregate publisher 尚待上述
  manifests 與 production backend 固定後接通；現有 CLI 不會假裝已能啟動正式全期批次。
- bottom-contact first/repeated density、failure density、bootstrap CI、paired-UTC HDR overlap、
  source–receptor matrix、學術圖表 registry/sidecar 與 known-source synthetic coverage 尚待 G3–G5。

## 5. 可立即進行的工作

在不放寬 gate 的前提下，可立即使用已部署 reference core 產生 canonical/reconstruction
validation、NWW full-hour analysis、實際 mesh/local/open-boundary/receptor manifests，並產製
expanded A 上游產品。NWW 方向無須再次抽樣裁決，只需在每一新版產品做一致性 QC。正式
config 必須繼續以
`--formal-release` fail closed；不得把 synthetic smoke 或 current-v3 pilot 描述為計畫成果。
