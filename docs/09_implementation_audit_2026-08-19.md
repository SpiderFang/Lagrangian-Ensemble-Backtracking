# 實作與 SERVER 驗證稽核（2026-08-19）

## 1. 結論

本專案已由純規劃狀態進入「可執行 reference core、可進資料修復與 pilot」階段。本機與
SERVER 均可由 `uv.lock` 重建環境、執行 50 項測試、產生 constant-flow backward
synthetic shard，並獨立驗證 checksum、CSR、Parquet、時間方向與停止狀態。現階段不可
啟動 2024–2025 正式 baseline；原因不是尚待使用者選擇科學設計，而是正式輸入、expanded
A 區、資料衍生 manifests、數值收斂與 production backend gate 尚未完成。

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
專案 `uv.lock`；測試結果為 `50 passed`，synthetic shard 的獨立 validator 回報
`valid=true`。唯讀全期報告保存在 SERVER：

```text
/home/mustlab/Workspace/Lagrangian-Ensemble-Backtracking/work/preflight-20260819.json
```

該報告只保存環境變數 path token 與相對路徑，不保存 SSH 密碼。報告涵蓋 4 domains ×
24 months × OCM/NWW 兩類產品，共 192 筆 inventory；共 188 項 finding，未出現 schema
major、必要陣列、實際 NPY header shape/dtype、domain ID 或 OCM/NWW time-axis mismatch：

| finding | 數量 | 解釋 |
|---|---:|---|
| `STATUS_NOT_READY` | 96 | 四個 domain 的 24 個 NWW analysis 月份全為 `trial_ready`，不是正式要求的 `ready` |
| `NWW_TIME_GAP_EXCEEDED` | 68 | 四個 domain 各 17 個月超過 1.5 h，最大 gap 72 h；四域月份集合相同 |
| `CACHE_KIND_REJECTED` | 16 | 四個 domain 的 OCM 202503、202505、202507、202511 為 `standard_partial_month` |
| `CROSS_MONTH_TIME_GAP_EXCEEDED` | 8 | 四個 domain 的 202403→202404 為 26 h、202404→202405 為 25 h |

超過 1.5 小時的 17 個月內時間軸為：202401–202406、202409–202412、202502、202503、
202505–202507、202510、202511；此外還有上述兩個跨月界缺口。這些缺口在 OCM 與 NWW
的對位時間軸上共同存在；執行器必須於缺口停止，不能內插 24–72 小時或用最近時次填補。
因而「目錄中有兩年 24 個月份」不等於
「可無條件執行連續兩年正式軌跡」。上游可選擇補齊缺時資料，或建立經審查的 coverage/
arrival-time 排除 manifest；本專案不替上游偽造缺值。

## 4. 正式批次前的未完成閘門

### 4.1 必須先處理的輸入與幾何

1. 將 NWW analysis 由 `trial_ready` 升版為具方向證據與完整 QC 的 `ready`；波向慣例目前
   仍是 inferred/adopted，未達正式 `confirmed/approved`。
2. 釐清並修復上述 17 個月份的 2–72 小時缺口；若 4 個 partial OCM 月份不可補齊，需
   形成明示 coverage 與排除理由，而非沿用 `standard_month` 名義。
3. 產生不覆寫現行 v3 的 expanded A flow domain；25 km baseline 與 35 km sensitivity
   對 OCM native、OCM surface、NWW analysis 都須有至少兩個共同有效格點的 outer margin。
4. 由實際 native mesh 產生 domain、static ocean、local-domain、open-boundary 與 receptor
   manifests。貢寮、龜山島各保留 20 receptors，各自 10,000 基礎情境，僅共用 A forcing
   與 outer boundary；local domains 可重疊。
5. 由有效 forcing coverage 產生五站各 50 個 arrival UTC；不能選在需跨未核准缺口的
   區間，且 arrival 的完整回溯窗必須另行檢查。

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

在不放寬 gate 的前提下，可立即使用已部署 reference core 進行單月/完整時段 no-Stokes 或
已知缺口外的資料接線測試、產生實際 mesh/local/open-boundary/receptor manifests、抽樣核對
OCM wetdry/Kz 與 NWW 方向，並準備 expanded A 上游產品。正式 config 必須繼續以
`--formal-release` fail closed；不得把 synthetic smoke 或 current-v3 pilot 描述為計畫成果。
