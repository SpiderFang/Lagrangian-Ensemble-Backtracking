# Lagrangian Ensemble Backtracking

本專案以 CWA-OCM 三維海流、CWA-NWW3 波浪衍生的 Stokes 漂流、向下沉降與次網格擴散，從受體位置與到達時間向過去建立海洋廢棄物的條件式來源足跡。

本頁是入口與狀態摘要；完整文件分類與閱讀路線見[文件總入口](docs/README.md)，實作狀態見[實作狀態](docs/implementation_status.md)，複雜 CLI 與 SERVER 操作見[CLI 參考](docs/operations/cli_reference.md)。

> 目前定位：工程執行流程、輸入／輸出契約與測試已可供持續維護；這不代表全期真資料批次或正式科學成果已完成。

## 1. 用途與目前狀態

專案處理完全沉沒、向下沉降的海廢代理，從受體位置與到達時間向過去建立結果。結果只能稱為「條件式來源足跡」或「相對來源權重」。

| 能力 | 已可做 | 仍待正式驗證 |
|---|---|---|
| 設定與輸入 | 設定、來源 manifest、OCM／NWW3 格式檢查與 UTC／幾何綁定 | accepted products、gap-safe manifest 與全期輸入清單 |
| 物理與軌跡 | 正向物理速度、signed-time RK4、獨立擴散、巢狀邊界、trajectory shard 與事件 | 解析解、時步／成員收斂、真實敏感度與獨立觀測驗證 |
| 續跑與產品 | `execution checkpoint`、亂數延續、固定格式、checksum、聚合統計基礎 | 發佈整合、真實敏感度與正式驗證證據 |
| 報告與部署 | 報告前置唯讀檢核、統計介面、pilot／synthetic workspace | 主線報告建立、完整報告渲染與正式 SERVER 科學發布 |

工程測試、pilot 與 synthetic 只證明可重現的介面或計算契約；它們不等於全期真資料研究或正式科學成果。

本機已有 20 粒子、203 筆觀測的真資料 pilot 示範；它可證明示範鏈路，不代表五站全期研究、參數收斂、獨立觀測驗證或正式來源結論。SERVER 現況以當次部署／run records 與唯讀 preflight 為準，本頁不代替現場紀錄。

## 2. 五站四區的沉降情境

四個流場資料／外層流場區域為 A–D；五個研究站點各自保留 `study_site_id`、受體、事件與統計，不因共用流場資料而合併成一站。

| 研究站點 | 流場區域 | 說明 |
|---|---|---|
| 貢寮 | A | 與龜山島共用 A 區流場資料；正式版須通過南擴與共同有效格網檢核 |
| 龜山島西側 | A | 與貢寮共用 A 區流場資料，但局部區域、受體與統計獨立 |
| 新竹外海 | B | 使用 B 流場區域 |
| 後灣海生館 | C | 使用 C 流場區域 |
| 連江 | D | 使用 D 流場區域 |

每站完整基礎設計是 `10 種材質／形狀 × 20 個三維受體 × 50 個到達時間 = 10,000` 個情境；若每個情境都採相同的成員數 `M`，五站執行量才可寫成 `50,000×M`。各情境成員數不同時，總量應寫成 `Σ M_s`；`M` 是隨機成員數，不是額外情境因子。實驗案例（例如 `no_stokes` 或擴散敏感度）另行編號。

所有本專案沉降速度均為負值、物理方向以向上為正；不允許上升物性，也不對完全沉沒物體加入風壓效應。缺少密度、阻力、再懸浮參數時，不宣稱已完成沉積—再懸浮動力。

貢寮與龜山島共用 A 區流場資料與最外層開放邊界，但各自保存局部區域入口、跨站診斷、情境身分與統計分母；穿越另一站局部區域不會轉移粒子所屬站點。A 區正式使用的擴張區域仍須由 OCM／NWW3 共同有效格網與邊界餘裕證明，不能只改名稱或範圍框。

每個受體×到達配對的實際初始深度來自到達 UTC 的已驗證動態紀錄；不以模板深度代替所有到達時間。投影座標、公尺距離、步長限制、統計網格與軌跡計算使用公尺制座標，圖面可將公尺座標轉為經緯度顯示，但經緯度不進入粒子物理運算。

## 3. 輸入、速度欄位與輸出版本

正式流程只讀已驗收產品，不讀 raw NetCDF 或轉移封存檔，也不以零值、最近值或未登錄的跨月／跨網格外插補缺。

| 輸入 | 必要內容 | 用途 |
|---|---|---|
| OCM native schema `3` | `ocm_native/<flow_domain_id>/grid` 與 `months/YYYYMM` 的 mesh、`hvel`、垂向速度、`zcor`、`elev`、wet/dry、diffusivity | 三維 current、垂向與擴散 |
| OCM surface schema `3` | `u_surface_mps`、`v_surface_mps`、`surface_z`、`eta_m`、mask、QC | 到達時間篩選與表層篩選器；不取代原生三維流場資料 |
| NWW3 analysis schema `1` | Hs、peak frequency、wave direction、mask、QC 與完整 UTC 軸 | Stokes 漂流與波浪檢核 |
| 版本化 manifest／設定 | geometry、receptor、arrival、dynamic pair、material、seed、來源綁定 | 建立可稽核的 pilot／正式執行 |

受體×到達配對的實際深度與三維初始條件來自到達 UTC 的 OCM 原生動態紀錄；OCM surface 只負責到達時間篩選，不代表配對的來源深度或三維流場資料。

### Observation 速度紀錄

`Observation` 同時涵蓋既有定期輸出紀錄與狀態／終止紀錄。只有當 Observation 的位置與 UTC 和已取得的步首參考取樣完全相同，才附上速度；沒有同點取樣時明示狀態並保留缺值，不借用 RK4 階段、內部步或其他位置的速度。這些是取樣值，不是位移除以時間的平均速度；位置單位為 m，時間為 UTC 奈秒，速度單位為 m/s；逆向積分不另行反號。

九個固定欄位依四類如下；所有速度單位均為 m/s：

| 分類 | 欄位 | 意義 |
|---|---|---|
| 總速度 | `total_u_mps`、`total_v_mps`、`total_w_mps` | 確定性海流、Stokes 與沉降合成後的正向物理速度 |
| OCM 海流 | `ocm_u_mps`、`ocm_v_mps`、`ocm_w_mps` | OCM 三維 current；`u/v/w` 為東／北／向上 |
| Stokes 水平 | `stokes_u_mps`、`stokes_v_mps` | NWW3 衍生的水平 Stokes drift |
| 沉降 | `settling_w_mps` | 以向上為正的垂向沉降速度；本專案沉降為負 |

`total_u = ocm_u + stokes_u`、`total_v = ocm_v + stokes_v`、`total_w = ocm_w + settling_w`，由明定容差驗證。Stokes 垂向與沉降水平不屬於欄位，不能假造。`CombinedMonthForcing` 的正式有效樣本提供完整分項；明確關閉 Stokes 可記有效的 0，自然海況計算結果也可能是有效的 0，但缺少 NWW3 波浪資料必須保留缺值並帶非零 QC，不能記成 0。只有總速度的 synthetic callback 標為 `total_only`，不代表 OCM／NWW3 來源；`not_sampled`、缺值與無效狀態分開保存。

### 版本與舊格式相容性

| 產品 | 新 writer | 舊格式相容性 |
|---|---|---|
| trajectory shard | `3.0.0`：固定位置、狀態、事件、環境與速度資料檔；實際產品含 `.npy`、`.parquet`、`.json`，並有 dtype／count／checksum 驗證器 | `1.0.0`、`2.0.0` 可唯讀；舊欄位沒有速度資料，不自動補速度。正式報告可使用 v2 或 v3，v1 不得作正式垂向驗證證據；同一執行混用 v2／v3 保守拒絕 |
| execution checkpoint | `2.2.0`：保存 Observation 11 個速度欄位、環境、事件、triangle hint 與 PCG64DXSM RNG | `2.0.0`／`2.1.0` 以固定舊欄位集合唯讀；缺少速度時回傳 `NOT_SAMPLED`／`None`，不受最新資料類別污染 |

缺值與版本界線是驗證證據的一部分：舊成果不會自動升格為含速度的新成果，synthetic／pilot 也不會升格為正式科學證據；事件與狀態保留 `DATA_GAP`、`NUMERICAL_FAILURE`、邊界事件、可用性與失敗原因，無效取樣必須有非零 QC，不能用靜水或終止位置掩蓋失敗查詢。trajectory shard 與 checkpoint 都是不可覆寫、以檔案大小／SHA-256 驗證的工程產品；checkpoint 只保存可恢復的粒子狀態、觀測、事件、triangle hint 與 RNG，大型流場資料、網格與幾何由相同來源綁定的 request factory 重建。

輸出分析前應先確認 manifest 的執行身分、schema、來源綁定、粒子／觀測／事件計數與 checksum，再解讀位置、停止類型或統計比例；不同版本、研究站點或實驗案例不能只因欄位名稱相同就直接合併。`observations.csv` 的速度欄位若為空，代表該點沒有可證明的同點速度資料，不表示靜水；有效的 0 可來自明確的 `no_stokes` 或自然海況計算結果，缺波浪、無效或未取樣情形必須依狀態與 QC 判讀，不能用 0 代替。

## 4. 快速安裝與 synthetic smoke

以下 smoke 不需真資料。必要條件是 Python `3.11+`、已安裝的 `uv` 與可執行 POSIX shell；`uv.lock` 是安裝依據。指令先建立暫存父目錄，再把尚不存在的 child 指定為輸出，不會把 `mktemp -d` 建立的父目錄直接當成輸出根目錄。

```bash
uv sync --frozen
LBT_SMOKE_PARENT=$(mktemp -d)
uv run pytest -q -p no:cacheprovider
uv run lbt synthetic-smoke --output "$LBT_SMOKE_PARENT/synthetic-smoke-v1"
uv run lbt validate-shard "$LBT_SMOKE_PARENT/synthetic-smoke-v1"
```

`synthetic-smoke` 只驗證 CLI、engine、trajectory I/O、Parquet、manifest 與 checksum；其 metadata 會標示非科學結果，不能替代已驗收 OCM／NWW3 或正式驗證。

若只要檢查速度契約，可閱讀 `tests/test_velocity_recording.py`；若要檢查舊 checkpoint 的讀取與亂數延續，閱讀 `tests/test_checkpoint_execution.py`。這些測試刻意不執行真實模型、不下載資料，也不修改既有 pilot 結果。

## 5. 實際資料與 SERVER 使用入口

正式執行所需根目錄由環境變數或 CLI 參數注入，例如 `OCM_NATIVE_ROOT`、`OCM_SURFACE_ROOT`、`NWW_ANALYSIS_ROOT`、`LBT_OUTPUT_ROOT`、`LBT_SCRATCH_ROOT` 與 `LBT_CHECKPOINT_ROOT`；本專案不在程式或文件硬編碼私有 SERVER 路徑。實際資料依序執行唯讀 `preflight` → strict `inputs-build`／`inputs-validate` → release 設定來源綁定 → `run-create` → `run-shard`／checkpoint resume → `run-reconcile` → `validate-run` → aggregate／report validator；完整參數與外置 checkpoint root 規則見[CLI 參考](docs/operations/cli_reference.md)，容量、鎖、部署與資料同步見[SERVER 執行手冊](docs/operations/06_server_runbook_plan.md)及[Git 部署與資料同步手冊](docs/operations/git_deployment_and_data_sync.md)。

本機 Git 是開發來源；SERVER 只部署核定且可追溯的 commit。Git、上游大型資料、執行工作區、trajectory、checkpoint、scratch 與發佈輸出分開管理；部署同步需核對 commit、已追蹤檔案、checksum、dirty flag、seed 與輸入清單。未完成該次 preflight 前，不啟動五站 `50,000×M` 正式 batch。

正式輸入的每個月份、UTC 時間軸、schema、單位／方向、mask、缺時形狀、geometry、容量與權限，都應在當次 preflight 留下可機讀紀錄；已知時間缺口只能採核准重建或缺口安全到達視窗，執行流程不臨時外插，不以最近值或零值補資料。

## 6. 文件導覽

| 主題 | 文件 |
|---|---|
| 文件分類、五條閱讀路線與完整原路徑對照 | [文件總入口](docs/README.md) |
| 當前實作、版本與正式驗證證據缺口 | [實作狀態](docs/implementation_status.md) |
| CLI、執行生命週期、輸入、release、checkpoint | [CLI 與執行參考](docs/operations/cli_reference.md) |
| SERVER 容量、鎖、執行與發布 | [SERVER 執行手冊](docs/operations/06_server_runbook_plan.md) |
| 原始碼模組關係與資料流程 | [互動式程式架構圖](docs/source_code_architecture_map.html) |

## 7. 開發檢查與交付界線

每次變更送審前至少檢查相關 pytest、Ruff、`git diff --check`、schema／checksum 資料拓撲與文件相對連結。新增模組時同步 architecture-map catalog；架構圖的產生與檢查應使用既有產生器及固定輸出流程。

測試結果要區分核心物理不變性、checkpoint／I/O round-trip、下游相容性與正式驗證證據。前兩者通過不能代替後兩者；報告中的有效分母與缺值狀態也不能由測試樣本外推到全期真資料。

任何程式、schema 或輸入方法變更，都應同時更新對應 docstring、需求追溯、版本說明與相容性測試。若變更會影響既有結果，建立新執行／schema 或明示 migration，不在原輸出上覆寫。

速度、checkpoint、正式驗證證據缺口與報告邊界以[實作狀態](docs/implementation_status.md)為準。需求 `REQ-007` 的「關閉單項敏感度、單位檢核」是持續驗收要求；速度單元測試不能取代正式科學敏感度驗收。

閱讀順序建議是先看本頁的限制，再看資料契約與科學方法，最後依 CLI 參考或 SERVER 執行手冊操作。歷史 audit、Phase／Slice 文件可追溯決策，但不得用其中的舊「下一步」句子推定今天的部署或資料狀態。

## 8. 方法與不採用範圍

1. 本專案沿用 BayTrace 可對應的工程思路：CPU SoA／batch／chunk、每粒子可重現亂數、triangle hint、可暫停 engine 與可驗證 I/O；不承諾整包 tracker output 的格式或物理範圍，也不把 BayTrace 的工程參照當成本案科學驗證。
2. 在沒有先驗、似然、調查努力量與獨立觀測前，任何逆向結果都不可稱為絕對來源機率、法律責任或因果歸因；報告必須保留原始分子／分母、有效分母、低樣本、零分母、`DATA_GAP` 與 `NUMERICAL_FAILURE`，不得將不可用狀態改成 0。
3. 確定性海流、Stokes 與沉降沿正向物理速度積分，時間方向由 signed-time step 表示；隨機擴散獨立於 RK4，不能插入 RK4 階段。材質名稱與沉降格點是條件式敏感度代理，不是來源先驗；缺少密度、阻力與再懸浮參數時，不宣稱完整沉積—再懸浮動力。方法理由與待驗證項目見[科學方法與驗證](docs/foundation/03_scientific_method_and_validation.md)。

## 9. B 區海岸預覽（既有示範）

`scripts/build_pilot_coastline_preview.py` 可對既有 B 區 r2 preview 重繪海岸總覽與 H1–H5 局部圖；它只讀 preview CSV／JSON、已核對幾何與海岸 GeoJSON，不讀流場資料或重跑模型。既有示範保存 20 粒子、203 筆模型紀錄，仍只是工程／展示資料。

預設 `legacy` 保留兩張水平 PNG；`--style baytrace` 另輸出水平圖、`depth_age.png`、`terminal_counts.png`、README 與 manifest，並拒絕覆寫既有輸出目錄。呈現參照是 BayTrace v4.7.2 套件 `scripts/analyze_cases.py` 的 `plot_case`；本專案的垂向與停止診斷是額外內容，不擴張 BayTrace 語意。完整命令、來源與限制見[單站 pilot 計畫的獨立海岸底圖重繪](docs/operations/pilot_run_plan.md#獨立海岸底圖重繪)。這些計畫、腳本與測試是持久化的 pilot／補圖入口；補圖不改原 preview 的資料、圖層或採樣序列，只在新的 sibling output directory 產生圖檔與 manifest，局部圖保留實際公尺刻度與 panel identity，缺值留空，不能以海岸圖的完整性替代軌跡／流場資料 QC。

舊文件中的 Phase／Slice 名稱是設計脈絡，不是自動更新的現在狀態；若與本頁、[實作狀態](docs/implementation_status.md)、程式碼或當次紀錄不一致，以較新的可驗證程式／manifest／測試證據為準，無法確認的 SERVER 事實只引用現場紀錄。本頁不保存 SERVER 密碼、私有絕對路徑、未提交的大型資料或模型結果；實際部署依已授權環境的唯讀盤點與核准流程辦理。
