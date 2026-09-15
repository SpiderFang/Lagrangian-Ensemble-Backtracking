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
| 向下沉降來源路徑圖 | `source-pathway-v1` 六面板圖、grid／boundary／outcomes sidecar、manifest／checksum validator | 真實五站科學解讀仍須 formal aggregate、收斂與獨立驗證；成果不是完整 report-v1 |

工程測試、pilot 與 synthetic 只證明可重現的介面或計算契約；它們不等於全期真資料研究或正式科學成果。

本機已有 20 粒子、203 筆觀測的真資料 pilot 示範；它可證明示範鏈路，不代表五站全期研究、參數收斂、獨立觀測驗證或正式來源結論。SERVER 現況以當次部署／run records 與唯讀 preflight 為準，本頁不代替現場紀錄。

## 2. 五站四區的沉降情境

四個流場資料／外層流場區域為 A–D；五個研究站點各自保留 `study_site_id`、受體、事件與統計，不因共用流場資料而合併成一站。

| 研究站點 | 流場區域 | 說明 |
|---|---|---|
| 貢寮 | A | 使用 `northeast_taiwan_common_cache_v3` 與 `formal_domain_policy=v3_local20km_20260909_v1`；12.5 km 受體核心、20 km 局部區域，與龜山島共用最外層停止邊界，正式驗收尚待共同邊界餘裕證據 |
| 龜山島西側 | A | 使用同一 v3 A 區海流資料與範圍規範；12.5 km 受體核心、20 km 局部區域，局部區域、受體與統計維持獨立 |
| 新竹外海 | B | 使用 `hsinchu_cache_v3` 流場區域；local domain 仍等於 flow domain，水平受體候選限於 `[120.45, 24.75]` 半徑 12.5 km 核心；24 小時展示 pilot 入口與參數見[紀錄](docs/results/14_hsinchu_2024-01-01_24h_pilot_parameter_record.md) |
| 後灣海生館 | C | 使用 C 流場區域 |
| 連江 | D | 使用 D 流場區域 |

每站完整基礎設計是 `10 種材質／形狀 × 20 個三維受體 × 50 個到達時間 = 10,000` 個情境；若每個情境都採相同的成員數 `M`，五站執行量才可寫成 `50,000×M`。各情境成員數不同時，總量應寫成 `Σ M_s`；`M` 是隨機成員數，不是額外情境因子。實驗案例（例如 `no_stokes` 或擴散敏感度）另行編號。

所有本專案沉降速度均為負值、物理方向以向上為正；不允許上升物性，也不對完全沉沒物體加入風壓效應。缺少密度、阻力、再懸浮參數時，不宣稱已完成沉積—再懸浮動力。

貢寮與龜山島共用 A 區流場資料與最外層開放邊界，但各自保存局部區域入口、跨站診斷、情境身分與統計分母；穿越另一站局部區域不會轉移粒子所屬站點。本期 A 區不南擴，沿用 `northeast_taiwan_common_cache_v3` 的 bbox `[121.306315,122.793685,24.600844,25.499156]`，以 anchor 半徑 12.5 km 的受體核心與 20 km 局部區域建立 `formal_domain_policy=v3_local20km_20260909_v1`，最外層停止邊界維持原 A 區設定。原先 25 km 基準、35 km 敏感度與南向擴張屬歷史規劃，移出本期；不自行加入 15 km 或 23 km case。舊 25 km 幾何、manifest、shard 與 hash 不沿用，必須依新規範重建。現有 v3 三類產品即使有 24 個月份目錄，也仍須由 `OCM native`、`OCM surface` 與 `NWW3` 的共同邊界餘裕證據完成實測驗收；目前程序只涵蓋受體的原生篩選，不能代替 20 km 邊界的三產品、兩共同格點證明，因此可先進行嚴格輸入準備，但正式驗收尚未通過、正式運算尚未開放。舊 `formal_domain_policy=expanded_domain_v1` 只保留舊設定相容讀取，不是本期 A 範圍。

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

月份目錄只是產品分割與延遲載入的索引；執行階段會依每個 OCM／NWW3 產品實際的
`time_utc_ns`，在必要時從相鄰月份尋找 before／after 時間端點。相鄰月份的時間軸連續且
未超過產品允許的最大時間間隔時，才進行合法時間內插；真正的時間缺口仍回傳
`TIME_GAP`，不以零值、最近值或外插補齊。OCM 與 NWW3 會各自依自身時間軸選取端點，
因此同一查詢時刻可以使用不同月份的 OCM／NWW3 原始資料，並在保留原始物理欄位後完成
Stokes 合成。正式運算設定的 `execution.max_resident_forcing_months` 應設定至少為
`2`（對應 manager constructor 的 `max_resident_months=2`），讓跨月兩端可常駐並避免
每個邊界 stage 反覆重載月份；設定為 `1` 仍維持數值正確性，但只適合記憶體受限的測試
或低頻取樣，可能產生月界 I/O thrash。
同月安全查詢會直接沿用 `CombinedMonthForcing`；暖機後 cache stats 的每次普通
no-Stokes sample 只增加一次 OCM hit，Stokes sample 再增加一次 NWW hit。跨月、月尾
或可能有 halo duplicate 的查詢才會記錄必要的 endpoint 探查命中。

OCM native 的垂向取樣在一般水柱內仍要求有效 `zcor` 上下層夾住 query z；針對移動海面，若固定 z 在某一個 before／after 端點高於該端點最高有限 `zcor`，端點可使用最高有效層的速度、垂向速度與 Kz，表示 OCM 最上層控制體的 surface hold。這不是任意最近值外插，也不把 top `zcor` 當成物理海面。所有海面上界查詢共用 `models.py` 的 `SURFACE_BOUNDARY_TOLERANCE_M = 5e-6 m`：只有 `z - eta` 不超過 5 微米時才先夾回 query-time `eta`，讓 endpoint top-layer 支援與 Stokes profile 使用同一表面；超過此尺度仍回傳 `VERTICAL_UNSUPPORTED`／保留原有失敗 QC。這個 5 微米尺度是為涵蓋 checkpoint-8 最大約 `3.367686e-6 m` 的海面邊界定位數值殘差（含浮點與積分／內插）而設，遠小於 OCM 垂向物理層距，並非可任意放大的物理緩衝。海床與「海床高於海面」的既有 1 微米幾何契約維持不變；乾點、缺值、域外與時間缺口也不因海面容差取得通行權。Smagorinsky 水平 current 取樣共用同一端點支援與 query-time 幾何範圍檢查。此為工程取樣政策與單元測試契約，不代表已完成真實資料的科學驗證。

上述 5 微米仍是一般環境取樣介面（`forcing API`）的唯一海面上界容許尺度，不因四階 Runge-Kutta 法（RK4）的中間點越界而放大。新竹 24 小時 r3 曾在 `dt_min=0.1 s` 的 k4 觀察約 `z-eta=10.37e-6 m` 上越；r4 又在 k2 觀察三筆約 `2.27583e-4` 至 `2.37189e-4 m` 的明確上越。r4 顯示步首參考樣本可先由一般取樣器依 5 微米契約判定有效，但移動 eta 到 k2 時已下降；若引擎再要求步首嚴格 `z<=eta`，反而會否定同一集中契約。故步首資格要求參考樣本有效，且 `z` 位於 `bed-1e-6 m` 至 `eta+5e-6 m`；eta／bed 相容性也沿用既有 1 微米契約。這只承接已驗證的步首微米級邊界位置，不會放寬失敗中間點：k2／k3／k4 仍須回傳精確 `VERTICAL_UNSUPPORTED` 且 `z>eta`，並只在下一次自適應折半會低於 `dt_min` 時建立一次性的中間點垂向速度包裝器（`SurfaceStageVelocityProvider`）。包裝器以 `z_reflected=2*eta-z` 將該深度鏡射進實際水柱，再於相同 x、y、世界協調時間（UTC）重查；鏡射點仍須嚴格位於 `[bed,eta]`。重查若為乾點、域外、時間缺口、缺值、組合品質旗標或其他無效狀態，整步維持封閉式失敗。只有完整確定性 RK4 成功後才消耗一次布朗運動（Brownian）亂數，最終提議位置仍交由一般海面／海床解析器處理。單獨的中間點上越不會建立終點事件，也未擴張軌跡／事件資料格式；這是反射數值邊界條件，不表示粒子可存在於海面以上，亦不代表 r4 已完成修正後重跑驗收。

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
| execution checkpoint | 新 writer 固定 `3.1.0`：以 `immutable history segments` 追加觀測／事件列，`compact_state.json.gz` 與 `history_segment.json.gz` 使用 deterministic gzip 保存 current／history；manifest 綁定壓縮檔 `st_size`、SHA-256 與解壓後 JSON 大小，完整 `checkpoint.json` SHA-256 chain 綁定 generation、provenance 與資料內容；已終止粒子的 current 與 history 在後代逐欄凍結 | loader 保留 `3.0.0` 未壓縮拓撲，允許 3.0→3.1 混合 chain；`2.0.0`／`2.1.0`／`2.2.0` 以固定舊欄位集合唯讀遷移；禁止 3.1 降回 3.0／2.x；舊目錄不原地升級；schema 3 仍逐次更新共享 `progress.lock` |

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

### 4.1 向下沉降來源路徑圖成果包

`source-pathway-build` 讀取已驗證 aggregate release 與同一 `AggregateSpec` hash 的
`ReportSpec`，只納入 `settling_velocity_mps < 0` 的情境。每站產生 300 dpi PNG／SVG／PDF
六面板圖：訪格比例、首次通過年齡、local first-exit KDE／raw count、每有效成員停留時數、
local boundary 弧長與完整停止／QC 結果；同時保存 `grid.parquet`、`boundary.parquet`、
`outcomes.parquet`、caption 與含 aggregate manifest SHA-256 的 `manifest.json`。
其中 C 的逆向 `local_first_exit` 在條件式解讀下對應正向潛在移入入口，E 呈現潛在移入
邊界區段；兩者都是邊界事件診斷，不能直接稱為確定來源。

圖面與 sidecar 的 pooled 結果是按本次已執行成員數加權、條件於情境設計與有效成員的
「條件式來源足跡／相對來源權重」，不推論材料自然比例、絕對來源機率、沉積質量或沉積濃度。
訪格每粒子每格只計首次訪問；停留時間保留重複迴游，兩者分圖。無樣本格留白，低樣本格以
斜線標示；`bed_first_contact_count` 與 `bed_repeated_contact_count` 是底床邊界接觸診斷，
不能解讀為沉積量。

```bash
uv run lbt source-pathway-build \
  --aggregate-release "$LBT_OUTPUT_ROOT/<run_id>.aggregate-v1" \
  --report-spec "$LBT_OUTPUT_ROOT/<run_id>.report-spec.json" \
  --destination "$LBT_OUTPUT_ROOT/<run_id>.source-pathway-v1" \
  --mplconfigdir "$LBT_SCRATCH_ROOT/mplconfig-source-pathway"

uv run lbt source-pathway-validate \
  "$LBT_OUTPUT_ROOT/<run_id>.source-pathway-v1"
```

`--mplconfigdir` 必須是 caller 先建立的絕對、可寫、非 symbolic link 目錄；source
pathway writer 不覆寫既有 final。可用 `tests/test_source_pathway_release.py` 的四個
synthetic tests 先做工程 round-trip、KDE available／低樣本、PNG metadata 與 tamper
檢查；測試建立的 PNG 可直接以 `view_image` 檢查版面，但 synthetic fixture 不代表正式
五站研究成果。

若只要檢查速度契約，可閱讀 `tests/test_velocity_recording.py`；若要檢查 v3 segment chain、舊 checkpoint 的唯讀讀取與亂數延續，閱讀 `tests/test_checkpoint_segments.py` 與 `tests/test_checkpoint_execution.py`。這些測試刻意不執行真實模型、不下載資料，也不修改既有 pilot 結果。

## 5. 實際資料與 SERVER 使用入口

正式執行所需根目錄由環境變數或 CLI 參數注入，例如 `OCM_NATIVE_ROOT`、`OCM_SURFACE_ROOT`、`NWW_ANALYSIS_ROOT`、`LBT_OUTPUT_ROOT`、`LBT_SCRATCH_ROOT` 與 `LBT_CHECKPOINT_ROOT`。SERVER 的結果儲存契約固定以 `/data/LBT` 為單一 `LBT_RESULT_NFS_ROOT`；execution package、run workspace、checkpoint、scratch、log、aggregate、report、視覺化成果與 UV／Matplotlib／XDG／temporary cache 都必須位於該 NFS mount 的嚴格子目錄。`/home` 只保留已追蹤的主專案功能模組、乾淨 checkout 與既有 `.venv`。tracked SERVER runner 會在任何 batch 寫入前檢查路徑、mount identity、剩餘空間、寫入、原子改名與跨程序鎖，失敗即停止。實際資料依序執行唯讀 `preflight` → strict `inputs-build`／`inputs-validate` → release 設定來源綁定 → `run-create` → `run-shard`／checkpoint resume → `run-reconcile` → `validate-run` → aggregate／report validator；完整參數與外置 checkpoint root 規則見[CLI 參考](docs/operations/cli_reference.md)，容量、鎖、部署與資料同步見[SERVER 執行手冊](docs/operations/06_server_runbook_plan.md)及[Git 部署與資料同步手冊](docs/operations/git_deployment_and_data_sync.md)。

本機 Git 是開發來源；SERVER 只部署核定且可追溯的 commit。Git checkout／`.venv` 與 `/data` 上的上游大型資料、execution package、執行工作區、trajectory、checkpoint、scratch 及發佈輸出分開管理；部署同步需核對 commit、已追蹤檔案、checksum、dirty flag、seed 與輸入清單。未完成該次儲存檢查與科學 preflight 前，不啟動五站 `50,000×M` 正式 batch。

效能改善以[正式完整母體效能工作線](docs/operations/16_performance_improvement_tracks.md)推進。首批提供 `run-worker` 接續執行同 run 的指定分片並重用流場管理器、正常步首速度樣本重用，以及 OCM 表面資料的少量格點取值；使用方法見[CLI 參考](docs/operations/cli_reference.md)。未來各區正式完整母體使用 `run-formal-parallel`：固定數量的長壽命 worker 依固定 run plan 確定分組，每個程序以單一 `run-worker` 連續執行自己的 shard 群，重用同程序流場管理器與已編譯 dispatcher。完整完成仍須全 shard lifecycle、child exit 與 `validate-run --require-complete` 同時通過；不做舊／新版倍率 A/B 比較，也不把局部試跑當完整成果。實際正式執行仍須先通過版本化輸入、乾淨 deployment provenance、SERVER NFS 儲存檢查與科學驗證；runner 還會即時核對本次 `scratch_root` 的 NFS source，避免誤用其他掛載點的 PASS 快照。設定可明示 `execution.physics_kernel_backend: numpy_v1` 或 `numba_cpu_v1`，OCM 內層插值另可明示 `execution.ocm_interpolation_backend: numpy_v1` 或 `numba_ocm_v1`。後端版本會進入設定與 run 身分；省略欄位的舊設定仍沿用 NumPy。Numba dispatcher 目前使用 `cache=False`，每個正式 worker 會在自己程序內呼叫 `warmup_numba_backend()` 一次，編譯結果只留在該程序記憶體，不能宣稱跨程序磁碟 cache 重用。若設定了 Numba backend，`NUMBA_CACHE_DIR` 仍必須明示為通過儲存檢查的 scratch 子目錄，並在匯入加速模組前設定；這是安全路徑契約，不代表目前核心會寫入 `.nbc`／`.nbi`。

龜山島單站 30 天工程測速的分片順序、外部監測、checkpoint／resume 與耗時解讀見[單站 H30 工程測速操作契約](docs/operations/17_engineering_window_benchmark.md)。該測速僅量測實際步進成本、資料讀寫與準備時間，不能升格為正式研究成果或改寫五站完整情境契約。checkpoint 的 segment 操作、故障恢復與容量指標見[checkpoint segment 操作契約](docs/operations/18_checkpoint_segment_storage.md)。工程配對若需讓 `no_stokes` 與 `finite_depth_stokes` 在相同 scenario、master seed、member 下共用可重現的擴散亂數，請在 `scripts/run_engineering_window.py run` 明示非空 `--random-stream-id <ID>`；ID 只替換 seed 導出案例命名空間，兩個 run 仍保留各自 `experiment_case_id`、`particle_id`、物理 request 與輸出，並由 execution checkpoint schema `3.1.0` 保存含 ID 的 `binding`／segment chain。省略時維持原案例 seed；resume stream 變更、空白或非字串均 fail-closed。這只是 common-random-number 工程控制，不代表物理案例具有相同軌跡或可合併其條件式來源足跡。

持續 worker 的快取計數按分片執行增量保存，續跑合併已保存增量，避免共用管理器的累計次數被報告重複相加；常駐位元組等狀態量以樣本最大值呈現。checkpoint 的 `checkpoint_bytes` 表示每次新 generation 的 lifetime logical bytes-written，`checkpoint_active_bytes` 表示目前所有保留 generation、`latest.json` 與 checkpoint 資料檔的已發布普通檔案 `st_size` 邏輯長度加總；它不包含目錄與 NFS 配置空間，不能取代 `du`／`df` 儲存閘門。兩者不可混為同一種容量。歷史缺少計數語意或量測不完整的紀錄須保留限制說明，不能作為精確總量。

回溯日數採通用參數：`inputs.backtrack_support_days` 指定共同輸入要篩選與驗證的正整日上限，`boundaries.max_backtrack_days` 指定本次實際回溯長度。先建置並驗證支援 30 日的共同輸入，即可由 `release-config-create --max-backtrack-days` 產生 7 日、30 日等獨立執行設定，保留同一批到達時刻與情境，不必重跑整套 `inputs-build`。天數不是限定選單；超出既有輸入上限時須另建並驗證較長版本。設定範例、步數預算與來源綁定限制見[輸入衍生契約](docs/operations/14_input_derivation_and_release_contract.md#31-通用回溯支援與共同比較母體)與[CLI 參考](docs/operations/cli_reference.md)。

### 5.1 多個回溯日數的一鍵共同母體

若要公平比較 30、60、90 日，使用 `horizon-suite-create` 一次建立共同母體與三份
release config。suite 先取 `--backtrack-days` 的最大值（此例為 90），由原始 template
產生 effective `common-config`，精確填入 `inputs.backtrack_support_days: 90`，再以三個
明示的 accepted product roots 嚴格執行一次 `inputs-build`。只有到達時刻的
`[arrival - 90 日, arrival]` inclusive UTC 窗口完整 gap-safe 時，該到達時刻才會進入共同
母體；OCM 缺時不得以零值或最近值補齊。接著 suite 從完全相同的 `common-input` 產生
30／60／90 日三份 release config，分別設定 `boundaries.max_backtrack_days`，並依
`ceil(days * 86400 / integration.dt_min_seconds) + 1` 設定
`boundaries.maximum_step_count`。

```bash
uv run lbt horizon-suite-create \
  --config-template "$FORMAL_CONFIG_TEMPLATE" \
  --backtrack-days 30 60 90 \
  --destination "$LBT_SCRATCH_ROOT/horizon-suite-2024-2025-h30-h60-h90-v1" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --formal-release
```

`--formal-release` 也可寫成 `--formal`；工程 pilot 請改用 `--pilot`。formal 仍須通過既有
A 區 v3/local20 正式閘門，suite 不得繞過；兩種模式都執行完整結構、來源與 SHA-256 檢查，pilot 保留 `generated`／pilot 狀態，不得解讀為正式
科學結果。建立後以同一批 accepted roots 執行唯讀的 `horizon-suite-validate`；省略三個 roots 時只驗
artifact closure，不代表重新核對 accepted source bytes／canonical UTC axis；正式／移機驗收必須明示三個 roots（完整命令見 [CLI 參考](docs/operations/cli_reference.md)）。
suite 目的地必須是不存在的新目錄，既有目的地不會覆寫；任一步驟失敗都不發布成功的 final，並保留失敗的 `.partial-*` 現場，不自動遞迴刪除。
共享同帳號 SERVER 上，人工清理前先確認 process、目錄擁有者、inode 與 final 狀態，不採用先 `stat` 再 `unlink` 的競態方式，且不得把 `.partial-*` 當成成功。
輸出包含 `source-template`、effective `common-config`、唯一的 `common-input`、`release-configs`、`validations`（common input 與各 release validator JSON），以及記錄各檔案
與 artifact hash 的 `horizon-suite-manifest.json` 和相鄰 `.sha256` 綁定檔。suite 只接受原範例文件化的
`scenarios.receptor_arrival_initial_condition_manifest` placeholder（值為 `manifests/receptor_arrival_initial_condition.json`）；若 template 已綁定
release／pilot，或其他欄位含非預期 derived path，應拒絕，不會猜測或改寫既有來源。
三個 horizon 共用 site、receptor、arrival、material、initial-condition、scenario 母體與
artifact hash，因此差異可歸因於回溯長度設定；這不保證每粒子實際走滿最長日數，粒子仍可
因海岸、域外、資料缺口或數值狀態停止。正式輸入只接受 OCM schema 3 `ocm_native`、OCM
schema 3 `ocm_surface` 與 NWW3 schema 1 `nww3_analysis`；raw NetCDF、transfer archive、
零值填補與最近值補齊都不在 suite 輸入範圍內。完整拓撲、驗證與限制見
[輸入衍生契約](docs/operations/14_input_derivation_and_release_contract.md#31-通用回溯支援與共同比較母體)及
[CLI 參考](docs/operations/cli_reference.md)。
正式輸入的每個月份、UTC 時間軸、schema、單位／方向、mask、缺時形狀、geometry、容量與權限，都應在當次 preflight 留下可機讀紀錄；已知時間缺口只能採核准重建或缺口安全到達視窗，執行流程不臨時外插，不以最近值或零值補資料。

ABCD 第一次 24 小時試跑的結果與限制見[四區試跑稽核](docs/results/15_four_region_first_pilot_audit.md)。
四區的明示 pilot registry 共用 `2024-01-02T01:00:00Z`、24 小時回溯與 25 個逐時節點；A
區必須同時選貢寮與龜山島，B／C／D 則各自選單站。既有試跑仍須以
`pilot-matrix-validate` 檢查共同設定；目前 A 使用 `no_stokes`，B／C／D 使用
`finite_depth_stokes`，因此不能把它們宣稱為同設定比較或正式研究結果。

在 NFS 上，preview／figure 的 `.complete` 只表示該成果目錄已通過逐檔位元組、manifest、程式
指紋與儲存閘門綁定，可供成果 reader 讀取；它不表示粒子 run 完成。run 的生命週期仍以
`run_progress.json`、`run-reconcile` 與 `validate-run --require-complete` 判定。

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

`scripts/build_pilot_coastline_preview.py` 可對既有 B 區 r2 preview 重繪海岸總覽與 H1–H5 局部圖；它只讀 preview CSV／JSON、已核對幾何與海岸 GeoJSON，不讀流場資料或重跑模型。`build_pilot_preview.py` 的來源入口接受既有 `pilot_exact`，或通過嚴格單站／單到達／單材質／全情境全選閘門的 `run_kind=pilot`、`scenario_selection.mode=full`；後者仍受 source／selected count/hash、無 sampling strata、本站受體一對一及粒子／觀測容量上限限制，不能用於 generic multi-site/full formal run。既有示範保存 20 粒子、203 筆模型紀錄，仍只是工程／展示資料。

預設 `legacy` 保留兩張水平 PNG；`--style baytrace` 另輸出水平圖、`depth_age.png`、`terminal_counts.png`、README 與 manifest，並拒絕覆寫既有輸出目錄。BayTrace 圖面契約 `1.3.0` 讓水平總覽、局部軌跡、垂向軌跡與停止統計共用停止狀態名稱，其中 `numerical_failure` 一律顯示為「數值失敗停止」；特定取樣原因僅保存在診斷欄位，不改寫圖例。呈現參照是 BayTrace v4.7.2 套件 `scripts/analyze_cases.py` 的 `plot_case`；本專案的垂向與停止診斷是額外內容，不擴張 BayTrace 語意。完整命令、來源與限制見[單站 pilot 計畫的獨立海岸底圖重繪](docs/operations/pilot_run_plan.md#獨立海岸底圖重繪)。這些計畫、腳本與測試是持久化的 pilot／補圖入口；補圖不改原 preview 的資料、圖層或採樣序列，只在新的 sibling output directory 產生圖檔與 manifest，局部圖保留實際公尺刻度與 panel identity，缺值留空，不能以海岸圖的完整性替代軌跡／流場資料 QC。若 geometry manifest 為 `status=generated`，BayTrace reader 只在 preview 已完整驗證並有 `.complete`、summary 明示 `run_kind=pilot` 且 `selection_mode` 為 `pilot_exact` 或受限 `full` 時放行；兩份 domain/open 的 canonical hash、站點／region／flow owner 與外框仍須全數通過。此情況的 figure manifest／README 會標示 `geometry_release_status=generated` 與 `engineering_only=true`，表示工程試跑圖面，不是 approved 正式 geometry、來源機率或因果歸因結果；既有 approved 路徑維持原相容性。

舊文件中的 Phase／Slice 名稱是設計脈絡，不是自動更新的現在狀態；若與本頁、[實作狀態](docs/implementation_status.md)、程式碼或當次紀錄不一致，以較新的可驗證程式／manifest／測試證據為準，無法確認的 SERVER 事實只引用現場紀錄。本頁不保存 SERVER 密碼、私有絕對路徑、未提交的大型資料或模型結果；實際部署依已授權環境的唯讀盤點與核准流程辦理。
