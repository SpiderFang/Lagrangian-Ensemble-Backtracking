# Lagrangian Ensemble Backtracking

本專案以 CWA-OCM 三維海流、CWA-NWW3 波浪衍生的 Stokes 漂流、向下沉降與次網格擴散，從受體位置與到達時間向過去建立海洋廢棄物的條件式來源足跡。

本頁是入口與狀態摘要；完整文件分類與閱讀路線見[文件總入口](docs/README.md)，實作狀態見[實作狀態](docs/implementation_status.md)，複雜 CLI 與 SERVER 操作見[CLI 參考](docs/operations/cli_reference.md)。

> 目前定位：工程執行流程、輸入／輸出契約與測試已可供持續維護；這不代表全期真資料批次或正式科學成果已完成。

可選的[已審模型岸界](docs/development/reviewed_native_coast.md)只覆蓋南灣主缺支援區的原生交界與連江指定112邊內孔，以首次碰岸停止；完整範圍、已審兩站40顆重算入口與 C2／C3 核准狀態見該文件，本頁不重複列出細節。

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
| 貢寮 | A | 使用 `northeast_taiwan_common_cache_v3` 與 `formal_domain_policy=v3_local20km_20260909_v1`；12.5 km 受體核心、20 km 局部區域，與龜山島共用最外層停止邊界；正式運算沿用不擴張的逐階段資料支援嚴格檢查 |
| 龜山島西側 | A | 使用同一 v3 A 區海流資料與範圍規範；12.5 km 受體核心、20 km 局部區域，局部區域、受體與統計維持獨立 |
| 新竹外海 | B | 使用 `hsinchu_cache_v3` 流場區域；local domain 仍等於 flow domain，水平受體候選限於 `[120.45, 24.75]` 半徑 12.5 km 核心；24 小時展示 pilot 入口與參數見[紀錄](docs/results/14_hsinchu_2024-01-01_24h_pilot_parameter_record.md) |
| 南灣 | C | 使用 `houwan_nmmba_cache_v3` C 區流場；`forcing` 產品 ID 保留歷史名稱，現行 study site 為 `nanwan` |
| 連江 | D | 使用 D 流場區域 |

現行五站 anchor（WGS84，經度／緯度）固定為：貢寮 `[121.9223889,25.0964444]`、龜山島 `[121.939084,24.845304]`（WP-51 C：由原陸地錨點 `[121.951606,24.843127]` 移到島西側海上，規則見下）、新竹 `[120.45,24.75]`、南灣 `[120.763161,21.946577]`、連江 `[119.95,26.2]`。正式母體五站的水平受體均由各站 persistent-wet/core 候選池以 `seeded_uniform_random_without_replacement_v1` 無放回抽樣；新竹 24 小時試跑的五個固定點只作歷史位置核對，不進 current formal。每個水平 face 再以獨立 stream 依 `seeded_uniform_random_open_interval_v1` 抽四個 `(0,1)` normalized fraction，random rank 不代表物理水層。C 區只改研究站 anchor 與名稱，不改 `houwan_nmmba_cache_v3` 的 `forcing bbox`。

龜山島錨點搬移（WP-51 C）：舊錨點落在龜山島陸地內，改為「`receptor_priority_polygon` 走廊內離舊錨點最近、50 個到達時刻皆為濕（persistent-wet）、距候選區邊界不小於 2 倍網格尺度、NWW 與 OCM surface 支撐有效且不在校閱海岸陸地內的 OCM 原生網格面中心」。因 geometry 在到達時刻與濕乾資料載入前就需要錨點，座標寫死於設定（`anchor_policy: snap_to_persistent_wet_mesh_within_priority_polygon`），推導腳本為 `scripts/derive_guishan_anchor.py`，推導紀錄（原錨點、面編號、距離、候選數）保存在設定的 `anchor_snap_provenance`，並寫入受體 manifest 的 `priority_receptors_by_site.guishan`。12.5 km 受體核心與 20 km local domain 都以新錨點為圓心；貢寮錨點與幾何不變。

每站完整基礎設計是 `10 種材質／形狀 × 20 個三維受體 × 50 個到達時間 = 10,000` 個情境；若每個情境都採相同的成員數 `M`，五站執行量才可寫成 `50,000×M`。各情境成員數不同時，總量應寫成 `Σ M_s`；`M` 是隨機成員數，不是額外情境因子。實驗案例（例如 `no_stokes` 或擴散敏感度）另行編號。

所有本專案沉降速度均為負值、物理方向以向上為正；不允許上升物性，也不對完全沉沒物體加入風壓效應。缺少密度、阻力、再懸浮參數時，不宣稱已完成沉積—再懸浮動力。

貢寮與龜山島共用 A 區流場資料與最外層開放邊界，但各自保存局部區域入口、跨站診斷、情境身分與統計分母；穿越另一站局部區域不會轉移粒子所屬站點。A 區沿用 `northeast_taiwan_common_cache_v3`、以 anchor 半徑 12.5 km 受體核心與 20 km 局部區域建立 `formal_domain_policy=v3_local20km_20260909_v1`；龜山島另有核定的 soft-priority 走廊，只在同一 12.5 km core 的 persistent-wet 候選中優先選點。A 區正式空間支援採 `runtime_spatial_support_policy=runtime_stage_fail_closed_no_expansion_v1`：OCM 或 NWW3 在實際查詢位置／時間無有效支援時立即停止，保留 `data_gap` 等原始狀態，不以零值、最近值或擴張資料域補齊。完整範圍歷史沿革、既有 24 小時 DEMO 產物邊界與正式 30 天重建需求見[全部可得資料與時間重建](docs/operations/10_available_data_time_reconstruction_and_a_expansion.md)。

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

月份目錄只是產品分割與延遲載入的索引；執行階段會依每個 OCM／NWW3 產品實際的 `time_utc_ns`，在必要時從相鄰月份尋找 before／after 時間端點並完成合法時間內插；真正的時間缺口仍回傳 `TIME_GAP`，不以零值、最近值或外插補齊。若明示啟用 execution-priority 的 `continuous_time_fallback=true`，只放寬純時間可用性、在已知短缺口採有效 before／after 內插並完整記錄來源、偏移秒數與策略識別碼；乾點、域外、垂向不支援、無效物理或非有限場仍照原規則失敗。完整條件、正式運算設定的 `execution.max_resident_forcing_months` 與稽核欄位見[CLI 參考](docs/operations/cli_reference.md)。

OCM 垂向取樣、移動海面附近的微米級容差（`SURFACE_BOUNDARY_TOLERANCE_M`）與 RK4 中間點越界時的一次性鏡射重查（`SurfaceStageVelocityProvider`）屬集中定義的工程取樣契約，只承接已驗證的步首邊界位置，不放寬失敗中間點；完整數值、限制與逐步驗證見[架構與資料契約](docs/foundation/02_architecture_and_data_contract.md)與[科學方法與驗證](docs/foundation/03_scientific_method_and_validation.md)。

### Observation 速度紀錄

`Observation` 同時涵蓋既有定期輸出紀錄與狀態／終止紀錄。只有當 Observation 的位置與 UTC 和已取得的步首參考取樣完全相同，才附上速度；沒有同點取樣時明示狀態並保留缺值，不借用 RK4 階段、內部步或其他位置的速度。這些是取樣值，不是位移除以時間的平均速度；位置單位為 m，時間為 UTC 奈秒，速度單位為 m/s；逆向積分不另行反號。

九個固定欄位依四類如下；所有速度單位均為 m/s：

| 分類 | 欄位 | 意義 |
|---|---|---|
| 總速度 | `total_u_mps`、`total_v_mps`、`total_w_mps` | 確定性海流、Stokes 與沉降合成後的正向物理速度 |
| OCM 海流 | `ocm_u_mps`、`ocm_v_mps`、`ocm_w_mps` | OCM 三維 current；`u/v/w` 為東／北／向上 |
| Stokes 水平 | `stokes_u_mps`、`stokes_v_mps` | NWW3 衍生的水平 Stokes drift |
| 沉降 | `settling_w_mps` | 以向上為正的垂向沉降速度；本專案沉降為負 |

`total_u = ocm_u + stokes_u`、`total_v = ocm_v + stokes_v`、`total_w = ocm_w + settling_w`，由明定容差驗證。Stokes 垂向與沉降水平不屬於欄位，不能假造。`CombinedMonthForcing` 的正式有效樣本提供完整分項；明確關閉 Stokes 可記有效的 0，缺少 NWW3 波浪資料必須保留缺值並帶非零 QC，不能記成 0。

### 版本與舊格式相容性

| 產品 | 新 writer | 舊格式相容性 |
|---|---|---|
| trajectory shard | `3.0.0`：固定位置、狀態、事件、環境與速度資料檔；實際產品含 `.npy`、`.parquet`、`.json`，並有 dtype／count／checksum 驗證器 | `1.0.0`、`2.0.0` 可唯讀；舊欄位沒有速度資料，不自動補速度。正式報告可使用 v2 或 v3，v1 不得作正式垂向驗證證據；同一執行混用 v2／v3 保守拒絕 |
| execution checkpoint | 新 writer 固定 `3.1.0`：以 `immutable history segments` 追加觀測／事件列，`compact_state.json.gz` 與 `history_segment.json.gz` 使用 deterministic gzip 保存 current／history；manifest 綁定壓縮檔 `st_size`、SHA-256 與解壓後 JSON 大小，完整 `checkpoint.json` SHA-256 chain 綁定 generation、provenance 與資料內容；已終止粒子的 current 與 history 在後代逐欄凍結 | loader 保留 `3.0.0` 未壓縮拓撲，允許 3.0→3.1 混合 chain；`2.0.0`／`2.1.0`／`2.2.0` 以固定舊欄位集合唯讀遷移；禁止 3.1 降回 3.0／2.x；舊目錄不原地升級；schema 3 仍逐次更新共享 `progress.lock` |

缺值與版本界線是驗證證據的一部分：舊成果不會自動升格為含速度的新成果，synthetic／pilot 也不會升格為正式科學證據；事件與狀態保留 `DATA_GAP`、`NUMERICAL_FAILURE`、邊界事件、可用性與失敗原因，無效取樣必須有非零 QC，不能用靜水或終止位置掩蓋失敗查詢。

輸出分析前應先確認 manifest 的執行身分、schema、來源綁定、粒子／觀測／事件計數與 checksum，再解讀位置、停止類型或統計比例；不同版本、研究站點或實驗案例不能只因欄位名稱相同就直接合併。

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

正式執行所需根目錄由環境變數或 CLI 參數注入，例如 `OCM_NATIVE_ROOT`、`OCM_RECONSTRUCTION_ROOT`、`OCM_SURFACE_ROOT`、`NWW_ANALYSIS_ROOT`、`LBT_OUTPUT_ROOT`、`LBT_SCRATCH_ROOT` 與 `LBT_CHECKPOINT_ROOT`。SERVER 的結果儲存契約固定以 `/data/LBT` 為單一 `LBT_RESULT_NFS_ROOT`，容量、鎖、部署與資料同步見[SERVER 執行手冊](docs/operations/06_server_runbook_plan.md)及[Git 部署與資料同步手冊](docs/operations/git_deployment_and_data_sync.md)。實際資料依序執行唯讀 `preflight` → OCM reconstruction patch 建置／驗證（若採核准重建）→ strict `inputs-build`／`inputs-validate` → release 設定來源綁定 → `run-create` → `run-shard`／checkpoint resume → `run-reconcile` → `validate-run` → aggregate／report validator；完整參數見[CLI 參考](docs/operations/cli_reference.md)。

OCM 重建建置器只讀上游 schema 3 cache，在另一個 NFS 目錄發布缺失 UTC rows，不改寫原始月份；預設 fail-closed 模式下未登錄缺口仍回傳 `data_gap`，只有明示核准 reconstruction policy 時才依連續時間策略取樣缺時，物理／空間失敗仍不放寬。四區建置操作與 gap-safe／gap-reconstruction manifest 綁定見[全部可得資料與時間重建](docs/operations/10_available_data_time_reconstruction_and_a_expansion.md#45-四區稀疏-patch-建置操作)。

本機 Git 是開發來源；SERVER 只部署核定且可追溯的 commit；未完成儲存檢查與科學 preflight 前，不啟動五站 `50,000×M` 正式批次。正式完整母體效能工作線、`run-worker`／`run-formal-parallel` 的 dynamic worker 佇列設計、`physics_kernel_backend`／`ocm_interpolation_backend` 後端選擇、Numba cache 契約與 `--defer-full-validation` 延後驗收旗標完整規則見[效能改善工作線](docs/operations/16_performance_improvement_tracks.md)與[CLI 參考](docs/operations/cli_reference.md)。

龜山島單站 30 天工程測速的分片順序、外部監測、checkpoint／resume、耗時解讀與 `--random-stream-id` common-random-number 工程控制見[單站 H30 工程測速操作契約](docs/operations/17_engineering_window_benchmark.md)；checkpoint segment、故障恢復、`checkpoint_bytes`／`checkpoint_active_bytes` 容量語意見[checkpoint segment 操作契約](docs/operations/18_checkpoint_segment_storage.md)。

正式範例登錄 random bed residence：`inputs.backtrack_support_days=180` 是 observation 選時包絡，`bed_residence_time.runtime_horizon_support_days=90` 與 `boundaries.max_backtrack_days=90` 是沉底後執行支援，不可混為粒子回溯天數；正式 H30/H60/H90、M=10 的 common-input source template 為[`configs/lagrangian_backtracking.formal_h30_h60_h90_m10.yaml`](configs/lagrangian_backtracking.formal_h30_h60_h90_m10.yaml)，固定最大沉底年齡 90 日、五站共用 50 個分層隨機整點小時與 seed `20260916`，只供 `horizon-suite-create --backtrack-days 30 60 90 --formal-release` 一次建立共同母體。正式範例的 `backtrack_mode` 預設為 `full_horizon_from_deposition`，只發布 H30/H60/H90 各一份共三份 release；`fixed_calendar_window` 仍保留為可選敏感度模式。沉底年齡抽樣範圍開關（`age_sampling_scope`）、`horizon-suite-resume`、新版母體 `forcing_years`／`replicate_rank` 母體規範等完整契約見[輸入衍生契約](docs/operations/14_input_derivation_and_release_contract.md#31-通用回溯支援與共同比較母體)。

ABCD 第一次 24 小時試跑的結果與限制見[四區試跑稽核](docs/results/15_four_region_first_pilot_audit.md)；A 區使用 `no_stokes`，B／C／D 使用 `finite_depth_stokes`，不能宣稱同設定比較。在 NFS 上，preview／figure 的 `.complete` 只表示成果目錄已通過驗證可供 reader 讀取，不表示粒子 run 完成；run 生命週期仍以 `run_progress.json`、`run-reconcile` 與 `validate-run --require-complete` 判定。

`native_runner` 子套件另外提供 `lbt native-plan`／`lbt native-run`／`lbt native-status`／`lbt native-postprocess`／`lbt native-derive-horizon` 五個子命令，以工作單元（區域、到達時刻、沉降速度、`study_site_id`）為顆粒度做規劃、持續 worker 排程、續跑與失敗隔離；`--backtrack-mode`、`--experiment-case`（`no_stokes`／`finite_depth_stokes`，固定使用工作手冊 eq.(10) Smagorinsky 擴散）、`--bed-no-deposition`（研究用敏感度案例：關閉「連續停留海床即沉積並終止」判定，不是鏡射反射也不是再懸浮動力學，見該文件 9.15 節）與設定檔不一致時 fail closed，須明示 `--acknowledge-config-mode-mismatch` 才放行。`native-postprocess --with-boundary-segments` 可依方位扇區／沿邊界弧長位置細分邊界進入統計。伺服器背景啟動入口見 `scripts/run_native_backtrack_server.sh`；完整工作單元模型、排程策略、實驗案例契約與已知限制見[native 逆向批次 orchestrator 操作契約](docs/operations/19_native_backward_batch_orchestrator.md)。此路徑是獨立於既有 `run-formal-parallel` 的排程／執行實作，不能取代既有正式母體流程。

## 6. 文件導覽

| 主題 | 文件 |
|---|---|
| 文件分類、五條閱讀路線與完整原路徑對照 | [文件總入口](docs/README.md) |
| 當前實作、版本與正式驗證證據缺口 | [實作狀態](docs/implementation_status.md) |
| CLI、執行生命週期、輸入、release、checkpoint | [CLI 與執行參考](docs/operations/cli_reference.md) |
| SERVER 容量、鎖、執行與發布 | [SERVER 執行手冊](docs/operations/06_server_runbook_plan.md) |
| 原始碼模組關係與資料流程 | [互動式程式架構圖](docs/source_code_architecture_map.html) |

### BayTrace 原版 CPU 批次（Q4）

明示 `execution.advection_backend: baytrace_native_cpu_v1` 時，執行環境對純流／流波均設定 `require_native=True`，直接執行 PI 原版 BayTrace CPU RK4／內插核心；未支援的場資料轉接拋出可恢復工程例外，不藉粒子數值失敗早停宣稱完成，真實資料 QC 保留原分類。有限水深波浪採 `finite_depth_node_fields_v1`，與舊粒子逐階段取樣波浪參數有離散差異，不宣稱軌跡完全等價。原版場快取、固定五站100顆／已審兩站岸界40顆展示入口、`run-reconcile` 排他鎖與逐分片索引改善、以及僅供既有失敗點唯讀比對的 `diagnose_failure_points_readonly.py`／`diagnose_batch_readonly.py` 診斷腳本，完整命令、環境變數與限制見[BayTrace CPU 對照](docs/development/baytrace_native_cpu_comparison.md)；已審兩站岸界40顆入口另見[已審模型岸界](docs/development/reviewed_native_coast.md)。

最新 synthetic no-Stokes 生命週期測速：`128×16×3` 為 1.500／0.578 秒、`512×16×3` 為 6.194／2.341 秒，最大誤差為 0／`8.88e-16 m`，比較通過，約 2.6 倍；結果涵蓋完整 `ProductionBatch` 生命週期但不含完整 SERVER I/O／真實流場驗證，歷史 direct-core 的 139 倍不可外推為完整執行速度。

## 7. 開發檢查與交付界線

每次變更送審前至少檢查相關 pytest、Ruff、`git diff --check`、schema／checksum 資料拓撲與文件相對連結。新增模組時同步 architecture-map catalog；架構圖的產生與檢查應使用既有產生器及固定輸出流程。OCM 原版轉接建場的節點／層批次遮罩、向量化拓樸檢核與唯讀步進診斷工具（`diagnose_failure_points_readonly.py`、`diagnose_batch_readonly.py`）完整規則見[BayTrace CPU 對照](docs/development/baytrace_native_cpu_comparison.md)。

測試結果要區分核心物理不變性、checkpoint／I/O round-trip、下游相容性與正式驗證證據。前兩者通過不能代替後兩者；報告中的有效分母與缺值狀態也不能由測試樣本外推到全期真資料。

任何程式、schema 或輸入方法變更，都應同時更新對應 docstring、需求追溯、版本說明與相容性測試。若變更會影響既有結果，建立新執行／schema 或明示 migration，不在原輸出上覆寫。

速度、checkpoint、正式驗證證據缺口與報告邊界以[實作狀態](docs/implementation_status.md)為準。需求 `REQ-007` 的「關閉單項敏感度、單位檢核」是持續驗收要求；速度單元測試不能取代正式科學敏感度驗收。

閱讀順序建議是先看本頁的限制，再看資料契約與科學方法，最後依 CLI 參考或 SERVER 執行手冊操作。歷史 audit、Phase／Slice 文件可追溯決策，但不得用其中的舊「下一步」句子推定今天的部署或資料狀態。

## 8. 方法與不採用範圍

1. 本專案沿用 BayTrace 可對應的工程思路：CPU SoA／batch／chunk、每粒子可重現亂數、triangle hint、可暫停 engine 與可驗證 I/O；不承諾整包 tracker output 的格式或物理範圍，也不把 BayTrace 的工程參照當成本案科學驗證。
2. 在沒有先驗、似然、調查努力量與獨立觀測前，任何逆向結果都不可稱為絕對來源機率、法律責任或因果歸因；報告必須保留原始分子／分母、有效分母、低樣本、零分母、`DATA_GAP` 與 `NUMERICAL_FAILURE`，不得將不可用狀態改成 0。
3. 確定性海流、Stokes 與沉降沿正向物理速度積分，時間方向由 signed-time step 表示；隨機擴散獨立於 RK4，不能插入 RK4 階段。十個沉降速度是「未校準的沉降速度敏感度格點」，iOcean 材質名稱只是附註，不是來源先驗；正式成果須以速度數值或極慢／慢／中／快分群為主詞，不得以材質名稱直接對應單一來源方向（例如不得寫「寶特瓶來自 SSE」）；`deposit_on_first_contact_and_stop` 沒有再懸浮或底流搬運參數，會使足跡系統性偏向較近來源。完整限制、依據數據與 `M` 下限估計見[科學方法與驗證第 11 節](docs/foundation/03_scientific_method_and_validation.md)。

## 9. B 區海岸預覽（既有示範）

`scripts/build_pilot_coastline_preview.py` 可對既有 B 區 r2 preview 重繪海岸總覽與 H1–H5 局部圖；它只讀 preview CSV／JSON、已核對幾何與海岸 GeoJSON，不讀流場資料或重跑模型。既有示範保存 20 粒子、203 筆模型紀錄，仍只是工程／展示資料。

預設 `legacy` 保留兩張水平 PNG；`--style baytrace` 另輸出水平圖、`depth_age.png`、`terminal_counts.png`、README 與 manifest，並拒絕覆寫既有輸出目錄；呈現參照是 BayTrace v4.7.2 套件 `scripts/analyze_cases.py` 的 `plot_case`，垂向與停止診斷是本專案額外內容。補圖不改原 preview 的資料、圖層或採樣序列，只在新的 sibling output directory 產生圖檔與 manifest；若 geometry manifest 為 `status=generated`，圖面會標示 `engineering_only=true`，不是 approved 正式 geometry 或來源結論。完整命令、來源、放行條件與限制見[單站 pilot 計畫的獨立海岸底圖重繪](docs/operations/pilot_run_plan.md#獨立海岸底圖重繪)。

舊文件中的 Phase／Slice 名稱是設計脈絡，不是自動更新的現在狀態；若與本頁、[實作狀態](docs/implementation_status.md)、程式碼或當次紀錄不一致，以較新的可驗證程式／manifest／測試證據為準，無法確認的 SERVER 事實只引用現場紀錄。本頁不保存 SERVER 密碼、私有絕對路徑、未提交的大型資料或模型結果；實際部署依已授權環境的唯讀盤點與核准流程辦理。
