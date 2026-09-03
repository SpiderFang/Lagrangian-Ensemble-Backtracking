# 正式成果 release 與科學圖表生產管線規劃

## 1. 目的、證據界線與完成條件

本文件把 `docs/07_results_visualization_plan.md` 的 F01–F12、T01–T06 轉成可由 CLI
重建、驗證及原子發布的生產管線。正式成果只能讀取已通過公開 validator 的 source run、
aggregate release、原始設定與明示的比較／驗證證據；不得從未驗證目錄、Notebook 暫存
變數或手工修改圖面取得數值。

本機沒有真實 OCM schema 3 與 NWW3 schema 1 產品，因此本機只允許建立
`synthetic_engineering_evidence`：用來驗證檔案拓撲、公式、單位、分母、繪圖程式、原子
發布與 reader／validator。這些圖表不得登錄成 F03–F12 或 T03–T06 的正式科學成果。
`server_scientific_evidence` 必須在 SERVER 重新完成真資料 preflight、pilot、restart、正式
run、aggregate release 與 report release 後才能成立。

SERVER 上已完成 formal baseline、但 F11/T05 comparison 或 F12/T06 validation evidence 尚未
齊備的中間 release，證據類別固定為 `server_formal_baseline_evidence`。它可以保存真實資料
baseline 結果及明示 unavailable 列，但不得改稱完整科學報告；只有兩組 optional evidence
都齊備、兩個 allow flag 均關閉後，才能發布為 `server_scientific_evidence`。

正式交付的最低完成條件分為三組：

1. 單一 baseline run 可產生 F01–F10 與 T01–T04；每項都必須保存底層 data sidecar、
   caption sidecar、有效樣本數、分母、單位、座標參考系統、輸入 checksum 與限制。
2. F11／T05 必須另有至少一份同資料期間、同研究站點契約且明示差異參數的 comparison
   release。沒有比較 release 時只能登錄 `unavailable_missing_comparison`，不得用 baseline
   自身製造零差異圖冒充敏感度分析。
3. F12／T06 必須有版本化 validation evidence，至少包含解析解／時間步收斂、member 收斂、
   known-source synthetic、checkpoint/restart、NumPy/Numba 一致性及正向驗證。沒有相應
   證據時只能登錄 `unavailable_missing_validation_evidence`。

## 2. 輸入責任與單一真相來源

預定公開入口：

```python
build_report_release(
    *,
    source_run_root: str | Path,
    aggregate_release_root: str | Path,
    config_path: str | Path,
    destination: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    comparison_release_roots: Sequence[str | Path] = (),
    validation_evidence_path: str | Path | None = None,
) -> Path

validate_report_release(path: str | Path) -> dict[str, object]
read_report_registry(path: str | Path) -> ReportRegistry
```

資料責任固定如下：

| 資料 | 唯一來源 | 不允許的替代方式 |
|---|---|---|
| run identity、case、M、shard、commit、config/input hash | 已驗證 source run plan/progress | CLI 重複輸入、由檔名猜測 |
| scenario 的 material／receptor／arrival／season／tide／初始條件 | aggregate `scenario_strata` 與 static input binding | 從 trajectory 欄位不完整地反推 |
| 公尺格網、邊界弧長、age 秒軸、KDE/HDR/bootstrap 設定 | 已驗證 `AggregateSpec` | 依目前軌跡 extent 自動縮放或使用繪圖預設 |
| 正文 KDE、低樣本遮罩、代表軌跡數與抽樣規則 | 已驗證 `ReportSpec` | 看過結果後選頻寬／門檻或使用 Matplotlib 預設 |
| 全域事件、pathway、來源—受體與停止結果 | 已驗證 aggregate payload | 重讀未驗證 Parquet/NPY、以零補缺 |
| 代表三維軌跡與 season/tide/material 細分 | 已驗證 source trajectory iterator | 一次載入全部軌跡、抽未登錄的「好看案例」 |
| forcing coverage、observed/reconstructed/unsupported 時段 | source `input_inventory.json` 與其版本化重建證據 | 用終止事件倒推完整 forcing 狀態 |
| 敏感度差異 | 明示 comparison releases | baseline 自減、未綁定的圖片 |
| 收斂、解析解、known-source 與正向驗證 | validation evidence release | 以「pytest 通過」文字取代定量指標 |

report builder 必須先在 source run exclusive gate 內重驗 complete run、static inputs、aggregate
release 的 source hashes、run ID、experiment case、M、scenario／shard ranges 與 trajectory
manifest。aggregate release 不保存 SERVER 絕對路徑，因此 caller 仍須明示 source run；
兩者若不是同一份 immutable source，必須在任何 report partial 建立前失敗。

## 3. 模組切分與資料流

### 3.1 版本化 ReportSpec

`AggregateSpec` 已固定格網、三個核密度估計頻寬、高密度區層級、時間箱與 bootstrap，
但它沒有決定正文採用哪一個頻寬、何時遮罩低樣本格，以及 F03 要抽出多少條代表軌跡。
這些選擇若留在 renderer 的預設值，會造成同一份 aggregate release 在不同機器產生不同
圖面，甚至形成看過結果後才調參的風險。因此正式 report builder 必須再讀取一份
`report_spec.json`，而且不得在缺少規格時自行補預設。

預定 `ReportSpec` schema 1 固定保存：

| 欄位 | 契約 |
|---|---|
| `run_id`、`aggregate_spec_canonical_sha256` | 精確綁定同一 source／aggregate 規格，不接受跨 run 沿用 |
| `primary_kde_bandwidth_m` | 必須精確等於 aggregate 已登錄三個頻寬之一；另外兩個仍輸出敏感度 sidecar |
| `minimum_kde_raw_count` | 低於門檻只保存 raw count 與 `not_estimable` 原因，不製造平滑密度 |
| `low_sample_min_member_count` | F05 等格網低樣本遮罩門檻；遮罩不會把原始計數改成零 |
| `vertical_depth_bin_edges_m` | 以瞬時海面向下為正的公尺邊界，從 0 開始且嚴格遞增；域外值另記 overflow，不 clip |
| `representative_trajectory_count_per_site` | 必須至少 8 且可被 8 整除；每站對 4 season × 2 spring/neap core strata 等額保留，貢寮與龜山島各自計算 |
| `representative_selection_policy` | 固定為 `stable_hash_core_season_tide_v1`；每層依 seed＋particle identity 的 SHA-256 優先序取最小者，不依軌跡「好看程度」挑選 |
| `representative_selection_seed` | 0 至 2^128-1 的原生整數，與 run seed 分開保存 |
| `travel_age_quantiles` | 精確為 0.05/0.25/0.5/0.75/0.95；結果明示為 age-bin 解析度近似 |
| `pathway_first_passage_quantiles` | 精確為 0.25/0.5/0.75，對應 F05 median/IQR |
| `figure_formats`、`raster_dpi` | 精確為 PNG/SVG/PDF 與 300 dpi；不可用互動圖取代靜態圖 |
| `renderer_style_version`、`language` | 固定學術圖面樣式版本與 `zh-TW`，確保標籤及 caption 可重建 |

公開入口預定為 `load_report_spec`、`write_report_spec` 與
`validate_report_spec_against_aggregate_spec`；CLI 增加 `report-spec-create`，所有研究判定
參數都必須明示，不能由目前資料範圍、事件數或圖面結果反推。`report-build` 必須保存規格
原始 SHA-256 與 canonical SHA-256，並在任何 partial 建立前完成 source rebind。

### 3.2 模組責任

| 模組 | 責任 | 輸入 | 輸出 |
|---|---|---|---|
| `report_spec.py` | 正文頻寬、低樣本門檻、代表軌跡與 renderer policy | 版本化 JSON、`AggregateSpec` | immutable `ReportSpec` |
| `report_records.py` | registry、caption、table 與 evidence 的 exact typed records | 原生 scalar/mapping | immutable dataclasses |
| `report_ratio_statistics.py` | 比例狀態與分箱分位數 | 已驗證計數、秒制 histogram／edges | immutable ratio/quantile products |
| `report_pathway_statistics.py` | 有效成員 pathway 格網 | `StreamingPathwayAggregate`、有效分母 | immutable pathway product |
| `report_kde_statistics.py` | KDE/HDR 頻寬敏感度 | `AggregateSpec`、`ReportSpec`、事件格網 | immutable KDE layers |
| `report_matrix_statistics.py` | 停止結果與方向性跨站連通矩陣 | 已驗證 aggregate counts／site denominators | immutable outcome/connectivity products |
| `report_source_receptor_statistics.py` | 來源段—受體條件式比例、event share 與旅行年齡 | source-receptor counts／histograms／receptor denominators | immutable source-receptor products |
| `report_statistics.py` | 統一匯出上述產品，並組合 F/T 所需差異統計 | 上述 typed products | renderer-facing facade |
| `report_trajectory_identity.py` | 有效成員政策與完整代表軌跡 identity/hash | `ParticleResult`、scenario strata | immutable identity records |
| `report_trajectory_selection.py` | 依固定雜湊與八層配額保存每站代表軌跡 | `ReportSpec`、單一有效 `ParticleResult` | `O(site×K)` selected paths |
| `report_material_statistics.py` | 十種材質的有效 member 分母、首次海床接觸／沉積計數與比例；depth-age／環境脈絡由後續 trajectory stream 補充 | 正式 v2 `Observation`、`BED_CONTACT`／`DEPOSITED` 事件與 scenario strata | immutable `study_site_id × material_id` products |
| `report_trajectory_stream.py` | season/tide/material 的一次串流 reducer 與有效 pathway 重算 | complete trajectory iterator、strata index | bounded-memory stratified products |
| `report_render.py` | 依固定 style/render contract 產生 PNG/SVG/PDF | typed products、caption metadata | 未發布靜態圖與 data sidecars |
| `report_release.py` | source binding、fixed topology、manifest、原子 writer/reader/validator | 所有 report products | immutable sibling report release |
| `report_pipeline.py` | 依 F/T dependency graph 排程、標記 unavailable、組合產品 | source、aggregate、config、optional evidence | 可交給 writer 的完整 payload |

### 3.3 軌跡環境脈絡與 schema 升級

現行 trajectory shard 只保存 `x_m/y_m/z_m`，不足以重建 F03/F09 所要求的海面、海床、
水深與離底高度；renderer 不得回頭用圖面或固定水深猜測。正式 report 統計開始前，
`Observation` 必須新增版本化環境取樣脈絡：

| 欄位 | 語意與限制 |
|---|---|
| `environment_sample_status` | `not_sampled`／`valid`／`invalid`，不得以全零浮點值代表缺值 |
| `eta_m`、`bed_z_m` | 海面與海床的公尺制 z-positive-up 值；只有 valid 時必須為有限值且 `bed_z_m <= z_m <= eta_m` |
| `forcing_month_id` | 實際取樣的 UTC `YYYYMM`，不是由輸出檔名反推 |
| `environment_qc_flags` | 原始 forcing 品質旗標；valid 時為 0，invalid 時保存非零原因 |

每次數值步開始已取得的 valid reference sample 會回填同一狀態的最新 observation，不另外
消耗 RNG，也不改變 RK4 或擴散。剛完成一步的 active output 在下一步開始補齊；終止點若
位於域外或邊界，可保持 `not_sampled`，但最後一個有效內部 observation 必須有 context。
同時 bump execution checkpoint 與 trajectory shard schema；新 writer 只寫新版，reader 可將
舊版明示載入為 `not_sampled` 以維持工程相容，但正式 F03/F09 gate 不接受舊版被當成完整
垂向證據。新增 NPY 欄位使用 status/code 分離缺值與 QC，NaN 只作 payload sentinel，不能
單獨決定資料狀態。

資料流固定為：

```text
validated source run ─┬─ static config/geometry/inventory ───────┐
                      └─ complete trajectory shards (one pass) ─┤
validated aggregate release ─ event/pathway/strata products ────┼─ report statistics
comparison releases ─ exact-compatible difference inputs ──────┤
validation evidence ─ registered quantitative metrics ─────────┘
                                                                   │
                                                                   v
                                                      figures/tables/sidecars
                                                                   │
                                                                   v
                                                    atomic report-v1 release
```

trajectory reducer 不得保存所有 shard、所有 `ParticleResult` 或所有 observation。每個 shard
讀回後立即更新固定站點／season／tide／material 統計與預先登錄的代表案例 reservoir；完成
後才產生 typed products。若某項統計需要 member-level bootstrap，使用 deterministic
counter/seed 或 bounded sufficient records，不能為方便而 materialize 全案 50,000×M 軌跡。

### 3.4 統計、一次串流與 bootstrap 的實作介面

報告統計拆成 ratio、pathway、KDE、matrix 四個小型計算模組與只作公開組合的
`report_statistics.py`。它們都不讀寫檔案，也不重新開啟 trajectory shard；拆分的目的只是
縮小模組責任與測試範圍，不改變 F01–F12／T01–T06 的資料契約。後續
renderer 只能讀這些 immutable typed products，不得在繪圖函式內重算分母、挑選頻寬或
把缺值改成零：

| 產品 | 固定內容與閘門 |
|---|---|
| `CountRatio` | raw numerator、raw denominator、比例與 `available`／`zero_denominator`／`low_count` 狀態；狀態是缺值語意來源，`None`／NaN 不得單獨決定語意 |
| `HistogramQuantileProduct` | raw count、固定 quantile、age-bin midpoint 秒數近似、bin resolution 與估計狀態；caption 必須明示不是未分箱精確分位數 |
| `KDESensitivityProduct` | 原始 `(y_cell, x_cell)` 計數、`AggregateSpec.kde_bandwidths_m` 全部帶寬、固定 HDR levels、正文 primary bandwidth 與低樣本狀態；低於門檻時保留 raw count，但不得建立假 KDE |
| `PathwayGridStatistics` | 有效成員分母、visit numerator/fraction、residence seconds、first-passage quantiles、low-sample mask；輸入 pathway 的 `input_particle_count` 必須等於有效分母 |
| `ConnectivityStatistics` | raw matrix、各列有效分母、逐目標 visit fraction 與 cross-site event 內部 share；對角線另存為不適用遮罩，不能把拓撲上的不適用解讀為零連通；兩種比例不可混稱，且每格 raw count 不得超過該列有效分母 |
| `OutcomeStatistics` | 每站各停止狀態 raw count、total denominator、比例及 data-gap／numerical failure exposure；失敗成員不能進入有效分母 |
| `SourceReceptorStatistics` | receptor×boundary-kind×segment raw count、有效 receptor denominator、條件式通過比例、同 receptor／kind 內部 event share 與分箱 travel-age quantiles；兩種比例不可混稱，bootstrap CI 由下述 member-level 流程補入 |

所有 typed product 的 NumPy 陣列都要 defensive-copy、設為唯讀並保留公尺／秒及
`(y_cell, x_cell[, age_bin])` 軸語意。`report_statistics.py` 可重用
`binned_gaussian_kde_2d()` 與 `first_passage_quantiles()`，但必須再次核對 spec binding；不得
接受 renderer 傳入臨時 bandwidth、quantile 或 denominator。

有效成員政策、完整 identity 與 canonical SHA-256 先封裝於
`report_trajectory_identity.py`；固定容量雜湊選樣由 `report_trajectory_selection.py`
負責。完整 shard 的
season/tide/material 與有效 pathway 累加則由 `report_trajectory_stream.py` 以
`TrajectoryReportAccumulator.add_shard(...)` 作為唯一更新
入口。caller 先依 immutable run plan 順序驗證 shard，再把單一 shard 的結果交入；accumulator
不得保存已處理 shard。資料流固定如下：

1. `DATA_GAP` 與 `NUMERICAL_FAILURE` 是無效成員；其餘已終止狀態才可進入 F03/F05/F08/F09
   的有效 pathway、代表軌跡與材料統計。這項政策必須與 event aggregate 的有效分母完全相同；
   local／outer boundary、source-receptor、bed-contact 與 cross-site numerator 也只能由相同的
   有效成員母體累加。失敗成員只保留 outcome 與 failure-grid 診斷，不能留下較早事件作為
   條件式來源 numerator，否則 numerator 與 denominator 將描述不同母體。
2. 每個 shard 先依站點分組有效結果，再以既有 `stream_pathway_first_passage()` 建立 bounded
   chunk 並交給每站 `StreamingPathwayAccumulator`；不得直接使用含失敗成員的 aggregate
   pathway numerator 搭配有效分母。
3. 代表軌跡固定為每站 `DJF/MAM/JJA/SON × spring_proxy/neap_proxy` 八層等額保留。優先序是
   `ReportSpec` selection seed、policy、完整 particle identity 的 canonical JSON SHA-256；每層
   只保留最小的 `K/8` 筆。`event` arrival 不得混入 core 八層，任一層不足即 fail closed。
4. 正式 v2 trajectory 中，每個有效成員的 `ACTIVE` observation 必須有 `VALID` 環境 context；
   boundary terminal 可為 `NOT_SAMPLED`，但不得拿固定水深補值。深度定義為
   `eta_m - z_m`（positive-down，m），離底高度為 `z_m - bed_z_m`（m）；結果依固定 age/depth
   edges 累加，另保存 terminal-unsampled、invalid、underflow 與 overflow 計數。
5. 材料統計依正式十種 `material_id` 保存有效分母、至少一次 bed contact 的 member numerator、
   deposited member numerator 及兩者比例；不得只以中文顯示名稱作 join key。正文主要列為
   `oca_fishinggear_open_mesh_bundle × near_bed`，但其他材料／垂向層仍須保留。現行最小純
   計算入口為 `MaterialStatisticsAccumulator`／`build_material_statistics`：以一條 iterable
   一次加入 `ParticleResult`，由 scenario strata 解析站點與材質，對同一
   `scenario_id × member_id` 重複輸入 fail-closed；`BED_CONTACT` 與 `DEPOSITED` 事件按
   member 去重。`DATA_GAP`／`NUMERICAL_FAILURE` 不進有效分母；零分母的比例保存為
   `None`。基線不要求 repeated-contact 欄位，只有 `bed_reflect`／再懸浮敏感度才將重複
   接觸作為核心結果。
6. `finalize()` 必須核對每站有效 pathway 分母、八個代表層、十種材料及所有計數守恆後才回傳
   immutable summary；不能以 pooled season/tide 或 aggregate all-member pathway 作 fallback。

exact nonparametric member bootstrap 另由 `report_bootstrap.py` 負責。trajectory 一次串流期間只
把每個有效 member 的最小 sufficient categorical record 寫入 report 自有的暫存 spool；不保存
完整 `ParticleResult` 或 observation。之後依 receptor／boundary group 逐組讀取，對每個 replicate
以 sequential conditional binomial 產生與 `Multinomial(N; 1/N, ..., 1/N)` 完全同分布的 member
weights：第 `i` 筆在尚餘 `R_i` 次抽樣時取
`w_i ~ Binomial(R_i, 1/(N-i))`，最後一筆取得全部餘數。亂數子流由
`AggregateSpec.bootstrap_seed`、group key 與 replicate contract 經 SHA-256 派生；輸入順序固定為
run plan/shard/particle order。bootstrap 陣列可以使用 report partial 內的 memory-mapped file 維持
固定 RAM，但 spool、seed、replicates、confidence level、group member count、輸出 checksum 與
清理狀態都必須進入 report provenance。不得把 aggregate count 的 Poisson 模擬標成 member
bootstrap，也不得因 RAM 不足降級成近似方法。

### 3.5 Renderer、release writer 與公開驗證介面

正式 renderer 與發布層的責任必須分開。`report_render.py` 只接受已驗證 typed products、
來源 metadata 與 writer 指定的空白目錄，不能自行決定 final 路徑、重新讀取 trajectory、
改變分母或降低缺證閘門。它回傳 `RenderedArtifact` records；每個 available figure 必須同時
具備 PNG／SVG／PDF、caption JSON 與至少一個最小可重繪 data sidecar，每個 available table
必須同時具備 Parquet／CSV 與欄位 metadata JSON。F11/T05 與 F12/T06 缺證時不繪製空白
佔位圖，也不建立零列假表，而是只在 registry 保存成對 unavailable 狀態。

圖面 reproducibility 固定如下：

1. Matplotlib 使用非互動 `Agg` backend；`ReportSpec.renderer_style_version`、300 dpi、固定
   色盤、線寬、面板順序與公尺／秒軸不得由執行環境覆蓋。
2. SVG 的 `hashsalt` 由 report spec canonical SHA-256 固定派生；PDF／PNG metadata 不保存
   wall-clock 時間。caption 保存 renderer 版本、selected CJK font family／font-file SHA-256、
   run/aggregate/report hashes、raw n、denominator、units、CRS 與限制。
3. 繁體中文字型只允許由版本化候選清單解析；SERVER preflight 找不到含必要 glyph 的字型時
   必須停止，不可用 tofu、下載時未登錄字型或改成英文掩蓋環境缺件。
4. renderer 只能把秒換成日、公尺換成公里作顯示；data sidecar 與 typed product 仍保存
   原始秒／公尺值。所有換算因子與顯示單位都進入 caption／table metadata。

SERVER CLI 執行前必須明示 `MPLCONFIGDIR`，且該路徑是 operator 核定、既有、可寫、
non-symlink 的 task-specific cache directory；正式流程不得退回使用 `$HOME/.matplotlib`，也不在
未登錄的系統暫存目錄建立隨機 cache。`report-build` 在任何 release partial 建立前完成這項
環境 preflight，並固定 `Agg` backend，避免 batch node 因沒有圖形介面或 home 權限而中途失敗。

`report_release.py` 提供 `ReportReleaseWriter`、`read_report_release()` 與
`validate_report_release()`。writer 在 final 同父目錄建立自己擁有的 UUID partial，依固定
名稱寫入來源快照與 renderer 產品，從實際 bytes 建立 `ReportProductRef`／`ReportRegistry`，
完成 strict source rebind、registry closure、檔案集合、大小與 SHA-256 驗證後才以
`os.replace` 發布。writer 不覆寫 final、不跟隨 symlink，也不刪除別次執行留下的 partial。
公開 reader／validator 的驗證順序固定為：

```text
fixed root/source topology
  -> report_manifest schema and exact file inventory
  -> no-follow regular-file size/SHA-256
  -> report-spec/aggregate/source-run hash rebind
  -> figure/table registry exact F01–F12/T01–T06 closure
  -> product media/schema and caption/table metadata
  -> evidence-class and optional comparison/validation gates
```

source snapshot 除既有 run plan、progress、normalized config、input inventory 與 aggregate
manifest 外，必須另含完整 `aggregate_spec.json` 與 `report_spec.json`；manifest 保存兩者 raw
與 canonical SHA-256。公開錯誤只回傳固定 stage／reason code，不包含 SERVER 絕對路徑。

`report_pipeline.py` 的公開 `build_report_release(...)` 在任何 partial 建立前完成 run、aggregate、
config、spec、trajectory schema 與 optional evidence 的全部 binding。正式 F03/F09 必須確認每個
source shard 的 `trajectory_schema_version` 精確等於目前 v2 常數；legacy v1 即使一般 reader
仍可載入，也不得進入 report build。pipeline 只呼叫一次 complete trajectory iterator，依序
更新有效 pathway、代表軌跡、season/tide/material、環境 completeness 與 bootstrap spool，然後
關閉來源 shard reference，再交給 renderer 與 writer。任一步失敗都不得留下可被 validator
接受的 final release。

### 3.6 Formal baseline 與 scientific evidence 的分階段發布

完成真實 SERVER baseline run、aggregate 與 F01–F10/T01–T04，不代表 comparison 與 validation
已自動成立。第一份可交付的真實資料發布可由 operator 明示兩個 allow flags，證據類別只能是
`server_formal_baseline_evidence`；F11/T05、F12/T06 以成對 unavailable records 保存，核心圖表
仍必須完整且使用真實資料。這不是本機 synthetic，也不是最終 scientific evidence，但可供 PI
先檢查來源足跡、路徑、旅行時間、材質與停止結果。

後續 comparison 至少要有一份同資料期間、同五站、同 scenario/receptor/arrival 基線與明示
單一物理差異的 aggregate release；pipeline 先做 exact compatibility matrix，再計算差異，不接受
把 baseline 自減或以不同 coverage 比較。validation evidence 使用版本化 quantitative schema，
至少包含解析解誤差、時間步收斂、member 收斂、known-source synthetic coverage、restart 差異、
NumPy/Numba 差異與 forward/backward 指標；只記錄「pytest 通過」不合格。所有門檻必須在看到
正式比較結果前由版本化設定核定，未核定時只能維持 formal baseline，不能由 renderer 臨時選擇
看起來會通過的 threshold。

只有 comparison、validation、正式 run、所有 F/T 產品與 provenance 同時通過，且兩個 allow
flags 都關閉，才可產生 `server_scientific_evidence`。因此「可在 SERVER 完整執行」的工程驗收
順序是：先完成 formal baseline release，再執行已登錄 comparison/validation cases，最後升級
scientific release；各階段 final 目錄互為 sibling，不覆寫前一階段證據。

## 4. F01–F12 依賴與計算契約

| ID | 必要輸入 | 主要計算與圖面 | 缺少輸入時狀態 |
|---|---|---|---|
| F01 | config、geometry、inventory | 四域五站、受體／local/outer、投影與 17,544 小時 coverage | baseline 正式 run 不得缺 |
| F02 | plan、scenario strata、config | 10×20×50、M、case、方法與計數 | baseline 正式 run 不得缺 |
| F03 | source trajectories、geometry、strata | 預先登錄 hash/quantile 規則選樣；plan view + depth–age | source trajectory 不完整即整項失敗 |
| F04 | event grids、boundary arrays、spec | raw exit、三帶寬 KDE、50/75/90% HDR、弧長分布與 bootstrap CI | baseline 正式 run 不得缺 |
| F05 | pathway arrays、site denominators | visit fraction、residence、first-passage median/IQR、低樣本遮罩 | baseline 正式 run 不得缺 |
| F06 | source-receptor、cross-site、denominators | row-normalized matrix、raw n、方向性 2×2 A 區診斷 | baseline 正式 run 不得缺 |
| F07 | boundary/source travel histograms | ECDF/分位數/censoring，秒轉日只在顯示層 | baseline 正式 run 不得缺 |
| F08 | trajectory stream + season/tide strata | 固定 4×2 small multiples、共同 extent/scale/denominator | strata 不完整即失敗，不做 pooled fallback |
| F09 | trajectory stream + material/vertical events | 10 類 proxy、`near_bed` 漁具優先 panel、首次 bed contact／assumed deposition、depth–age quantiles | vertical 資料不完整即失敗；定性簡報不得充當數量先驗 |
| F10 | outcomes、failure grids、inventory timeline | raw count + denominator + failure/reconstruction exposure | 無 provenance 時明示 unavailable 子 panel |
| F11 | baseline + comparison releases | difference map、HDR overlap、rank/travel/contact 差 | `unavailable_missing_comparison` |
| F12 | validation evidence | M/dt 收斂、known-source、restart、backend 差異 | `unavailable_missing_validation_evidence` |

KDE 必須使用 `AggregateSpec.kde_bandwidths_m` 的三個公尺帶寬；正文採用哪一個帶寬由
核准 spec／report policy 明示，不能在看過結果後挑選。HDR 以 probability mass 0.5、0.75、
0.9 建立 nested mask。訪格比例的分母是有效 members；停留時間保留秒；首次抵達時間由
age histogram 估計時必須在 caption 明示 bin resolution，不能宣稱為未分箱的精確中位數。

bootstrap 使用 `AggregateSpec.bootstrap_seed`、replicates 與 confidence level。來源段／受體
CI 需要 member-level 重新抽樣時，必須從已驗證 trajectory stream 建立可重現 sample units；
不得對已加總 count 做 Poisson 假設後仍標成 nonparametric member bootstrap。

## 5. T01–T06 依賴與欄位

| ID | 來源 | 生產契約 |
|---|---|---|
| T01 | config、input inventory、preflight evidence | domain/month/schema/checksum、rows、coverage、gap/reconstruction/NWW rebuild |
| T02 | scenario strata、plan | 五站 10×20×50、A 區 20,000、全案 50,000、M、缺／重列 |
| T03 | run progress/benchmark、outcomes | released/effective/completed、raw+denominator+percentage、wall/CPU/RSS/I/O/steps |
| T04 | source-receptor、travel hist、bootstrap | receptor/segment/raw n/有效分母/relative weight/CI/travel/rank |
| T05 | comparison releases | case/參數/HDR overlap/rank/travel/contact difference/限制 |
| T06 | validation evidence | analytic/dt/M/synthetic/restart/backend/forward-validation quantitative metrics |

CSV 只作交換格式；正式 typed table 以 Parquet 保存欄位型別、nullability 與 units sidecar。
所有比例欄位必須同列保存 raw numerator 與 denominator。缺值、資料缺口、數值失敗與不適用
狀態使用不同欄位／狀態碼，不以 0 或空字串代替。

## 6. report release 固定拓撲與原子發布

預定 final 路徑是 source run 同層的 `<run_id>.report-v1`；不得寫進 source run 或 aggregate
release。writer 只建立自己擁有的同父目錄 partial，完成所有 product checksum、schema、
registry、caption/data sidecar 與 source rebind 後才 `os.replace`。既有 final、symbolic link、
broken link 或其他 partial 一律不覆寫、不清理。

```text
<run_id>.report-v1/
├── report_manifest.json
├── figure_registry.json
├── table_registry.json
├── source/
│   ├── aggregate_manifest.json
│   ├── run_plan.json
│   ├── run_progress.json
│   ├── normalized_config.json
│   └── input_inventory.json
├── figures/main/                 # F01–F12；每項依 registry 列出 PNG/SVG/PDF
├── figures/supplement/
├── tables/                       # T01–T06 Parquet + CSV
├── caption_sidecars/             # 一項一份 canonical JSON
└── data_sidecars/                # 最小可重繪 typed arrays/tables
```

因不同 figure 的 sidecar 數量不一，report schema 不以「掃描任意檔案」驗證；manifest 必須
列出 exact relative path、role、size、SHA-256、media type、figure/table ID 與 schema。
validator 先驗 topology/manifest/checksum，再驗 source binding、registry closure、sidecar
schema，最後才讀圖表產品。公開失敗報告只回固定 stage，不包含 SERVER 路徑。

## 7. registry 狀態與不可用項目政策

每個 F01–F12、T01–T06 必須恰有一列，不能省略。允許狀態固定為：

- `available`：所有必要輸入、產品、caption、data sidecar 與 checksum 皆完整；
- `unavailable_missing_comparison`：只適用 F11/T05，且 comparison input 為空；
- `unavailable_missing_validation_evidence`：只適用 F12/T06；
- `unavailable_missing_provenance_subpanel`：只允許 F10 的明確子 panel，主停止結果仍需 available；
- `not_applicable_by_registered_design`：只有研究設計明示不適用且提供版本化理由時可用。

`available` 列必須保存非空輸入 SHA-256 集合、原始有效樣本數、具名分母與分母計數；
`unavailable` 列則不得留下這三類數值，避免把缺證狀態誤讀成零樣本統計。同一份 release
中每個產品相對路徑只能被一個 F/T 列引用，不能用同一張圖或同一個 sidecar 重複充當
不同成果。F11/T05 與 F12/T06 分別共享同一 comparison／validation evidence 依賴，故狀態
必須成對為 `available`，或在相應 allow flag 開啟時成對為對應的 unavailable 狀態；不接受
一圖可用但其定量表缺證的半套發布。

不得使用 `todo`、`unknown`、`best_effort` 或沒有原因的 `skipped`。正式 release gate 預設要求
F01–F10/T01–T04 available；是否允許 F11/F12/T05/T06 unavailable 必須由 CLI 明示的
`--allow-missing-comparison`／`--allow-missing-validation-evidence` 控制，且 manifest 保存該
政策。使用任一 allow flag 的 formal run 必須標為 `server_formal_baseline_evidence`，不得標成
`server_scientific_evidence`；最終送 PI 的完整報告不得使用這兩個 allow flags。

## 8. CLI、驗證與 SERVER 順序

正式入口與必要旗標：

```text
lbt report-build --run ... --aggregate-release ... --report-spec ... --output ... \
  --evidence-class {synthetic_engineering_evidence,server_pilot_evidence,
                    server_formal_baseline_evidence,server_scientific_evidence} \
  [--comparison-release ...] [--validation-evidence ...] [--checkpoint-root ...] \
  [--allow-missing-comparison] [--allow-missing-validation-evidence]
lbt report-validate <report-release>
```

`--output` 必須是不存在的明示 final 路徑，CLI 不從目前工作目錄猜測發布位置；相同 final、
symlink 或 broken symlink 一律失敗。`--evidence-class` 必填，pipeline 仍會依 run kind、SERVER
來源 provenance、comparison／validation closure 與 allow flags 重新驗證，不能靠文字旗標把
synthetic 或 pilot 產品升級。`server_scientific_evidence` 禁止兩個 allow flags，且 comparison
與 validation 必須實際存在並通過 exact binding。`report-build` 成功時只輸出 JSON-safe final
摘要；任何失敗回傳非零，且不把絕對 SERVER path、token 或帳密寫進 stdout/stderr。

本機驗證順序：typed records/statistics → synthetic single-run products → unavailable 狀態 →
atomic report smoke → validator tamper tests → renderer image/PDF visual QA → full pytest/Ruff/wheel。
這一階段只形成 `synthetic_engineering_evidence`。

SERVER 順序：唯讀環境與資料 preflight → 小型真資料 pilot → 人工中止/resume/reconcile →
aggregate release → report release → validator → 逐圖表科學 QC。pilot 通過後才依量測核定正式
M/shard/資源並執行五站 50,000 base scenarios。SERVER 上每份 F/T 的 registry row 必須標為
`server_scientific_evidence` 並綁定當次 run/release checksum；不得沿用本機 synthetic 圖檔。
