# 涵蓋 2024-01-01 的新竹外海 24 小時工程試跑參數與執行紀錄（r5）

> 文件狀態：SERVER r5 工程試跑已完成，`run lifecycle=COMPLETE`、revision `204`、
> 4/4 shards 完成；執行計數、後處理驗證、工程 preview 證據與 BayTrace 參照呈現 v3
> 的主代理圖面審查均已回填。本文件仍是 pilot-only、可重建的工程設定與執行紀錄，
> 不是全期研究成果，也不是正式 48+2 arrival selection 的替代品。本文所稱「來源足跡」
> 均指條件式來源足跡或相對來源權重的工程描述；不得由本工程試跑推論絕對來源、因果
> 歸因或全期代表性。

## 1. 重建入口與版本界線

本案例由 `inputs-build` 的版本化明示入口建立。`--pilot-arrival-utc` 目前只接受下列
一組固定 site／UTC pair；不接受 offset、非整點、任意替換時刻或 `--formal-release`：

```bash
uv run lbt inputs-build \
  --config "$PILOT_CONFIG_TEMPLATE" \
  --destination "$LBT_SCRATCH_ROOT/hsinchu-2024-01-01-24h-inputs-v1" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --pilot-arrival-utc hsinchu=2024-01-02T01:00:00Z
```

建置前必須先通過 config schema、OCM／NWW3 accepted-product metadata 與時間軸檢查；
建置時再逐筆檢查 25 個 UTC 節點的產品軸、OCM native gap-safe、OCM surface scalar，
以及 NWW metric location 四角支援。任一節點缺失、無效或跨缺口即 fail closed；不使用
2024-01-01T00:00:00Z 作補件，不使用最近值、零值或未登錄的跨缺口時間內插。若要跨越已知
缺口，只能先完成版本化重建、blocked cross-validation 與 Lagrangian sensitivity，並保存
observed／reconstructed provenance；未通過時只能改選 gap-safe arrival window。

## 2. 可引用的固定參數與證據

下表把 r5 已確認的 calibration-bound 候選與 execution scalar 直接列出，避免 PI
報告只看到「沿用設定」而無法核對數值。這些數值是參數紀錄，不會把範例 template
中的 `null` 自動視為可執行值；真正執行時，仍以生成後的 config 與
`pilot_execution_binding`（包含 `candidate_values`／`applied_fields`）為唯一證據。
r5 已提供的 calibration、config 與 preflight 指紋列於第 5 節，SERVER 執行證據列於
第 6 節，後處理與工程 preview 驗證列於第 7 節；已完成並經主代理圖面審查的 BayTrace
參照呈現 v3 證據列於第 8 節。

| 項目 | 固定設定／本案例數值 | config key 或 manifest 證據 | 設定理由與限制 |
|---|---|---|---|
| 研究範圍 | `study_site_id=hsinchu`、analysis region B | `study_sites[].study_site_id`、`analysis_region_id`；`arrival.json`、`receptor.json` | 只做新竹單站展示；不改五站正式分母。 |
| flow／local domain | `hsinchu_cache_v3`；local domain 與 flow domain 完全相同 | `study_sites[hsinchu].flow_domain_id`、`local_domain_policy=same_as_flow_domain`；`domain.json`／`local.json` 的 `local_equals_flow=true` | 核心只限制水平 receptor candidate polygon，不縮小 B 區 forcing 或 local 邊界。 |
| 核心中心與半徑 | `(120.45, 24.75)`；`12,500 m` | `study_sites[hsinchu].anchor_lonlat`、`receptor_core_radius_m`；`receptor.json` 的實際 lon/lat 以同一 flow-domain AEQD 投影核對 | 距離在公尺制 AEQD 計算；不以經緯度差近似。候選區是 core 圓與既有 local/static-ocean polygon 交集。 |
| receptor 位置／水層 | 歷史 24 小時試跑沿用 5 個固定水平位置 × 4 個固定水層 = 20 個 scenario 起始位置 | `scenarios.expected_receptor_count_per_site=20`、`horizontal_receptor_count_per_site=5`、`vertical_receptor_count_per_horizontal_location=4`；歷史 `receptor.json` 20 筆 hsinchu records | 僅保留位置核對／pilot provenance；current formal 已改為五個 random horizontal face × 每面四個 random vertical draw，不回寫此歷史固定點。 |
| arrival 方向與 25 時次 | 到達 `2024-01-02T01:00:00Z`，反向至 `2024-01-01T01:00:00Z`，共 24 h；inclusive 25 個逐時節點；`max_backtrack_days=1.0` | CLI `--pilot-arrival-utc`；`arrival.json` 的 `explicit_pilot_window`；`ocm_gap_safe_arrival.json` 的 `horizon_start_utc`、`horizon_end_utc`、`expected_step_count=25`、`supported_step_count=25` | OCM native／surface 此月第一個有效節點是 `2024-01-01T01:00:00Z`，因此本案例不得寫成從 `2024-01-01T00:00:00Z` 開始；NWW 雖從 00Z 起，也不能補 OCM 00Z。 |
| pilot selection identity | policy `hsinchu_explicit_24h_window_replacement_v1`；新 arrival ID 由 site、UTC、policy、`design_version` 穩定雜湊；metadata 保存原／替換 ID | `arrival.json` record metadata：`pilot_replaced_arrival_time_id`、`pilot_replacement_arrival_time_id`、`pilot_selection_scope`；artifact index `source_bindings.pilot_selection_scope` | 決定性替換一筆既有 hsinchu selection；label 使用 `pilot_explicit_window`／`explicit_24h_window`，不冒充潮汐 phase 或事件。 |
| 全案輸入計數 | 250 arrivals、5,000 dynamic pairs | `arrival.json` 250 records；`initial_conditions.json` 5,000 records；`validate_input_derivatives` summary | 替換不改五站 50×5 的 source coverage；本案例的 20 個 run scenarios 由 run-create 精確 selector 產生。 |
| OCM native／surface 欄位 | native：mesh、`zcor`、`elev`、`wetdry_elem`、`hvel`、`vertical_velocity`、`diffusivity`；surface：`eta_m`、`u_surface_mps`、`v_surface_mps`、`surface_z`、`valid_mask_surface`、`qc_flags` | `forcing_inventory.json` 的 OCM schema 3 `ocm_native` products、source files、canonical time；`initial_conditions.json` 的 `ocm_time_origin=observed`；native arrays 包含 `vertical_velocity.npy` | native dynamic pair 使用實際 UTC 的 OCM 原生切片；`vertical_velocity` 是 OCM 垂向速度（m/s、z positive-up），`diffusivity` 才是 Kz 候選；surface 只供 arrival scalar 與逐時支援檢查，不以 native 全域 `hvel` 代替。2024-01 整月 OCM 為 `740/744`，缺 4 個時次：`2024-01-01T00:00:00Z`、`2024-01-14T00:00:00Z`、`2024-01-18T00:00:00Z`、`2024-01-31T00:00:00Z`；本 pilot exact window 仍為 25/25。 |
| NWW3 欄位與空間支援 | `significant_wave_height`、`peak_frequency`、`peak_direction_raw_deg`、`valid_mask_wave`；metric location 每小時四角 static／dynamic／有限值／物理條件均有效 | `forcing_inventory.json` 的 NWW3 schema 1 `nww3_analysis`；arrival metadata 的 `metric_location_*`；pilot build 對 25 小時重做 exact-hour 四角 gate | 不尋找最近有效格點、不補零、不做時間內插；四角任一小時失敗即拒絕。2024-01 整月 NWW 為 `744/744`。 |
| 材質、行為與沉降 | run-create 精確指定 `oca_fishinggear_open_mesh_bundle`；`behavior=sinking`；`settling_velocity_mps=-0.002 m/s`（z 軸向上為正） | `physics.settling.material_classes[material_id=oca_fishinggear_open_mesh_bundle].settling_velocity_mps`；`material.json`；run-create `--pilot-material-id` | 這是已驗收的嚴格負沉降候選，仍屬 provisional proxy，不是 OCA 單體量測；只納入設定描述的沉沒適用條件。 |
| Stokes／experiment | `experiment_case_id=finite_depth_stokes`；Stokes 啟用，有限水深 bulk formulation | run-create `--experiment-case finite_depth_stokes`；`physics.stokes.enabled=true`；`physics.stokes.formulation=finite_depth_monochromatic_bulk`；NWW manifest／runtime input binding | 案例不切換 engine 或另造波浪參數；有限水深公式與 `invalid_wave_policy=stop_with_wave_data_gap` 由生成 config／binding 固定，本紀錄不把 BayTrace 範例值帶入。 |
| RK4、步長、輸出與步數上限 | 四階 Runge-Kutta 法（RK4）；自適應 `dt_min_seconds=0.1 s` 至 `dt_max_seconds=30 s`；raw output `output_interval_seconds=300 s`；`maximum_step_count=1,000,000` | generated pilot config 的 `integration.deterministic_method`、`integration.dt_min_seconds`、`integration.dt_max_seconds`、`integration.output_interval_seconds`、`boundaries.maximum_step_count`；`pilot_execution_binding.execution_scalar_snapshot` | 這些是 r5 execution scalars；不改 engine，不採 BayTrace demo 的 Euler／`3600 s`。最大步數是執行保護上限，不是收斂證明。 |
| Kh／Kz | `Kh=0.4429482105965188 m²/s`；`Kz=0.0013526236792521886 m²/s`；Smagorinsky `floor=0 m²/s`、`cap=12.77813526595494 m²/s` | generated pilot config 的 `physics.horizontal_diffusion.constant_kh_m2ps=0.4429482105965188`、`physics.vertical_diffusion.constant_kz_m2ps=0.0013526236792521886`、`physics.horizontal_diffusion.smagorinsky.kh_floor_m2ps=0`、`physics.horizontal_diffusion.smagorinsky.kh_cap_m2ps=12.77813526595494`；`pilot_execution_binding.candidate_values`、`applied_fields` | **Kh** 是工項 PDF 式 (9)、(10) 明列的水平渦動擴散（horizontal eddy diffusivity）。**Kz** 是本專案三維隨機運動額外採用的垂向擴散參數，候選取自 OCM `diffusivity`；工項 PDF 未明列 Kz，不得寫成工項明列參數。 |
| M／seed | `M=1`；`master_seed=20260831` | generated pilot config 的 `scenarios.members_per_scenario=1`、`scenarios.master_seed=20260831`；run plan 的 seed／scenario selection binding | M 是同一 scenario 的 member 數，不是額外情境因子；單一 M 不代表收斂已完成。 |
| 分片、作用中 chunk、checkpoint 與 forcing 常駐 | `4 shards × 5 scenarios`；`shard_scenario_count=5`；`active_chunk_size=5`；每 `1,000 sweeps` checkpoint；`max_resident_forcing_months=2` | generated pilot config 的 `execution.shard_scenario_count`、`execution.active_chunk_size`、`execution.checkpoint_interval_sweeps`、`execution.max_resident_forcing_months`；run plan／checkpoint binding | 20 個 scenario 以每 shard 5 個切成 4 個分片；chunk 與 checkpoint 是可重啟及資源控制設定，不改變 scenario identity 或科學參數。 |
| 邊界 | surface `reflect_and_record_contact`；sinking bed `deposit_on_first_contact_and_stop`；coast `stop_and_record_contact`；flow boundary `stop_at_first_crossing` | generated config：`boundaries.surface_suspended_or_sinking`、`boundaries.bed_sinking_or_near_bed`、`boundaries.coast_baseline`、`boundaries.flow_domain_open_boundary`；`open.json`、`local.json`、`domain.json` | 這些是本案例沿用的已驗收邊界政策；核心圓不是 local boundary，不因 pilot 另寫一套邊界規則，也不把跨另一站 local 視為站點轉移。 |
| gap 政策與 exact window 品質 | `inclusive_observed_ocm_only_no_cross_gap`；exact window `25/25` hourly nodes、`crossed_gap=false`、`missing_utc=[]` | `ocm_gap_safe_arrival.json`；arrival metadata `explicit_pilot_window_time_support` | 任一 OCM native／surface／NWW exact-hour 或 NWW 四角支援缺失都 fail closed；不跨越已知缺口，不以零值、最近值或未登錄內插補齊。 |
| release／formal 界線 | artifact 可由 hash、sidecar、artifact index、closure 與 input validator 驗證；release config 維持 `config_status=generated` | `artifact_index.json`、`artifact_bindings.json`、`release_binding`、`release_approval`；formal validator error `formal_pilot_explicit_window_not_48_plus_2` | pilot component 可作工程輸入紀錄，但非正式 48+2 selection；`--formal-release` 與 pilot 入口互斥。 |

## 3. run-create 精確選取

建置並驗證 input artifact 後，先以 generated pilot execution config 綁定 calibration
candidate，再以三個識別碼精確建立一個 pilot run。`ARRIVAL_ID` 應取自
`arrival.json` 中 `pilot_replacement_policy_id=hsinchu_explicit_24h_window_replacement_v1`
的 record：

```bash
uv run lbt run-create \
  --config "$PILOT_CONFIG" \
  --input-inventory "$PILOT_INPUT_ARTIFACT" \
  --destination "$LBT_OUTPUT_ROOT/runs" \
  --run-id "$RUN_ID" \
  --run-kind pilot \
  --experiment-case "$PILOT_EXPERIMENT_CASE" \
  --pilot-study-site-id hsinchu \
  --pilot-arrival-id "$ARRIVAL_ID" \
  --pilot-material-id oca_fishinggear_open_mesh_bundle
```

這個精確 selector 對 hsinchu 五個水平位置各取四個既有垂向 receptor，故來源情境數為
20；它不改 `arrival.json` 的 250 筆母體，也不把 20 筆執行樣本寫回正式 selection。
run-create 仍須重新驗證完整 input artifact、config hash、component hash、seed 與
scenario selection binding。

## 4. BayTrace backward example 對照

BayTrace backward example 只作參數呈現參照，不是本專案 runtime 設定或輸入資料契約：

| 對照項目 | BayTrace example | 本案例 |
|---|---|---|
| 粒子／情境 | 3 particles | run-create 精確選取 20 scenarios；M=1，實際粒子數仍由 run plan 產生 |
| 回溯長度 | 12 h | 24 h；2024-01-02 01Z 回到 2024-01-01 01Z |
| `dtm`／步長 | `dtm=120 s` | calibration-bound adaptive RK4：0.1–30 s |
| spool／輸出 | `nspool=30`（1 h） | `output_interval_seconds=300`（5 min） |
| `ndeltp` | 1 | 不作本案例的 engine 參數；M 由 `members_per_scenario=1` 明示 |
| advection | passive backward；CLI default Euler，需 `--advection 2` 才為 RK4 | 專案 config 明示 `integration.deterministic_method=rk4`；不使用 BayTrace default |

因此 BayTrace 對照只說明命令列選項與時間／輸出尺度的差異，不構成 OCM／NWW3 欄位、
沉降、Stokes、Kh/Kz、邊界或來源解讀的替代定義。

本 pilot 的 20 scenarios、24 h、0.1–30 s 自適應 RK4 與 300 s 輸出，都是本專案的展示
設定；BayTrace 只作介面與行為參照，不是本 pilot 的輸入契約或科學參數來源。

## 5. r5 calibration 與可重建指紋

r5 calibration 共保存 `5,000/5,000 valid` pair samples。這表示校準輸入 pair 均通過
該階段的有效性條件，不表示 20 個 pilot scenario 已完成軌跡、步長收斂、member
收斂或科學驗收。`Kh`、`Kz`、floor 與 cap 的數值須與下表的 calibration report、
manifest 及 pair parquet 共同追溯，不能只引用本文件的手抄值。

| 證據項目 | r5 指紋／狀態 | 追溯用途與限制 |
|---|---|---|
| calibration report SHA-256 | `f2eaa090c6154fbcf7c53b4189e4447a7d421a4a8cd0af8f65646ddcce9246b4` | 核對 Kh、Kz、floor、cap 與 5,000/5,000 valid 摘要的報告內容。 |
| calibration manifest SHA-256 | `a2d9b45fa576428372d8267af461164569c8ba99b7b55b9cc5100a46c5b39df7` | 核對校準輸入來源、版本與 pair 清單；此項不是 pilot 輸出 checksum。 |
| calibration pair parquet SHA-256 | `5250a9cb0e9d126d038d44060a250edd63034671af8a4f36412157845299479f` | 核對逐 pair 的校準樣本；不可用來宣稱軌跡結果已完成。 |
| config semantic hash | `dfc8c5a1fcdbf07c293d93bc2204fd17d11749b66a0d6579c07088cf618d6909` | 核對設定語意；需與 generated config／`pilot_execution_binding` 一起使用。 |
| config raw SHA-256 | `8b8882df0d2e7ffb014dcdcdc7df3e033dbc6fdd9715d1ee04127c160666924c` | 核對設定檔原始位元組；格式或序列化變更也可能造成差異。 |
| preflight raw SHA-256 | `a921ef1bf850f8db66eda6c283eeba8eb793204817bf389f7d0409bccafb9c08` | 核對 preflight 原始證據；不代替本次 run 的結果驗證。 |
| Git commit | `4fa055d0b3fd27edd749856a2075e14f4f9103c2` | 固定程式版本邊界。 |
| deployment tree | `fd98f746b150db886a8bfc295d6eb2e51a31a811a4c2a2310a24b8401a75b322` | 固定部署樹內容指紋；不可解讀為輸出資料 checksum。 |
| dirty flag | `false` | r5 執行版本無未提交工作樹變更。 |

## 6. SERVER r5 工程試跑執行證據

本節記錄 SERVER 上已完成的工程 run lifecycle。`COMPLETE`、shard 完成、粒子計數與
資源量測證明本次執行依計畫收束；不等於正式研究成果、來源歸因、時間步長／系集
收斂或獨立觀測驗證。尤其所有粒子的 `forcing_start` 只表示回溯抵達本案例選定的
forcing window 起點，不是初始離域、數值失敗或科學來源結果。

| 證據項目 | r5 SERVER 實際值 | PI 解讀與限制 |
|---|---|---|
| run identity | `run_id=b-hsinchu-core-fishinggear-m1-24h-20240101-r5` | 單站、單材質、`M=1` 的工程試跑識別碼；不代表正式研究 run。 |
| 程式版本 | Git commit `4fa055d0b3fd27edd749856a2075e14f4f9103c2` | 與第 5 節 provenance 指紋一致，固定本次執行的程式版本邊界。 |
| run lifecycle | `COMPLETE`；revision `204`；`errors=[]` | lifecycle 完成只證明執行控制與收束紀錄完整，不等同科學驗收。 |
| shard 完成 | `4/4 shards COMPLETE`；`errors=[]` | 四個分片均完成；不可只以單一 shard 推論整個 pilot。 |
| scenarios／particles | `20 scenarios / 20 particles` | 本次為 20 個精確選取 scenario、每個 scenario 一個 member；不是正式五站母體。 |
| observations | 每粒子 `289` 筆；總計 `5,780` 筆 | 保存逐粒子觀測計數；不把觀測列數直接解讀成有效科學樣本或來源權重。 |
| selected forcing window | `2024-01-01T01:00:00Z` 至 `2024-01-02T01:00:00Z`；25 個整點皆有 OCM／NWW；`missing_utc=[]`；`crossed_gap=false` | 本案例由 01Z 起算，未使用 00Z 補件；選定視窗內不跨已知缺口。 |
| 終止狀態 | 全部粒子為 `forcing_start`；回溯年齡約 `86,400 s` | 表示粒子抵達選定 forcing window 的時間起點；**不是**初始離域、數值失敗，也不可描述成科學來源結果。 |
| steps | 依 shard 回報順序：`188,457`、`211,623`、`205,892`、`219,098`；總計 `825,070` | 保存四個 shard 的實際步數；總步數是執行量測，不是軌跡科學品質證明。 |
| shard wall seconds | 依 shard 回報順序：`924.2677`、`936.0910`、`922.2272`、`1065.8163 s` | 分片並行時整體 wall time 以最慢 shard `1065.8163 s`，約 `17.76 min` 呈現；不可將四個 wall seconds 相加。 |
| aggregate CPU | `3832.2683 CPU-s`，約 `63.87 CPU-min` | 這是 aggregate CPU 使用量，不是並行整體 wall time。 |
| peak per-process RSS | `426,991,616 bytes`，約 `407 MiB` | 單一 process 峰值常駐記憶體；不等於所有 process 記憶體總和。 |
| run plan／final progress | run plan created `2026-09-07T08:12:53.147654Z`；final progress `2026-09-07T08:31:58.474911Z` | 兩時間戳差值包含啟動與排程，不等同純運算 wall time；不得用它取代 shard wall evidence。 |

## 7. r5 後處理驗證與工程 preview 證據

本節補入已驗證的 run validator、事件統計與工程 preview 證據。這些結果證明輸出拓撲、
事件／觀測計數與 preview artifact 可被稽核；仍不把工程試跑升格為正式研究成果或來源
歸因。

| 後處理證據 | r5 實際值 | PI 解讀與限制 |
|---|---|---|
| `lbt-validate-run --require-complete` | `valid=true`；`errors=[]`；`completed_shard_count=4`；`particle_count=20`；`scenario_count=20`；`shard_count=4`；`run_lifecycle=COMPLETE` | 證明 run workspace、分片覆蓋與核心計數通過完整性驗證；不等於物理或科學驗收。 |
| `final_status` | `forcing_start=20` | 20 顆粒子均抵達選定 forcing window 起點；此狀態不可描述成科學來源結果。 |
| `event_type` | `forcing_start=20`；`surface_contact=42,699`；`event_total=42,719` | `surface_contact` 是反射／接觸事件計數，不是 42,699 顆粒子，也不是停止原因；本次沒有 `surface_regime_exit`。事件總數不可直接當粒子數或來源權重。 |
| minimum clamp | `minimum_clamp_total=0` | 本次沒有記錄最小步長 clamp；這是執行診斷，不是收斂證明。 |
| observations | `5,780`；20 顆粒子各 `289` 筆 | 與第 6 節粒子／觀測計數一致；觀測列數不直接等於有效科學樣本或來源權重。 |
| 工程 preview 發布 | `artifact_kind=pilot-preview-v1`；`schema=1.0.0`；`output_count=8` | 8 個輸出包含 1 個 manifest 與 7 個產品；這是工程展示 artifact，不是正式研究報告。 |
| preview manifest 自身 SHA-256 | `f92bea2d3192098e3faa7467df541ffe59ae52e918deb1b91dac2eb6dbab2659` | 用於核對 manifest 本身；不與產品檔案 checksum 混淆。 |
| 產品依 manifest 本機重算 | `checked=7`；`bad=[]` | 7 個產品均通過本機逐檔 checksum 比對。 |

### 7.1 工程 preview 產品 checksum

以下 SHA-256 是 preview manifest 所列 7 個產品的本機重算結果；不代表圖面已完成
forcing_start 語意審查，也不代表正式研究輸出已發布。

| 產品 | SHA-256 |
|---|---|
| `horizontal.png` | `c3c6f79f68d4bf9e4446691feb675159ff291dcb72198a5e742d3e5ccf19555b` |
| `depth_age.png` | `0aa8681dda6770976b882f79962f49a97757294b2067ad2d6c765d6912fa8094` |
| `terminal_counts.png` | `5779eb976ee8703fc5c057ac6b2411f9c8a989af851f09dd1ded0e3ca04d0d0f` |
| `summary.json` | `342c91f2764ce3793c733f20a56ff7260a403f29565c3d25105be72a4c5816d9` |
| `observations.csv` | `58d7ccc42a0ebc0c360f84d02342269763da3f03016007f0576db75907f4f910` |
| `particles.csv` | `a7590e0b7b554b3eebf6d3c38d88e2ea1d93c3b49e5fd6904320576453c8c0f1` |
| `README.md` | `0e2e2b27cd5700c93c7e999d59cea2cae27b85f11f6c06442d8ea1299c787818` |

### 7.2 `surface_contact` 的禁止性解讀

`surface_contact=42,699` 只表示粒子在海面邊界發生反射／接觸的事件次數。它不是
42,699 顆粒子、不是 42,699 個停止案例，也不是來源量或來源權重；本次沒有
`surface_regime_exit` 事件，不得把兩者當成同義詞。真正的
粒子終止摘要仍是 `final_status=forcing_start:20`，其語意是抵達選定 forcing window
起點，而非科學來源結果。

## 8. BayTrace 參照呈現 v3 圖面審查證據

BayTrace 參照呈現 v3 已完成並經主代理圖面審查。這是沿用同一份 r5 工程試跑來源的
圖面與語意呈現，不是重新執行 BayTrace 模型，也不是正式研究成果。輸出目錄為：

`work/pi-pilot-20260907/hsinchu-core-24h-20240101-r5/figures-baytrace-v3`

| 呈現契約 | v3 實際值 | 解讀與限制 |
|---|---|---|
| 樣式版本 | `style_version=1.2.0` | 固定本次 BayTrace 參照呈現的圖面與標籤語意版本。 |
| 來源不變 | `source_unchanged=true` | v3 只修正呈現與文字語意，未改寫 r5 軌跡或後處理來源。 |
| `forcing_start` 年齡容差 | `forcing_start_age_tolerance_seconds=0.0001` | 以 0.0001 秒容差判定是否完成本次 24 小時回溯；不改變積分結果。 |
| `forcing_start` 年齡狀態計數 | `forcing_start_age_state_counts={"completed":20}` | 20 顆粒子均被呈現為完成本次資料時間窗內的 24 小時回溯。 |

### 8.1 `forcing_start` 與 `max_age` 的語意邊界

| 狀態 | 本文件採用的中文語意 | 禁止性解讀 |
|---|---|---|
| `forcing_start` | 到達本次資料時間窗起點（完成 24 小時回溯）。本 r5 工程試跑共 20 顆粒子符合此狀態。 | 不得解讀為找到科學來源、初始離域或數值失敗。 |
| `max_age` | 到達設定回溯時間上限（24 小時）。 | 與資料時間窗起點是不同停止語意，不得與 `forcing_start` 合併標示或互相替代。 |

### 8.2 BayTrace 參照呈現 v3 SHA-256

| 檔案 | SHA-256 |
|---|---|
| `horizontal_overview.png` | `50ad8668a4bd9cd32ddd90bc43d49d52b63be276d303a362425773f4e60982aa` |
| `horizontal_local.png` | `7a72ab7b5cbae56b651b7c68ea828e513dd08146c78f1033f2dd112d3da5e400` |
| `depth_age.png` | `b8924fe1f910504eefc0e7b0798141309d4063a99ca70ca5960df811f4b2e0cc` |
| `terminal_counts.png` | `2255b6ec1bcbdffe0c20356141d9cfed9c43a115ba6ee7457cdef0b0ee54b0c1` |
| `README.md` | `ec7e2ffdce3c43ac60506db3856bb6fa45d36e521e32f7ac2e27192f59946f58` |
| `manifest.json` | `a8e07dc32d65b06e8e289513b853e66ef6fc3f4fb23f2464cd5df1759b5c26ea` |

此 v3 圖面審查只確認呈現契約、終止標籤語意與檔案指紋；不改變 `M=1` 的工程試跑
限制，也不構成來源機率、因果歸因、member convergence 或正式科學驗證。

## 9. 展示限制與驗收證據

- 本次 `COMPLETE` 是工程試跑的 lifecycle 結果，不是正式研究完成、來源歸因、科學
  驗證或全期代表性證明；`forcing_start` 也不具科學來源語意。
- 這是新竹單站、單一明示 24 小時視窗、單一材質與 `M=1` 的工程展示；不代表兩年
  17,544 小時母體、正式 48+2 strata、材質敏感度或 member convergence 已完成。
- `generated`、`approved component` 或輸入 validator 通過只代表檔案／schema／hash／
  cross-reference 契約成立，不等於 OCM／NWW3 科學驗收或軌跡結果驗證。
- 不以本案例宣稱全期來源歸因；任何報告應使用條件式來源足跡、相對來源權重及明示
  分母，並保留 data-gap、boundary、數值失敗與未取樣狀態。
- 直接驗證應包含 `tests/test_input_derivation.py` 的合法替換、25 節點中間小時缺失、
  dynamic pair／gap manifest closure 與 formal zero-write 測試，以及
  `tests/test_config.py` 的核心欄位早期 schema gate；真實 SERVER 建置仍須保存當次
  config、source manifest、Git dirty flag、hash、seed、資源與 QC 紀錄。
