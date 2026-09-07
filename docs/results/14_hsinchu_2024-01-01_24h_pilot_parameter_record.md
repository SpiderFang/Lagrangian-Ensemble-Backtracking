# 新竹 2024-01-01 起始 24 小時展示 pilot 參數紀錄

> 文件狀態：pilot-only、可重建的工程設定紀錄；不是全期研究成果，也不是正式
> 48+2 arrival selection 的替代品。本文所稱「來源足跡」均指條件式來源足跡或相對
> 來源權重的工程描述；不得由本展示推論絕對來源、因果歸因或全期代表性。

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
2024-01-01T00:00:00Z 作補件，不使用最近值、零值或跨缺口時間內插。

## 2. 可引用的固定參數與證據

下表把本次已驗收的 calibration-bound 候選實際值直接列出，避免 PI 報告只看到
「沿用設定」而無法核對數值。這些數值是參數紀錄，不會把範例 template 中的
`null` 自動視為可執行值；真正執行時，仍以生成後的 config 與
`pilot_execution_binding`（包含 `candidate_values`／`applied_fields`）為唯一證據。
在 SERVER 實際建置完成後，應回填該 config、binding 與輸入 manifest 的 SHA-256；
本文件不以手抄數值取代 hash 或 runtime binding。

| 項目 | 固定設定／本案例數值 | config key 或 manifest 證據 | 設定理由與限制 |
|---|---|---|---|
| 研究範圍 | `study_site_id=hsinchu`、analysis region B | `study_sites[].study_site_id`、`analysis_region_id`；`arrival.json`、`receptor.json` | 只做新竹單站展示；不改五站正式分母。 |
| flow／local domain | `hsinchu_cache_v3`；local domain 與 flow domain 完全相同 | `study_sites[hsinchu].flow_domain_id`、`local_domain_policy=same_as_flow_domain`；`domain.json`／`local.json` 的 `local_equals_flow=true` | 核心只限制水平 receptor candidate polygon，不縮小 B 區 forcing 或 local 邊界。 |
| 核心中心與半徑 | `(120.45, 24.75)`；`12,500 m` | `study_sites[hsinchu].anchor_lonlat`、`receptor_core_radius_m`；`receptor.json` 的實際 lon/lat 以同一 flow-domain AEQD 投影核對 | 距離在公尺制 AEQD 計算；不以經緯度差近似。候選區是 core 圓與既有 local/static-ocean polygon 交集。 |
| receptor 位置／水層 | 5 個水平位置 × 4 個水層 = 20 個 scenario 起始位置 | `scenarios.expected_receptor_count_per_site=20`、`horizontal_receptor_count_per_site=5`、`vertical_receptor_count_per_horizontal_location=4`；`receptor.json` 20 筆 hsinchu records | 保留正式五站／每站 20 receptors 契約；pilot 只在 run-create 精確選取一個 arrival 與一個 material。 |
| arrival 方向與 25 時次 | 到達 `2024-01-02T01:00:00Z`，反向至 `2024-01-01T01:00:00Z`，inclusive 25 個逐時節點；`max_backtrack_days=1.0` | CLI `--pilot-arrival-utc`；`arrival.json` 的 `explicit_pilot_window`；`ocm_gap_safe_arrival.json` 的 `horizon_start_utc`、`horizon_end_utc`、`expected_step_count=25`、`supported_step_count=25` | OCM native／surface 2024 第一個可用節點是 Jan 1 01Z；NWW 從 00Z 起，但展示視窗不得補 OCM 00Z。 |
| pilot selection identity | policy `hsinchu_explicit_24h_window_replacement_v1`；新 arrival ID 由 site、UTC、policy、`design_version` 穩定雜湊；metadata 保存原／替換 ID | `arrival.json` record metadata：`pilot_replaced_arrival_time_id`、`pilot_replacement_arrival_time_id`、`pilot_selection_scope`；artifact index `source_bindings.pilot_selection_scope` | 決定性替換一筆既有 hsinchu selection；label 使用 `pilot_explicit_window`／`explicit_24h_window`，不冒充潮汐 phase 或事件。 |
| 全案輸入計數 | 250 arrivals、5,000 dynamic pairs | `arrival.json` 250 records；`initial_conditions.json` 5,000 records；`validate_input_derivatives` summary | 替換不改五站 50×5 的 source coverage；本案例的 20 個 run scenarios 由 run-create 精確 selector 產生。 |
| OCM native／surface 欄位 | native：mesh、`zcor`、`elev`、`wetdry_elem`、`hvel`、`diffusivity`；surface：`eta_m`、`u_surface_mps`、`v_surface_mps`、`surface_z`、`valid_mask_surface`、`qc_flags` | `forcing_inventory.json` 的 OCM schema 3 products、source files、canonical time；`initial_conditions.json` 的 `ocm_time_origin=observed` | native dynamic pair 使用實際 UTC 的 OCM 原生切片；surface 只供 arrival scalar 與逐時支援檢查，不以 native 全域 `hvel` 代替。 |
| NWW3 欄位與空間支援 | `significant_wave_height`、`peak_frequency`、`peak_direction_raw_deg`、`valid_mask_wave`；metric location 每小時四角 static／dynamic／有限值／物理條件均有效 | `forcing_inventory.json` 的 NWW3 schema 1；arrival metadata 的 `metric_location_*`；pilot build 對 25 小時重做 exact-hour 四角 gate | 不尋找最近有效格點、不補零、不做時間內插；四角任一小時失敗即拒絕。 |
| 材質與沉降 | run-create 精確指定 `oca_fishinggear_open_mesh_bundle`；`settling_velocity_mps=-0.002 m/s`（z 軸向上為正） | `physics.settling.material_classes[material_id=oca_fishinggear_open_mesh_bundle].settling_velocity_mps`；`material.json`；run-create `--pilot-material-id` | 這是已驗收的嚴格負沉降候選，仍屬 provisional proxy，不是 OCA 單體量測；只納入設定描述的沉沒適用條件。 |
| Stokes／experiment | `experiment_case_id=finite_depth_stokes`；Stokes 啟用，有限水深 bulk formulation | run-create `--experiment-case finite_depth_stokes`；`physics.stokes.enabled=true`；`physics.stokes.formulation=finite_depth_monochromatic_bulk`；NWW manifest／runtime input binding | 案例不切換 engine 或另造波浪參數；有限水深公式與 `invalid_wave_policy=stop_with_wave_data_gap` 由生成 config／binding 固定，本紀錄不把 BayTrace 範例值帶入。 |
| RK4、步長與輸出 | RK4；`dt_min_seconds=0.1 s`、`dt_max_seconds=30 s`、raw output `output_interval_seconds=300 s` | generated pilot config 的 `integration.deterministic_method`、`integration.dt_min_seconds`、`integration.dt_max_seconds`、`integration.output_interval_seconds`；`pilot_execution_binding.execution_scalar_snapshot` | 這些是已驗收 execution scalars；不改 engine，不採 BayTrace demo 的 Euler／3600 s。 |
| Kh／Kz | `Kh=0.5546152926568088 m²/s`；`Kz=0.001552638244840742 m²/s` | generated pilot config 的 `physics.horizontal_diffusion.constant_kh_m2ps=0.5546152926568088`、`physics.vertical_diffusion.constant_kz_m2ps=0.001552638244840742`；`pilot_execution_binding.candidate_values`、`applied_fields` | 這是已驗收 calibration-bound 候選值；範例 template 的 `null` 不構成執行值，實際數值以 generated config／binding 為唯一證據，避免手抄值與 calibration artifact 漂移。 |
| M／seed | `M=1`；`master_seed=20260831` | generated pilot config 的 `scenarios.members_per_scenario=1`、`scenarios.master_seed=20260831`；run plan 的 seed／scenario selection binding | M 是同一 scenario 的 member 數，不是額外情境因子；單一 M 不代表收斂已完成。 |
| 邊界 | surface `reflect_and_record_contact`；sinking bed `deposit_on_first_contact_and_stop`；coast `stop_and_record_contact`；flow boundary `stop_at_first_crossing` | generated config：`boundaries.surface_suspended_or_sinking`、`boundaries.bed_sinking_or_near_bed`、`boundaries.coast_baseline`、`boundaries.flow_domain_open_boundary`；`open.json`、`local.json`、`domain.json` | 這些是本案例沿用的已驗收邊界政策；核心圓不是 local boundary，不因 pilot 另寫一套邊界規則，也不把跨另一站 local 視為站點轉移。 |
| gap 政策 | `inclusive_observed_ocm_only_no_cross_gap`；pilot row 25/25、`crossed_gap=false`、`missing_utc=[]` | `ocm_gap_safe_arrival.json`；arrival metadata `explicit_pilot_window_time_support` | 任一 OCM native／surface／NWW exact-hour 或 NWW 四角支援缺失都 fail closed；不跨越已知缺口。 |
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

## 5. 展示限制與驗收證據

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
