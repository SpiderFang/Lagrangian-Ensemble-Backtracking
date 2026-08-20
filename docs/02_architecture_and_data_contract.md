# 架構與資料契約

## 1. 設計目標

本專案採「上游快取唯讀、forcing adapter 與計算核心分離、輸出不可變」架構。大型 OCM/NWW3 陣列留在 SERVER 原位置，以 memory-map 與時間窗載入；本專案只保存索引、manifest、軌跡分片、事件與聚合成果。

## 2. 正式輸入閘門

### 2.1 OCM native schema 3

根目錄：`$OCM_NATIVE_ROOT/<flow_domain_id>/`

靜態 grid 最低需求：

| 檔案 | 維度 | 用途 |
|---|---|---|
| `source_lon.npy`, `source_lat.npy` | `(node,)` | 原生 node WGS84 座標；preflight 後投影至 domain metric CRS |
| `source_face_nodes_local.npy` | `(face,4)` | 原生 face connectivity，`-1` 表示無第四點 |
| `source_face_node_count.npy` | `(face,)` | 區分 triangle／quad，不以陣列值臆測 |
| `source_depth_m.npy` | `(node,)` | 水深正向向下；用於海床位置與 Stokes 有限水深 |
| `source_node_bottom_index.npy` | `(node,)` | 每 node 最低有效 layer 的輔助 QC |
| `source_face_global_index.npy` | `(face,)` | 與上游 grid 保持可追溯 |
| `metadata.json` | JSON | schema、domain bbox、node/face/edge count、單位與 provenance |

每月 forcing 最低需求：

| 檔案 | 維度 | 物理語意與限制 |
|---|---|---|
| `time_utc_ns.npy` | `(time,)` | UTC epoch ns；嚴格遞增、唯一 |
| `hvel.npy` | `(time,node,layer,2)` | 東／北向水平速度，目標單位 m/s；正式 run 前須通過單位依據檢查 |
| `vertical_velocity.npy` | `(time,node,layer)` | SCHISM 垂向速度，m/s；專案統一使用 z positive-up |
| `zcor.npy` | `(time,node,layer)` | 每時每 node 的物理 z，m positive-up；不得以固定 layer 當固定深度 |
| `elev.npy` | `(time,node)` | 自由水面 z，m |
| `wetdry_elem.npy` | `(time,face)` | 動態濕乾原值；0/1 語意須由 metadata／參考文件或可稽核測試固定 |
| `diffusivity.npy` | `(time,node,layer)` | SCHISM tracer eddy diffusivity，目標單位 m²/s；作 Kz 候選而非無條件真值 |
| `metadata.json`, `quality_report.json` | JSON | status、cache kind、source coverage、array schema 與 QC |

正式 run 要求 schema major=3、`status=ready`，並接受 `standard_month` 與
`standard_partial_month`。後者不是待供應者補件的瑕疵版本，而是「2024–2025 全部可得
資料」母體的一部分；原始標籤、缺時清單與 coverage 必須原樣保存。各月份 UTC 先以
stable sort 與 `prefer_last` 去重建立 canonical 軸，再由版本化重建 patch 或 gap-safe
arrival window 提供連續 forcing；不得覆寫上游快取，也不得在 runtime 臨時跨缺口外插。

### 2.2 NWW3 analysis schema 1

根目錄：`$NWW_ANALYSIS_ROOT/<flow_domain_id>/months/YYYYMM/`

| 檔案 | 維度 | 用途 |
|---|---|---|
| `time_utc_ns.npy` | `(time,)` | 已對位 OCM target time 的 UTC 軸 |
| `significant_wave_height.npy` | `(time,y,x)` | bulk Hs，m |
| `peak_frequency.npy` | `(time,y,x)` | `fp`，Hz；只在有限且 `fp>0` 時計算 Tp |
| `peak_direction_raw_deg.npy` | `(time,y,x)` | 峰值波向 raw degree；必須以明示方向慣例轉向量 |
| `valid_mask_wave.npy` | `(time,y,x)` | 核心波浪欄位共同有效遮罩 |
| `qc_flags.npy` | `(time,y,x)` | 靜態無效、空間不支援、欄位缺失與時間不支援 |
| `metadata.json`, `quality_report.json` | JSON | schema、target grid、native spacing、插值、方向限制與 coverage |

正式 run 要求 schema major=1，並依本研究的 available-data contract 接受 `ready` 與
`trial_ready`。這裡的 `trial_ready` 僅表示上游 lexical cycle selection 未宣稱供應者認定的
最佳 forecast lead；在原始提供者與額外 metadata 已不可取得的條件下，不再把改寫上游
status 當成正式成果阻擋。SERVER 的 `nww3_native` 已具有 2024-01-01 00:00 至
2025-12-31 23:00 共 17,544 個連續逐時 UTC，故正式 analysis 由 native 產品重採樣到各
OCM 靜態格網的完整逐時軸，不沿用舊 OCM target-time 缺口。`flow_domain_id`、target grid
geometry 與正式 OCM domain 仍須相容。NWW3 是約 0.025° 原生波場重採樣到約 1 km OCM
grid；重採樣不提升有效物理解析度，圖說與 metadata 必須保留此限制。

### 2.3 方向與單位基線

- SCHISM 官方物理式使用 `z` positive-up，`hvel` 與 `vertical_velocity` 為 m/s；相鄰專案保存的 SCHISM 變數表亦記載 `elev [m]`、`wind_speed [m/s]`、`dahv [m/s]`、`vertical_velocity [m/s]`、`diffusivity [m²/s]` 與 `hvel [m/s]`。正式 preflight 必須把依據版本與檔案 hash 寫入 manifest。
- NWW3 已採用 `nww3_dp_wnd_two_typhoon_adopted_v1`：`DP` 為自正北順時針的
  wave-from，傳播去向 `theta_to=(DP+180°) mod 360°`；`.wnd` planes 1/2 分別為東向與
  北向風分量。此結論由山陀兒與康芮兩個獨立颱風事件交叉判定，研究展示與後續計算均採
  同一慣例，config/manifest 必須明示契約 ID，不能在讀檔器內隱藏。
- 未知供應者欄位保持 `unknown`，不得虛構 provider confirmation。只有新的實證與上述
  研究端契約直接衝突時才停止受影響 run、建立新版 decision record；「無法再詢問供應者」
  本身不是阻擋。

## 3. Domain 與座標

目前上游有四個 forcing domain：

| `flow_domain_id` | 支援研究區 |
|---|---|
| `northeast_taiwan_common_cache_v3` | 龜山島與貢寮等東北台灣受體 |
| `hsinchu_cache_v3` | 新竹外海受體 |
| `houwan_nmmba_cache_v3` | 後灣／海生館受體 |
| `lienchiang_common_cache_v3` | 北竿、南竿與連江島群受體 |

每個 domain 由 WGS84 polygon 與一個局地 metric CRS 組成。建議以 domain 中心建立 Azimuthal Equidistant CRS，避免連江 domain 跨 UTM zone 邊界時出現不必要的分區。正式 CRS 保存 PROJJSON/WKT、中心、轉換版本與 round-trip 誤差。

`flow_domain`、`study_site`、`local_domain`、`receptor`、`open_boundary_segment` 與 `reporting_region` 是不同物件：

- `flow_domain`：forcing 支撐與最外層停止邊界；貢寮與龜山島共用同一個 A 區 domain version 及 outer boundary，以保留兩地互通的水動力背景。
- `study_site`：情境、seed、主要 local event 與成果的第一層獨立統計單元；貢寮與龜山島可共用 flow domain 而不共用情境。
- `local_domain`：辨識移入關注海域入口的巢狀邊界；貢寮／龜山島採 anchor 半徑 25 km 與有效海域的交集，兩者允許重疊。每條軌跡只以自己的 local domain 產生主要 first-exit，另一站 local domain 的 crossing 只屬非終止連通診斷。
- `receptor`：終端觀測位置／小 polygon、深度及不確定性。
- `open_boundary_segment`：排除海岸後可穿越的命名邊界，用於 first crossing 與弧長密度。
- `reporting_region`：下游彙整單元，不改變 forcing 或軌跡。

正式 geometry manifest 必須分別保存每個 local/flow polygon、其 exterior 中可交換水體的
`open_boundary_segment` LineString/MultiLineString，以及剩餘海岸段。執行器只把交點落在
命名 open-water 線段上的 crossing 分類為 `local_domain_first_exit` 或
`flow_domain_open_exit`；其他 polygon exit 一律是終止性的 `coast_contact`。只提供 polygon
而未提供 open-boundary 子集合的相容模式僅限合成測試，不得通過正式發布閘門。

### 3.1 A 區 domain version 契約

現行 `northeast_taiwan_common_cache_v3` 的 bbox 南界為 `24.600844°N`。SERVER preflight 顯示龜山島 25 km local boundary 至該南界僅餘約 1.64 km，未達兩個約 1 km OCM surface／NWW 共同格點，因此它的 `domain_role` 固定為 `development_and_pilot`。正式 release 不得就地改寫此上游識別碼或 metadata，而須引用新的 expanded `flow_domain_id`。

expanded A 區的機器可驗證契約至少包含：

- 新 ID 固定為 `northeast_taiwan_common_cache_v4_lbt_south_expanded`，候選 bbox 為
  `[121.306315, 122.793685, 24.480000, 25.499156]`；龜山島 35 km geodesic 南緣約
  `24.527152°N`，至名目南界約保留 5.22 km。bbox 是產製目標，正式驗收仍以實際共同
  有效格網為準；
- OCM native triangle、OCM surface 與 NWW analysis 對 25 km baseline 及 35 km sensitivity 的 open-water arcs 均至少保留兩個共同有效格點；
- OCM/NWW 的 grid、month、time、mask、schema、status、input fingerprint 與方向／單位決策均重新進入 G0/G1，而不是沿用舊 domain 的通過紀錄；
- 含 Stokes baseline 不得只利用已延伸的 OCM native source margin，因現行 NWW analysis 並未覆蓋該 margin；
- `flow_domain_id` 與 outer-boundary segment IDs 隨新版本建立，所有 scenario 仍保留原本的五站點與 50,000 基礎情境定義。

## 4. OCM native mesh 取樣器

### 4.1 靜態索引

1. 將 node 座標投影到 metric CRS。
2. 依 `source_face_node_count` 建立三角形；quad 使用可重現的對角線規則切成兩個 triangle，並保存 `triangle_to_source_face`。
3. 拒絕零面積、翻轉、重複 node 或跨 domain 的 triangle；輸出 mesh QC。
4. 建立 uniform-bin 或等價的可序列化 spatial index；粒子優先沿用前一 triangle，失敗才 fallback 全域索引。
5. 保留 barycentric tolerance，只有落在允許數值誤差內的點才視為 triangle 內部。

不得以 OCM surface grid 的 `source_face_index.npy` 取代本索引，因相鄰專案已明載該欄位是 SciPy Delaunay simplex index，不是 SCHISM face ID。

### 4.2 四維速度取樣

對每個 `(x,y,z,t)`：

1. 在 canonical 軸找到前後兩個時間 slice；slice 可來自 immutable observed 月份或
   approved reconstruction patch，且每一時次保存 origin label。若時間超出核定軸，或
   manifest 聲稱可重建但 patch/checksum/支撐實際不存在，回報 `data_gap`；不得由 sampler
   自行最近值填補或跨未登錄缺口外插。
2. 找到 native triangle 與 source face；檢查動態濕乾狀態。
3. 在三個 node 上，使用各自 `zcor` 找到包夾 z 的上下有效 layer，線性取樣 `hvel`、`vertical_velocity` 與候選 `diffusivity`；禁止海面以上、海床以下或單側外插。
4. 三個 node 全部有效後以 barycentric 權重做水平內插；任一必要支撐缺值時保持無效。
5. 在前後時間 slice 線性內插；每次 RK stage 都使用其真正 stage time。

這一順序與既有 SVD 的「先 node 垂向、再重心水平」政策一致，但粒子使用任意位置與原生 face，不依賴規則格網 cell。

## 5. NWW3 與 Stokes 取樣器

正式 NWW3 `nww3_analysis` 與 OCM 靜態 surface grid 對位，時間支撐則由完整 native 軸
重建為 17,544 個逐時 UTC。粒子位置先轉成該 grid 的 `(y,x)`，以 mask-aware bilinear
interpolation 取樣 Hs、fp、DP；方向採單位向量的圓形內插，不直接平均角度。四角只要有
必要欄位缺值，該 stage 的 Stokes forcing 無效；基線不以最近格點補值。

OCM 與 NWW3 缺值政策分開：

- 已知 OCM 整時缺口在正式 run 前完成短缺口／EOF-state-space 重建與 blocked
  cross-validation；若未達門檻，baseline 改用不跨缺口的分層 arrival windows 與最短已
  收斂 horizon。已知缺口不能在正式 runtime 才讓全部粒子停止。
- NWW 既有 analysis 的時間缺口只因舊版沿用 OCM target time；正式版由完整 native 軸
  重建，無須對波浪時間作統計補值。空間必要欄位若仍無效，含 Stokes baseline 標
  `wave_data_gap`；no-Stokes 是獨立 sensitivity，不能混入 baseline 分母。
- `current_data_gap`／`wave_data_gap` 保留給 manifest 外缺檔、checksum 改變、局部重建
  失敗、空間支撐無效或 I/O 損毀，並在成果中以 failure/coverage 圖揭露。

## 6. 預定 Python 套件分層

| 模組 | 職責 |
|---|---|
| `config` | Pydantic/YAML schema、標準化 JSON、hash 與決策狀態 gate |
| `preflight` | SERVER 路徑、月份、metadata、shape、time、coverage、磁碟與資源 inventory |
| `time_axis` | 跨月份 stable sort、prefer-last 去重、來源映射與缺口形狀盤點 |
| `reconstruction` | OCM 短缺口與多變量 EOF-harmonic state-space patch、blocked validation、posterior forcing members 與 provenance |
| `geometry` | CRS、共用 flow outer domain、站點 local domain、receptor、open-boundary 幾何與 own/foreign crossing |
| `mesh` | SCHISM face triangulation、spatial index、barycentric locator |
| `forcing.ocm` | OCM month window、4D current/z/elev/diffusivity sampler |
| `forcing.nww3` | NWW3 analysis-grid time/space sampler 與 QC |
| `physics.stokes` | dispersion solver、bulk finite-depth profile、方向轉換 |
| `physics.diffusion` | Smagorinsky Kh、Kz、gradient drift 與 stochastic increment |
| `integrators` | NumPy reference RK4、stochastic split、CFL/dt controller |
| `boundaries` | 海面、海床、海岸、開放邊界、data-gap 與 first-crossing event |
| `scenarios` | 五站點各自 material/receptor/arrival 的 10×20×50 完整矩陣、member 配置與 seed 派生 |
| `engine` | 分片執行、checkpoint、restart、Numba production kernel |
| `outputs` | trajectory/event column arrays、manifest、checksum、原子發布 |
| `aggregation` | exit/pathway/residence/bottom-contact、KDE/HDR、跨站 local-domain connectivity、bootstrap |
| `visualization` | 學術地圖、比較圖、caption/provenance sidecar |

## 7. 設定與情境契約

### 7.1 Material manifest

每列至少包含：

| 欄位 | 說明 |
|---|---|
| `material_id`, `version` | 穩定識別碼 |
| `settling_velocity_mps` | z positive-up；沉降負、上浮正 |
| `behavior_class` | suspended / sinking / rising / near_bed |
| `source_basis` | 文獻、實驗或研究團隊決策 |
| `uncertainty` | 範圍或分布；無依據時標 `provisional` |

### 7.2 Receptor manifest

每個 receptor 保存 `receptor_id`、WGS84 geometry、位置誤差、`vertical_reference`、目標水柱比例、實際 `z_m`／`height_above_bed_m`、垂向誤差、`study_site_id`、`analysis_region_id`、source face、版本與生成狀態。五站點各有 20 個、全案共 100 個；貢寮與龜山島各自完整保留 20 個，不共享 ID 或在 A 區內分配。

每站點 20 個受體由 5 個水平位置 × 4 個垂向層位產生。貢寮／龜山島的水平候選限於 anchor 半徑 12.5 km receptor core，其餘站點限於 flow/local domain；第一點由 anchor 或 flow-domain center snap 至 persistent-wet mesh，其餘使用固定 tie-break 的 metric maximin。垂向目標為海面下 `0.10H`、`0.40H`、`0.70H` 與最低有效 OCM layer 中心；實際位置必須對五十個到達時間均有合法 `zcor` 支撐。

### 7.3 Arrival-time manifest

每列保存 UTC ns、ISO UTC、年份、季節、潮汐類別、波況／流況標籤、選取依據、forcing availability、seed 及版本。顯示可另附 UTC+8，但運算只用 UTC。

### 7.4 Scenario 與 member

基礎 `scenario_id = hash(study_site_id, material_id, receptor_id, arrival_time_id, design_version)`。五站點各自的三因子完整交叉必須恰好產生 10,000 個唯一 ID；A 區兩站聯集為 20,000，全案聯集恰有 50,000 個。no-Stokes、Kh/Kz、domain 等敏感度由 `experiment_case_id` 區分；它們不能暗中改變基礎情境的定義。

貢寮與龜山島 scenario 共用相同 A 區 forcing adapter、time-window cache 與 outer-boundary geometry，但不得因此共用 receptor ID、local-boundary state 或統計分母。若軌跡穿越另一站 local domain，事件列保存 `related_study_site_id` 與 crossing direction；原始 `study_site_id`、`scenario_id`、seed 與 primary local-exit state 全程不變。這使實作可以重用昂貴的 OCM/NWW I/O，又不會把兩個條件式受體問題混成同一情境。

每個基礎情境可有 `member_id = 0..M_s-1`，seed 由 master seed、`scenario_id`、`experiment_case_id` 與 `member_id` 派生。`M_s` 是同一固定參數組合下的獨立隨機實現數，不是第四個計畫書因子：

```text
N_base_per_site = 10 × 20 × 50 = 10,000
N_base_region_A = 2 × N_base_per_site = 20,000
N_base_total = 5 × N_base_per_site = 50,000
N_trajectory_total_per_experiment = sum_s(M_s)
N_trajectory_total_per_experiment = 50,000 × M  # 僅在所有 M_s 相同時
```

完全確定性試驗使用 `M_s=1`；隨機擴散或 forcing／受體微擾試驗的正式 `M` 必須由收斂測試決定。`scenario_table`、`seed_table` 與 run manifest 需同時保存 site、region、每站點／A 區／全案基礎情境數、各情境 member 數及 experiment case，避免以「系集數」一詞混用不同數量。

## 8. 輸出資料契約

```text
$LBT_OUTPUT_ROOT/particles/<run_id>/
├── run_manifest.json
├── normalized_config.json
├── input_inventory.json
├── scenario_table.parquet
├── seed_table.parquet
├── shards/
│   └── part-00000/
│       ├── trajectory_offsets.npy
│       ├── particle_id.npy
│       ├── time_utc_ns.npy
│       ├── x_m.npy
│       ├── y_m.npy
│       ├── z_m.npy
│       ├── status_code.npy
│       ├── event_table.parquet
│       ├── shard_manifest.json
│       └── checksums.sha256
├── aggregates/
│   ├── boundary_exit_points.parquet
│   ├── local_domain_entry_crossings.parquet
│   ├── local_boundary_arclength_density.parquet
│   ├── local_entry_kde.npy
│   ├── boundary_arclength_density.parquet
│   ├── boundary_exit_kde.npy
│   ├── pathway_density.npy
│   ├── residence_time_s.npy
│   ├── travel_time_summary.parquet
│   ├── source_receptor_connectivity.parquet
│   ├── cross_site_local_domain_connectivity.parquet
│   ├── cross_site_pathway_hdr_overlap.parquet
│   ├── bottom_contact_density.npy
│   ├── run_outcome_summary.parquet
│   ├── aggregate_grid.json
│   └── uncertainty_summary.json
└── figures/
    ├── figure_registry.json
    └── caption_sidecars/
```

軌跡採 CSR 類型 ragged column arrays：`trajectory_offsets` 指出每個 particle 在共同一維 observation arrays 的起訖，避免 object dtype。事件與情境適合以 Parquet 保存可查詢欄位；所有大型輸出都需 checksum、shape、dtype、單位及分片 row count。

### 8.1 必要事件欄位

- `particle_id`, `scenario_id`, `member_id`, `study_site_id`, `analysis_region_id`, `receptor_id`。
- `event_type`：local_domain_first_exit / other_site_local_domain_enter / other_site_local_domain_exit / flow_domain_open_exit / coast_contact / surface_contact / surface_regime_exit / bed_contact / deposited / data_gap / max_age / forcing_start / numerical_failure。
- `related_study_site_id`：只供 foreign-local crossing 使用；不得覆寫原始 `study_site_id`，主要事件則為 null。
- event 前後的 `time_utc_ns`, `x_m`, `y_m`, `z_m` 與內插 crossing 座標。
- `boundary_segment_id`, `boundary_s_m`（適用時）。
- `source_face_id`, `triangle_id`, `forcing_month_id` 與 QC flags。

## 9. 發布與相容性

- 所有正式 run 先寫 `.partial-<uuid>`，完成 schema、row count、checksum 與 aggregate QC 後原子發布為 `<run_id>`。
- 已發布 run 不覆寫；設定、input manifest、method、seed policy 或 geometry 改變即產生新 run ID。
- checkpoint 是未發布工作資料，綁定 config hash、input inventory、shard range 與 code commit；不相容 checkpoint 必須拒絕續跑。
- 上游 cache 被標記 `superseded` 時，對應 run 必須列入 impact report，不可靜默沿用。
