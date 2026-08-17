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

正式 run 要求 schema major=3、`status=ready`。`standard_partial_month` 只有在缺日清單、影響期間與研究團隊核准均寫入 run manifest 時才可納入；粒子不得跨缺口外插。

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

正式 run 要求 schema major=1、`status=ready`，且 `flow_domain_id`、target grid geometry、target time source 與 OCM domain 相容。NWW3 是約 0.025° 原生波場重採樣到約 1 km OCM grid；重採樣不提升有效物理解析度，圖說與 metadata 必須保留此限制。

### 2.3 方向與單位基線

- SCHISM 官方物理式使用 `z` positive-up，`hvel` 與 `vertical_velocity` 為 m/s；相鄰專案保存的 SCHISM 變數表亦記載 `elev [m]`、`wind_speed [m/s]`、`dahv [m/s]`、`vertical_velocity [m/s]`、`diffusivity [m²/s]` 與 `hvel [m/s]`。正式 preflight 必須把依據版本與檔案 hash 寫入 manifest。
- NWW3 的已採用推定慣例是 `DP` 為自正北順時針的 wave-from；傳播去向 `theta_to=(DP+180°) mod 360°`。此慣例須以 config enum 明示，不能在讀檔器內隱藏。
- 若 SERVER 正式 metadata 與上述基線衝突，停止正式 run、建立 decision record，不自動轉換。

## 3. Domain 與座標

目前上游有四個 forcing domain：

| `flow_domain_id` | 支援研究區 |
|---|---|
| `northeast_taiwan_common_cache_v3` | 龜山島與貢寮等東北台灣受體 |
| `hsinchu_cache_v3` | 新竹外海受體 |
| `houwan_nmmba_cache_v3` | 後灣／海生館受體 |
| `lienchiang_common_cache_v3` | 北竿、南竿與連江島群受體 |

每個 domain 由 WGS84 polygon 與一個局地 metric CRS 組成。建議以 domain 中心建立 Azimuthal Equidistant CRS，避免連江 domain 跨 UTM zone 邊界時出現不必要的分區。正式 CRS 保存 PROJJSON/WKT、中心、轉換版本與 round-trip 誤差。

`flow_domain`、`receptor`、`open_boundary_segment` 與 `reporting_region` 是不同物件：

- `flow_domain`：forcing 支撐與停止邊界。
- `receptor`：終端觀測位置／小 polygon、深度及不確定性。
- `open_boundary_segment`：排除海岸後可穿越的命名邊界，用於 first crossing 與弧長密度。
- `reporting_region`：下游彙整單元，不改變 forcing 或軌跡。

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

1. 找到 OCM 前後兩個時間 slice；超出時間軸或跨越核定缺口即回報 `data_gap`，不外插。
2. 找到 native triangle 與 source face；檢查動態濕乾狀態。
3. 在三個 node 上，使用各自 `zcor` 找到包夾 z 的上下有效 layer，線性取樣 `hvel`、`vertical_velocity` 與候選 `diffusivity`；禁止海面以上、海床以下或單側外插。
4. 三個 node 全部有效後以 barycentric 權重做水平內插；任一必要支撐缺值時保持無效。
5. 在前後時間 slice 線性內插；每次 RK stage 都使用其真正 stage time。

這一順序與既有 SVD 的「先 node 垂向、再重心水平」政策一致，但粒子使用任意位置與原生 face，不依賴規則格網 cell。

## 5. NWW3 與 Stokes 取樣器

NWW3 `nww3_analysis` 已與 OCM surface grid/time 對位。粒子位置先轉成該 grid 的 `(y,x)`，以 mask-aware bilinear interpolation 取樣 Hs、fp、DP；四角只要有必要欄位缺值，該 stage 的 Stokes forcing 無效。基線不以最近格點補值。

OCM 與 NWW3 缺值政策分開：

- OCM 速度缺失：停止粒子並標 `current_data_gap`。
- NWW3 缺失：基準含 Stokes run 標 `wave_data_gap`；另有 no-Stokes sensitivity 可繼續，但不得把它混入基準 run。

## 6. 預定 Python 套件分層

| 模組 | 職責 |
|---|---|
| `config` | Pydantic/YAML schema、標準化 JSON、hash 與決策狀態 gate |
| `preflight` | SERVER 路徑、月份、metadata、shape、time、coverage、磁碟與資源 inventory |
| `geometry` | CRS、domain/receptor/open-boundary 幾何與 crossing |
| `mesh` | SCHISM face triangulation、spatial index、barycentric locator |
| `forcing.ocm` | OCM month window、4D current/z/elev/diffusivity sampler |
| `forcing.nww3` | NWW3 analysis-grid time/space sampler 與 QC |
| `physics.stokes` | dispersion solver、bulk finite-depth profile、方向轉換 |
| `physics.diffusion` | Smagorinsky Kh、Kz、gradient drift 與 stochastic increment |
| `integrators` | NumPy reference RK4、stochastic split、CFL/dt controller |
| `boundaries` | 海面、海床、海岸、開放邊界、data-gap 與 first-crossing event |
| `scenarios` | material/receptor/arrival 的 10×20×50 完整矩陣、member 配置與 seed 派生 |
| `engine` | 分片執行、checkpoint、restart、Numba production kernel |
| `outputs` | trajectory/event column arrays、manifest、checksum、原子發布 |
| `aggregation` | exit/pathway/residence/bottom-contact、KDE/HDR、bootstrap |
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

每個 receptor 保存 `receptor_id`、WGS84 geometry、位置誤差、`vertical_reference`、`z_m` 或 `height_above_bed_m`、垂向誤差、調查時間來源、site/reporting region、版本、核定者與狀態。

### 7.3 Arrival-time manifest

每列保存 UTC ns、ISO UTC、年份、季節、潮汐類別、波況／流況標籤、選取依據、forcing availability、seed 及版本。顯示可另附 UTC+8，但運算只用 UTC。

### 7.4 Scenario 與 member

基礎 `scenario_id = hash(material_id, receptor_id, arrival_time_id, design_version)`，三個因子的完整交叉必須恰好產生 10,000 個唯一 ID。no-Stokes、Kh/Kz、domain 等敏感度由 `experiment_case_id` 區分；它們不能暗中改變基礎情境的定義。

每個基礎情境可有 `member_id = 0..M_s-1`，seed 由 master seed、`scenario_id`、`experiment_case_id` 與 `member_id` 派生。`M_s` 是同一固定參數組合下的獨立隨機實現數，不是第四個計畫書因子：

```text
N_base_scenario = 10 × 20 × 50 = 10,000
N_trajectory_per_experiment = sum_s(M_s)
N_trajectory_per_experiment = 10,000 × M   # 僅在所有 M_s 相同時
```

完全確定性試驗使用 `M_s=1`；隨機擴散或 forcing／受體微擾試驗的正式 `M` 必須由收斂測試決定。`scenario_table`、`seed_table` 與 run manifest 需同時保存基礎情境數、各情境 member 數及 experiment case，避免以「系集數」一詞混用三種數量。

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
│   ├── boundary_arclength_density.parquet
│   ├── boundary_exit_kde.npy
│   ├── pathway_density.npy
│   ├── residence_time_s.npy
│   ├── travel_time_summary.parquet
│   ├── source_receptor_connectivity.parquet
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

- `particle_id`, `scenario_id`, `member_id`, `receptor_id`。
- `event_type`：open_boundary_exit / coast_contact / surface_contact / bed_contact / deposited / data_gap / max_age / forcing_start / numerical_failure。
- event 前後的 `time_utc_ns`, `x_m`, `y_m`, `z_m` 與內插 crossing 座標。
- `boundary_segment_id`, `boundary_s_m`（適用時）。
- `source_face_id`, `triangle_id`, `forcing_month_id` 與 QC flags。

## 9. 發布與相容性

- 所有正式 run 先寫 `.partial-<uuid>`，完成 schema、row count、checksum 與 aggregate QC 後原子發布為 `<run_id>`。
- 已發布 run 不覆寫；設定、input manifest、method、seed policy 或 geometry 改變即產生新 run ID。
- checkpoint 是未發布工作資料，綁定 config hash、input inventory、shard range 與 code commit；不相容 checkpoint 必須拒絕續跑。
- 上游 cache 被標記 `superseded` 時，對應 run 必須列入 impact report，不可靜默沿用。
