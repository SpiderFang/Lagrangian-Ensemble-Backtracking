# 架構與資料契約

## 1. 設計目標

本專案採「上游快取唯讀、forcing adapter 與計算核心分離、輸出不可變」架構。大型 OCM/NWW3 陣列留在 SERVER 原位置，以 memory-map 與時間窗載入；本專案只保存索引、manifest、軌跡分片、事件與聚合成果。

### 1.1 BayTrace 可用部分整合界線

本專案已採用 BayTrace 可對應本地 CPU 執行的工程思路：CPU SoA／batch／chunk、每粒子可
重現亂數、SCHISM triangle hint、可暫停 engine，以及 schema 2 checkpoint/restart。未採用
GPU/CUDA、BayTrace raw `schout`／`bp` I/O、oil/weathering、droptime、共享記憶體
multiprocessing，也未放寬 backward round-trip 成功判定。`ptrack4a` 僅保留為未來具備完整
相容 fixture 時的 golden reference，不是目前的正式驗證結果。

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
| `manifests` | versioned material/receptor/arrival/geometry JSON 的 strict schema、cross-reference、canonical hash 與正式 coverage gate |
| `forcing.ocm` | OCM month window、4D current/z/elev/diffusivity sampler |
| `forcing.nww3` | NWW3 analysis-grid time/space sampler 與 QC |
| `forcing_window` | 單一 flow domain 的 UTC 月份 lazy loader、LRU resident window、material facade 與 cache/resource stats |
| `provenance` | 不含絕對路徑的 Git/deployment tree、uv.lock、Python 與套件版本指紋；formal 只接受 Git clean 或 declared deployment commit |
| `run_control` | immutable run plan、atomic progress、schema 2 checkpoint generation、trajectory publish 與 reconcile |
| `run_validation` | run plan/progress、scenario/seed table、shard range、checkpoint/output checksum 與工程 benchmark 唯讀驗證 |
| `physics.stokes` | dispersion solver、bulk finite-depth profile、方向轉換 |
| `physics.diffusion` | Smagorinsky Kh、Kz、gradient drift 與 stochastic increment |
| `integrators` | NumPy reference RK4、stochastic split、CFL/dt controller |
| `boundaries` | 海面、海床、海岸、開放邊界、data-gap 與 first-crossing event |
| `scenarios` | 五站點各自 material/receptor/arrival 的 10×20×50 完整矩陣、member 配置與 seed 派生 |
| `engine` | 單粒子 reference step、可暫停 execution 與停止事件；CPU/NumPy batch 由 `production` 編排 |
| `outputs` | trajectory/event column arrays、manifest、checksum、原子發布 |
| `aggregation` | exit/pathway/residence/bottom-contact、KDE/HDR、跨站 local-domain connectivity、bootstrap |
| `visualization` | 學術地圖、比較圖、caption/provenance sidecar |

## 7. 設定與情境契約

### 7.1 Material manifest

每列至少包含：

| 欄位 | 說明 |
|---|---|
| `material_id`, `design_version` | 穩定識別碼與 `design_baseline_v2_non_rising_oca_proxy` 版本 |
| `oca_category_zh` | iOcean 十個海廢統計項目之一；十筆必須一對一且不可重複 |
| `material_family_zh` | 代表材質；用於說明代理條件，不宣稱為官方分類的唯一組成 |
| `representative_shape_zh` | 代表形狀、開口或含水狀態；終端速度對非球形幾何敏感，故不得省略 |
| `settling_velocity_mps` | z positive-up；正式 v2 僅接受嚴格負值，零值與正值在設定驗證階段拒絕 |
| `behavior_class` | 正式 v2 固定為 `sinking`；通用引擎的其他類別不屬於本情境矩陣 |
| `applicability_condition_zh` | 說明吸水、進水或生物附著等納入條件；未轉為負浮力的物件必須排除 |
| `calibration_status`, `evidence_grade` | 目前為 `provisional_proxy`，另記 B／C／D 證據等級與可外推限制 |
| `classification_source`, `velocity_source` | 分開記錄官方分類來源與研究敏感度格點來源，禁止把清除統計誤作速度量測 |

### 7.2 Receptor manifest

每個 receptor 保存 `receptor_id`、WGS84 geometry、位置誤差、`vertical_reference`、目標水柱比例、模板代表 `z_m_positive_up`、垂向誤差、`study_site_id`、`analysis_region_id`、source face、版本與生成狀態。五站點各有 20 個、全案共 100 個；貢寮與龜山島各自完整保留 20 個，不共享 ID 或在 A 區內分配。此處的 `z_m_positive_up` 只是水平受體與 `vertical_id` 模板的候選代表值，不是所有 arrival UTC 的正式實際深度。

每站點 20 個受體由 5 個水平位置 × 4 個垂向層位產生。貢寮／龜山島的水平候選限於 anchor 半徑 12.5 km receptor core，其餘站點限於 flow/local domain；第一點由 anchor 或 flow-domain center snap 至 persistent-wet mesh，其餘使用固定 tie-break 的 metric maximin。垂向模板目標為海面下 `0.10H`、`0.40H`、`0.70H` 與最低有效 OCM layer 中心；每個 arrival 的正式實際 z 必須改由 dynamic pair manifest 的 OCM `eta`／`zcor`／`wetdry` 計算與驗證。

### 7.3 Arrival-time manifest

每列保存 UTC ns、ISO UTC、年份、季節、潮汐類別、波況／流況標籤、選取依據、forcing availability、seed 及版本。顯示可另附 UTC+8，但運算只用 UTC。

Phase 3A 將上述文字契約固定成可讀取的 schema；component JSON 的未知欄位、重複
object key、NaN／Infinity、bool numeric、空白識別碼及無法追溯的來源均 fail-fast。
`load_scenario_inputs` 以三個固定 component 檔案的 raw SHA-256 與 canonical JSON SHA-256
綁定輸入；載入 dynamic pair manifest 時再加入固定 component key
`receptor_arrival_initial_condition`，並使用既有 `build_scenarios` 建立 tuple，不建立
NumPy object array。pair records 只保存 5,000 筆，不因十種 material 複製成 50,000 筆。

#### 7.3.1 Phase 3A component manifest 形狀

新增 manifest 的版本字串固定為 `1.0.0`，所有 root 均為 JSON object；`status` 只接受
`approved`、`pilot` 或 `generated`，正式載入必須是 `approved`。`provenance` 至少含
非空 `method_id`、UTC ISO8601 `created_at_utc` 與非空 `source_hashes`；後者每個 value
必須是 64 位小寫 SHA-256。相對 path 一律以 config YAML 所在目錄解析。

material 為既有 CLI `schema_version: "2.0.0"` 的相容文件，root key 必須恰為
`schema_version`、`design_version`、`classification_source`、`velocity_unit`、
`velocity_source`、`positive_or_zero_velocity_policy`、`calibration_scope`、`records`；
每筆 record 必須恰含既有 `Behavior` 九欄。此格式沒有新增 `status` 或共同 provenance；
三個既有來源欄位共同保存分類、速度與校準範圍的 provenance。formal loader 依 config
design version、十個一對一 iOcean 分類及嚴格負值 `sinking` gate 判定，速度仍只是
條件式材質／形狀敏感度代理。

receptor v1 root 固定如下：

```json
{
  "manifest_kind": "receptor_manifest",
  "schema_version": "1.0.0",
  "status": "approved",
  "design_version": "design_baseline_v2_non_rising_oca_proxy",
  "coordinate_reference": "EPSG:4326",
  "vertical_reference": "z_m_positive_up",
  "generation_method_id": "receptor_from_persistent_wet_mesh_v1",
  "provenance": {
    "method_id": "receptor_generation_v1",
    "created_at_utc": "2026-08-28T00:00:00Z",
    "source_hashes": {"ocm_grid": "<64-lowercase-hex>"}
  },
  "records": [{
    "receptor_id": "gongliao_r00",
    "study_site_id": "gongliao",
    "analysis_region_id": "A",
    "lon": 121.92807,
    "lat": 25.11245,
    "z_m_positive_up": -12.0,
    "vertical_id": "upper_water_column",
    "metadata": {"candidate_rank": 0}
  }]
}
```

receptor 的經緯度只作 WGS84 交換；`lon/lat` 必須在 bounds 內，`z_m_positive_up` 為
公尺且可有限，ID 全案唯一，站點與 region 必須和 config 對應。formal 為五站各 20、
全案 100；pilot 可是 config 站點的非空子集，但每個選中站點必須有 record。

arrival-time v1 root 與 receptor 相似，但固定使用 `manifest_kind: "arrival_time_manifest"`、
`time_standard: "UTC"`、`selection_method_id`、`provenance` 與 `records`；每筆 record
恰含 `arrival_time_id`、`study_site_id`、`time_utc_ns`、`year`、`season`、`tide_class`、
`phase_or_event`、`metadata`。`time_utc_ns` 必須是真正 integer，UTC year 必須等於
`year`，season 只接受正式 selector 的 `DJF`／`MAM`／`JJA`／`SON`，ID 全案唯一且同站
UTC 不重複。formal 除五站各 50、全案 250 外，每站還必須逐格符合以下 48+2 契約：
`config.inputs.years` 的兩年 × 四個 season × `spring_proxy`／`neap_proxy` ×
`fastest_rising`／`fastest_falling`／`slack_proxy` 各一筆，共 48 筆；另以
`tide_class: "event"` 恰含 `high_wave_event` 與 `strong_current_event` 各一筆。pilot 可為
config 站點的非空子集，不強制補齊 48+2，但所有 season label 仍須使用上述四值。相同 UTC
可在不同站點各自出現。

#### 7.3.2 Receptor×arrival dynamic initial-condition manifest

正式 `Receptor` 的 `z_m_positive_up` 只代表模板候選深度；實際 runtime 初始深度必須
來自每個 receptor×arrival pair 在 arrival UTC 的 OCM 摘要。此文件的 manifest root 必須
恰含 `manifest_kind`、`schema_version`、`status`、`design_version`、`vertical_reference`、
`time_standard`、`generation_method_id`、`provenance`、`records`，固定為
`receptor_arrival_initial_condition_manifest`、`1.0.0`、`z_m_positive_up`、`UTC`；formal
status 必須是 `approved`。每筆 record 恰含：
`receptor_id`、`arrival_time_id`、`study_site_id`、`analysis_region_id`、`flow_domain_id`、
`time_utc_ns`、`vertical_id`、`z_m_positive_up`、`eta_m_positive_up`、
`bed_z_m_positive_up`、`water_column_height_m`、`height_above_bed_m`、
`zcor_lower_m_positive_up`、`zcor_upper_m_positive_up`、`vertical_bracket_alpha`、
`source_face_local_index`、`source_face_global_index`、`wetdry_elem_value`、
`wetdry_semantics_id`、`ocm_month_yyyymm`、`ocm_source_time_index`、`ocm_time_origin`。

formal coverage 恰為五站各 `20×50=1,000`、全案 5,000 pairs；250 筆 arrival-time 與 100
筆 receptor template 仍分開保存。每站十種 material 共用同一 pair record，所以 scenario
仍是 `10×20×50`，全案 50,000，不能把 pair manifest 直接展成 50,000 筆。pilot 若明示
`require_dynamic_initial_conditions=true`，則對當次載入的站點子集要求完整 Cartesian
coverage。

loader 會逐欄核對 receptor／arrival 的站點、region、UTC ns、`vertical_id` 與 pair 唯一性，
並以 `DomainConfig.resolved_flow_domain_id` 統一解析 flow domain：pilot 使用 base ID；formal
若設定 `formal_release_flow_domain_id` 則使用該正式 ID，否則使用 base ID。formal／可執行
pilot 只接受 `wetdry_elem_value=0`、固定語意
`schism_wetdry_elem_0_wet_1_dry` 與 `ocm_time_origin=observed`。所有 z 與高度以公尺、z
向上為正；必須滿足 `eta>bed`、`water=eta-bed`、`bed≤z≤eta`、
`height=z-bed`、`bed≤zcor_lower<zcor_upper≤eta`、`lower≤z≤upper` 及
`alpha=(z-lower)/(upper-lower)∈[0,1]`。固定 `1e-8` 公差只吸收 JSON 浮點序列化尾差，
不允許實質域外 bracket 或物理不等式錯誤。

公開入口為 `load_receptor_arrival_initial_condition_manifest(path, config, receptors,
arrivals, formal=False)`；`ScenarioInputs.initial_conditions` 保存 immutable tuple，
`initial_conditions_by_pair[(receptor_id, arrival_time_id)]` 是唯讀 mapping。此 Phase 3A2
只驗證已產出的 JSON、保存 raw/canonical hash 與 pair mapping；不讀取 OCM、不產生 manifest。
pilot/formal runtime 都由 `RuntimeRequestFactory` 消費 pair actual `z_m_positive_up`；formal
另在 runtime initializer/open controller 重新驗證正式 inventory topology、時間軸與
gap-safe/full-product 支援。這代表程式入口已接通，不代表已完成 SERVER 科學批次。

#### 7.3.3 Lazy forcing window 契約

三種 geometry manifest 共用 root 的 `manifest_kind`、`schema_version`、`status`、
`design_version`、`coordinate_reference: "EPSG:4326"`、`provenance` 與 `records`。
geometry object 僅允許 GeoJSON `type`／`coordinates`，不得含 Z 座標、空 geometry 或
invalid geometry。Shapely 解析前，每個位置必須恰為有限數值 `[lon, lat]`，不得使用 bool、
文字、NaN／Infinity、混合 nesting 或第三維；經度範圍為 `[-180, 180]`，緯度範圍為
`[-90, 90]`。投影後會再檢查 geometry 非空、有效且所有公尺座標有限。最小 record 形狀如下：

| 文件 | `manifest_kind` | 每筆 record 欄位 |
|---|---|---|
| domain | `domain_geometry_manifest` | `analysis_region_id`, `flow_domain_id`, `geometry`（Polygon）, `source_geometry_id` |
| local | `local_geometry_manifest` | `study_site_id`, `analysis_region_id`, `flow_domain_id`, `local_equals_flow`, `geometry`（Polygon）, `source_geometry_id` |
| open-boundary | `open_boundary_manifest` | `owner_kind`（`flow_domain`／`local_domain`）, `owner_id`, `analysis_region_id`, `segment_id`, `geometry`（LineString／MultiLineString）, `source_geometry_id` |

loader 先以各 flow domain config center 建立 AEQD 公尺投影，再檢查 local polygon 是否在
flow polygon 的 `coordinate_round_trip_tolerance_m` 內；`local_equals_flow` 必須拓撲一致。
open-water line 必須落在對應 polygon boundary 的公尺容許帶。A 區兩站只把相同
`flow_domain_id` 的另一站 local polygon 放入 `foreign_local_domains`；不同 forcing
domain 絕不互相加入。B-D 若 `local_equals_flow=true` 且沒有重複 local line，使用已驗證
的 flow line 作 own-local line；A 區或其他不相等 local 則必須有自己的 line。formal 要求
四 domains、五 sites 與有效的 flow／local open-boundary coverage；pilot 可省略未選站點，
但每個已選站點仍須有有效 own/flow boundary。

#### 7.3.2 Lazy forcing window 契約

`ForcingWindowManager` 綁定單一 flow domain、固定 `DomainProjection`、`NativeMesh`、
OCM/NWW roots 與正整數 `max_resident_months`。production root layout 為
`<root>/<flow_domain_id>/grid` 及 `months/YYYYMM`；OCM mesh 只在
`from_roots` 建構時載入一次。每一個 RK stage 以 UTC `YYYYMM` 嚴格選月，沒有月份時
OCM 回 `OUTSIDE_TIME_RANGE`；OCM 存在但 include-Stokes 所需 NWW 整月不存在時回
`WAVE_UNSUPPORTED`。已存在目錄但缺 array、schema、shape 或 physics 不合法會上拋，
不能被轉成零速度或另一種 QC。

`provider(settling_velocity_mps, include_stokes)` 回傳只保存 manager／參數的 facade，
可直接交給 `ReferenceParticleRequest` 或 `HintTrackingVelocityProvider`；sample 的
`triangle_hint` 完整傳入既有 OCM locator。不同 material、Stokes policy、hint 或 facade
共用同月 OCM；NWW 只有第一次有 Stokes 需求才載入。LRU 淘汰會移除同月所有
`CombinedMonthForcing` cache，facade 不保留大型 month reference。`cache_stats` 是 immutable
snapshot，包含 OCM/NWW load、hit/miss、eviction、resident `YYYYMM` 與估計 ndarray bytes；
每個 process/worker 必須各建一個 manager，instance 本身不保證 thread safety。

Slice 2B2 另提供 `smagorinsky_provider(settings)` 與
`ManagedSpatialDiffusionProvider`。它只保存 manager／frozen `SmagorinskySettings`，每次
以 UTC 月份回到同一 OCM LRU，直接呼叫 OCM native mesh 的 P1 nodal Smagorinsky sample；
不呼叫 `_ensure_nww`、`nww_loader` 或 `CombinedMonthForcing`。因此「diffusion provider
不讀 NWW」是資料路徑契約，但 `smagorinsky_cs_010`、`015`、`020` 的 velocity case 仍
包含有限水深 Stokes，request 的 velocity facade 仍需 NWW root。OCM 整月缺失時，擴散
sample 以非零 `OUTSIDE_TIME_RANGE`、固定方法名稱、`forcing_month_id` 與缺失原因回報，
不以零 Kh 偽裝有效樣本。

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

### 7.5 Run plan、progress 與 restart 契約

Phase 3B2a 的 `initialize_run_workspace` 將依版本化
`analysis_region_arrival_utc_site_material_receptor_scenario_v1` 排序的 `Scenario` 交給
既有 `plan_scenario_shards`，先按 `(analysis_region_id, arrival_time_utc_ns)` 分 execution
group，再在每個 group 內切 shard；同一 group 不會與下一個 group 合併。這只改變 I/O
locality 與恢復邊界，不改科學樣本、`scenario_id`、particle ID 或 seed。它以同一 parent
的 partial directory 原子發布
`normalized_config.json`、`input_inventory.json`、`scenario_table.parquet`、
`seed_table.parquet`、`run_plan.json`、`run_progress.json` 與固定 `locks/` topology。plan schema 2 保存 raw input
inventory SHA-256、component／geometry canonical hashes、`CodeProvenance`、experiment／
master seed、`M`、shard range/hash、checkpoint interval、active chunk 與 run file checksum；
並保存 ordering policy、execution group metadata 與固定 `lock_root=locks`；四個 immutable
input checksum 不包含 lock file。seed table 以 32 位 hex 保存 128-bit PCG64DXSM seed，不能
以列順序或 worker 編號重新產生。schema 1 舊 workspace 不支援 resume，必須重建，不能猜測
ordering。

設定中的 checkpoint cadence 欄位固定命名為 `checkpoint_interval_sweeps`（完整 sweep），
不是已淘汰的 `checkpoint_interval_output_steps`；`active_chunk_size` 與
`max_resident_forcing_months` 只描述 CPU/NumPy 執行的分塊與 cache 上限。Slice 3B2a-A
已接通 pilot/formal runtime：`initialize_run`、`initialize_formal_run`、
`initialize_pilot_run`、`open_run_controller` 與兩種相容 wrapper 共同遵守 immutable plan、
formal inventory semantic gate、lazy forcing 與 checkpoint root forwarding；CLI 提供
`lbt run-create`、`lbt run-shard`、`lbt run-reconcile`。example config 仍因缺少正式
SERVER release manifests 而 fail-closed；五個 runtime experiment cases 由唯一 immutable
registry 登錄，`finite_depth_stokes` 是 formal baseline 候選，Smagorinsky cases 只作已接通
的研究敏感度，仍須通過 pilot、well-mixed、PDE、收斂與 floor/cap 閘門。本輪未登入或
執行 SERVER，也沒有正式結果。

progress schema 1 是唯一可變檔案，以 monotonically increasing `revision` 原子替換；run 與
shard lifecycle 只接受 `PLANNED`、`RUNNING`、`PAUSED`、`COMPLETE`、`FAILED`。checkpoint
固定位於 `checkpoint_root/run_id/shard_id/checkpoint-########`，不覆寫既有 generation；
`latest.json` 保存相對 directory token、sequence、`checkpoint.json` SHA-256、精確
`particle_steps`、`sweeps_completed` 與 counter 來源。generation 已完成但 pointer 尚未更新的
crash window，可由 execution 的每粒子 `step_count` 精確重建 particle steps；schema 2 未保存
run-level sweep 時只能保守記錄 sweep 下界，並以
`sweeps_recovered_lower_bound=true` 提供 machine-readable 標記。後續 checkpoint cadence 永遠
從上一個合法 generation 起最多再前進 plan interval，不以此下界取模，因此不改變物理狀態、
亂數狀態或下一次 checkpoint 時機。checkpoint
schema 2 的 `CheckpointBinding.input_inventory_hash` 使用 composite hash（raw inventory、
component／geometry、deployment tree、uv.lock），plan 仍單獨保存 raw inventory SHA。

只有 batch terminal、trajectory shard validator 通過後才可把 shard 標成 COMPLETE；PAUSED
只保存 checkpoint，不發布 partial trajectory。若 output 已發布但 progress 尚未更新，
`RunController.reconcile` 會在 binding、完整 RunUnit identity/order、checksum 與 formal/pilot
tracked-run metadata 通過時
採認；progress 已 COMPLETE 但 output 遺失或損壞則 fail-fast。`run_validation.validate_run`
是純讀檢查，不會替現場修復 latest 或刪除 unknown/partial 檔案；合法但尚未 reconcile 的
published output、missing/stale latest 與 RUNNING orphan generation 會回報可恢復但仍然
`valid=false`。

workspace 的鎖檔恰為 `run_gate.lock`、`progress.lock` 與每一個 shard 的 `<shard_id>.lock`，
均須是預建、零長度、非 symlink 普通檔案。worker 先取 run gate shared，再取 shard
exclusive；每次 progress mutation 再取 progress exclusive 並重讀最新 revision，避免不同
shard process lost update。reconcile 取 run gate exclusive，若有 worker 持有 shared gate
便立即 busy，不做部分修復。這個契約依賴 Unix `fcntl.flock`；NFS/NAS 的鎖定語意仍須在
目標 SERVER preflight 實測，不能以本機測試代替。

SERVER 可在建立 controller、執行 `validate_run` 或輸出 benchmark 時傳入 runtime external
checkpoint root；此絕對路徑不進入 plan/progress/output JSON。progress 宣告 generation 而
指定 root 找不到時，必須在 request factory 與 forcing I/O 前 fail-closed；PLANNED shard
看到任何 generation 亦拒絕採認。RUNNING、PAUSED、FAILED 都必須由 caller 明示 resume，
COMPLETE 僅重驗已發布 shard。unknown directory、partial、symlink、checksum、binding 或
RunUnit order 不符一律保留現場並拒絕恢復。

## 8. 輸出資料契約

上方 tree 是完整成果／aggregate 的長期交接拓撲；Phase 3B2a 目前先發布下列 run workspace
輸入與 CPU shard 子集：`shards/<shard_id>/` 仍使用本文件既有的 trajectory shard schema，
而 `run_control` 另保存 `checkpoints/`、`failures/` 與 immutable plan/progress。這個 slice
不宣稱已完成 formal SERVER forcing、gap-safe/full-product 科學 release、streaming aggregate
或 figures；pilot/formal CLI 已可建立、執行與 reconcile 通過各自輸入 gate 的 workspace，
但 example config 與未核准 inventory 仍會 fail-closed。

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
