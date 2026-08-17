# 五站點情境與巢狀邊界設計基線

## 1. 文件地位

本文件記錄 2026-08-17 完成的設計裁決。使用者已明示沒有可再提供的科學數據，並授權計畫書未明定之處依本專案最佳科學與工程判斷處理。因此，下列項目是可直接實作的 `design_baseline_v1`，不再列為待研究團隊選擇的問題；只有必須由 SERVER 實際資料、OCM 網格或先導試驗計算出的數值，保留為「衍生閘門」。

本裁決的核心原則是：**四個 forcing flow domains 不等於四個情境統計單元**。貢寮與龜山島共用同一套東北台灣 OCM/NWW forcing，但兩者是獨立研究站點，各自具有 20 個受體、50 個到達時間與完整 `10×20×50` 情境矩陣。

## 2. 上游依據與版本

| 來源 | SHA-256 | 本專案採用內容 |
|---|---|---|
| `OCM-Data-Preprocessing/configs/ocm_flow_domains.json` | `b8db61c38138d5690d203bf1b3785c6b2e581572d08f097573772c554ef373b3` | 四個正式 flow domain 的識別碼、中心與 bbox |
| `OCM-SVD-Analysis/configs/guishan_gongliao_northeast_taiwan_flow_domain_water_column_svd_available_2024_2025.json` | `daeb9f876eb8a62996b2f7b762e5cee0e03298adf010c802f03261527bdf67e0` | 貢寮與龜山島共用完整東北台灣水柱聯合 SVD／flow domain，不重複建立 forcing |
| `OCM-SVD-Analysis/configs/gongliao_surface_svd_available_2024_2025.json` | `70fc29a16bfd25f468f7a9aac5ea431fea604aed1b47ac88314324c1b1767c7d` | 貢寮 anchor 與舊候選框 provenance；舊框不作本專案 local domain |
| `OCM-SVD-Analysis/configs/guishan_surface_svd_available_2024_2025.json` | `58d79fc374aff88841ac354f34d407256048bff24d46dfb840d73c82c398bcf4` | 龜山島西側 anchor 與舊候選框 provenance；舊框不作本專案 local domain |

上游兩個候選框的線性尺度不足以作逆向傳輸的 local boundary。依使用者最新裁決，本專案只沿用其 anchor 與 provenance，**不沿用候選 bbox**；改以公尺制等距緩衝建立較大的 Lagrangian local domain。這不回寫或改變上游 SVD 的核定狀態。

## 3. 四個 flow domains 與五個獨立站點

### 3.1 Forcing 層

| `analysis_region_id` | flow domain | bbox `[lon_min, lon_max, lat_min, lat_max]` | 中心 |
|---|---|---|---|
| A | `northeast_taiwan_common_cache_v3` | `[121.306315, 122.793685, 24.600844, 25.499156]` | `[122.05, 25.05]` |
| B | `hsinchu_cache_v3` | `[119.70812, 121.19188, 24.300844, 25.199156]` | `[120.45, 24.75]` |
| C | `houwan_nmmba_cache_v3` | `[120.16671, 121.62, 21.550844, 22.449156]` | `[120.893355, 22.0]` |
| D | `lienchiang_common_cache_v3` | `[119.19912, 120.70088, 25.750844, 26.649156]` | `[119.95, 26.2]` |

四個 flow domains 是 forcing 支撐與最外層停止邊界。不得為貢寮與龜山島複製兩份相同 A 區 OCM/NWW 資料，兩站點只在情境、local domain、受體與成果分層上分開。

### 3.2 站點層

| `study_site_id` | 中文名稱 | region | local domain | 幾何政策 | anchor |
|---|---|---|---|---|---|
| `gongliao` | 貢寮 | A | `gongliao_local_domain_v1` | anchor 公尺制半徑 25 km buffer 與靜態 OCM 海域 polygon 的交集 | `[121.92807, 25.11245]` |
| `guishan` | 龜山島西側 | A | `guishan_west_local_domain_v1` | anchor 公尺制半徑 25 km buffer 與靜態 OCM 海域 polygon 的交集 | `[121.951606, 24.843127]` |
| `hsinchu` | 新竹外海 | B | `hsinchu_flow_domain_v1` | local domain 與 flow domain 相同 | flow-domain center 經 wet-mesh snap |
| `houwan` | 後灣海生館 | C | `houwan_flow_domain_v1` | local domain 與 flow domain 相同 | flow-domain center 經 wet-mesh snap |
| `lienchiang` | 連江 | D | `lienchiang_flow_domain_v1` | local domain 與 flow domain 相同 | flow-domain center 經 wet-mesh snap |

貢寮與龜山島均另設半徑 12.5 km 的 `receptor_core_v1`，五個水平受體位置只在核心內選取；半徑 25 km 的 local domain 則用於辨識正向移入關注海域的入口方向。此「受體核心：local boundary = 1:2」的巢狀尺度可避免剛釋放便觸及 local boundary，亦顯著大於原 SVD 候選框。正式分析預先登錄 20 km 與 35 km local-domain 半徑敏感度；若主要入口排名或 HDR 對尺度不穩定，報告必須呈現範圍而非單一邊界結論。

龜山島 anchor 至 A 區 bbox 南界依近似大圓距離僅約 26.9 km，因此 25 km baseline 必須由實際 mesh 證明 local open boundary 與 flow-domain 外界仍保有至少兩個局地 OCM 網格尺度的 forcing margin。35 km 案例不得裁切後假稱完整 35 km local domain；它只可在擴充 forcing domain 通過 G1 後執行。若 25 km baseline 的兩格 margin 未通過，優先建立擴充 A 區 forcing，而不是靜默縮回舊候選框。

所有圓形距離都在以 anchor 為中心的 Azimuthal Equidistant CRS 計算，不以經緯度差近似公里。local-domain polygon 使用固定的 OCM native mesh／海岸拓撲建立 `static_ocm_ocean_polygon`，不得隨到達時間改變；陸地、島體及無有效三角形區域必須剔除。動態濕乾只用於受體與逐步 forcing 有效性 gate，避免讓 local boundary 因五十個時次的選取結果而循環改變。兩 anchor 的近似大圓距離約 30.0 km，故兩個半徑 25 km local domains 將自然形成重疊區；此重疊代表相連水動力環境，不代表情境合併，所有受體、scenario、seed、事件及統計仍由 `study_site_id` 隔離。

`local_domain_first_exit` 只配置在 25 km 圓周所形成且連接有效外海的 open-water arcs；矩形／圓形與海岸相交形成的岸線仍是 `coast_contact`，不可計入 local entry KDE。此分類使「移入關注海域入口」不會被陸地邊界污染。

## 4. 情境矩陣與識別碼

五個站點分別採完整交叉：

```text
N_base_per_site = 10 materials × 20 receptors × 50 arrival times
                = 10,000

N_base_region_A = 2 sites × 10,000 = 20,000
N_base_total    = 5 sites × 10,000 = 50,000
```

若所有基礎情境採相同隨機成員數 `M`：

```text
N_trajectory_per_site = 10,000 × M
N_trajectory_region_A = 20,000 × M
N_trajectory_total    = 50,000 × M
```

若各情境使用不同 `M_s`，正確總數為 `sum_s(M_s)`。no-Stokes、擴散、domain expansion 或邊界敏感度以 `experiment_case_id` 另行計算，不混入 50,000 個 baseline scenarios。

基礎識別碼固定為：

```text
scenario_id = hash(
    study_site_id,
    material_id,
    receptor_id,
    arrival_time_id,
    design_version
)
```

`analysis_region_id` 保留為 forcing 與跨站點彙整欄位，但不得取代 `study_site_id`，否則貢寮與龜山島的同名受體或時間可能碰撞。

## 5. 每站點 20 個受體的產生規則

### 5.1 結構

每站點採 **5 個水平位置 × 4 個垂向層位 = 20 個三維 receptors**。這比在三維空間任意散布 20 點更容易檢驗水平與垂向覆蓋，也能以相同設計比較五站點。

### 5.2 水平位置

1. 貢寮與龜山島的候選集合限於各自半徑 12.5 km `receptor_core_v1`；其餘三站點限於各自 local/flow domain。所有候選均須具有效 OCM triangle、非陸地且可支援全部 50 個到達時間的 persistent-wet 節點或三角形中心。此 persistent-wet 條件只篩受體，不改變固定 local-domain polygon。
2. 貢寮與龜山島以各自 anchor 的最近有效海洋位置作第一點；若 anchor 本身無有效三角形，只允許在兩個局地代表網格尺度內 snap，並保存原始 anchor、實際位置及距離。
3. 其餘四點以固定 seed 的 metric-space maximin 演算法依序選取，使最小點間距最大；並以 `lon, lat, source_face_id` 作 tie-break，確保重跑結果相同。
4. 新竹、後灣與連江以 flow-domain center 的最近有效海洋位置作第一點，再使用相同 maximin 規則選四點。
5. 候選點距海岸、無效 triangle 或 flow-domain 外界至少一個局地代表網格尺度；若此限制使候選不足，先降低為半個尺度並記錄 QC，不以陸地最近鄰補值。

### 5.3 垂向層位

每個水平位置以當時局地總水深 `H=eta+depth` 建立四個目標：

- `upper_water_column`：自海面向下 `0.10H`；
- `mid_upper_water_column`：自海面向下 `0.40H`；
- `mid_lower_water_column`：自海面向下 `0.70H`；
- `near_bed`：最低有效 OCM layer 的中心；同時保存實際 height above bed。

實際 `z` 需 snap 至可被各 node `zcor` 包夾的有效層位，不允許海面上或海床下外插。若兩個目標落入同一有效層，選擇相鄰可用層以維持四個不同 receptor；若無法形成四個有效層位，該水平位置淘汰並改選下一個 maximin 候選。manifest 必須保存目標比例、實際 z/HAB、調整原因與所有 50 個到達時間的有效性。

## 6. 十種浮沉行為基線

在缺少特定廢棄物材質、尺寸、密度與生物附著量測時，不應假造十種具名材料。baseline 將其定義為涵蓋三個數量級的**垂向行為類別**，供敏感度與傳輸機制比較；數值不是對任何特定廢棄物的量測校準。

| `material_id` | `settling_velocity_mps` | 行為 |
|---|---:|---|
| `sink_100mmps` | -0.100 | 快速沉降 |
| `sink_030mmps` | -0.030 | 沉降 |
| `sink_010mmps` | -0.010 | 沉降 |
| `sink_003mmps` | -0.003 | 緩慢沉降 |
| `sink_001mmps` | -0.001 | 近中性沉降 |
| `neutral_000mmps` | 0.000 | 中性懸浮 |
| `rise_001mmps` | +0.001 | 近中性上浮 |
| `rise_003mmps` | +0.003 | 緩慢上浮 |
| `rise_010mmps` | +0.010 | 上浮 |
| `rise_030mmps` | +0.030 | 快速上浮 |

座標採 z positive-up，故負值沉降、正值上浮。生物附著造成的隨時間變速、粒徑分布或材質先驗若未來取得，必須建立新 `experiment_case_id` 或 design version，不可靜默改寫本表。

## 7. 每站點 50 個到達時間

每站點使用固定且可重現的 `48+2` 設計：

```text
48 core = 2 years × 4 seasons × 2 spring/neap classes × 3 intra-tidal phases
2 extremes = 1 high-wave event + 1 strong-current event
```

三個潮內相位使用當地潮位站或 OCM elevation 的可重現 proxy：最快上升、最快下降與最接近轉流的 slack。它們代表潮位相位分層，不直接宣稱為實測三維最大漲／退潮流。每一 `year × season × spring/neap` cell 各選三個不同 UTC；同分時依 forcing completeness、距資料邊界安全度、時間先後排序決定，不用隨機抽樣。

兩個事件補充時次分別由站點 local domain 的 NWW3 有效 `Hs` 與 OCM 三維流速代表統計量選取；兩者必須互異、未出現在核心 48 時次、且向前具有完整回溯 forcing 支撐。貢寮與龜山島在 coverage 允許時共用配對 UTC，以利隔離空間差異；每站點的潮況與極端標籤仍獨立計算。

確切 50 個 UTC 是 SERVER 資料衍生的 manifest，不是仍待使用者提供的科學選擇。

## 8. 巢狀邊界、接觸與停止條件

| 事件 | 基線行為 | 科學用途 |
|---|---|---|
| `local_domain_first_exit` | 首次離開站點 local domain 時記錄 crossing、segment、弧長與 backward age；貢寮／龜山島繼續積分 | 對應廢棄物在正向時間「移入關注海域」的主要入口方向 |
| `flow_domain_open_exit` | 首次離開外層 flow domain 時記錄並停止 | 遠域條件式潛在來源與主要傳輸走廊 |
| `coast_contact` | 記錄首次接觸並停止 | 潛在沿岸來源；避免粒子穿陸 |
| `bed_contact_deposit` | sinking／near-bed 類首次接觸海床後沉積並停止 | 沉積廢棄物來源足跡；不宣稱含再懸浮 |
| `bed_contact_reflect` | neutral／suspended 類數值越界時反射並記錄 | 維持完全沉沒懸浮狀態；另報接觸率 |
| `surface_regime_exit` | rising 類到達海面時停止 | 表示已離開「完全沉沒」模型適用範圍；不得在未含 windage 時繼續當表面漂流 |
| `surface_reflect` | neutral／sinking 類因擴散越過海面時反射並記錄 | 數值障壁處理，不改變物性類別 |
| `forcing_start` | 到達可用 forcing 最早時次即 censor 並停止 | 禁止時間外插 |
| `forcing_gap` | 任一必要 OCM／NWW 支撐超過核定 gap 即停止 | 避免把缺資料解讀為低來源 |
| `max_age` | 到先導試驗核定的最大回溯日數即 censor 並停止 | 防止封閉流線無限計算 |
| `numerical_failure` | NaN、定位失敗、步數上限或 CFL 無法滿足時停止 | 與物理停止原因分離 |

新竹、後灣與連江的 local domain 與 flow domain 相同，因此 `local_domain_first_exit` 與 `flow_domain_open_exit` 是同一 crossing，只寫一列具雙重語意的事件，避免重複計數。

## 9. 仍需計算、但不需使用者再確認的衍生閘門

| 衍生項目 | 決定方法 | 阻擋範圍 |
|---|---|---|
| SERVER 路徑、24 個月份、schema、方向、濕乾語意與容量 | 唯讀 preflight、metadata、實值 QC | 未通過前只可做合成測試與 TRIAL |
| 12.5/25 km 巢狀 ocean polygons 與 100 個三維 receptor records | anchor-centered metric buffers、OCM static ocean polygon、50 時次 wet/dry gate 與 deterministic selector | 未產出前不可凍結正式 scenario table |
| 五站點各 50 個確切 UTC | 到達時間 selector 與 forcing coverage gate | 未產出前不可啟動正式 batch |
| 常數 `Kh/Kz` baseline | Brownian／well-mixed 驗證與文獻合理範圍 pilot | 未通過時只跑無擴散解析或標記 trial |
| `M` | exit ranking、HDR、travel time、path density 的 member-convergence | 決定正式總軌跡數 |
| local/outer boundary margin | 以實際投影 mesh 驗證 25 km local boundary 至 A 區外界至少兩個局地網格尺度；35 km 只在 expanded forcing domain 執行 | margin 不足時阻擋該 domain version，不縮回舊候選框 |
| `max_backtrack_days` | 比較 7、14、30、60 日的 exit/censor、HDR 與排名穩定性，取最小穩定值 | 決定正式 horizon，不改變 50,000 個基礎情境 |
| `dt`、output interval、shard、checkpoint 與並行度 | dt 收斂、particle-step benchmark、RAM/I/O/容量 | 決定數值與工程配置 |

這些項目是資料品質與數值驗收程序，不是未解的研究範圍。除非 preflight 發現上游資料與已記錄契約矛盾，後續實作可依表中規則自行完成，不需再次詢問使用者選項。

## 10. 成果第一層與次要彙整

所有正式圖表與統計先輸出五個站點層級。A 區可另提供貢寮與龜山島的 pooled product，但它只是次要彙整；預設採兩站點等權，不以其中有效 member 較多者自動取得較大權重，且必須同時保留兩站點原始分母、成功率與不確定性。

最低合規成果為：local-domain raw entry crossings、沿邊界弧長密度、2D Gaussian KDE/HDR、外層來源出口、路徑訪格比例、代表性三維軌跡、旅行時間、懸浮與沉積分圖，以及 member/dt/domain/physics 敏感度。這一組合直接回應計畫書「視覺化懸浮與沉積廢棄物移入關注海域的主要潛在來源路徑」之成果要求。
