# 五站點情境與巢狀邊界設計基線

> **閱讀提示**
> - 文件類型：研究設計基線與衍生驗收閘門。
> - 它回答：五站點、情境數、沉降代理、幾何與資料閘門固定為何。
> - 建議先讀：[需求追溯](01_requirements_traceability.md)，再核對[實作狀態](../implementation_status.md)。

## 1. 文件地位

本文件記錄 2026-08-17 完成、2026-08-27 依研究主持人意見修訂，並於 2026-09-09 由 A 區範圍裁決更新的設計基線。原 `design_baseline_v2_non_rising_oca_proxy` 的十個非上浮材質／形狀代理、嚴格負沉降速度與物性限制仍沿用；本期 A 區幾何與 domain 身分改採 `design_baseline_v3_non_rising_a_v3_local20_20260909` 及 `formal_domain_policy=v3_local20km_20260909_v1`。使用者已明示沒有可再提供的單體物性資料，因此只有必須由 SERVER 實際資料、OCM 網格、現地樣本或先導試驗計算出的數值保留為「衍生閘門」。

本裁決的核心原則是：**四個 forcing flow domains 不等於四個情境統計單元**。貢寮與龜山島共用同一套東北台灣 OCM/NWW forcing，但兩者是獨立研究站點，各自具有 20 個受體、50 個到達時間與完整 `10×20×50` 情境矩陣。

## 2. 上游依據與版本

| 來源 | SHA-256 | 本專案採用內容 |
|---|---|---|
| `OCM-Data-Preprocessing/configs/ocm_flow_domains.json` | `b8db61c38138d5690d203bf1b3785c6b2e581572d08f097573772c554ef373b3` | 四個正式 flow domain 的識別碼、中心與 bbox |
| `OCM-SVD-Analysis/configs/guishan_gongliao_northeast_taiwan_flow_domain_water_column_svd_available_2024_2025.json` | `daeb9f876eb8a62996b2f7b762e5cee0e03298adf010c802f03261527bdf67e0` | 貢寮與龜山島共用完整東北台灣水柱聯合 SVD／flow domain，不重複建立 forcing |
| `OCM-SVD-Analysis/configs/gongliao_surface_svd_available_2024_2025.json` | `70fc29a16bfd25f468f7a9aac5ea431fea604aed1b47ac88314324c1b1767c7d` | 貢寮 anchor 與舊候選框 provenance；舊框不作本專案 local domain |
| `OCM-SVD-Analysis/configs/guishan_surface_svd_available_2024_2025.json` | `58d79fc374aff88841ac354f34d407256048bff24d46dfb840d73c82c398bcf4` | 龜山島西側 anchor 與舊候選框 provenance；舊框不作本專案 local domain |
| [海洋保育署 iOcean 海洋廢棄物管理頁](https://iocean.oca.gov.tw/OCA_OceanConservation/PUBLIC/Marine_Litter_v2.aspx) | 動態網頁；2026-08-27 查閱 | 採用查詢介面顯示的十個海廢項目作情境分類名稱；不採用重量／件數推估物性或來源先驗 |

上游兩個候選框的線性尺度不足以作逆向傳輸的 local boundary。依使用者最新裁決，本專案只沿用其 anchor 與 provenance，**不沿用候選 bbox**；改以公尺制等距緩衝建立版本化的 Lagrangian local domain。這不回寫或改變上游 SVD 的核定狀態。

## 3. 四個 flow domains 與五個獨立站點

### 3.1 Forcing 層

| `analysis_region_id` | flow domain | bbox `[lon_min, lon_max, lat_min, lat_max]` | 中心 |
|---|---|---|---|
| A | `northeast_taiwan_common_cache_v3` | `[121.306315, 122.793685, 24.600844, 25.499156]` | `[122.05, 25.05]` |
| B | `hsinchu_cache_v3` | `[119.70812, 121.19188, 24.300844, 25.199156]` | `[120.45, 24.75]` |
| C | `houwan_nmmba_cache_v3` | `[120.16671, 121.62, 21.550844, 22.449156]` | `[120.893355, 22.0]` |
| D | `lienchiang_common_cache_v3` | `[119.19912, 120.70088, 25.750844, 26.649156]` | `[119.95, 26.2]` |

四個 flow domains 是 forcing 支撐與最外層停止邊界。不得為貢寮與龜山島複製兩份相同 A 區 OCM/NWW 資料，兩站點只在情境、local domain、受體與成果分層上分開。兩站使用同一個 A 區 outer boundary：逆向軌跡離開自己的 local domain 後仍沿同一套 A 區水動力場積分，只有首次穿越共用 A 區 open boundary 才觸發 `flow_domain_open_exit`。此設計刻意保留兩地水動力互通性，避免用任意站點分界截斷可能的共享傳輸走廊。

### 3.2 站點層

| `study_site_id` | 中文名稱 | region | logical local_domain_id | 幾何政策 | anchor |
|---|---|---|---|---|---|
| `gongliao` | 貢寮 | A | `gongliao_local_domain_v1` | 本期 policy 的 anchor 公尺制半徑 20 km buffer 與靜態 OCM 海域 polygon 的交集 | `[121.92807, 25.11245]` |
| `guishan` | 龜山島西側 | A | `guishan_west_local_domain_v1` | 本期 policy 的 anchor 公尺制半徑 20 km buffer 與靜態 OCM 海域 polygon 的交集 | `[121.951606, 24.843127]` |
| `hsinchu` | 新竹外海 | B | `hsinchu_flow_domain_v1` | local domain 與 flow domain 相同；受體候選為 `[120.45,24.75]` 半徑 12.5 km 核心與既有 local 候選區的交集 | `[120.45, 24.75]` |
| `houwan` | 後灣海生館 | C | `houwan_flow_domain_v1` | local domain 與 flow domain 相同 | flow-domain center 經 wet-mesh snap |
| `lienchiang` | 連江 | D | `lienchiang_flow_domain_v1` | local domain 與 flow domain 相同 | flow-domain center 經 wet-mesh snap |

貢寮、龜山島與新竹均明示半徑 12.5 km 的 `receptor_core_v1`，五個水平受體位置只在
各自核心圓與既有 local／static-ocean 候選區的交集內選取。貢寮與龜山島本期另以半徑 20 km
local domain 辨識正向移入關注海域的入口方向；新竹的 local domain 仍與
`hsinchu_cache_v3` flow domain 相同，核心圓只限制受體候選，不改變 flow/local 邊界。
本期 20 km local domain 與 12.5 km receptor core 是新的 geometry／design identity；表中
logical `local_domain_id` 可延續既有站點標籤，但其 geometry binding 必須帶入本期 policy、
新 config／design 身分與新 hash。原 25 km baseline 與 20/35 km 敏感度規劃已移出本期，亦不
自行加入 15 km 或 23 km case。
若後續要恢復其他半徑，必須建立新的範圍決策與完整 manifest，不得沿用舊 geometry 或 hash。

現行 A 區 `northeast_taiwan_common_cache_v3` 的 bbox 固定為
`[121.306315,122.793685,24.600844,25.499156]`，本期不南擴。2026-09-15 裁決正式流程
沿用 A 區工程試跑的空間支援核心方案
`runtime_spatial_support_policy=runtime_stage_fail_closed_no_expansion_v1`，並以
`formal_release_domain_status=no_expansion_runtime_stage_fail_closed` 鎖定狀態，但不沿用
24 小時 DEMO 的資料證據：OCM surface
負責完整母體的 arrival 篩選，OCM native／NWW analysis 則由 runtime 在每個 RK4 stage
依實際位置、深度、UTC 與 mask 嚴格判定。任一必要 forcing 無效時立即停止並保存原始
狀態，不使用零值、最近值、未登錄外插或擴張資料域。30 天正式輸入仍必須重新通過完整
四區五站母體、field、mask、時間軸、逐 arrival gap-safe 與來源雜湊驗收。

原先的 `northeast_taiwan_common_cache_v4_lbt_south_expanded` 與 bbox
`[121.306315,122.793685,24.480000,25.499156]` 只作南向擴張的歷史候選，不是本期正式
domain；舊 expanded geometry、manifest、shard 與 hash 均不相容，不能沿用。舊
`formal_domain_policy=expanded_domain_v1` 僅保留為 legacy configuration 的預設相容身分，
不改變本期 A 範圍；B–D 的 `expanded_domain` 敏感度仍依各自 gate 驗證。

所有圓形距離都在以 anchor 為中心的 Azimuthal Equidistant CRS 計算，不以經緯度差近似公里。local-domain polygon 使用固定的 OCM native mesh／海岸拓撲建立 `static_ocm_ocean_polygon`，不得隨到達時間改變；陸地、島體及無有效三角形區域必須剔除。動態濕乾只用於受體與逐步 forcing 有效性 gate，避免讓 local boundary 因五十個時次的選取結果而循環改變。兩 anchor 的近似大圓距離約 30.0 km，故兩個半徑 20 km local domains 將自然形成重疊區；此重疊代表相連水動力環境，不代表情境合併，所有受體、scenario、seed、主要事件及統計仍由 `study_site_id` 隔離。

每條軌跡只以其 `study_site_id` 所屬 local boundary 定義主要 `local_domain_first_exit`。軌跡穿越另一站 local domain 時不得停止、不得改變 `study_site_id`、不得轉移 scenario 或併入另一站主要入口分母；可另寫 `other_site_local_domain_enter`／`other_site_local_domain_exit` 非終止事件，供計算跨站穿越比例、共享傳輸走廊與 local-domain footprint overlap。此診斷回答「到達某站的軌跡是否曾經過另一站周邊」，不把它誤稱為兩站之間的實測交換率或絕對轉移機率。

`local_domain_first_exit` 只配置在 20 km 圓周所形成且連接有效外海的 open-water arcs；矩形／圓形與海岸相交形成的岸線仍是 `coast_contact`，不可計入 local entry KDE。此分類使「移入關注海域入口」不會被陸地邊界污染。

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

1. 貢寮、龜山島與新竹的候選集合限於各自明示 anchor 半徑 12.5 km `receptor_core_v1`
   與既有 local／static-ocean 候選區的交集；後灣與連江等未明示核心的站點則限於各自
   local/flow domain。所有候選均須具有效 OCM triangle、非陸地且可支援全部 50 個到達
   時間的 persistent-wet 節點或三角形中心。此 persistent-wet 條件只篩受體，不改變固定
   local-domain polygon。
2. 貢寮、龜山島與新竹均以各自明示 anchor 作 deterministic maximin 的第一點排序依據，
   並在 manifest 保存原始 anchor、實際受體位置及 `anchor_snap_distance_m`。目前
   `input_derivation.py` 傳入受體 selector 的 policy 會保存實際 snap distance，但沒有把
   「最多兩個局地代表網格尺度」作為受體 anchor 的 runtime hard gate；因此本版不把該
   上限宣稱為已執行條件。這不要與 arrival NWW metric proxy 的「最大兩倍實際 NWW
   grid scale」政策混同。新竹 anchor 固定為已登錄的 `[120.45, 24.75]`。
3. 其餘四點以固定 seed 的 metric-space maximin 演算法依序選取，使最小點間距最大；並以 `lon, lat, source_face_id` 作 tie-break，確保重跑結果相同。
4. 後灣與連江以 flow-domain center 的最近有效海洋位置作第一點，再使用相同 maximin 規則
   選四點；新竹雖與 B 區 flow center 重合，仍依設定明示的 anchor 與核心候選區執行。
5. 候選點距海岸、無效 triangle 或 flow-domain 外界至少一個局地代表網格尺度；若此限制使候選不足，先降低為半個尺度並記錄 QC，不以陸地最近鄰補值。

### 5.3 垂向層位

本節的 `depth` 不是每月 forcing 內另存的一個欄位，而是來自 OCM 靜態網格的
`source_depth_m.npy`。此欄位代表海床相對垂向基準面的水深，採正值向下；上游
前處理設定可能以 `bathymetry_m.npy` 命名，但進入本專案資料契約後統一使用
`source_depth_m.npy`。`eta` 則來自每月 OCM 的時變自由水面高程 `elev.npy`，
採 `z` positive-up。兩者的資料欄位、單位與 `zcor` 的關係見
[資料契約](02_architecture_and_data_contract.md)。

對水平 receptor 所在的原生 OCM face，先以同一組 barycentric 權重內插靜態與時變
欄位。令三個 face 節點為 `i`、水平權重為 `w_i`、到達時間為 `t_a`，定義：

```text
h_r       = sum_i(w_i * source_depth_m[i])
eta_r(t_a) = sum_i(w_i * elev[t_a, i])
H_r(t_a)  = h_r + eta_r(t_a)
```

其中 `h_r` 是水平位置的靜態水深，`eta_r(t_a)` 是該到達時刻的自由水面高程，
`H_r(t_a)` 才是建立比例層位時使用的局地總水深。`H_r(t_a)` 必須為正且所有節點
均有效；不得以零值、最近鄰或單側外插補上無效水深／水面。這也表示同一水平
receptor 在不同到達時間的實際 `z` 可能因潮位而略有變化，manifest 必須逐一保存
到達時間對應的 `H_r` 與實際層位。

每個水平位置以該到達時間的局地總水深 `H_r(t_a)` 建立四個目標：

- `upper_water_column`：目標 `z = eta_r(t_a) - 0.10H_r(t_a)`；
- `mid_upper_water_column`：目標 `z = eta_r(t_a) - 0.40H_r(t_a)`；
- `mid_lower_water_column`：目標 `z = eta_r(t_a) - 0.70H_r(t_a)`；
- `near_bed`：最低有效 OCM layer 的中心；同時保存實際 height above bed。

上述目標的實際位置必須使用同一到達時間、同一 face 的 `zcor` 做垂向有效性檢查
與 layer snap；`zcor` 是實際 OCM 物理層座標，不可把固定 layer index 當成固定深度，
也不可以 `source_depth_m` 取代 `zcor`。不得將粒子放在海面以上、海床以下或只有單側
支撐的層位。若兩個目標落入同一有效層，選擇相鄰可用層以維持四個不同 receptor；若
無法形成四個有效層位，該水平位置淘汰並改選下一個 maximin 候選。manifest 必須保存
`h_r`、`eta_r(t_a)`、`H_r(t_a)`、目標比例、目標 `z`、snap 後 `zcor` 層位、實際
`z_m`／HAB、調整原因與所有 50 個到達時間的有效性。

## 6. 十種非上浮海廢材質／形狀代理基線

海洋保育署 iOcean 頁面可支持「我國清除統計使用哪些海廢項目名稱」，但沒有單體密度、尺寸、投影面積、形狀因子、阻力係數、生物附著量或沉降速度。因此，本版將官方十項分類與十個嚴格負值速度一對一配對，目的在建立可辨識、可重現的材質／形狀敏感度矩陣；**配對順序與數值不是官方量測、類別平均值或由清除重量回歸而得**。

| `material_id` | iOcean 項目 | 代表材質與形狀代理 | `settling_velocity_mps` | 適用條件 |
|---|---|---|---:|---|
| `oca_styrofoam_porous_fragment` | 保麗龍 | EPS 多孔不規則碎塊 | -0.0001 | 限吸水或生物附著後仍完全沉沒者 |
| `oca_wood_waterlogged_elongated` | 竹木 | 水浸飽和之細長枝條或片狀木屑 | -0.0002 | 限整體密度已高於周圍海水者 |
| `oca_wastepaper_folded_fiber` | 廢紙 | 濕潤摺疊紙片或纖維團 | -0.0005 | 不模擬持續解體或溶散 |
| `oca_nonrecyclable_flexible_sheet` | 其他／不可回收 | 進水薄膜、軟片或皺摺複合包材 | -0.001 | 分類內異質性另列主要限制 |
| `oca_fishinggear_open_mesh_bundle` | 廢漁網漁具 | 展開／覆網網片、繩索或纏結纖維束的共同代理 | -0.002 | 不解析網目實度、纏結、展開、掛礁與姿態變化；僅作沉降敏感度代理 |
| `oca_pet_waterfilled_bottle` | 寶特瓶 | 進水或壓扁之 PET 中空瓶體 | -0.005 | 不含密閉含氣瓶體 |
| `oca_other_recyclable_irregular_fragment` | 其他／可回收 | 混合可回收材質之不規則片塊 | -0.010 | 只作異質類別代理，不作類別平均 |
| `oca_aluminum_crushed_cylinder` | 鋁罐 | 進水壓扁之薄壁中空圓筒 | -0.020 | 不含密閉含氣罐體 |
| `oca_steel_rigid_cylinder` | 鐵罐 | 進水之剛性圓筒或金屬片 | -0.050 | 姿態與腐蝕效應未顯式解析 |
| `oca_glass_bottle_or_fragment` | 玻璃瓶 | 進水瓶體或緻密銳角碎片 | -0.100 | 完整瓶與碎片差異納入未來校準 |

座標採 `z` positive-up，故十個速度皆為負值且代表物理時間向前的沉降；設定驗證必須拒絕 `settling_velocity_mps >= 0`。速度格點涵蓋 `10^-4–10^-1 m/s` 三個數量級，其中 `-0.001` 至 `-0.005 m/s` 鄰近 van der Molen et al. (2021) 對沉降 PS 顆粒採用的 `-0.0015、-0.004、-0.006 m/s` 敏感度範圍，但此相近性只支持量級測試，不足以校準上述十類大型或複合海廢。各類另以 B／C／D 記錄近似實驗、機制量級或無可轉用數值的證據等級，完整文獻對照見 [iOcean 海廢十類與非上浮沉降代理之文獻備查](../../data/marine_litter_classification/README.md)。若後續取得密度、尺寸、終端速度或現地樣本，應建立新 `experiment_case_id` 或 design version，保存舊版 run 為 `superseded`，不可靜默改寫本表。

### 6.1 沉底漁業用具的報告優先層

合作團隊簡報照片（`S__20529161.jpg`，SHA-256
`bff3187f875a8e312aa9744ae5488401fc7cfb5110e0675a0f21c23d901693ed`）指出海底廢棄物中漁業用具類
為最大宗，主管另口頭表示特別關注沉底漁業用具。這兩項目前都只能列為定性、待正式文件確認的
研究優先項，不能改寫十類代理的速度、出現率或來源先驗。

正文統計固定先讀取 `oca_fishinggear_open_mesh_bundle × near_bed`，並以
`report_material_statistics.py` 依 `study_site_id × material_id` 產生有效 member 分母、首次
海床接觸 member 計數／比例及沉積 member 計數／比例。其他九種材質與三個較上層受體不得刪除，
只是在正文外顯示為同尺度比較。`BED_CONTACT` 若多次發生，按 member 只計一次；
`deposit_on_first_contact_and_stop` 基線可只有 `DEPOSITED` terminal event，不要求 repeated
contact。只有另行登錄的 `bed_reflect`／再懸浮敏感度，才可將重複接觸作為核心結果。

## 7. 每站點 50 個到達時間

每站點使用固定且可重現的 `48+2` 設計：

```text
48 core = 2 years × 4 seasons × 2 spring/neap classes × 3 intra-tidal phases
2 extremes = 1 high-wave event + 1 strong-current event
```

Spring/Neap classes:
大潮(Spring tide)與小潮(Neap tide)是因太陽、地球與月球相對位置改變，引潮力相加或互相抵銷，導致水位落差(潮差)最大或最小的周期現象。
大潮發生於農曆初一與十五前後(潮差最大)；小潮發生於農曆初七、八(上弦月)與二十二、二十三(下弦月)前後(潮差最小)。

intra-tidal phases:
三個潮汐階段可根據當地潮位站或 OCM 海面高度資料，以固定且可重複的方法選出：潮位上升最快的時刻、下降最快的時刻，以及最接近漲退潮交替的時刻。這三個時刻只是用來區分潮汐階段，不能直接當作實測的最大漲潮流、最大退潮流或真正的轉流時刻。每一個 `year × season × spring/neap` 格子各選三個不同的 UTC 時間；若有多個候選時刻相同，依資料完整程度、離資料邊界的安全距離和時間先後順序決定，不使用隨機抽樣。

兩個事件補充時次分別由站點 local domain 的 NWW3 有效 `Hs` 與 OCM 三維流速代表統計量選取；兩者必須互異、未出現在核心 48 時次、且向前具有完整回溯 forcing 支撐。貢寮與龜山島在 coverage 允許時共用配對 UTC，以利隔離空間差異；每站點的潮況與極端標籤仍獨立計算。

確切 50 個 UTC 是 SERVER 資料衍生的 manifest，不是仍待使用者提供的科學選擇。

## 8. 巢狀邊界、接觸與停止條件

| 事件 | 基線行為 | 科學用途 |
|---|---|---|
| `local_domain_first_exit` | 首次離開站點 local domain 時記錄 crossing、segment、弧長與 backward age；貢寮／龜山島繼續積分 | 對應廢棄物在正向時間「移入關注海域」的主要入口方向 |
| `other_site_local_domain_enter/exit` | 穿越同一 A 區內另一站 local domain 時另記錄非終止事件；不得改變 scenario 所屬或主要入口分母 | 描述貢寮—龜山島共享傳輸走廊與條件式跨站連通診斷 |
| `flow_domain_open_exit` | 首次離開外層 flow domain 時記錄並停止 | 遠域條件式潛在來源與主要傳輸走廊 |
| `coast_contact` | 記錄首次接觸並停止 | 潛在沿岸來源；避免粒子穿陸 |
| `bed_contact_deposit` | 十個 sinking 代理類首次接觸海床後沉積並停止；報告層只計首次海床接觸與 assumed deposition | 沉積廢棄物來源足跡；不宣稱含再懸浮或 repeated-contact 動力 |
| `surface_reflect` | sinking 類因亂流擴散越過海面時反射並記錄 | 數值障壁處理；不代表材料具有上浮速度 |
| `forcing_start` | 到達可用 forcing 最早時次即 censor 並停止 | 禁止時間外插 |
| `forcing_gap` | 已知 OCM 缺時在 run 前由 approved reconstruction 或 gap-safe arrival/horizon 處理；正常 baseline 不在這些缺口停止 | 只有 manifest 外缺檔、checksum/I/O 損毀、空間必要欄位無效或局部重建失敗才停止，並以 origin/exposure/failure 圖避免誤讀 |
| `max_age` | 到先導試驗核定的最大回溯日數即 censor 並停止 | 防止封閉流線無限計算 |
| `numerical_failure` | NaN、定位失敗、步數上限或 CFL 無法滿足時停止 | 與物理停止原因分離 |

新竹、後灣與連江的 local domain 與 flow domain 相同，因此 `local_domain_first_exit` 與 `flow_domain_open_exit` 是同一 crossing，只寫一列具雙重語意的事件，避免重複計數。

## 9. 仍需計算、但不需使用者再確認的衍生閘門

| 衍生項目 | 決定方法 | 阻擋範圍 |
|---|---|---|
| SERVER 路徑、24 個月份、schema、濕乾語意與容量 | 唯讀 preflight、metadata、實值 QC；現有資料為完整 available 母體，不要求供應者補件 | manifest 外 schema/checksum/I/O 異常未排除前只可做合成測試與 TRIAL |
| OCM canonical 軸與缺時重建 | stable sort/prefer-last；依 1/23/24/25/49-step 實際缺口做多變量 EOF-harmonic state-space blocked validation與 Lagrangian skill 檢定 | 未通過者以 gap-safe arrival/horizon 作 baseline，不得 runtime 臨時補值 |
| NWW full-hour analysis | 從 17,544/17,544 完整 native UTC 重採樣到四個 OCM 靜態格網；方向依既定契約作圓形內插 | 產物未通過時含 Stokes run 不啟動；不對波浪時間作統計填補 |
| 12.5/20 km 巢狀 ocean polygons 與 100 個三維 receptor records | 依本期 A policy 建立 12.5 km receptor core、20 km local domain；OCM static ocean polygon、50 時次 wet/dry gate 與 deterministic selector | 未產出前不可凍結正式 scenario table |
| 五站點各 50 個確切 UTC | 到達時間 selector 與 forcing coverage gate | 未產出前不可啟動正式 batch |
| 常數 `Kh/Kz` baseline | Brownian／well-mixed 驗證與文獻合理範圍 pilot | 未通過時只跑無擴散解析或標記 trial |
| `M` | exit ranking、HDR、travel time、path density 的 member-convergence | 決定正式總軌跡數 |
| A 區版本化 domain policy | 依 `formal_domain_policy=v3_local20km_20260909_v1` 使用 v3 bbox、12.5 km receptor core 與 20 km local domain；禁止 v4、南擴及未登錄半徑 | domain／site ID、bbox、半徑、geometry identity 或 no-expansion policy 任一漂移即阻擋 formal |
| A 區沿途 forcing 空間支援 | OCM surface 先驗證完整 arrival 母體；OCM native／NWW analysis 在每個 RK4 stage 依實際位置、深度、UTC、mask 與物理值 fail closed | 無效支援立即停止並保留 `data_gap`／既有狀態；禁止補零、最近值、未登錄外插或改用 expanded domain。24 小時、20 情境 DEMO 不替代 30 天正式母體與時間證據 |
| `max_backtrack_days` | 比較 7、14、30、60 日的 exit/censor、HDR 與排名穩定性，取最小穩定值 | 決定正式 horizon，不改變 50,000 個基礎情境 |
| `dt`、output interval、shard、checkpoint 與並行度 | dt 收斂、particle-step benchmark、RAM/I/O/容量 | 決定數值與工程配置 |

這些項目是資料品質與數值驗收程序，不是未解的研究範圍。除非 preflight 發現上游資料與已記錄契約矛盾，後續實作可依表中規則自行完成，不需再次詢問使用者選項。

## 10. 成果第一層與次要彙整

所有正式圖表與統計先輸出五個站點層級。A 區可另提供貢寮與龜山島的 pooled product，但它只是次要彙整；預設採兩站點等權，不以其中有效 member 較多者自動取得較大權重，且必須同時保留兩站點原始分母、成功率與不確定性。跨站連通產品另列兩站各自的 foreign-local-domain crossing 比例、配對 UTC 的 pathway/HDR overlap 與共享走廊；其分母仍是來源站點的有效 members，不能把兩站軌跡先合併再計算。

最低合規成果為：local-domain raw entry crossings、沿邊界弧長密度、2D Gaussian KDE/HDR、外層來源出口、路徑訪格比例、代表性三維軌跡、旅行時間、懸浮與沉積分圖，以及 member/dt/domain/physics 敏感度。這一組合直接回應計畫書「視覺化懸浮與沉積廢棄物移入關注海域的主要潛在來源路徑」之成果要求。
