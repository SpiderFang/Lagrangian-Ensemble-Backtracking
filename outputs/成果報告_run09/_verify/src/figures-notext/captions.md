# WP-21 native 正式成果圖表：圖說（2026-10-20 10:35 UTC）

文字層級：`--figure-text none`。本檔案逐圖列出圖說——這張圖是什麼、資料來源欄位、關鍵設定（站點、粒子數、M、終止邊界定義、物理案例）；文字與 `manifest.json` 對應圖檔的 `caption` 欄位完全相同，可用 SHA-256 核對圖檔與圖說是否對得上同一個版本。

## 貢寮（gongliao）

### `horizontal_overview.png`

代表粒子水平軌跡總覽，涵蓋整個 flow domain 範圍，標示 flow domain 開放邊界（終止邊界）與依 final_status_code 著色的終止標記；不畫 local domain 邊界或 local domain 首次離開標記（WP-25 裁決，該事件仍完整保存於 local_domain_departures.parquet 等資料表，只是不在這張圖呈現）。資料來源：trajectory_samples_by_site.parquet 的軌跡座標與時間、departures.parquet 的 final_status_code、boundary_arc_segments_geometry_by_site 的開放邊界折線幾何。 貢寮（gongliao），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `horizontal_local.png`

受體附近局部放大圖，座標為經緯度相對代表粒子回溯起點平均位置的偏移量，同樣只畫 flow domain 開放邊界與終止標記，不畫 local domain 邊界或首次離開標記（WP-25 裁決，理由同水平軌跡總覽圖）。資料來源與水平軌跡總覽圖相同：trajectory_samples_by_site.parquet、departures.parquet、boundary_arc_segments_geometry_by_site。 貢寮（gongliao），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `depth_age.png`

代表粒子深度對回溯時齡曲線，深度為海面以下之公尺數（海面為零、向上為正），顏色依代表粒子起始深度在同批抽樣中的排序序位分色，序位只用來分色，不對應固定 物理水層。資料來源：trajectory_samples_by_site.parquet 的 trajectory_z_m 與 trajectory_utc_ns。 貢寮（gongliao），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `terminal_counts.png`

本站實際出現之停止類別（含成功穿越開放邊界）的粒子計數與比例長條圖，計數為 0 的類別不畫，分母為 boundary_summary_by_site.total_particle_count。資料來源：departures.parquet 的 final_status_code 逐列統計，並與 boundary_summary_by_site 的 fraction_* 欄位 互相核對。 貢寮（gongliao），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `boundary_azimuth_rose.png`

16 方位扇區的相對來源權重極座標玫瑰圖（正北向上、順時針），標出前三大方位扇區名稱與百分比。相對來源權重＝該扇區成功穿越次數除以 全站成功穿越開放邊界的總次數。資料來源：departures.parquet 的 departure_azimuth_sector_index 與 departure_azimuth_sector_name。 貢寮（gongliao），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `boundary_arc_segments.png`

終止邊界（flow domain 開放邊界）依權重色階著色，共 20 段弧段，並以長條圖並列比較相對來源權重；相對來源權重＝該弧段成功穿越次數除以全站成功穿越開放邊界的總次數，只計在終止邊界的弧段上。boundary_arc_segments_geometry_by_site 另外登記 local domain 開放邊界的弧段幾何（共 20 段），純供地圖疊線的地理參照，權重恆為 0、不計入終止統計；local_equals_flow=True 的測站這條 local 邊界與上面的終止邊界是同一個幾何（見 boundary_summary_by_site 的 local_equals_flow 欄位），local_equals_flow=False 的測站則是各自站點獨立的 local domain 邊界。資料來源：boundary_arc_segments_geometry_by_site 的弧段幾何與 departures.parquet 的 boundary_segment_id、boundary_arc_segment_index。長條圖橫軸的「流場邊界 n」「局部邊界 n」，n 為資料表 arc_segment_index 加 1（圖面序號從 1 起算，資料表從 0 起算）。 貢寮（gongliao），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `source_footprint_kde.png`

條件式來源足跡核密度圖，背景為每平方公尺密度，疊加 50%／75%／90% 高密度區間（HDR，Highest Density Region）等值線，並疊海岸線與開放邊界作地理參照。資料來源：kde_by_site.parquet。 貢寮（gongliao），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `settling_velocity_breakdown.png`

十個沉降速度類別各自的穿越開放邊界比例長條圖，與各類別內方位扇區相對來源權重熱圖並列比較；縱軸每一列是一個類別，刻度為類別顯示名稱加上該類粒子實際用到的沉降速度範圍（最小值–最大值，取絕對值，單位公尺／秒），由上而下依類別代表沉降速度由慢到快排列。每類有 10 個候選沉降速度，每顆粒子抽用其中一個，各占該類粒子的十分之一；比例與權重以類別為單位彙整，不拆到個別速度。類別名稱為圖面顯示名稱：「不可回收塑膠」對應材質清冊的「其他/不可回收」，「可回收塑膠」對應「其他/可回收」，材質識別碼與清冊記錄不受影響。資料來源：departures.parquet 的 settling_velocity_mps（類別代表沉降速度，用於分組與排序）、particle_settling_velocity_mps（逐粒子實際速度，用於標示範圍）、material_id（類別識別）與 departure_azimuth_sector_index，依類別重新分組聚合。 貢寮（gongliao），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `source_footprint_residence.png`

停留時數密度圖，呈現回溯過程中這一站全部粒子逐小時所在位置的累計分布。停留時數的定義：每顆粒子在回溯的每個整點時刻，只要仍在漂流（尚未穿越開放邊界、擱淺、沉底或因其他原因終止），就在它所在的網格記 1 個粒子小時；回溯起點（到達時刻）的位置計入；粒子終止之後凍結的位置不再計入。各網格的粒子小時數除以本站粒子小時總數，再除以網格面積（km²），單位為「% 停留時數／km²」，全部網格對面積積分為 100%。網格為經緯度規則網格，解析度 0.02°（約 2.2 km），面積依緯度以 cos φ 計算。色階為對數色階，上限為最大值向上取整到 10 的整數次方（10^1），下限為上限的 1/10000（10^-3），每個 10 的次方標一個刻度，色條與地圖同高；低於下限的網格不著色（其數量與所占停留時數比例記錄在 manifest）。圖上的線為高密度區間（Highest Density Region, HDR）50%、75%、90%：依密度由高到低累加網格，累計到該比例的停留時數所需的最小面積範圍；本站三個區間的面積依序為 50% 為 1,249 km²、75% 為 3,225 km²、90% 為 5,536 km²。與舊版邊界核密度圖的差別：舊圖的輸入只有粒子穿越開放邊界的那一個點，這些點全部落在一維的邊界線上，二維核密度不適用；本圖改用回溯全程逐小時的位置，反映粒子在流場範圍內實際停留的地方。底圖為陸地（灰）與海岸線（黑），中灰細線為 200 公尺（實線）、1000 公尺（虛線）等深線，200 公尺為陸棚邊緣（水深取自計算網格的節點水深；為避免干擾高密度區間，只畫這些水深，地圖上不標水深數字，水深與線型見圖例），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：貢寮，受體位置（星號）為研究點錨點 121.9224°E, 25.0964°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `source_footprint_residence_site_scale.png`

停留時數密度圖，呈現回溯過程中這一站全部粒子逐小時所在位置的累計分布。停留時數的定義：每顆粒子在回溯的每個整點時刻，只要仍在漂流（尚未穿越開放邊界、擱淺、沉底或因其他原因終止），就在它所在的網格記 1 個粒子小時；回溯起點（到達時刻）的位置計入；粒子終止之後凍結的位置不再計入。各網格的粒子小時數除以本站粒子小時總數，再除以網格面積（km²），單位為「% 停留時數／km²」，全部網格對面積積分為 100%。網格為經緯度規則網格，解析度 0.02°（約 2.2 km），面積依緯度以 cos φ 計算。色階為對數色階，上限為最大值向上取整到 10 的整數次方（10^0），下限為上限的 1/10000（10^-4），每個 10 的次方標一個刻度，色條與地圖同高；低於下限的網格不著色（其數量與所占停留時數比例記錄在 manifest）；本圖採用本站自訂色階（本站最大值為 0.422 % 停留時數／km²），只適合看本站內部的分布形狀。圖上的線為高密度區間（Highest Density Region, HDR）50%、75%、90%：依密度由高到低累加網格，累計到該比例的停留時數所需的最小面積範圍；本站三個區間的面積依序為 50% 為 1,249 km²、75% 為 3,225 km²、90% 為 5,536 km²。與舊版邊界核密度圖的差別：舊圖的輸入只有粒子穿越開放邊界的那一個點，這些點全部落在一維的邊界線上，二維核密度不適用；本圖改用回溯全程逐小時的位置，反映粒子在流場範圍內實際停留的地方。底圖為陸地（灰）與海岸線（黑），中灰細線為 200 公尺（實線）、1000 公尺（虛線）等深線，200 公尺為陸棚邊緣（水深取自計算網格的節點水深；為避免干擾高密度區間，只畫這些水深，地圖上不標水深數字，水深與線型見圖例），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：貢寮，受體位置（星號）為研究點錨點 121.9224°E, 25.0964°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `lookback_snapshot.png`

回溯快照圖，2×2 四個面板依序為回溯第 72、168、336、720 小時仍在漂流的粒子所在位置的條件式來源足跡。密度先累加到與停留時數密度相同的 0.02° 網格，再以寬度 2 km 的高斯核做輕度平滑；平滑只在海域內進行（分子分母都乘上海域遮罩，陸地與流場範圍外的格不參與也不會得到密度），平滑後總質量重新正規化為 1。每個面板只計入該時刻仍在漂流的粒子，密度的分母是該時刻仍在漂流的粒子數，因此各面板的總和都是 1；已經終止的粒子（含回溯總時間不足該時刻者）不在面板內，各面板標題列出仍在漂流的比例（分母是本站全部粒子）。仍在漂流的粒子少於 200 顆時該面板不畫密度，並在面板中註明粒子數。色階為對數色階，四個面板共用同一組上下限與一條和面板同高的色條，單位為「% 仍在漂流粒子／km²」，上限為 10^1，下限為 10^-4，低於下限的位置不著色；線為該面板的高密度區間（HDR）50%、75%、90%，算法與停留時數密度相同。陸地畫在密度之上，密度不會出現在陸地上。底圖為陸地（灰）與海岸線（黑），中灰細線為 200 公尺（實線）、1000 公尺（虛線）等深線，200 公尺為陸棚邊緣（水深取自計算網格的節點水深；為避免干擾高密度區間，只畫這些水深，地圖上不標水深數字，水深與線型見圖例），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：貢寮，受體位置（星號）為研究點錨點 121.9224°E, 25.0964°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `boundary_departure_along_arc.png`

穿越位置沿開放邊界的一維分布圖。橫軸是沿流場範圍開放邊界折線自起點量起的弧長（km），起點與前進方向取自邊界折線的頂點順序，邊界全長 499 km；直條為每 10 km 弧長內穿越的粒子占全部穿越粒子的百分比，棕色曲線與右軸為累積百分比；淡色底色區分西、南、東、北各邊，虛線標出各邊之間的轉角。本站共 100,000 顆粒子穿越開放邊界。橫軸從 0 起算到邊界全長向上取整，左縱軸為 0 到 50%，右縱軸為 0 到 100%，三條軸的兩端都標值。穿越點全部落在一維的邊界線上，因此以一維分布呈現，取代舊版把這些點做二維核密度的圖。 站點：貢寮，受體位置（星號）為研究點錨點 121.9224°E, 25.0964°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

## 龜山島（guishan）

### `horizontal_overview.png`

代表粒子水平軌跡總覽，涵蓋整個 flow domain 範圍，標示 flow domain 開放邊界（終止邊界）與依 final_status_code 著色的終止標記；不畫 local domain 邊界或 local domain 首次離開標記（WP-25 裁決，該事件仍完整保存於 local_domain_departures.parquet 等資料表，只是不在這張圖呈現）。資料來源：trajectory_samples_by_site.parquet 的軌跡座標與時間、departures.parquet 的 final_status_code、boundary_arc_segments_geometry_by_site 的開放邊界折線幾何。 龜山島（guishan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `horizontal_local.png`

受體附近局部放大圖，座標為經緯度相對代表粒子回溯起點平均位置的偏移量，同樣只畫 flow domain 開放邊界與終止標記，不畫 local domain 邊界或首次離開標記（WP-25 裁決，理由同水平軌跡總覽圖）。資料來源與水平軌跡總覽圖相同：trajectory_samples_by_site.parquet、departures.parquet、boundary_arc_segments_geometry_by_site。 龜山島（guishan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `depth_age.png`

代表粒子深度對回溯時齡曲線，深度為海面以下之公尺數（海面為零、向上為正），顏色依代表粒子起始深度在同批抽樣中的排序序位分色，序位只用來分色，不對應固定 物理水層。資料來源：trajectory_samples_by_site.parquet 的 trajectory_z_m 與 trajectory_utc_ns。 龜山島（guishan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `terminal_counts.png`

本站實際出現之停止類別（含成功穿越開放邊界）的粒子計數與比例長條圖，計數為 0 的類別不畫，分母為 boundary_summary_by_site.total_particle_count。資料來源：departures.parquet 的 final_status_code 逐列統計，並與 boundary_summary_by_site 的 fraction_* 欄位 互相核對。 龜山島（guishan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `boundary_azimuth_rose.png`

16 方位扇區的相對來源權重極座標玫瑰圖（正北向上、順時針），標出前三大方位扇區名稱與百分比。相對來源權重＝該扇區成功穿越次數除以 全站成功穿越開放邊界的總次數。資料來源：departures.parquet 的 departure_azimuth_sector_index 與 departure_azimuth_sector_name。 龜山島（guishan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `boundary_arc_segments.png`

終止邊界（flow domain 開放邊界）依權重色階著色，共 20 段弧段，並以長條圖並列比較相對來源權重；相對來源權重＝該弧段成功穿越次數除以全站成功穿越開放邊界的總次數，只計在終止邊界的弧段上。boundary_arc_segments_geometry_by_site 另外登記 local domain 開放邊界的弧段幾何（共 20 段），純供地圖疊線的地理參照，權重恆為 0、不計入終止統計；local_equals_flow=True 的測站這條 local 邊界與上面的終止邊界是同一個幾何（見 boundary_summary_by_site 的 local_equals_flow 欄位），local_equals_flow=False 的測站則是各自站點獨立的 local domain 邊界。資料來源：boundary_arc_segments_geometry_by_site 的弧段幾何與 departures.parquet 的 boundary_segment_id、boundary_arc_segment_index。長條圖橫軸的「流場邊界 n」「局部邊界 n」，n 為資料表 arc_segment_index 加 1（圖面序號從 1 起算，資料表從 0 起算）。 龜山島（guishan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `source_footprint_kde.png`

條件式來源足跡核密度圖，背景為每平方公尺密度，疊加 50%／75%／90% 高密度區間（HDR，Highest Density Region）等值線，並疊海岸線與開放邊界作地理參照。資料來源：kde_by_site.parquet。 龜山島（guishan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `settling_velocity_breakdown.png`

十個沉降速度類別各自的穿越開放邊界比例長條圖，與各類別內方位扇區相對來源權重熱圖並列比較；縱軸每一列是一個類別，刻度為類別顯示名稱加上該類粒子實際用到的沉降速度範圍（最小值–最大值，取絕對值，單位公尺／秒），由上而下依類別代表沉降速度由慢到快排列。每類有 10 個候選沉降速度，每顆粒子抽用其中一個，各占該類粒子的十分之一；比例與權重以類別為單位彙整，不拆到個別速度。類別名稱為圖面顯示名稱：「不可回收塑膠」對應材質清冊的「其他/不可回收」，「可回收塑膠」對應「其他/可回收」，材質識別碼與清冊記錄不受影響。資料來源：departures.parquet 的 settling_velocity_mps（類別代表沉降速度，用於分組與排序）、particle_settling_velocity_mps（逐粒子實際速度，用於標示範圍）、material_id（類別識別）與 departure_azimuth_sector_index，依類別重新分組聚合。 龜山島（guishan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `source_footprint_residence.png`

停留時數密度圖，呈現回溯過程中這一站全部粒子逐小時所在位置的累計分布。停留時數的定義：每顆粒子在回溯的每個整點時刻，只要仍在漂流（尚未穿越開放邊界、擱淺、沉底或因其他原因終止），就在它所在的網格記 1 個粒子小時；回溯起點（到達時刻）的位置計入；粒子終止之後凍結的位置不再計入。各網格的粒子小時數除以本站粒子小時總數，再除以網格面積（km²），單位為「% 停留時數／km²」，全部網格對面積積分為 100%。網格為經緯度規則網格，解析度 0.02°（約 2.2 km），面積依緯度以 cos φ 計算。色階為對數色階，上限為最大值向上取整到 10 的整數次方（10^1），下限為上限的 1/10000（10^-3），每個 10 的次方標一個刻度，色條與地圖同高；低於下限的網格不著色（其數量與所占停留時數比例記錄在 manifest）。圖上的線為高密度區間（Highest Density Region, HDR）50%、75%、90%：依密度由高到低累加網格，累計到該比例的停留時數所需的最小面積範圍；本站三個區間的面積依序為 50% 為 754 km²、75% 為 2,215 km²、90% 為 4,543 km²。與舊版邊界核密度圖的差別：舊圖的輸入只有粒子穿越開放邊界的那一個點，這些點全部落在一維的邊界線上，二維核密度不適用；本圖改用回溯全程逐小時的位置，反映粒子在流場範圍內實際停留的地方。底圖為陸地（灰）與海岸線（黑），中灰細線為 200 公尺（實線）、1000 公尺（虛線）等深線，200 公尺為陸棚邊緣（水深取自計算網格的節點水深；為避免干擾高密度區間，只畫這些水深，地圖上不標水深數字，水深與線型見圖例），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：龜山島，受體位置（星號）為研究點錨點 121.9391°E, 24.8503°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `source_footprint_residence_site_scale.png`

停留時數密度圖，呈現回溯過程中這一站全部粒子逐小時所在位置的累計分布。停留時數的定義：每顆粒子在回溯的每個整點時刻，只要仍在漂流（尚未穿越開放邊界、擱淺、沉底或因其他原因終止），就在它所在的網格記 1 個粒子小時；回溯起點（到達時刻）的位置計入；粒子終止之後凍結的位置不再計入。各網格的粒子小時數除以本站粒子小時總數，再除以網格面積（km²），單位為「% 停留時數／km²」，全部網格對面積積分為 100%。網格為經緯度規則網格，解析度 0.02°（約 2.2 km），面積依緯度以 cos φ 計算。色階為對數色階，上限為最大值向上取整到 10 的整數次方（10^0），下限為上限的 1/10000（10^-4），每個 10 的次方標一個刻度，色條與地圖同高；低於下限的網格不著色（其數量與所占停留時數比例記錄在 manifest）；本圖採用本站自訂色階（本站最大值為 0.207 % 停留時數／km²），只適合看本站內部的分布形狀。圖上的線為高密度區間（Highest Density Region, HDR）50%、75%、90%：依密度由高到低累加網格，累計到該比例的停留時數所需的最小面積範圍；本站三個區間的面積依序為 50% 為 754 km²、75% 為 2,215 km²、90% 為 4,543 km²。與舊版邊界核密度圖的差別：舊圖的輸入只有粒子穿越開放邊界的那一個點，這些點全部落在一維的邊界線上，二維核密度不適用；本圖改用回溯全程逐小時的位置，反映粒子在流場範圍內實際停留的地方。底圖為陸地（灰）與海岸線（黑），中灰細線為 200 公尺（實線）、1000 公尺（虛線）等深線，200 公尺為陸棚邊緣（水深取自計算網格的節點水深；為避免干擾高密度區間，只畫這些水深，地圖上不標水深數字，水深與線型見圖例），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：龜山島，受體位置（星號）為研究點錨點 121.9391°E, 24.8503°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `lookback_snapshot.png`

回溯快照圖，2×2 四個面板依序為回溯第 72、168、336、720 小時仍在漂流的粒子所在位置的條件式來源足跡。密度先累加到與停留時數密度相同的 0.02° 網格，再以寬度 2 km 的高斯核做輕度平滑；平滑只在海域內進行（分子分母都乘上海域遮罩，陸地與流場範圍外的格不參與也不會得到密度），平滑後總質量重新正規化為 1。每個面板只計入該時刻仍在漂流的粒子，密度的分母是該時刻仍在漂流的粒子數，因此各面板的總和都是 1；已經終止的粒子（含回溯總時間不足該時刻者）不在面板內，各面板標題列出仍在漂流的比例（分母是本站全部粒子）。仍在漂流的粒子少於 200 顆時該面板不畫密度，並在面板中註明粒子數。色階為對數色階，四個面板共用同一組上下限與一條和面板同高的色條，單位為「% 仍在漂流粒子／km²」，上限為 10^1，下限為 10^-4，低於下限的位置不著色；線為該面板的高密度區間（HDR）50%、75%、90%，算法與停留時數密度相同。陸地畫在密度之上，密度不會出現在陸地上。底圖為陸地（灰）與海岸線（黑），中灰細線為 200 公尺（實線）、1000 公尺（虛線）等深線，200 公尺為陸棚邊緣（水深取自計算網格的節點水深；為避免干擾高密度區間，只畫這些水深，地圖上不標水深數字，水深與線型見圖例），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：龜山島，受體位置（星號）為研究點錨點 121.9391°E, 24.8503°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `boundary_departure_along_arc.png`

穿越位置沿開放邊界的一維分布圖。橫軸是沿流場範圍開放邊界折線自起點量起的弧長（km），起點與前進方向取自邊界折線的頂點順序，邊界全長 499 km；直條為每 10 km 弧長內穿越的粒子占全部穿越粒子的百分比，棕色曲線與右軸為累積百分比；淡色底色區分西、南、東、北各邊，虛線標出各邊之間的轉角。本站共 100,000 顆粒子穿越開放邊界。橫軸從 0 起算到邊界全長向上取整，左縱軸為 0 到 50%，右縱軸為 0 到 100%，三條軸的兩端都標值。穿越點全部落在一維的邊界線上，因此以一維分布呈現，取代舊版把這些點做二維核密度的圖。 站點：龜山島，受體位置（星號）為研究點錨點 121.9391°E, 24.8503°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

## 新竹外海（hsinchu）

### `horizontal_overview.png`

代表粒子水平軌跡總覽，涵蓋整個 flow domain 範圍，標示 flow domain 開放邊界（終止邊界）與依 final_status_code 著色的終止標記；不畫 local domain 邊界或 local domain 首次離開標記（WP-25 裁決，該事件仍完整保存於 local_domain_departures.parquet 等資料表，只是不在這張圖呈現）。資料來源：trajectory_samples_by_site.parquet 的軌跡座標與時間、departures.parquet 的 final_status_code、boundary_arc_segments_geometry_by_site 的開放邊界折線幾何。 新竹外海（hsinchu），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `horizontal_local.png`

受體附近局部放大圖，座標為經緯度相對代表粒子回溯起點平均位置的偏移量，同樣只畫 flow domain 開放邊界與終止標記，不畫 local domain 邊界或首次離開標記（WP-25 裁決，理由同水平軌跡總覽圖）。資料來源與水平軌跡總覽圖相同：trajectory_samples_by_site.parquet、departures.parquet、boundary_arc_segments_geometry_by_site。 新竹外海（hsinchu），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `depth_age.png`

代表粒子深度對回溯時齡曲線，深度為海面以下之公尺數（海面為零、向上為正），顏色依代表粒子起始深度在同批抽樣中的排序序位分色，序位只用來分色，不對應固定 物理水層。資料來源：trajectory_samples_by_site.parquet 的 trajectory_z_m 與 trajectory_utc_ns。 新竹外海（hsinchu），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `terminal_counts.png`

本站實際出現之停止類別（含成功穿越開放邊界）的粒子計數與比例長條圖，計數為 0 的類別不畫，分母為 boundary_summary_by_site.total_particle_count。資料來源：departures.parquet 的 final_status_code 逐列統計，並與 boundary_summary_by_site 的 fraction_* 欄位 互相核對。 新竹外海（hsinchu），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `boundary_azimuth_rose.png`

16 方位扇區的相對來源權重極座標玫瑰圖（正北向上、順時針），標出前三大方位扇區名稱與百分比。相對來源權重＝該扇區成功穿越次數除以 全站成功穿越開放邊界的總次數。資料來源：departures.parquet 的 departure_azimuth_sector_index 與 departure_azimuth_sector_name。 新竹外海（hsinchu），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `boundary_arc_segments.png`

終止邊界（flow domain 開放邊界）依權重色階著色，共 20 段弧段，並以長條圖並列比較相對來源權重；相對來源權重＝該弧段成功穿越次數除以全站成功穿越開放邊界的總次數，只計在終止邊界的弧段上。boundary_arc_segments_geometry_by_site 另外登記 local domain 開放邊界的弧段幾何（共 20 段），純供地圖疊線的地理參照，權重恆為 0、不計入終止統計；local_equals_flow=True 的測站這條 local 邊界與上面的終止邊界是同一個幾何（見 boundary_summary_by_site 的 local_equals_flow 欄位），local_equals_flow=False 的測站則是各自站點獨立的 local domain 邊界。資料來源：boundary_arc_segments_geometry_by_site 的弧段幾何與 departures.parquet 的 boundary_segment_id、boundary_arc_segment_index。長條圖橫軸的「流場邊界 n」「局部邊界 n」，n 為資料表 arc_segment_index 加 1（圖面序號從 1 起算，資料表從 0 起算）。 新竹外海（hsinchu），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `source_footprint_kde.png`

條件式來源足跡核密度圖，背景為每平方公尺密度，疊加 50%／75%／90% 高密度區間（HDR，Highest Density Region）等值線，並疊海岸線與開放邊界作地理參照。資料來源：kde_by_site.parquet。 新竹外海（hsinchu），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `settling_velocity_breakdown.png`

十個沉降速度類別各自的穿越開放邊界比例長條圖，與各類別內方位扇區相對來源權重熱圖並列比較；縱軸每一列是一個類別，刻度為類別顯示名稱加上該類粒子實際用到的沉降速度範圍（最小值–最大值，取絕對值，單位公尺／秒），由上而下依類別代表沉降速度由慢到快排列。每類有 10 個候選沉降速度，每顆粒子抽用其中一個，各占該類粒子的十分之一；比例與權重以類別為單位彙整，不拆到個別速度。類別名稱為圖面顯示名稱：「不可回收塑膠」對應材質清冊的「其他/不可回收」，「可回收塑膠」對應「其他/可回收」，材質識別碼與清冊記錄不受影響。資料來源：departures.parquet 的 settling_velocity_mps（類別代表沉降速度，用於分組與排序）、particle_settling_velocity_mps（逐粒子實際速度，用於標示範圍）、material_id（類別識別）與 departure_azimuth_sector_index，依類別重新分組聚合。 新竹外海（hsinchu），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `source_footprint_residence.png`

停留時數密度圖，呈現回溯過程中這一站全部粒子逐小時所在位置的累計分布。停留時數的定義：每顆粒子在回溯的每個整點時刻，只要仍在漂流（尚未穿越開放邊界、擱淺、沉底或因其他原因終止），就在它所在的網格記 1 個粒子小時；回溯起點（到達時刻）的位置計入；粒子終止之後凍結的位置不再計入。各網格的粒子小時數除以本站粒子小時總數，再除以網格面積（km²），單位為「% 停留時數／km²」，全部網格對面積積分為 100%。網格為經緯度規則網格，解析度 0.02°（約 2.2 km），面積依緯度以 cos φ 計算。色階為對數色階，上限為最大值向上取整到 10 的整數次方（10^1），下限為上限的 1/10000（10^-3），每個 10 的次方標一個刻度，色條與地圖同高；低於下限的網格不著色（其數量與所占停留時數比例記錄在 manifest）。圖上的線為高密度區間（Highest Density Region, HDR）50%、75%、90%：依密度由高到低累加網格，累計到該比例的停留時數所需的最小面積範圍；本站三個區間的面積依序為 50% 為 1,787 km²、75% 為 3,764 km²、90% 為 6,015 km²。與舊版邊界核密度圖的差別：舊圖的輸入只有粒子穿越開放邊界的那一個點，這些點全部落在一維的邊界線上，二維核密度不適用；本圖改用回溯全程逐小時的位置，反映粒子在流場範圍內實際停留的地方。底圖為陸地（灰）與海岸線（黑），本圖未繪等深線（本站繪圖範圍內沒有所要求水深的等深線），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：新竹外海，受體位置（星號）為研究點錨點 120.4500°E, 24.7500°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `source_footprint_residence_site_scale.png`

停留時數密度圖，呈現回溯過程中這一站全部粒子逐小時所在位置的累計分布。停留時數的定義：每顆粒子在回溯的每個整點時刻，只要仍在漂流（尚未穿越開放邊界、擱淺、沉底或因其他原因終止），就在它所在的網格記 1 個粒子小時；回溯起點（到達時刻）的位置計入；粒子終止之後凍結的位置不再計入。各網格的粒子小時數除以本站粒子小時總數，再除以網格面積（km²），單位為「% 停留時數／km²」，全部網格對面積積分為 100%。網格為經緯度規則網格，解析度 0.02°（約 2.2 km），面積依緯度以 cos φ 計算。色階為對數色階，上限為最大值向上取整到 10 的整數次方（10^0），下限為上限的 1/10000（10^-4），每個 10 的次方標一個刻度，色條與地圖同高；低於下限的網格不著色（其數量與所占停留時數比例記錄在 manifest）；本圖採用本站自訂色階（本站最大值為 0.139 % 停留時數／km²），只適合看本站內部的分布形狀。圖上的線為高密度區間（Highest Density Region, HDR）50%、75%、90%：依密度由高到低累加網格，累計到該比例的停留時數所需的最小面積範圍；本站三個區間的面積依序為 50% 為 1,787 km²、75% 為 3,764 km²、90% 為 6,015 km²。與舊版邊界核密度圖的差別：舊圖的輸入只有粒子穿越開放邊界的那一個點，這些點全部落在一維的邊界線上，二維核密度不適用；本圖改用回溯全程逐小時的位置，反映粒子在流場範圍內實際停留的地方。底圖為陸地（灰）與海岸線（黑），本圖未繪等深線（本站繪圖範圍內沒有所要求水深的等深線），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：新竹外海，受體位置（星號）為研究點錨點 120.4500°E, 24.7500°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `lookback_snapshot.png`

回溯快照圖，2×2 四個面板依序為回溯第 72、168、336、720 小時仍在漂流的粒子所在位置的條件式來源足跡。密度先累加到與停留時數密度相同的 0.02° 網格，再以寬度 2 km 的高斯核做輕度平滑；平滑只在海域內進行（分子分母都乘上海域遮罩，陸地與流場範圍外的格不參與也不會得到密度），平滑後總質量重新正規化為 1。每個面板只計入該時刻仍在漂流的粒子，密度的分母是該時刻仍在漂流的粒子數，因此各面板的總和都是 1；已經終止的粒子（含回溯總時間不足該時刻者）不在面板內，各面板標題列出仍在漂流的比例（分母是本站全部粒子）。仍在漂流的粒子少於 200 顆時該面板不畫密度，並在面板中註明粒子數。色階為對數色階，四個面板共用同一組上下限與一條和面板同高的色條，單位為「% 仍在漂流粒子／km²」，上限為 10^1，下限為 10^-4，低於下限的位置不著色；線為該面板的高密度區間（HDR）50%、75%、90%，算法與停留時數密度相同。陸地畫在密度之上，密度不會出現在陸地上。底圖為陸地（灰）與海岸線（黑），本圖未繪等深線（本站繪圖範圍內沒有所要求水深的等深線），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：新竹外海，受體位置（星號）為研究點錨點 120.4500°E, 24.7500°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `boundary_departure_along_arc.png`

穿越位置沿開放邊界的一維分布圖。橫軸是沿流場範圍開放邊界折線自起點量起的弧長（km），起點與前進方向取自邊界折線的頂點順序，邊界全長 499 km；直條為每 10 km 弧長內穿越的粒子占全部穿越粒子的百分比，棕色曲線與右軸為累積百分比；淡色底色區分西、南、東、北各邊，虛線標出各邊之間的轉角。本站共 100,000 顆粒子穿越開放邊界。橫軸從 0 起算到邊界全長向上取整，左縱軸為 0 到 50%，右縱軸為 0 到 100%，三條軸的兩端都標值。穿越點全部落在一維的邊界線上，因此以一維分布呈現，取代舊版把這些點做二維核密度的圖。 站點：新竹外海，受體位置（星號）為研究點錨點 120.4500°E, 24.7500°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

## 連江（lienchiang）

### `horizontal_overview.png`

代表粒子水平軌跡總覽，涵蓋整個 flow domain 範圍，標示 flow domain 開放邊界（終止邊界）與依 final_status_code 著色的終止標記；不畫 local domain 邊界或 local domain 首次離開標記（WP-25 裁決，該事件仍完整保存於 local_domain_departures.parquet 等資料表，只是不在這張圖呈現）。資料來源：trajectory_samples_by_site.parquet 的軌跡座標與時間、departures.parquet 的 final_status_code、boundary_arc_segments_geometry_by_site 的開放邊界折線幾何。 連江（lienchiang），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `horizontal_local.png`

受體附近局部放大圖，座標為經緯度相對代表粒子回溯起點平均位置的偏移量，同樣只畫 flow domain 開放邊界與終止標記，不畫 local domain 邊界或首次離開標記（WP-25 裁決，理由同水平軌跡總覽圖）。資料來源與水平軌跡總覽圖相同：trajectory_samples_by_site.parquet、departures.parquet、boundary_arc_segments_geometry_by_site。 連江（lienchiang），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `depth_age.png`

代表粒子深度對回溯時齡曲線，深度為海面以下之公尺數（海面為零、向上為正），顏色依代表粒子起始深度在同批抽樣中的排序序位分色，序位只用來分色，不對應固定 物理水層。資料來源：trajectory_samples_by_site.parquet 的 trajectory_z_m 與 trajectory_utc_ns。 連江（lienchiang），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `terminal_counts.png`

本站實際出現之停止類別（含成功穿越開放邊界）的粒子計數與比例長條圖，計數為 0 的類別不畫，分母為 boundary_summary_by_site.total_particle_count。資料來源：departures.parquet 的 final_status_code 逐列統計，並與 boundary_summary_by_site 的 fraction_* 欄位 互相核對。 連江（lienchiang），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `boundary_azimuth_rose.png`

16 方位扇區的相對來源權重極座標玫瑰圖（正北向上、順時針），標出前三大方位扇區名稱與百分比。相對來源權重＝該扇區成功穿越次數除以 全站成功穿越開放邊界的總次數。資料來源：departures.parquet 的 departure_azimuth_sector_index 與 departure_azimuth_sector_name。 連江（lienchiang），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `boundary_arc_segments.png`

終止邊界（flow domain 開放邊界）依權重色階著色，共 20 段弧段，並以長條圖並列比較相對來源權重；相對來源權重＝該弧段成功穿越次數除以全站成功穿越開放邊界的總次數，只計在終止邊界的弧段上。boundary_arc_segments_geometry_by_site 另外登記 local domain 開放邊界的弧段幾何（共 20 段），純供地圖疊線的地理參照，權重恆為 0、不計入終止統計；local_equals_flow=True 的測站這條 local 邊界與上面的終止邊界是同一個幾何（見 boundary_summary_by_site 的 local_equals_flow 欄位），local_equals_flow=False 的測站則是各自站點獨立的 local domain 邊界。資料來源：boundary_arc_segments_geometry_by_site 的弧段幾何與 departures.parquet 的 boundary_segment_id、boundary_arc_segment_index。長條圖橫軸的「流場邊界 n」「局部邊界 n」，n 為資料表 arc_segment_index 加 1（圖面序號從 1 起算，資料表從 0 起算）。 連江（lienchiang），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `source_footprint_kde.png`

條件式來源足跡核密度圖，背景為每平方公尺密度，疊加 50%／75%／90% 高密度區間（HDR，Highest Density Region）等值線，並疊海岸線與開放邊界作地理參照。資料來源：kde_by_site.parquet。 連江（lienchiang），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `settling_velocity_breakdown.png`

十個沉降速度類別各自的穿越開放邊界比例長條圖，與各類別內方位扇區相對來源權重熱圖並列比較；縱軸每一列是一個類別，刻度為類別顯示名稱加上該類粒子實際用到的沉降速度範圍（最小值–最大值，取絕對值，單位公尺／秒），由上而下依類別代表沉降速度由慢到快排列。每類有 10 個候選沉降速度，每顆粒子抽用其中一個，各占該類粒子的十分之一；比例與權重以類別為單位彙整，不拆到個別速度。類別名稱為圖面顯示名稱：「不可回收塑膠」對應材質清冊的「其他/不可回收」，「可回收塑膠」對應「其他/可回收」，材質識別碼與清冊記錄不受影響。資料來源：departures.parquet 的 settling_velocity_mps（類別代表沉降速度，用於分組與排序）、particle_settling_velocity_mps（逐粒子實際速度，用於標示範圍）、material_id（類別識別）與 departure_azimuth_sector_index，依類別重新分組聚合。 連江（lienchiang），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `source_footprint_residence.png`

停留時數密度圖，呈現回溯過程中這一站全部粒子逐小時所在位置的累計分布。停留時數的定義：每顆粒子在回溯的每個整點時刻，只要仍在漂流（尚未穿越開放邊界、擱淺、沉底或因其他原因終止），就在它所在的網格記 1 個粒子小時；回溯起點（到達時刻）的位置計入；粒子終止之後凍結的位置不再計入。各網格的粒子小時數除以本站粒子小時總數，再除以網格面積（km²），單位為「% 停留時數／km²」，全部網格對面積積分為 100%。網格為經緯度規則網格，解析度 0.02°（約 2.2 km），面積依緯度以 cos φ 計算。色階為對數色階，上限為最大值向上取整到 10 的整數次方（10^1），下限為上限的 1/10000（10^-3），每個 10 的次方標一個刻度，色條與地圖同高；低於下限的網格不著色（其數量與所占停留時數比例記錄在 manifest）。圖上的線為高密度區間（Highest Density Region, HDR）50%、75%、90%：依密度由高到低累加網格，累計到該比例的停留時數所需的最小面積範圍；本站三個區間的面積依序為 50% 為 142 km²、75% 為 1,078 km²、90% 為 2,563 km²。與舊版邊界核密度圖的差別：舊圖的輸入只有粒子穿越開放邊界的那一個點，這些點全部落在一維的邊界線上，二維核密度不適用；本圖改用回溯全程逐小時的位置，反映粒子在流場範圍內實際停留的地方。底圖為陸地（灰）與海岸線（黑），本圖未繪等深線（本站繪圖範圍內沒有所要求水深的等深線），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：連江，受體位置（星號）為研究點錨點 119.9500°E, 26.2000°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `source_footprint_residence_site_scale.png`

停留時數密度圖，呈現回溯過程中這一站全部粒子逐小時所在位置的累計分布。停留時數的定義：每顆粒子在回溯的每個整點時刻，只要仍在漂流（尚未穿越開放邊界、擱淺、沉底或因其他原因終止），就在它所在的網格記 1 個粒子小時；回溯起點（到達時刻）的位置計入；粒子終止之後凍結的位置不再計入。各網格的粒子小時數除以本站粒子小時總數，再除以網格面積（km²），單位為「% 停留時數／km²」，全部網格對面積積分為 100%。網格為經緯度規則網格，解析度 0.02°（約 2.2 km），面積依緯度以 cos φ 計算。色階為對數色階，上限為最大值向上取整到 10 的整數次方（10^1），下限為上限的 1/10000（10^-3），每個 10 的次方標一個刻度，色條與地圖同高；低於下限的網格不著色（其數量與所占停留時數比例記錄在 manifest）；本圖採用本站自訂色階（本站最大值為 2 % 停留時數／km²），只適合看本站內部的分布形狀。圖上的線為高密度區間（Highest Density Region, HDR）50%、75%、90%：依密度由高到低累加網格，累計到該比例的停留時數所需的最小面積範圍；本站三個區間的面積依序為 50% 為 142 km²、75% 為 1,078 km²、90% 為 2,563 km²。與舊版邊界核密度圖的差別：舊圖的輸入只有粒子穿越開放邊界的那一個點，這些點全部落在一維的邊界線上，二維核密度不適用；本圖改用回溯全程逐小時的位置，反映粒子在流場範圍內實際停留的地方。底圖為陸地（灰）與海岸線（黑），本圖未繪等深線（本站繪圖範圍內沒有所要求水深的等深線），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：連江，受體位置（星號）為研究點錨點 119.9500°E, 26.2000°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `lookback_snapshot.png`

回溯快照圖，2×2 四個面板依序為回溯第 72、168、336、720 小時仍在漂流的粒子所在位置的條件式來源足跡。密度先累加到與停留時數密度相同的 0.02° 網格，再以寬度 2 km 的高斯核做輕度平滑；平滑只在海域內進行（分子分母都乘上海域遮罩，陸地與流場範圍外的格不參與也不會得到密度），平滑後總質量重新正規化為 1。每個面板只計入該時刻仍在漂流的粒子，密度的分母是該時刻仍在漂流的粒子數，因此各面板的總和都是 1；已經終止的粒子（含回溯總時間不足該時刻者）不在面板內，各面板標題列出仍在漂流的比例（分母是本站全部粒子）。仍在漂流的粒子少於 200 顆時該面板不畫密度，並在面板中註明粒子數。色階為對數色階，四個面板共用同一組上下限與一條和面板同高的色條，單位為「% 仍在漂流粒子／km²」，上限為 10^1，下限為 10^-4，低於下限的位置不著色；線為該面板的高密度區間（HDR）50%、75%、90%，算法與停留時數密度相同。陸地畫在密度之上，密度不會出現在陸地上。底圖為陸地（灰）與海岸線（黑），本圖未繪等深線（本站繪圖範圍內沒有所要求水深的等深線），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：連江，受體位置（星號）為研究點錨點 119.9500°E, 26.2000°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `boundary_departure_along_arc.png`

穿越位置沿開放邊界的一維分布圖。橫軸是沿流場範圍開放邊界折線自起點量起的弧長（km），起點與前進方向取自邊界折線的頂點順序，邊界全長 499 km；直條為每 10 km 弧長內穿越的粒子占全部穿越粒子的百分比，棕色曲線與右軸為累積百分比；淡色底色區分西、南、東、北各邊，虛線標出各邊之間的轉角。本站共 95,857 顆粒子穿越開放邊界。橫軸從 0 起算到邊界全長向上取整，左縱軸為 0 到 50%，右縱軸為 0 到 100%，三條軸的兩端都標值。穿越點全部落在一維的邊界線上，因此以一維分布呈現，取代舊版把這些點做二維核密度的圖。 站點：連江，受體位置（星號）為研究點錨點 119.9500°E, 26.2000°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

## 南灣（nanwan）

### `horizontal_overview.png`

代表粒子水平軌跡總覽，涵蓋整個 flow domain 範圍，標示 flow domain 開放邊界（終止邊界）與依 final_status_code 著色的終止標記；不畫 local domain 邊界或 local domain 首次離開標記（WP-25 裁決，該事件仍完整保存於 local_domain_departures.parquet 等資料表，只是不在這張圖呈現）。資料來源：trajectory_samples_by_site.parquet 的軌跡座標與時間、departures.parquet 的 final_status_code、boundary_arc_segments_geometry_by_site 的開放邊界折線幾何。 南灣（nanwan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `horizontal_local.png`

受體附近局部放大圖，座標為經緯度相對代表粒子回溯起點平均位置的偏移量，同樣只畫 flow domain 開放邊界與終止標記，不畫 local domain 邊界或首次離開標記（WP-25 裁決，理由同水平軌跡總覽圖）。資料來源與水平軌跡總覽圖相同：trajectory_samples_by_site.parquet、departures.parquet、boundary_arc_segments_geometry_by_site。 南灣（nanwan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `depth_age.png`

代表粒子深度對回溯時齡曲線，深度為海面以下之公尺數（海面為零、向上為正），顏色依代表粒子起始深度在同批抽樣中的排序序位分色，序位只用來分色，不對應固定 物理水層。資料來源：trajectory_samples_by_site.parquet 的 trajectory_z_m 與 trajectory_utc_ns。 南灣（nanwan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `terminal_counts.png`

本站實際出現之停止類別（含成功穿越開放邊界）的粒子計數與比例長條圖，計數為 0 的類別不畫，分母為 boundary_summary_by_site.total_particle_count。資料來源：departures.parquet 的 final_status_code 逐列統計，並與 boundary_summary_by_site 的 fraction_* 欄位 互相核對。 南灣（nanwan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `boundary_azimuth_rose.png`

16 方位扇區的相對來源權重極座標玫瑰圖（正北向上、順時針），標出前三大方位扇區名稱與百分比。相對來源權重＝該扇區成功穿越次數除以 全站成功穿越開放邊界的總次數。資料來源：departures.parquet 的 departure_azimuth_sector_index 與 departure_azimuth_sector_name。 南灣（nanwan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `boundary_arc_segments.png`

終止邊界（flow domain 開放邊界）依權重色階著色，共 20 段弧段，並以長條圖並列比較相對來源權重；相對來源權重＝該弧段成功穿越次數除以全站成功穿越開放邊界的總次數，只計在終止邊界的弧段上。boundary_arc_segments_geometry_by_site 另外登記 local domain 開放邊界的弧段幾何（共 20 段），純供地圖疊線的地理參照，權重恆為 0、不計入終止統計；local_equals_flow=True 的測站這條 local 邊界與上面的終止邊界是同一個幾何（見 boundary_summary_by_site 的 local_equals_flow 欄位），local_equals_flow=False 的測站則是各自站點獨立的 local domain 邊界。資料來源：boundary_arc_segments_geometry_by_site 的弧段幾何與 departures.parquet 的 boundary_segment_id、boundary_arc_segment_index。長條圖橫軸的「流場邊界 n」「局部邊界 n」，n 為資料表 arc_segment_index 加 1（圖面序號從 1 起算，資料表從 0 起算）。 南灣（nanwan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `source_footprint_kde.png`

條件式來源足跡核密度圖，背景為每平方公尺密度，疊加 50%／75%／90% 高密度區間（HDR，Highest Density Region）等值線，並疊海岸線與開放邊界作地理參照。資料來源：kde_by_site.parquet。 南灣（nanwan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `settling_velocity_breakdown.png`

十個沉降速度類別各自的穿越開放邊界比例長條圖，與各類別內方位扇區相對來源權重熱圖並列比較；縱軸每一列是一個類別，刻度為類別顯示名稱加上該類粒子實際用到的沉降速度範圍（最小值–最大值，取絕對值，單位公尺／秒），由上而下依類別代表沉降速度由慢到快排列。每類有 10 個候選沉降速度，每顆粒子抽用其中一個，各占該類粒子的十分之一；比例與權重以類別為單位彙整，不拆到個別速度。類別名稱為圖面顯示名稱：「不可回收塑膠」對應材質清冊的「其他/不可回收」，「可回收塑膠」對應「其他/可回收」，材質識別碼與清冊記錄不受影響。資料來源：departures.parquet 的 settling_velocity_mps（類別代表沉降速度，用於分組與排序）、particle_settling_velocity_mps（逐粒子實際速度，用於標示範圍）、material_id（類別識別）與 departure_azimuth_sector_index，依類別重新分組聚合。 南灣（nanwan），100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為 flow domain 開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），粒子身分表標籤：design_baseline_v3_non_rising_a_v3_local20_observation_2025_gap_censored_20260916（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據），未啟用「海床不沉積」敏感度案例。

### `source_footprint_residence.png`

停留時數密度圖，呈現回溯過程中這一站全部粒子逐小時所在位置的累計分布。停留時數的定義：每顆粒子在回溯的每個整點時刻，只要仍在漂流（尚未穿越開放邊界、擱淺、沉底或因其他原因終止），就在它所在的網格記 1 個粒子小時；回溯起點（到達時刻）的位置計入；粒子終止之後凍結的位置不再計入。各網格的粒子小時數除以本站粒子小時總數，再除以網格面積（km²），單位為「% 停留時數／km²」，全部網格對面積積分為 100%。網格為經緯度規則網格，解析度 0.02°（約 2.2 km），面積依緯度以 cos φ 計算。色階為對數色階，上限為最大值向上取整到 10 的整數次方（10^1），下限為上限的 1/10000（10^-3），每個 10 的次方標一個刻度，色條與地圖同高；低於下限的網格不著色（其數量與所占停留時數比例記錄在 manifest）。圖上的線為高密度區間（Highest Density Region, HDR）50%、75%、90%：依密度由高到低累加網格，累計到該比例的停留時數所需的最小面積範圍；本站三個區間的面積依序為 50% 為 151 km²、75% 為 2,071 km²、90% 為 5,193 km²。與舊版邊界核密度圖的差別：舊圖的輸入只有粒子穿越開放邊界的那一個點，這些點全部落在一維的邊界線上，二維核密度不適用；本圖改用回溯全程逐小時的位置，反映粒子在流場範圍內實際停留的地方。底圖為陸地（灰）與海岸線（黑），中灰細線為 200 公尺（實線）、1000 公尺（虛線）等深線，200 公尺為陸棚邊緣（水深取自計算網格的節點水深；為避免干擾高密度區間，只畫這些水深，地圖上不標水深數字，水深與線型見圖例），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：南灣，受體位置（星號）為研究點錨點 120.7707°E, 21.9204°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `source_footprint_residence_site_scale.png`

停留時數密度圖，呈現回溯過程中這一站全部粒子逐小時所在位置的累計分布。停留時數的定義：每顆粒子在回溯的每個整點時刻，只要仍在漂流（尚未穿越開放邊界、擱淺、沉底或因其他原因終止），就在它所在的網格記 1 個粒子小時；回溯起點（到達時刻）的位置計入；粒子終止之後凍結的位置不再計入。各網格的粒子小時數除以本站粒子小時總數，再除以網格面積（km²），單位為「% 停留時數／km²」，全部網格對面積積分為 100%。網格為經緯度規則網格，解析度 0.02°（約 2.2 km），面積依緯度以 cos φ 計算。色階為對數色階，上限為最大值向上取整到 10 的整數次方（10^1），下限為上限的 1/10000（10^-3），每個 10 的次方標一個刻度，色條與地圖同高；低於下限的網格不著色（其數量與所占停留時數比例記錄在 manifest）；本圖採用本站自訂色階（本站最大值為 5.21 % 停留時數／km²），只適合看本站內部的分布形狀。圖上的線為高密度區間（Highest Density Region, HDR）50%、75%、90%：依密度由高到低累加網格，累計到該比例的停留時數所需的最小面積範圍；本站三個區間的面積依序為 50% 為 151 km²、75% 為 2,071 km²、90% 為 5,193 km²。與舊版邊界核密度圖的差別：舊圖的輸入只有粒子穿越開放邊界的那一個點，這些點全部落在一維的邊界線上，二維核密度不適用；本圖改用回溯全程逐小時的位置，反映粒子在流場範圍內實際停留的地方。底圖為陸地（灰）與海岸線（黑），中灰細線為 200 公尺（實線）、1000 公尺（虛線）等深線，200 公尺為陸棚邊緣（水深取自計算網格的節點水深；為避免干擾高密度區間，只畫這些水深，地圖上不標水深數字，水深與線型見圖例），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：南灣，受體位置（星號）為研究點錨點 120.7707°E, 21.9204°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `lookback_snapshot.png`

回溯快照圖，2×2 四個面板依序為回溯第 72、168、336、720 小時仍在漂流的粒子所在位置的條件式來源足跡。密度先累加到與停留時數密度相同的 0.02° 網格，再以寬度 2 km 的高斯核做輕度平滑；平滑只在海域內進行（分子分母都乘上海域遮罩，陸地與流場範圍外的格不參與也不會得到密度），平滑後總質量重新正規化為 1。每個面板只計入該時刻仍在漂流的粒子，密度的分母是該時刻仍在漂流的粒子數，因此各面板的總和都是 1；已經終止的粒子（含回溯總時間不足該時刻者）不在面板內，各面板標題列出仍在漂流的比例（分母是本站全部粒子）。仍在漂流的粒子少於 200 顆時該面板不畫密度，並在面板中註明粒子數。色階為對數色階，四個面板共用同一組上下限與一條和面板同高的色條，單位為「% 仍在漂流粒子／km²」，上限為 10^1，下限為 10^-4，低於下限的位置不著色；線為該面板的高密度區間（HDR）50%、75%、90%，算法與停留時數密度相同。陸地畫在密度之上，密度不會出現在陸地上。底圖為陸地（灰）與海岸線（黑），中灰細線為 200 公尺（實線）、1000 公尺（虛線）等深線，200 公尺為陸棚邊緣（水深取自計算網格的節點水深；為避免干擾高密度區間，只畫這些水深，地圖上不標水深數字，水深與線型見圖例），星號為受體位置，深藍色虛線圓為抽樣範圍（以研究地點錨點為圓心、半徑 10 km 的測地線圓，即受體抽樣範圍），洋紅色虛線為流場範圍開放邊界（終止邊界）。 站點：南灣，受體位置（星號）為研究點錨點 120.7707°E, 21.9204°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。

### `boundary_departure_along_arc.png`

穿越位置沿開放邊界的一維分布圖。橫軸是沿流場範圍開放邊界折線自起點量起的弧長（km），起點與前進方向取自邊界折線的頂點順序，邊界全長 499 km；直條為每 10 km 弧長內穿越的粒子占全部穿越粒子的百分比，棕色曲線與右軸為累積百分比；淡色底色區分西、南、東、北各邊，虛線標出各邊之間的轉角。本站共 99,999 顆粒子穿越開放邊界。橫軸從 0 起算到邊界全長向上取整，左縱軸為 0 到 50%，右縱軸為 0 到 100%，三條軸的兩端都標值。穿越點全部落在一維的邊界線上，因此以一維分布呈現，取代舊版把這些點做二維核密度的圖。 站點：南灣，受體位置（星號）為研究點錨點 120.7707°E, 21.9204°N（取自正式釋放設定），粒子自錨點周圍半徑 10 km 的抽樣範圍內（方位×距離分層抽樣）的各受體點釋放，100,000 顆粒子，M=10（每情境 10 個獨立隨機系集成員，含隨機擴散），終止邊界為流場範圍開放邊界，物理案例：finite_depth_stokes（含 Stokes 漂移，擴散方法：smagorinsky_workbook_eq10），未啟用「海床不沉積」敏感度案例。
