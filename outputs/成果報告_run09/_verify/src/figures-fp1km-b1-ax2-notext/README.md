# WP-21：native 正式成果圖表（2026-10-22 08:32 UTC）

本目錄由 `native-figures` 命令列由既有 native 後處理輸出（`departures.parquet`
等，見下方「輸入來源」）獨立繪製，**不重跑任何積分、不讀取 forcing、不改寫任何
輸入檔**；輸入檔前後 SHA-256 已核對一致（見 `manifest.json`）。文字層級為
`--figure-text none`（見下方「每站八張圖」一節）。

**WP-25：輸出格式與色階**——每張圖同時輸出 `.png`（600 dpi）與同名
`.svg`（向量），對齊相鄰專案 `OCM-SVD-Analysis` 的正式成圖規範（見
`docs/results/17_native_figures_formal_output_rules.md`）；`source_footprint_
kde`、`boundary_arc_segments`、`boundary_azimuth_rose`、`settling_velocity_
breakdown` 四張圖之色階上下限與刻度固定，
超過上限的值以最高色顯示、不裁切資料，實際狀態記錄於 `manifest.json`
的 `color_scales` 區塊（`over_limit_count`／`over_limit_fraction`／
`display_saturation`）。

## 科學定義與範圍

- **系集成員數（M）**：M=10；每個系集成員皆含隨機擴散，
  差異只來自隨機數種子。M 是後處理實測值（`boundary_summary_by_site.
  ensemble_member_count`，相異 `member_id` 數量），不是本模組寫死的常數
  （WP-37）——同一套管線可能產出 `M=1` 的確定性試驗，也可能產出 `M>1` 的
  隨機系集。
- 「條件式來源足跡」與「相對來源權重」的定義：給定站點、受體、沉降速度與到達
  時刻等情境設定後，代表粒子穿越各方位扇區／各邊界弧段的相對頻率，是描述這批
  M 個系集成員在既定設定下分布的統計量。來源機率與因果歸因是另一層科學推論，
  需要另外建立先驗、概似與觀測驗證，本批圖表不提供這一層推論。
- 圖上的終止標記位置是回溯積分停止處的座標，對應 `final_status_code` 互斥
  狀態之一（圖例只列本站實際出現者）；停止處是積分終止的座標，不是另外
  查證過的實際污染來源。
- 海岸與開放邊界幾何目前的版本狀態是工程生成版：由既有海岸資料與弧段幾何直接
  取得，尚未經研究團隊另立正式核准發布流程。

## 站點

- `gongliao`（貢寮）
- `guishan`（龜山島）
- `hsinchu`（新竹外海）
- `lienchiang`（連江）
- `nanwan`（南灣）

## WP-23／WP-25：flow domain 終止邊界與 local domain 首次離開（資料仍在，圖上已不畫）

研究團隊裁決（`AGENTS.md`、`docs/development/05_decisions_and_risks.md`
D008）：**終止一律以 flow domain 開放邊界為準；離開站點 local domain 是非終止
事件，只記錄、粒子繼續回溯**。`departures.parquet` 的 `crossed_interest_region_
boundary`／`final_status_code` 因此一律是 flow domain 語意；`local_domain_departures.parquet`、`local_domain_first_exit_summary_by_
site` 與 `departures.parquet` 的 `also_local_domain_first_exit` 欄位完整保存
每顆粒子「首次離開自己站點 local domain」的位置與統計，不受下一段圖面裁決
影響。

**WP-25 裁決**：`horizontal_overview.png`／`horizontal_local.png` 不再畫 local
domain 邊界或 local domain 首次離開標記，圖上只保留 **flow domain 開放邊界**（藍色虛線，真正的終止邊界，
`terminal_counts.png` 等圖的「穿越開放邊界（正常結束）」指的就是這裡）與依
`final_status_code` 著色的終止標記。若需要 local domain 首次離開的空間分布，
請直接查詢 `local_domain_departures.parquet`（僅 `local_equals_flow=False`
的站有列）或 `local_domain_first_exit_summary_by_site` 的
`fraction_local_first_exit`（各站皆有一列彙總統計）。

## 每站八張圖

各站資料夾（`<site_id>/`）內固定輸出以下八張 PNG。`--figure-text full`（預設）
在圖面上同時畫出標題、副標、圖例與座標軸標籤；`--figure-text none` 只畫座標軸
與圖例，不畫標題與副標，供正式報告排版——每張圖對應的文字改由 `captions.md`
承載（逐圖一段，內容與下列條列一致，並與 `manifest.json` 的
`output_png_sha256.<site_id>/<stem>.png.caption` 完全相同）：

1. `horizontal_overview.png`：代表粒子水平軌跡總覽（經緯度顯示，涵蓋整個 flow
   domain 範圍），**只畫 flow domain 開放邊界**（見上一節；WP-25 起不再畫
   local domain 邊界或首次離開標記）。資料來源：
   `trajectory_samples_by_site`（軌跡座標、狀態）、`departures.parquet`
   （`final_status_code`）、`boundary_arc_segments_geometry_by_site`
   （開放邊界折線）。綠點＝回溯起點（受體到達位置）；終點依
   `final_status_code` 用不同顏色與標記形狀表示；線色依代表粒子起始深度序位
   （淺→深）。
2. `horizontal_local.png`：受體附近局部放大，座標為經緯度相對代表粒子回溯起點
   平均位置的偏移量，**同樣只畫 flow domain 開放邊界與終止標記**（WP-25 起不
   再疊 local domain 邊界或首次離開標記）。資料來源同上。
3. `depth_age.png`：深度（`z_m`，海面為 0、向上為正）對回溯時齡（小時）曲線，
   顏色依代表粒子的垂向抽樣序位分色——**序位不代表固定物理水層**，native 後
   處理沒有保留原始的 `vertical_id`。資料來源：`trajectory_samples_by_site`。
4. `terminal_counts.png`：本站實際出現之停止類別（含成功穿越開放邊界，計數為 0 者不畫），計數
   與佔全站粒子比例；分母＝`boundary_summary_by_site.total_particle_count`，與
   該表的 `fraction_*` 欄位互相印證。資料來源：`departures.parquet`（逐列統計）、
   `boundary_summary_by_site`（分母核對）。
5. `boundary_azimuth_rose.png`：16 方位扇區「相對來源權重」極座標玫瑰圖（正北
   向上、順時針），並標出前三大扇區名稱與百分比。**相對來源權重**＝該扇區穿越
   次數 ÷ 全站成功穿越邊界的總次數，聚合公式與 `postprocess._sector_entry_table`
   完全相同，只是直接由 `departures.parquet` 的
   `departure_azimuth_sector_index`／`_name` 逐列重新聚合，不依賴另外的聚合表。
6. `boundary_arc_segments.png`：開放邊界折線依弧段著色（權重色階）＋弧段權重長
   條圖，弧段編號與座標取自 `boundary_arc_segments_geometry_by_site`；權重聚合
   公式與 `postprocess._arc_segment_entry_table` 相同，一樣直接由
   `departures.parquet` 的 `boundary_segment_id`／`boundary_arc_segment_index`
   逐列重新聚合。
7. `source_footprint_kde.png`：由 `kde_by_site` 畫出**條件式來源足跡**——背景是
   核密度（每平方公尺），疊加 50%／75%／90% HDR（高密度區間，Highest Density
   Region）等值線，再疊海岸線與開放邊界作地理參照。**這是條件式來源足跡，不是
   絕對來源機率**。
8. `settling_velocity_breakdown.png`：沉降速度類別（十類，每類 10 個候選沉降速度、每顆粒子
   抽用其中一個，各占該類粒子的十分之一）各自的穿越開放邊界比例（長條圖）與各類別內的方位扇區
   相對來源權重（熱圖）。縱軸每一列是一個類別，刻度為類別顯示名稱（「不可回收塑膠」「可回收
   塑膠」等圖面顯示名稱，材質識別碼與材質清冊不變）加上該類粒子實際用到的沉降速度範圍
   （最小值–最大值，絕對值，公尺／秒），由上而下依類別代表沉降速度由慢到快排列；比例與
   權重以類別為單位彙整，不拆到個別速度。資料來源：`departures.parquet` 的
   `settling_velocity_mps`（類別代表速度，分組與排序）、`particle_settling_velocity_mps`
   （逐粒子實際速度，標示範圍）與 `material_id`（類別識別）。

## WP-38：停留時數密度、回溯快照與邊界弧長分布

後處理目錄含足跡產品（`native-postprocess --with-footprint`）時，每站另外輸出四張圖：

1. `source_footprint_residence.png`：停留時數密度（對數色階）。停留時數＝回溯過程中逐小時、仍在漂流（尚未終止）的粒子所在網格的粒子小時數，終止後凍結的位置不計入；色階單位 % 停留時數／km²，上限為最大值向上取整到 10 的整數次方、下限為上限的 1/10000，每個 10 的次方標一個刻度，低於下限不著色；疊高密度區間（HDR）50%／75%／90%。
2. `source_footprint_residence_site_scale.png`：同一份資料，色階上限改用各站自己的最大值，只適合看單站內部形狀。
3. `lookback_snapshot.png`：回溯 72／168／336／720 小時仍在漂流的粒子的條件式來源足跡（2×2 面板，與停留時數密度同網格的直方圖，可選海域遮罩下的輕度平滑），標題列出仍在漂流的比例；仍在漂流粒子數低於門檻的面板不畫密度。
4. `boundary_departure_along_arc.png`：穿越位置沿開放邊界弧長的一維分布（直條＋累積曲線），標出各邊與轉角。

舊的 `source_footprint_kde.png` 照常輸出，但不再是報告主圖：它的輸入只有粒子穿越開放邊界的那一點，這些點全部落在一維的邊界線上，二維核密度不適用。

等深線取自 `--bathymetry` 指定的 GeoJSON（由 `scripts/export_isobaths.py` 依 SCHISM 計算網格的節點水深匯出）。底圖為陸地（灰）、海岸線（黑 0.5 pt）、中灰細等深線（預設只畫 200 m 實線與 1000 m 虛線，地圖上不標水深數字）、深藍色虛線圓（抽樣範圍）、星號（研究地點錨點，位置取自正式釋放設定，圖面不寫站名）與洋紅色虛線（流場範圍開放邊界）。色條與地圖軸同高，座標軸範圍向外擴到刻度倍數、首末刻度落在框邊。`manifest.json` 的 `footprint_figures` 區塊記錄各圖的色階上下限、低於下限的網格統計、色條與地圖軸的上下緣量測值、座標軸範圍與刻度、等深線狀態，以及每站受體位置與來源。

## 已知限制與方法揭露

- **系集成員數（M）**：M=10（見上方「科學定義與範圍」）；
  本批圖表不論 M 為何，一律把同一情境下的全部系集成員直接併入同一批統計，
  不呈現、也不能呈現系集內部的離散度或不確定性區間。
- 方位扇區與弧段的「相對來源權重」分母是**同一分組內成功穿越開放邊界的粒子
  數**，不是該分組全部粒子數；未穿越邊界的粒子不參與方位／弧段統計，但仍計入
  `terminal_counts.png` 與 `boundary_summary_by_site` 的整體分母。
- `boundary_segment_id`（離開點最終判定所屬的開放邊界）採用「離開點到候選開放
  邊界線最短距離」的啟發式（見 `postprocess.classify_boundary_segment`
  docstring），是近似判定，不是精確幾何交點。
- 海岸與開放邊界幾何目前的版本狀態見上方「科學定義與範圍」；`horizontal_overview`
  等圖以「開放邊界折線（由弧段起訖點重建）」近似完整 domain 外框，本管線不讀
  domain／open manifest，因此不畫閉合陸域邊界，只畫開放水域邊界。
- 深度—時齡曲線的「序位」只是同一批代表粒子按起始深度排序後的名次，同一序位
  在不同站、不同受體之間**不對應同一水層**。
- `boundary_azimuth_rose.png`／`boundary_arc_segments.png` 目前只呈現 flow
  domain 語意的相對來源權重；local domain 首次離開的方位／弧段分布**尚未**
  另外出圖，只能從 `local_domain_departures.parquet`（或
  `local_domain_first_exit_summary_by_site` 的 `fraction_local_first_exit`）
  自行查閱。

## 輸入來源

見 `manifest.json` 的 `inputs_sha256_before`／`inputs_sha256_after`（相同代表本
批繪圖沒有改動任何輸入檔）與 `output_png_sha256`（每張 PNG 的 SHA-256）；八個
必要輸入表見本模組 docstring（`departures.parquet`、
`trajectory_samples_by_site.parquet`、`kde_by_site.parquet`、
`boundary_summary_by_site`、`boundary_arc_segments_geometry_by_site`、
`local_domain_departures.parquet`、`local_domain_first_exit_summary_by_site`、
`manifest.json`）。

## 重建指令

```
lbt native-figures --postprocess-dir /home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/postprocess-fp1km --coastline /home/mustlab/Workspace/lbt-inputs/coastline-taiwan.geojson --bathymetry /home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/isobaths.geojson --config /home/mustlab/Workspace/lbt-inputs/formal/v8-run09/20261019_final-sites_h180_common_m10_v8/release-configs/release-180d-full_horizon_from_deposition.yaml --output-dir /home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/figures-fp1km-b1-ax2-notext --figure-text none --footprint-colorbar-range 1e-3 1e3 --footprint-colormap-scheme nonuniform --axes-unified
```
