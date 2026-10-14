# `native-figures` 正式成圖規則（版本化）

> 文件狀態：**v1.1.0，2026-09-29 核定，現行版本**（v1.0.0 為 2026-09-24 首次核定；v1.1.0 新增第 6 節，
> 涵蓋 WP-38 三種新圖）。本文件第 1～6 節對應
> `src/lagrangian_backtracking/native_runner/figures.py` 的
> `build_native_figures` 產出的八張站點圖（`FIGURE_STEMS`），是 WP-25 交付的
> 一部分。機器可讀契約見 `docs/data/native_figures_formal_output_policy_v1.json`，
> 兩者欄位以 `tests/test_native_figures.py`
> `test_json_policy_matches_figures_py_runtime_constants` 逐項自動比對，
> 不得互相脫節。

## 1. 緣起與適用範圍

本專案是相鄰專案 `OCM-SVD-Analysis` 的延續。該專案已有一套經過多輪審查、
版本化的正式成圖規範
（`water_column_svd_formal_spatial_figure_rules.md`，版本 1.8.5）。研究團隊
要求 LBT 的成果圖對齊該規範的**適用部分**，而非另立一套互不相干的標準。

`OCM-SVD-Analysis` 的圖是**水體 SVD 空間模態圖**（流速模態、eta 模態、時間
係數、模態間比較），LBT 的圖是**軌跡與來源足跡圖**（水平總覽、深度—時齡、
終止狀態、邊界方位與弧段、條件式來源足跡 KDE、沉降速度分解）。兩者資料性質
不同，因此本文件逐條檢視對方規則後，只採納**與資料呈現政策相關、與 SVD
特定物理量無關**的部分（見第 3 節），並對不適用的條目逐一寫明理由（見
第 4 節），不強行套用。

本文件涵蓋的八張圖固定檔名如下（`FIGURE_STEMS`，與模組原始碼、JSON 契約、
manifest 三處完全一致）：

1. `horizontal_overview`——水平總覽
2. `horizontal_local`——水平局部
3. `depth_age`——深度—時齡曲線
4. `terminal_counts`——終止狀態計數
5. `boundary_azimuth_rose`——邊界方位玫瑰圖
6. `boundary_arc_segments`——邊界弧段權重圖
7. `source_footprint_kde`——條件式來源足跡 KDE
8. `settling_velocity_breakdown`——沉降速度分解

五個站點固定為 `gongliao`（貢寮）、`guishan`（龜山島西側）、`hsinchu`
（新竹外海）、`nanwan`（南灣）、`lienchiang`（連江），中文標籤見
`SITE_LABELS_ZH`。

## 2. 任務 0：水平軌跡圖移除 local domain 圖層（優先事項，已完成）

研究團隊指示：`horizontal_overview` 與 `horizontal_local` 兩張圖過去會畫出
（a）local domain 的橘色點線邊界，與（b）local 首次離開的橘色空心菱形標記，
使得 A 區（貢寮／龜山島，`local_equals_flow=False`）與其他三站（新竹／南灣／
連江，`local_equals_flow=True`）圖面不一致。此舉已完成，決定與範圍如下：

- **圖面層移除**：兩張圖不再畫 local domain 邊界線與 local 首次離開標記；
  圖例不再列出這兩個項目；副標不再有解釋 local／flow 差異的那一行。移除後
  五站在這兩張圖上呈現**完全一致的版面結構**：只有 flow domain 開放邊界
  （藍色虛線）與依終止狀態著色的終止標記。
- **資料層不變**：`local_domain_departures.parquet`、
  `local_domain_first_exit_summary_by_site.parquet` 與
  `departures.parquet` 的 `also_local_domain_first_exit` 欄位，三者的產出
  邏輯完全未被本次修改觸碰（`postprocess.py` 未變更），資料照常產出、照常
  可稽核，只是不再畫在這兩張圖上。
- **不受影響的圖**：`boundary_arc_segments`（邊界弧段權重圖）與
  `source_footprint_kde`（含站點邊界疊圖時）仍然依需要畫出 local domain
  邊界，因為那兩張圖的設計目的正是呈現 local／flow 兩種邊界幾何的關係，
  移除範圍僅限研究團隊明確指出的 `horizontal_overview`／`horizontal_local`
  兩張圖。
- `captions.md` 的圖說已同步更新，不再提到圖面上的 local domain 圖層，但
  說明文字仍會提及對應資料表仍完整保留（避免讀者誤以為資料也被移除）。

驗證方式：`tests/test_native_figures.py` 內
`test_horizontal_plots_no_longer_draw_local_domain_boundary_or_marker`、
`test_local_first_exit_marker_style_and_caption_helper_removed`、
`test_horizontal_overview_caption_no_longer_mentions_local_domain_layer`
三個測試分別從繪圖輸出（SVG 文字不含 local 邊界色碼）、原始碼結構（對應
的樣式常數與輔助函式已整個刪除，非僅棄用）、圖說文字三個角度確認移除
落實且沒有殘留。另以合成資料在 `local_equals_flow=False` 與 `=True` 兩種
站點各跑一次 `native-figures`，用 Read 工具目視比對 PNG，確認兩種站點的
`horizontal_overview.png`／`horizontal_local.png` 版面結構一致（見
`reports/WP-25.md` 目視檢查章節）。

## 3. 採納自 OCM-SVD-Analysis 的規則

以下規則judged適用於 LBT，已實作進 `figures.py`：

### 3.1 雙格式正式輸出：PNG 600 dpi ＋ SVG

`OCM-SVD-Analysis` 的正式交付規定 `output_formats: ["png", "svg"]`、
`png_dpi: 600`（見其 policy JSON `formal_delivery` 區塊）。LBT 採納此規則：

- `_save_exclusive` 對每張圖同時輸出 `{stem}.png` 與 `{stem}.svg`，兩者以
  排他（`"xb"`）檔案模式寫入，避免覆蓋既有輸出。
- CLI `--dpi` 旗標預設值為 `600`（見 `cli.py` 第 1996 行），對齊
  OCM-SVD-Analysis 的正式 dpi。
- manifest 的 `output_formats` 欄位固定為 `["png", "svg"]`，
  `output_png_sha256`／`output_svg_sha256` 分別記錄兩種格式每個檔案的
  SHA-256 與位元組數，供事後稽核兩種格式確實都產出且未被竄改。

### 3.2 固定色階與超限處理（不得視覺裁切卻不留紀錄）

`OCM-SVD-Analysis` 對流速模態圖採用固定 `vmin=0`／`vmax=0.024` 的色階，
超過上限的值飽和顯示為最高色（`display_saturation`），但底層數值不被
裁切，且在 metadata 記錄 `over_limit_count`／`over_limit_fraction`。LBT
採納這個「固定色階＋超限記錄，不視覺裁切」的核心政策，實作為
`ColorScale` 資料類別：

- `vmin` 固定為 `0.0`（KDE 密度與相對來源權重皆為非負量）。
- `vmax` 由 `_compute_shared_color_scales` 對本次執行、本批被選取站點的
  實際最大值決定（與 OCM-SVD-Analysis 用「經審查的歷史常數」不同，見
  第 5 節誠實揭露的差異）。
- `tick_count=7`，於 `[vmin, vmax]` 間等距分布（含頭尾），與
  OCM-SVD-Analysis 流速模態圖的 7 個刻度數量一致。
- `ColorScale.over_limit()` 計算 `(over_limit_count, over_limit_fraction,
  display_saturation)`，分母僅計入有限值筆數；`matplotlib` 的
  `Normalize(vmin, vmax)` 語意本身就會把超過 `vmax` 的值飽和顯示為色階
  最頂端顏色，不需要額外裁切程式碼，因此本類別不做任何資料修改，只做
  統計與提供上下限。
- `colorbar_extend` 固定為 `"neither"`（矩形色條，不畫三角形延伸箭頭），
  與 OCM-SVD-Analysis 對流速模態圖的色條樣式一致。
- 每站每張適用圖的 `sample_count`／`over_limit_count`／
  `over_limit_fraction`／`display_saturation` 皆寫入 manifest 的
  `color_scales.{stem}.per_site.{site_id}` 區塊，供人工稽核。

### 3.3 跨站色階刻度一致化

`OCM-SVD-Analysis` 對四海域使用共同的流速色階，讓不同海域可以直接比較
顏色深淺。LBT 採納相同精神，對以下四張圖建立跨站共用色階
（`_SHARED_COLOR_SCALE_STEMS`）：

- `source_footprint_kde`（條件式來源足跡密度）
- `boundary_arc_segments`（邊界弧段相對來源權重）
- `boundary_azimuth_rose`（方位扇區相對來源權重）
- `settling_velocity_breakdown`（依沉降速度分組的方位扇區相對來源權重）

這四張圖的色階數值本質上都是「相對來源權重」或「密度」，量綱在站點之間
可比；`_compute_shared_color_scales` 對本次選取的站點做一次 preflight，
取跨站最大值作為共同 `vmax`，讓每一站的色條範圍、刻度完全相同，讀者可以
直接比較不同站點的顏色深淺，不需要另外核對色條數字。

**未採納跨站共用色階的四張圖與理由**：`horizontal_overview`／
`horizontal_local`／`depth_age` 三張圖的顏色代表「代表粒子起始深度序位」
（`rank_index`／`rank_total`），不同站的代表粒子抽樣數本來就可能不同，
序位不是可跨站比較的物理量或比例，共用色階反而會誤導讀者以為序位數值
可以站間對照；`terminal_counts` 用固定的狀態別類別色
（`_STATUS_COLORS`），與資料範圍無關，天生就跨站一致，不需要額外機制。

### 3.4 既有行為維持不變

`--figure-text {full,none}` 與 `captions.md` 機制（WP-24b 建立）未被本次
變更觸碰：`full` 模式下標題／副標／圖例／座標軸標籤照常畫出；`none` 模式
下只畫座標軸與圖例，標題／副標改由 `captions.md` 承載，供正式報告排版。
兩種模式下的圖說文字內容完全相同，只差在是否畫進圖檔本身。

## 4. 判定不適用、明確不採納的規則（附理由）

以下 `OCM-SVD-Analysis` 規則經檢視後，判定**不適用於 LBT**，理由逐條列出：

| OCM-SVD-Analysis 規則 | 不適用理由 |
|---|---|
| 固定 `vmax=0.024`（流速模態專屬常數，經四海域 science run 校準） | 該常數是對「流速模態振幅」這個特定物理量、特定資料集校準出的數字，與 LBT 的「相對來源權重」／「條件式來源足跡密度」量綱、數值範圍完全不同，直接套用該常數沒有物理意義。LBT 改為「本次執行、本批站點的實際最大值」（見第 5 節），架構上保留未來若研究團隊核定跨 run 固定常數時可直接替換的彈性。 |
| eta 模態對稱發散色階（以 0 為中心，正負對稱） | LBT 的四張共用色階圖（KDE 密度、相對來源權重）皆為非負量，沒有正負對稱的物理意義，`vmin` 固定為 0 已是恰當設計，套用對稱發散色階會製造不存在的「負值」語意。 |
| 共同 quiver（風／流箭頭）幾何與箭頭比例尺 | LBT 的八張圖沒有向量場箭頭圖，此規則整條不適用。 |
| PC（時間係數）顯示規則（`Sigma@V.T`，不做標準化） | LBT 沒有 SVD 分解、沒有時間係數的概念，此規則整條不適用。 |
| 模態正交性／Gram 矩陣 QA 檢查 | 該檢查是 SVD 分解本身的數學性質驗證（確認左右奇異向量正交），LBT 的軌跡與足跡統計不涉及矩陣分解，此規則整條不適用。 |
| 模式比較 2×2 版面資產（4 模態 × 4 區域比較用的固定排版） | 該版面是為了在同一張圖比較四個海域的同一模態而設計，LBT 的圖已經是逐站輸出、逐圖比較，沒有「多模態」的維度需要用 2×2 版面壓縮呈現，此規則不適用。 |
| PROPACK（SVD 求解器）相關的數值方法規則 | 純屬 SVD 分解求解器層級的規則，LBT 完全沒有 SVD 分解步驟，此規則整條不適用。 |
| 流速模態色條刻度標示為 ×10⁻² 科學記號格式 | 該格式是配合 `vmax=0.024` 這個特定數量級選擇的顯示方式；LBT 的相對來源權重與密度數值量級因站點與情境而異，固定套用 ×10⁻² 格式在數值量級不同時反而造成誤讀，色條刻度改用 `ColorScale.ticks()` 產生的實際數值直接標示。 |
| 1% 超限硬性上限、超標即停止並要求重新送審 | 該硬性門檻是 OCM-SVD-Analysis 針對已核定歷史常數的既定情境設計的品管機制。LBT 目前的 `vmax` 定義本身就是「本次執行的實際最大值」，數學上不會有超限發生（超限一定為 0），套用同樣的硬性門檻沒有實質意義；`over_limit_count`／`over_limit_fraction` 欄位仍完整保留，架構上支援未來改用固定常數後啟用相同門檻。 |

## 5. 誠實揭露：與 OCM-SVD-Analysis 的關鍵差異

**`vmax` 的決定方式不同，這是本次交付有意識的取捨，不是規則遺漏。**
OCM-SVD-Analysis 的 `vmax=0.024` 是先用真實四海域 science run 資料做過
一次 preflight、經審查後寫死在規則文件裡的常數；之後每次重繪都固定使用
同一數字，新資料若超過它才會出現非零的 over-limit 紀錄，這個機制才具備
「跨 run 比較」與「品管警示」的實質意義。

LBT 目前是第一次建立這套跨站共用色階機制，**沒有一個經過審查、可以直接
套用到未來所有 run 的歷史 `vmax` 常數**。因此 `_compute_shared_color_scales`
把 `vmax` 定義為「這次執行、這批站點的實際最大值」——這樣做在單次執行內
必然不會出現超限（因為上限本來就取自這批資料本身），已達成「同一批圖之間
色階刻度一致化、超限只記在 metadata、不做視覺裁切」的核心行為，但**尚不
具備跨 run 比較或品管警示的能力**。`over_limit_count` 等欄位在架構上完整
保留；未來若研究團隊審閱多次 run 的結果後核定一個跨 run 固定的常數，只
需要在呼叫端改用該常數建構 `ColorScale`，不必更動統計或繪圖邏輯。

## 6. WP-38 三種新圖的規則（v1.1.0 新增）

適用對象：停留時數密度（`source_footprint_residence`，含各站色階的 `_site_scale` 版）、
回溯快照（`lookback_snapshot`，2×2 面板）與邊界弧長分布
（`boundary_departure_along_arc`）。`--figure-text full` 與 `none` 兩種模式一律符合。
機器可讀版本見 JSON 契約的 `wp38_footprint_figures` 區塊。結果一律稱「條件式來源足跡」
或「相對來源權重」，等值線稱「高密度區間（HDR）」。

### 6.1 等深線不干擾高密度區間

- 只畫 200 m（陸棚邊緣，實線）與 1000 m（虛線），中灰（`#7f7f7f`）、線寬 0.4 pt；
  圖層在密度色面之上、HDR 等值線之下，HDR 的深棕色粗線才是主角。
- **地圖上不放等深線數字標籤**（標籤是干擾高密度區間的主因）；水深與線型改由圖例
  （「等深線 200 m（實線）」「等深線 1000 m（虛線）」）與 `captions.md` 交代。
- 某站範圍內沒有 1000 m 等深線時，圖例不列 1000 m。水深可由 `--isobath-levels` 指定，
  但正式出圖用預設值。

### 6.2 受體位置準確、地圖與圖例不寫站名

- 星號必須是正式設定中該站的研究點錨點（`study_sites[].anchor_lonlat`），由 `--config`
  提供；**正式出圖必須帶 `--config`，省略時 fail-closed 拒絕**。僅診斷用途可加
  `--allow-unanchored-receptor`，此時星號退回釋放點平均位置，圖例改寫「受體（釋放點平均位置）」，
  manifest 標註 `formal_anchor_used=false`。
- manifest 記錄設定檔路徑與 SHA-256、每站的錨點座標與 `receptor_core_radius_m`。
- **同時畫出實際釋放範圍**：全部相異釋放點（後處理足跡幾何表的受體水平位置）的凸包，
  淡紫色半透明填色加細框，圖例「粒子釋放範圍」。設定檔只定義受體核心圓（錨點加 12.5 km 半徑），
  沒有釋放多邊形，所以以實際釋放點的外包絡為準。manifest 與摘要記錄：錨點到釋放點重心的距離（km）、
  釋放範圍最大半徑（km，釋放點到重心）、釋放點到錨點的最大距離、錨點是否落在海上（海岸線
  GeoJSON 判定）。釋放點是在核心圓內以站點獨立亂數無放回抽取的少數幾點，重心因此不會剛好落在錨點。
- 圖例只寫「受體（研究點錨點）」；地圖上沒有任何站名文字。`full` 模式的圖標題可保留站名
  作為圖面識別，`none` 模式沒有標題。

### 6.3 繪圖規範（對齊 OCM-SVD-Analysis）

出處：`OCM-SVD-Analysis/docs/gongliao_2025_svd_report_interpretation.md` 約第 56 行
（「經緯度軸首尾刻度就是實際繪圖邊界；色條首尾刻度就是該圖 vmin／vmax」）、
`svd_figure_reference_log.md` 約第 316 行（「經緯度軸與色條把實際上下限列為首尾刻度」，
本專案數值邊界稽核決策）、`water_column_svd_formal_spatial_figure_rules.md` 第 3.3 節
（等距刻度含 0 與 vmax、矩形色條 `extend='neither'`、超限僅記錄於 metadata、不裁切資料）。

1. **色條與主圖同高**：色條軸的上下緣與地圖軸的上下緣，在 draw 之後以 display 座標量測，
   相差不超過 1 px（`COLORBAR_ALIGNMENT_TOLERANCE_PX`），量測值寫入 manifest。2×2 快照的共用色條
   與整個面板網格同高（上排面板上緣至下排面板下緣）。
2. **色條上下限都以刻度標出**：對數色階，上限取大於或等於資料最大值的 10 的整數次方，
   下限為上限的 10⁻⁴，每個 10 的次方一個刻度，首尾刻度就是 vmin／vmax；矩形色條、
   `extend='neither'`。低於下限的格子透明，manifest 記錄格數與所占比例；`over_limit_count`
   （高於上限）理論上為 0，仍記錄。各站色階版與跨站共用色階版規則相同。
3. **主圖 X、Y 軸邊界標明**：範圍向外擴到刻度間距的整數倍，首尾刻度落在框邊並標數字與 °E／°N；
   間距依範圍自動選 0.1、0.2、0.25 或 0.5 度，每軸 4～7 個刻度（2×2 快照每軸最多 6 個）。
   2×2 面板每格範圍一致；共用軸只在外側標數字，框邊仍有刻度。
4. **弧長分布圖**：x 軸 0 至總弧長（取整到刻度倍數）、左 y 軸 0 至取整後最大值、右 y 軸 0～100%，
   首尾刻度都落在框邊。
5. 陸地與海岸線沿用現有樣式；地圖上不畫經緯度格線。

驗證：`tests/test_footprint_figures.py` 以實際 draw 後的量測檢查同高、色條與軸的首尾刻度、
地圖無等深線數字標籤與站名、圖例無站名、星號等於錨點、釋放範圍包含全部釋放點、缺 `--config`
時拒絕，以及 JSON 契約與常數一致。

## 6.1 沿開放邊界之相對來源權重曲線（WP-55，`native-arc-by-group`）

獨立指令 `lbt native-arc-by-group`（程式 `native_runner/arc_by_group.py`、`arc_by_group_figures.py`），只讀後處理輸出，
不屬於 `native-figures` 的八張＋四張清單，檔名另行登記於 `ARC_BY_GROUP_FIGURE_STEMS`。每站每案六張圖：
依沉降速度類別、依沉底時刻的季節、依沉底時刻的大潮／小潮，各一張疊圖（`*_overlay`）與一張分格圖（`*_panels`，
類別 2×5、季節 2×2、大小潮 1×2），各自輸出含文字（`figures/`）與無文字（`figures-notext/`）兩版，PNG 與 SVG 並出。

1. 縱軸為「相對來源權重（%／km）」，每條曲線對弧長積分為 100%（分母為該組穿越粒子數）；不使用「機率」一詞。
2. 平滑為一維高斯核，頻寬預設 5 km（`--sigma-km`）。邊界折線首尾閉合時環狀接續；未閉合時於端點鏡射（質量守恆）。
   閉合與否由 `footprint_geometry_by_site` 的折線頂點判定，每站各自記錄於輸出表與圖說。
3. 季節與大小潮只用沉底時刻計算的 `deposition_season`、`deposition_tide_class`；`observation_*` 一律不用。
4. 圖上不寫站名（站名只在檔名與圖說）；X、Y 軸兩端都標數字；同一分組的圖在本次執行的全部站與全部案例共用
   同一組軸範圍與刻度（記錄於 manifest 的 `axes`），圖面與圖說不另加說明。
5. 依類別的兩張圖，速度標示依資料而定：後處理表有逐粒子沉降速度欄位（v7，每類 10 個速度各占十分之一）時，
   圖例與分格標題寫該類穿越粒子實際用到的「最小值–最大值 m/s」（絕對值、三位有效數字），圖例標題為「沉降速度範圍」；
   沒有該欄位（v6 以前，每類單一速度）時寫單一值。類別排序一律依類別代表速度由慢到快。表中記錄各類速度的最小值、
   最大值與相異速度個數。
6. 類別名稱走 `native_runner/display_names.py` 的顯示名稱對照表（只有「其他/不可回收」顯示為「不可回收塑膠」、
   「其他/可回收」顯示為「可回收塑膠」）；`material_id`、材質清冊的 `oca_category_zh` 與所有指紋、雜湊不變。
7. 量化表（`tables/<站>/<分組>_summary.json` 與 `_curves.csv`，摘要同時寫入 `manifest.json`）：各組曲線格點值、
   南／東／北／西各邊占比、組間兩兩總變異距離，以及抽樣雜訊基準：同組隨機對半切（粒子層級與沉底事件層級）；
   類別另有「同一釋放點交換類別標籤」的配對置換，季節與大小潮另有沉底事件層級的標籤置換。
8. 驗證：`tests/test_native_arc_by_group.py`（環狀與鏡射平滑積分為 100%、閉合判定、總變異距離、置換基準、
   圖檔名登記、端到端輸出）與 `tests/test_native_display_names.py`（顯示名稱不影響識別值）。

## 6.2 沉降速度細分圖的類別標示（WP-56，`native-figures`）

`settling_velocity_breakdown` 的列仍以類別代表速度（`settling_velocity_mps`）分組，排序不變（由上而下由慢到快）。
縱軸刻度與軸名依資料而定：

1. 後處理表有逐粒子沉降速度欄位 `particle_settling_velocity_mps`（v7，每類 10 個候選速度逐粒子抽樣）時，
   左側長條圖的刻度寫「類別顯示名稱＋該類粒子實際用到的速度範圍（最小值–最大值，絕對值，三位有效數字）」，
   例如「不可回收塑膠 0.000897–0.108」，軸名為「沉降速度類別（下沉速度範圍，公尺／秒）」；右側熱圖只標類別顯示名稱。
   類別名稱走 `native_runner/display_names.py`，沒有中文字型而退成英文時改用 `material_id`。
   代表速度只是分組鍵，不再當刻度。該欄位含缺值或非有限值時 fail-closed。
2. 沒有該欄位（v6 以前，每類單一速度）時，刻度、軸名、圖說與 README 與歷來逐字相同。
3. 圖說與 README 在 v7 模式改寫為「N 個沉降速度類別、每類 10 個候選沉降速度、各占該類粒子十分之一，
   比例與權重以類別為單位彙整」；舊版「十個未校準沉降速度敏感度格點」只用於舊版資料。
   `manifest.json` 另有 `settling_velocity_breakdown_labels` 區塊，記錄每列的 `material_id`、顯示名稱、代表速度與實際速度範圍。
4. 終止類別圖（`terminal_counts`）規則不變：只畫實際出現的終止類別（例如南灣含 Stokes 案的 2 顆「沉底」），標籤為中文。
5. 驗證：`tests/test_native_figures_velocity_labels.py`（有、無逐粒子速度欄位、英文退場、缺值 fail-closed）。

## 7. 版本異動紀錄

| 版本 | 日期 | 異動內容 |
|---|---|---|
| v1.0.0 | 2026-09-24 | 首次核定。完成任務 0（水平軌跡圖移除 local domain 圖層）；建立本規則文件與對應 JSON 契約 `docs/data/native_figures_formal_output_policy_v1.json`；採納 OCM-SVD-Analysis 的雙格式輸出（PNG 600 dpi＋SVG）、固定色階＋超限記錄、跨站色階一致化三項規則；明確排除八項判定不適用的規則（第 4 節）。 |
| v1.1.0 | 2026-09-29 | 新增第 6 節，涵蓋 WP-38 三種新圖（停留時數密度含各站色階版、回溯快照、邊界弧長分布）：等深線只畫 200 m 實線與 1000 m 虛線且不標數字、星號取研究點錨點並強制 `--config`、圖上畫粒子釋放範圍（釋放點凸包）且 manifest 記錄錨點與釋放範圍核對數值、地圖與圖例不寫站名、色條與地圖同高（1 px 內，量測寫入 manifest）、對數色階上限取 10 的整數次方且下限為 10⁻⁴、座標軸首尾刻度即框邊、弧長分布圖三軸取整；附 OCM-SVD-Analysis 出處。JSON 契約升為 1.1.0，新增 `wp38_footprint_figures` 區塊。 |
| v1.2.0 | 2026-10-14 | 新增第 6.1 節：WP-55 沿開放邊界之相對來源權重曲線（依類別、季節、大小潮分線）。僅文件說明；`docs/data/native_figures_formal_output_policy_v1.json` 未更動。 |
