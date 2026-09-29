# WP-38 報告：以停留時數密度取代邊界點 KDE 的「條件式來源足跡」後處理與出圖

分支 `wp38-footprint`（自 main `552ff4f` 分出，未推送、未改 main）。

## 1. 為什麼要做

既有 `kde_by_site` 的輸入只有粒子穿越開放邊界的那一點，全部落在一維邊界線上；二維 KDE（Scott 頻寬）把它們
抹成跨出計算域的圓團，HDR 大片落在域外，物理上站不住。本工作包改用每顆粒子的完整軌跡，逐小時累加「仍在漂流」
的位置到固定經緯度網格，得到停留時數密度並導出 HDR；另補回溯快照 KDE（點散在域內，二維 KDE 適用）與沿開放邊界
弧長的離開分布（一維）。舊的 `kde_by_site`、`source_footprint_kde` 保留不動。

## 2. 提交

| 提交 | 內容 |
|---|---|
| `03f9226` | 同步 .203 已改的圖例中文化（WP-42／43），作為 golden 基準 |
| `063704d` | 後處理：足跡累加器、CLI 旗標、水深讀取與等深線匯出、測試 |
| `1147068` | `native-figures`：三種新圖、底圖與等深線、圖說、測試 |
| `7faebcf` | 圖面修飾：定義域裁切、快照與弧長圖標題／標籤 |
| `9d1eb73` | 快照圖色條改固定位置、副標分行 |
| `4a11c21` | 重產 `docs/source_code_architecture_map.html`（新增模組後既有測試要求同步） |

基準同步結果：Mac 主樹與 .203 `~/Workspace/lbt-repo` 的 `figures.py` 差異只有 WP-42／43 的圖例字串，已收進 `03f9226`；
舊產品的 golden digest 均以 `03f9226` 的舊程式產生。

## 3. 新增與修改的檔案

- 新增 `src/lagrangian_backtracking/native_runner/footprint.py`：`FootprintAccumulator`（`add_batch`／`finalize`）、
  `make_grid`、`compute_hdr`、`total_variation_distance`、`summary_table`。
- 新增 `src/lagrangian_backtracking/native_runner/bathymetry.py`：`BathymetryField`、`load_ocm_grid_bathymetry`、
  `resolve_site_grid_dirs`、`build_isobath_geojson`。
- 新增 `scripts/export_isobaths.py`。
- 修改 `native_runner/postprocess.py`（在 `add_batch` 串流路徑接入累加器、寫出新表與 manifest 區塊）、
  `cli.py`（旗標見第 9 節）、`native_runner/figures.py`（三種新圖、底圖、圖說、manifest 登記）。
- 新增測試 `tests/test_footprint.py`（38 項）、`tests/test_footprint_figures.py`（10 項）、`tests/wp38_fixture.py`。
- 重產 `docs/source_code_architecture_map.html`。

## 4. 產品定義

| 產品（表／圖） | 定義 |
|---|---|
| `residence_density_by_site`（含 `_member_half`、`_settling_velocity`） | 停留時數＝逐小時、`status==0` 的粒子樣本數（終止後凍結的位置不計）；到達時刻樣本計入。正規化＝格內停留時數 ÷ 該站總停留時數（總和為 1，容差 1e-9，不符即 fail-closed）；密度＝正規化值 ÷ 格面積（km²，`R²·Δλ·Δφ·cosφ`，R=6371.0088 km）。 |
| `residence_hdr_by_site` | 依密度由高到低累加到 50／75／90% 停留時數的最小面積網格集合；累積比較用整數計數。 |
| `lookback_snapshot_kde_by_site` | 回溯第 72／168／336／720 小時仍在漂流（`status==0`）的粒子位置，用既有 `conditional_kde_2d`（Scott 頻寬，96×96 格）；分母為全站粒子；軌跡長度不足者算已終止；粒子少於 30 顆不估計；落在 flow domain 外的 KDE 質量超過 10% 於 manifest 警示。 |
| `boundary_departure_along_arc_by_site`、`boundary_edge_intervals_by_site` | 終止於開放邊界的粒子離開點以 `shapely.line_locate_point` 投影到邊界折線，10 km 一組的直方圖與累積比例；以流場外框的經緯度邊界標示北／東／南／西邊與角落的弧長區間。 |
| `footprint_summary_by_site` | 每站一列：總粒子數、總停留小時、HDR 50／75／90 面積、質心與受體的距離／方位、水深小於 200 m 的停留比例（與水深缺漏比例）、各快照的仍在漂流比例與域外質量、成員分半 TVD 與 HDR50 面積相對差、兩種雜訊底（成員隨機分半、到達單元隨機分半，各 64 次，seed 20260929）的 TVD 平均與 p95、0.01／0.02／0.04° 的 HDR50／90 面積。 |
| `footprint_grid_by_site`、`footprint_geometry_by_site` | 網格定義與 flow domain、開放邊界幾何，供出圖。 |
| 圖 `source_footprint_residence`（跨站共用色階）、`source_footprint_residence_site_scale`（各站自訂色階）、`lookback_snapshot`、`boundary_departure_along_arc` | 規格見第 7 節；兩種色階版本在 manifest 的 `color_scales` 與圖說中明示。 |

串流與位元相同：`add_batch` 每批只做整數 `bincount`，狀態大小與粒子總數、軌跡長度無關；座標到網格用每站一張 200 m 查表
（`lut_cell_m`），不逐點呼叫 pyproj。因此分塊累加與一次累加位元相同，且既有輸出在不加 `--with-footprint` 時位元不變。

## 5. 驗證

- 新測試：`tests/test_footprint.py` 38 項、`tests/test_footprint_figures.py` 10 項，Mac 全過；`test_native_figures.py`＋
  `test_native_postprocess.py`（排除 1 項既有環境失敗）86 項通過。
- 位元相同的回歸防護：`GOLDEN_LEGACY_DIGESTS` 是以舊程式 `03f9226` 產生的既有產品規範內容摘要；
  未開 `--with-footprint` 時新程式輸出與其逐一相等。
- 合成資料測試涵蓋：正規化總和、HDR 的最小面積性質與整數比較、終止後凍結位置不計、到達樣本開關、分塊切分位元相同、
  TVD 性質與雜訊底、網格敏感度、快照過短視為終止、域外質量警示、弧長直方圖與邊／角標示、LUT 邊界行為、
  fail-closed（正規化失敗、缺欄位）。
- 獨立重算（試產子集，直接讀 npz）：`trajectory_status==0` 的總樣本數與 manifest 的 `total_particle_hours` 五站逐一相等
  （貢寮 334,866；龜山島西側 342,855；新竹外海 157,984；南灣 614,581；連江 1,149,928）；`final_status==1`（離開流場）的
  粒子數與 `fraction_crossed_interest_region_boundary` 一致（連江 3,907／4,000＝0.97675；南灣 3,998／4,000＝0.9995；其餘 4,000）。
- 邊界堆積檢查：弧長直方圖沒有任何一組落在角落區間；貢寮最高一組是南邊界 70–80 km 處的 37.8%，該處有實質的離開通量，
  不是終止位置被重複累加（終止樣本不計）。停留時數密度圖在開放邊界上沒有沿邊界的高值帶。
- 全套件（.203，於 `1147068` 上分 8 段跑）：4,027 通過、15 失敗、4 錯誤、134 略過。失敗全部屬既有環境問題：
  `test_input_derivation`（13）、`test_native_postprocess` 的 provenance 一項（1）、`test_engineering_window`（4 錯誤），
  以及當時尚未重產的 `test_committed_html_is_source_generated`（1，已於 `4a11c21` 修正，7 項通過）。
  對 `03f9226` 乾淨樹在同機重跑相同檔案，得到同一組失敗（該樹另有數項 `test_input_derivation` 在高負載下失敗，
  `test_hsinchu_explicit_pilot_rejects_middle_hour_product_support[nww3_analysis]` 單獨在兩棵樹都通過，判定為負載相依的不穩定測試）。
- `ruff check`（行長 110）：新增與修改的檔案全部通過；全倉庫另有 `forcing.py` 一項 import 排序與 `test_dynamic_execution.py` 兩項
  E501，皆非本工作包造成、未動。

## 6. 試產（.203，run04 no_stokes 子集）

子集：100 個工作單元、20,000 顆粒子（全量身分表 500,000 顆），用符號連結取自 run04 結果，不寫入 run04。

| 步驟 | 耗時 | 尖峰記憶體 |
|---|---|---|
| 後處理不含足跡（對照） | 34.3 s | 1.478 GB |
| 後處理含足跡 | 37.1 s | 1.475 GB |
| 等深線匯出 | 10.0 s | 1.455 GB |
| 出圖 full（600 dpi＋SVG） | 11 分 45 秒（同時段主機負載平均約 41） | 1.749 GB |
| 出圖 none | 見 `checks/wp38/logs/figures-notext.log` | 約 1.75 GB |

子集觀察（僅供參考，每站只有 20 個到達單元）：

| 站 | HDR50／75／90 面積 (km²) | 質心距受體 (km)／方位 (°) | 水深<200 m 停留比例 | 72 h 仍在漂流 |
|---|---|---|---|---|
| 貢寮 | 806／1,845／2,893 | 14.0／75 | 0.55 | 65.7% |
| 龜山島西側 | 274／933／1,873 | 11.8／29 | 0.84 | 54.6% |
| 新竹外海 | 463／922／1,341 | 37.4／241 | 1.00 | 0% |
| 連江 | 616／1,423／2,740 | 14.7／359 | 1.00 | 85.7% |
| 南灣 | 266／925／2,145 | 35.8／321 | 0.56 | 34.0% |

- 成員分半 TVD（0.032–0.065）與成員隨機分半雜訊底平均（0.032–0.057）相當：前後兩半成員的差異不比隨機分半更大。
  到達單元隨機分半的雜訊底（0.12–0.16）較大，因為子集每站只有 20 個單元；全量後應顯著下降。
- HDR50 面積對網格明顯敏感：新竹外海 0.01／0.02／0.04° 為 367／463／594 km²，貢寮 748／806／896 km²；HDR90 較穩定
  （新竹外海 1,173／1,341／1,583 km²）。報告面積時必須註明網格解析度。
- 快照 KDE 域外質量：貢寮 72 h 13.6%、南灣 720 h 24.6%（該時刻仍在漂流的只有 5.95%，約 238 顆），皆超過 10% 門檻，
  manifest 有警示，面板標題標註「範圍外 N% 未繪」。
- 新竹外海 100% 停留在 200 m 以淺，且粒子在 72 h 內全部離開流場，四個快照面板皆為「粒子數不足，未估計」。
- KDE 不含陸地遮罩：南灣 168 h 快照的 HDR 線有一段跨過恆春半島的陸地（陸地灰底蓋在密度上，但等值線仍畫出）。
  這是 KDE 平滑的性質，圖說已註明；沒有做陸地遮罩重新正規化，避免改變 KDE 的定義。

## 7. 圖面規格與檢視

依交辦：陸地 #d9d9d9、海岸線黑色 0.5 pt、50／100／200／1000 m 等深線灰色細線（200 m 較粗）並標數字、受體星號加站名、
洋紅色 1.2 pt 虛線開放邊界、色階 YlGnBu 對數、下限為最大值的 1e-4 且下限以下不著色、色條「% 停留時數／km²」、
HDR 用同一色系（深棕）的實線／虛線／點線、經緯度刻度、`full`／`none` 兩種文字模式、dpi 與既有圖一致。
圖例文字採「高密度區間 50%」「流場範圍開放邊界（終止邊界）」。

已檢視 5 站 × 新圖（100 dpi 縮圖，`_verify/wp38/v100c/`，共 20 張）：底圖、陸地、海岸線、等深線、受體、邊界、
色階均正確；中文標題與圖例無截斷。已修正並重新檢視的問題：快照圖色條蓋到右欄面板邊緣；快照與停留時數圖的副標過長貼邊；
弧長圖標題與副標重疊、角落標籤過小；「粒子數不足」文字壓住受體；密度網格與 HDR 線超出 flow domain（改以 flow domain 多邊形裁切）。

## 8. 底圖與水深的處理

水深取自正式運算用的 OCM 標準化子網格（每個 flow domain 的 `grid/` 快取目錄：節點經緯度、節點水深、三角形單元）。
`BathymetryField` 以三角網格線性內插（matplotlib `LinearTriInterpolator`）；等深線用 `tricontour` 在 CRS84 上等值化，
輸出 GeoJSON。網格缺漏時等深線與水深統計整體略過並寫入 manifest，不以預設值頂替。

- 五站對應的網格由 `identity_table.npz` 與 `region_to_flow_domain.json` 解出，路徑根為 `~/Workspace/lbt-local-data/ocm_native`。
- 試產等深線檔：.203 `~/Workspace/lbt-runs/checks/wp38/isobaths.geojson`（360 KB）；正式重跑腳本匯出到
  `~/Workspace/lbt-runs/run04-h90-m10-v3/isobaths.wp38.geojson`（不存在才匯出，跨案例共用）。
- 「水深小於 200 m」的停留比例以網格中心水深計，分母只含水深可內插的停留小時；水深無法內插（網格中心落在 OCM 三角網格之外）的停留小時
  另以 `hours_fraction_depth_missing` 列出（貢寮 0.9%、龜山島西側 0.8%、連江 6.0%、南灣 17.7%，新竹外海 0）。
  南灣的缺漏比例較高，正文引用該站的比例時需一併註明缺漏比例。
- 繪圖前以 `shapely.clip_by_rect` 把等深線裁到圖框，避免長線段被丟棄。

## 9. 全量重跑（未執行，待主代理同意）

新增旗標：`native-postprocess` 的 `--with-footprint`（需同時給 `--config` 與 `--with-boundary-segments`）、
`--footprint-grid-deg`（預設 0.02）、`--footprint-padding-deg`、`--footprint-lookback-hours`、`--footprint-arc-bin-km`（預設 10）、
`--bathymetry-grid-root` 與 `--region-flow-domain-map`（須同時提供）、`--trial-subset-from-results`（僅試產）；
`native-figures` 的 `--bathymetry <isobaths.geojson>`。

腳本：`_verify/wp38/run_post_figures_wp38.sh <no_stokes|finite_depth_stokes>`（未追蹤，另有副本供 .203 使用），
其做法與 `run_post_figures_wp37.sh` 相同，差異是程式碼樹改為乾淨的 `~/Workspace/lbt-repo-wp38`（`--formal --project-root`）、
輸出寫到新目錄，且任一新目錄已存在就中止：

```
$R/postprocess.wp38/        # 後處理（含足跡）
$R/figures.wp38/            # 出圖 full
$R/figures-notext.wp38/     # 出圖 none
# $R = ~/Workspace/lbt-runs/run04-h90-m10-v3/{no_stokes,finite_depth_stokes}
```

指令（兩案各一次，可循序）：

```
cd ~/Workspace/lbt-repo-wp38 && git status --porcelain      # 必須為空
bash run_post_figures_wp38.sh no_stokes
bash run_post_figures_wp38.sh finite_depth_stokes
```

估計（非實測全量）：
- 後處理：run04 既有基準 582.6 s、1.79 GB。試產子集含／不含足跡為 37.1 s／34.3 s，加上的成本約 8%；
  全量約 25 倍子集，估計每案 10–12 分鐘、尖峰記憶體約 1.8 GB（足跡狀態與粒子數無關，只增加不到 0.1 GB）。
- 出圖：圖的輸入是彙總表，耗時與粒子數無關；試產 11 分 45 秒（主機忙碌時），單獨執行估計 7–10 分鐘，每種文字模式各一次，
  尖峰記憶體約 1.75 GB。
- 等深線匯出約 10 s。

## 10. 偏離、未解與未驗證

1. 報告檔名用 `WP-38_footprint.md`：`reports/WP-38.md` 已存在（未追蹤、被 gitignore 的 v3 紀錄），不覆蓋。`reports/` 被 gitignore，需 `git add -f`。
2. 新產品採 opt-in（`--with-footprint`），不加旗標時既有輸出位元不變；若要改成預設開啟，需另行決定。
3. 網格查表以 200 m 量化，距網格線 200 m 內的點可能歸到相鄰格；相對 0.02°（約 2 km）的格寬，影響限於邊界格，不影響總量（總和由整數計數保證）。
4. `_verify/wp38/t.txt` 是早先留下的暫存檔（未追蹤）。
5. 環境既有測試失敗（見第 5 節）與本工作包無關，未修。
6. 未驗證：`--formal --project-root` 路徑與 `--with-footprint` 同時使用（試產走 `--trial-subset-from-results`，非正式模式）；
   全量規模下的實際耗時與記憶體；全量下雜訊底與網格敏感度的數值（子集只有 20 個到達單元）。
7. 出圖檢視用 100 dpi 縮圖；正式 600 dpi 的 PNG 已在 .203 產出（`checks/wp38/figures*/`），未逐張以 600 dpi 目視。
8. Mac 端 `git stash`／`git index.lock` 在此環境會出現無法解除鎖檔的警告，提交是用 `g2.sh` 包裝完成，不影響歷史。
