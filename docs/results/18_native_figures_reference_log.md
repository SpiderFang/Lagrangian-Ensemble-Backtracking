# `native-figures` 圖面做法文獻依據紀錄

> 文件狀態：**v1.0.0，2026-09-24 建立**。比照 `OCM-SVD-Analysis` 的
> `svd_figure_reference_log.md` 建立，記錄 LBT `native-figures` 八張圖
> （`FIGURE_STEMS`，見
> `docs/results/17_native_figures_formal_output_rules.md`）的呈現方式，
> 哪些有文獻依據、哪些純屬本專案工程決策。
>
> **依據來源限定**：本文件只引用
> `data/results_visualization_literature/README.md` 已完整登記
> （原文與紅框標註副本、DOI、SHA-256 皆已核對存檔）的八篇文獻，不另外
> 引用未經此程序驗證的文獻或 DOI。凡本文件找不到對應文獻紅框頁支持的
> 呈現選擇，一律列在第 3 節「純工程決策，非文獻依據」，不勉強牽附。

## 1. 文獻與圖面對應總表

| 圖（`FIGURE_STEMS`） | 對應文獻 | 文獻具體支持的呈現方式 | 本專案採用方式與必要限制 |
|---|---|---|---|
| `horizontal_overview`／`horizontal_local` | Carlson et al. (2017)，第 4 頁 Figure 2 | 以少量時間著色的逆向軌跡說明路徑，不用全部軌跡製造密度假象 | `max_trajectories_per_site` 限制每站畫出的代表軌跡數，以「代表粒子起始深度序位」著色；圖說須交代抽樣規則、受體站點與回溯方向。**不得**宣稱畫出的軌跡是「所有軌跡的代表分布」，只是有限樣本的路徑示意。 |
| `boundary_azimuth_rose`／`boundary_arc_segments` | van Sebille et al. (2018)，第 14–15 頁 Figure 4／Figure 5；Rypina et al. (2014)，第 5 頁 Figure 3 | 明確區分「全部粒子位置計數後正規化」與「曾至少訪問一次的不同粒子比例」兩種呈現；空間訪問分率須與樣本數、網格（此處為方位扇區／邊界弧段）解析度一併說明 | 兩張圖呈現的是**相對來源權重**（依 `postprocess.py` 既有定義的加權統計），圖說與 manifest 皆須寫明分母是穿越開放邊界的粒子數，不與「曾訪問」定義的分率混用。方位扇區數（`azimuth_sector_count`，預設 16）與弧段幾何解析度已在 manifest／README 交代，供讀者判斷統計穩定度。 |
| `source_footprint_kde` | van Sebille et al. (2018)，同上；Rypina et al. (2014)，同上 | 空間訪問機率圖搭配網格大小與樣本數說明 | KDE 密度為連續化的相對來源權重呈現，HDR（50%／75%／90%）等值線標示高密度區域範圍。圖名與圖說固定使用「條件式來源足跡」，**不使用**「來源機率」或「posterior probability」語意——依 `data/results_visualization_literature/README.md` 第 7 節既有限制，本專案尚未建立來源先驗與觀測似然，不具備使用後驗機率語意的條件（Pierard et al. 2022 之後驗機率格式僅在完成正式貝氏層後方可採用，目前不適用）。 |
| `depth_age` | 無直接對應文獻紅框頁 | — | 純工程決策，見第 3 節。 |
| `terminal_counts` | Carlson et al. (2017)，第 6 頁 Figure 4；表格分開列出擱淺、仍漂浮、開放邊界等狀態 | 終止狀態應分類列出，不合併為單一「未到達」類別 | `terminal_counts` 依 `postprocess.py` 既有七種終止狀態分類（成功穿越、達最大回溯、擱淺、沉積、缺時、數值失敗、窗口前沉積）逐一計數呈現，符合「乾點、域外、資料缺口與數值失敗分開，不可合併」的文獻建議精神。 |
| `settling_velocity_breakdown` | Ko et al. (2018)，第 6／10 頁 Figure 2／Figure 5 | 依類別分組的長條圖比較，附樣本數與誤差說明 | 依沉降速度分組後在各方位扇區呈現相對來源權重，呈現形式上類似該文獻的「分組長條比較」精神，但本專案的分組變數（沉降速度）與該文獻（來源國／月份）不同，屬呈現手法借用，非直接沿用其定義。 |

## 2. 跨圖共通的分母與語意規則（文獻支持）

以下規則不專屬單一圖，而是 `data/results_visualization_literature/README.md`
文獻歸納表對所有圖的共通要求，`native-figures` 全數遵守：

- **未到達者不得填零、不得併入分母**：`postprocess.py`
  既有的七種終止狀態分類本身已將「成功穿越」與其餘六種未穿越狀態分開
  儲存，`native-figures` 直接沿用這個分類，不自行合併或補零（van Sebille
  et al. 2018；Rypina et al. 2014）。
- **語意固定為條件式來源足跡／相對來源權重，不稱機率**：本專案目前沒有
  來源先驗、觀測似然或獨立驗證，`figures.py` 全部圖說與 README 固定使用
  「條件式來源足跡」「相對來源權重」用語，不使用「來源機率」
  「posterior probability」或任何隱含貝氏後驗語意的字眼（Pierard et al.
  2022 對比說明；`data/results_visualization_literature/README.md` 第 7
  節總結）。
- **樣本數與分母須可稽核**：manifest 的 `color_scales.{stem}.per_site`
  區塊記錄每站每張圖的 `sample_count`，README 與圖說交代
  `max_trajectories_per_site`、`azimuth_sector_count` 等抽樣／解析度參數
  （Rypina et al. 2014 對網格解析度與最小有效成員數的要求）。

## 3. 純工程決策，非文獻依據

以下呈現選擇是本專案的工程／版面決策，**沒有**在已登記的八篇文獻中找到
直接對應的紅框頁支持，明確標示為工程決策而非文獻依據，避免誤導讀者以為
所有呈現選擇都有文獻背書：

- **`depth_age` 圖的線形＋起始深度序位著色版面**：這是本專案設計的呈現
  形式，用來展示同一批代表粒子的深度—時齡關係。已登記文獻中沒有與此
  版面直接對應的紅框頁。
- **AEQD（方位等距）投影的地圖呈現方式**：屬於地理投影選擇，非成果統計
  呈現方式，不在本文獻庫涵蓋範圍內。
- **固定色階＋超限飽和顯示、超限記錄於 manifest**：此做法是**採納自
  相鄰專案 `OCM-SVD-Analysis` 的成圖規則**（見
  `docs/results/17_native_figures_formal_output_rules.md` 第 3.2 節），
  不是來自本文獻庫的海洋科學文獻，屬於跨專案工程規範對齊，而非學術文獻
  依據，兩者性質不同、依據來源不同，此處特別區分以免混淆。
- **PNG 600 dpi ＋ SVG 雙格式輸出**：同樣是採納自 `OCM-SVD-Analysis`
  的正式交付規範，屬於出版格式工程決策，非科學呈現方法的文獻依據。
- **16 方位扇區、HDR 50%／75%／90% 等值線的具體數值選擇**：扇區數與 HDR
  百分位數是本專案既有設計參數（`postprocess.py`
  `DEFAULT_AZIMUTH_SECTOR_COUNT`），已登記文獻支持「應呈現空間訪問分率」
  這個方向，但沒有文獻指定必須使用 16 扇區或這三個特定百分位數，具體
  數值屬本專案工程決策。

## 4. 版本異動紀錄

| 版本 | 日期 | 異動內容 |
|---|---|---|
| v1.0.0 | 2026-09-24 | 首次建立，對應 WP-25。全部引用取自 `data/results_visualization_literature/README.md` 既有登記之八篇文獻，未新增任何文獻或 DOI。 |
