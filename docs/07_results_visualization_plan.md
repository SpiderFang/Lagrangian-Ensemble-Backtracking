# 成果呈現與學術視覺化規格

## 1. 文件目的與適用範圍

計畫書只要求辨識主要潛在來源路徑與 KDE，未完整規定圖表。本文依海洋 Lagrangian、逆向溯源、海洋廢棄物及 ensemble drift 的同儕審查研究，制定本專案的正式成果呈現方式。下列圖組是從文獻常見做法推導出的本專案規格，並非把文獻個案的物理假設直接移植至本研究。

主要讀者為教授與博士級計畫主持人，因此圖表必須同時回答：模型輸入與假設為何、粒子可能從何處而來、需時多久、季節／潮況／物性如何改變結論、結果是否收斂，以及資料或數值失敗是否造成偏誤。

## 2. 文獻中的成果呈現方式

| 研究 | 文獻採用的主要呈現 | 對本專案的可用設計 |
|---|---|---|
| [van Sebille et al. (2018), *Lagrangian ocean analysis: Fundamentals and practices*](https://doi.org/10.1016/j.ocemod.2017.11.008) | 以粒子位置或「每粒子每格只計一次」建立網格密度／機率圖，並以 transit-time distribution 呈現傳輸時間；文中亦特別區分停留造成的高計數與實際訪格粒子比例 | `pathway_unique_particle_fraction` 與 `residence_time` 必須分圖；旅行時間以分布而非單一平均值呈現 |
| [Carlson et al. (2017), *Combining Litter Observations with a Regional Ocean Model...*](https://doi.org/10.3389/fmars.2017.00078) | 少量代表逆向／正向軌跡依漂移時間著色；以「觀測 transect × 海岸段」矩陣顯示到達比例；另有網格粒子數圖，以及擱淺／漂浮／開放邊界比例與 bootstrap 信賴區間表 | 代表軌跡只用於說明機制；主結論改用來源—受體矩陣、空間密度、停止結果比例與 CI |
| [Ko et al. (2018), *Monitoring multi-year macro ocean litter dynamics and backward-tracking simulation...*](https://doi.org/10.1088/1748-9326/aaaf21) | 以南海區域路徑圖搭配月份、可能來源區與 windage／深度案例的色帶或比例，並列表比較各來源區漂移日數；同時以觀測的月／年統計驗證季節性敘事 | 台灣周邊成果應用共同底圖做季節小多圖，並同步呈現來源區比例與 travel-time 統計，不只展示單一路徑 |
| [Rypina et al. (2014), *Drifter-based estimate of the 5 year dispersal of Fukushima-derived radionuclides*](https://doi.org/10.1002/2014JC010306) | 將「到達／訪格機率圖」與「平均首次抵達時間圖」配對，使高可能路徑與傳輸速度可同時解讀 | 每張主要 pathway footprint 應配一張 first-passage-time 圖，避免把高密度誤讀為快速到達 |
| [Cedarholm et al. (2019), *Investigating Subsurface Pathways of Fukushima Cesium in the Northwest Pacific*](https://doi.org/10.1029/2019GL082500) | 對 subsurface backward/forward trajectories 並列每格平均到達日期與軌跡通過百分比，直接比較逆向與正向結果 | 三維成果除平面密度外，需加入深度—時間剖面；合成驗證應並列 forward/backward footprint 與時間差異 |
| [Pierard et al. (2022), *Attribution of Plastic Sources Using Bayesian Inference...*](https://doi.org/10.3389/fmars.2022.925437) | 以多來源 likelihood／posterior 地圖、指定採樣點的來源權重—粒子年齡曲線與標準差陰影，以及沿岸來源比例堆疊圖呈現 | 若未加入來源先驗，只沿用版式而稱為「條件式足跡／相對權重」；受體可用來源權重—回溯時間曲線與不確定性帶 |
| [Hammoud et al. (2021), *Moving source identification in an uncertain marine flow...*](https://doi.org/10.1016/j.oceaneng.2020.108435) | ensemble 流場產生 forward probability map，再由 backward inverse maps 與目標函數辨識來源位置／時間，並檢查觀測面積、觀測時間與流場變異等敏感度 | 逆向 footprint 必須伴隨來源時間與物理參數敏感度；若日後有已知候選來源，可另建 objective/ranking，而非只靠目視熱區 |
| [Assessing ocean ensemble drift predictions by comparison with observed oil slicks (2023)](https://doi.org/10.3389/fmars.2023.1122192) | 以 rank histogram 與 spread–error relationship 驗證 ensemble 是否能表達 forcing 不確定性 | 若導入 forcing ensemble，需額外報告 reliability／spread–error，不能只畫 ensemble 外包絡 |

文獻的共同結構是「代表軌跡說明傳輸機制，統計密度或連通矩陣承載主結論，旅行時間與不確定性限制解讀」。因此本專案不將每區 `10,000×M`、全案 `40,000×M` 條軌跡全部疊在單張圖上；大量軌跡以聚合統計呈現，原始軌跡仍保留供追溯。

## 3. 名詞與統計量的呈現界線

### 3.1 條件式足跡，不預設為後驗來源機率

本專案從已知受體、到達時間與物性逆推，若沒有各候選來源的先驗排放量、觀測 likelihood 與抽樣努力量，正規化 KDE 只能稱為：

- 條件式來源足跡（conditional source footprint）；
- 相對來源權重（relative source weight）；
- 逆向粒子訪格比例或停留密度。

只有另立 Bayesian 方法版本、明定 prior、likelihood、候選來源全集與觀測模型後，圖例才能使用 posterior probability。任何圖說均不得把「軌跡較常出現」寫成「該處確定為來源」。

### 3.2 三種容易混淆的空間統計必須分開

1. **訪格比例**：每一 member 在每格最多計一次，分母是有效 members；表示多少比例的條件式軌跡曾通過該格。
2. **停留時間密度**：累積每條軌跡在格內的實際時間；可凸顯滯留區，但不能解讀成獨立粒子數。
3. **首次抵達時間**：對第一次進入每格或邊界段的回溯年齡取 median、IQR 或其他分位數；表示傳輸時間尺度。

三者可共用空間格網，但不得共用未說明的「density」欄位或圖名。

## 4. 正式核心圖組

### F01 研究範圍、forcing 與受體設計圖

- A-D 四個分析海域、對應的四個 forcing domains、每區 20／全案 80 個 receptors、受體深度／HAB、開放邊界段、海岸與水深。可另以次要符號標示貢寮、龜山島、新竹、後灣海生館與連江五個調查位置，但不得把它們畫成五個研究區域。
- inset 顯示台灣周邊相對位置；局部圖以各 domain 的 metric CRS 呈現。
- 圖例與表格確認 A-D 各有 20 個受體，並列出 A 區 20 個受體在貢寮／龜山島子地點間的實際分配。
- companion panel 顯示 2024-2025 forcing 可用率與缺口，不將資料缺口區畫成低來源區。

### F02 方法、情境矩陣與計數圖

- 概念圖顯示 OCM current、NWW3-derived Stokes、浮沉速度、Kh/Kz、反向時間、海面／海床／海岸／開放邊界事件。
- 情境方塊明列每區 `10 materials × 20 receptors × 50 arrival times = 10,000 base scenarios`，以及四區合計 `4 × 10,000 = 40,000`。
- 另列 `M` 為每情境隨機 members；單一 experiment case 的總軌跡數為 `sum(M_s)`，一致 M 時為每區 `10,000×M`、全案 `40,000×M`。
- no-Stokes、domain expansion 等以獨立 experiment cases 顯示，不混入基礎情境數。

### F03 代表性三維逆向軌跡

每個分析海域選少量預先登錄的代表案例；A 區必要時分列貢寮與龜山島子地點。每個案例至少使用兩個同步 panel：

1. plan-view 地圖：軌跡按 backward age 著色，受體以星號、first exit／停止點以不同符號表示；背景含水深與主要流向。
2. depth–time 或沿軌跡距離–深度剖面：顯示海面、海床、粒子 `z`、浮沉類別、first bed contact 與有效水深。

可附 3D 透視圖作補充，但不能作唯一三維成果，因透視遮蔽與視角會妨礙定量比較。全體軌跡只畫經固定 seed 或分位數規則抽出的代表 subset，圖說標示抽樣規則與 `n/N`。

### F04 邊界條件式來源足跡

- 主圖疊加 raw first-exit points、2D KDE／訪格比例與 50%、75%、90% HDR contours。
- companion panel 沿 open-boundary arc length 顯示 1D density／比例，標註 boundary segment 與 bootstrap CI。
- 至少比較三種 bandwidth；正文使用預先核定者，補充資料呈現敏感度。
- 圖說列出分母（所有有效 members 或成功 exits）、raw `n`、成功率、bandwidth、格網解析度、experiment case 與回溯上限。

### F05 路徑訪格比例與首次抵達時間配對圖

- 左圖：每 member 每格只計一次的訪格比例，凸顯共同傳輸走廊。
- 右圖：相同有效格的 median first-passage time；可加 IQR 或不確定性遮罩。
- 兩圖 extent、海岸、水深與 mask 完全一致。訪格比例可用 sequential 或經合理門檻的 log normalization；時間圖必須使用具單位的連續色階。
- 低樣本格以 hatch／透明遮罩表示，不用鮮明色彩暗示可靠結論。

### F06 來源—受體連通矩陣

- rows 為 receptor 或 receptor group，columns 為 boundary/source segments；cell 為在該受體條件下到達該來源段的相對權重或比例。
- baseline 使用 receptor-row normalization，使每列分母清楚；另表保存 raw count 與有效 member 數。
- 若比較 10 種物性、季節或潮況，使用 small multiples 或差異矩陣，不將所有維度塞入單張 unreadable heatmap。
- 可依地理弧長排序 source segments，避免聚類排序破壞空間連續性；若採 clustering，須另保留地理順序版本。

### F07 旅行時間分布

- 以 ECDF、violin/box 或 ridgeline 呈現從受體回溯至 boundary/source segment 的 first-passage time distribution。
- 報告 median、IQR、5–95% interval、censoring/max-age 比例與有效 `n`；分布偏斜時不只報平均值。
- 可依地點／季節／物性分面，但所有比較 panel 共用 x 軸範圍與單位。
- 未出界 members 屬 right-censored 或獨立停止類別，不得從分母靜默刪除。

### F08 季節、潮況與到達時間小多圖

- 以四季 × 大／小潮建立固定 4×2 small multiples；同一比較組使用共同 extent、色階範圍、bandwidth 與 denominator 定義。
- 每 panel 顯示來源足跡或來源段相對權重，並標示實際 arrival-time 數、members 與 forcing coverage。
- 每區 50 個到達時間的個別結果保留在 supplement／scenario browser；正文以分區分層摘要與代表個案呈現。

### F09 物性、垂向行為、停留與底部接觸

- 10 種浮沉速度以有物理順序的 small multiples 或「速度 × 指標」曲線呈現，不用十種任意類別色造成辨識負荷。
- 對沉降／近底類別並列 first bed-contact density、repeated contact 或 deposited fraction；對上浮／懸浮類別並列 surface-contact 與 vertical occupancy。
- 深度分布使用 depth–time heatmap 或 quantile ribbon，明示 z positive-up、深度／HAB 基準及水深變化。
- 不具再懸浮參數時，圖名只能使用 contact/deposition-under-assumed-policy，不能宣稱完整底床沉積動力。

### F10 停止結果、資料品質與失敗圖

- 以 stacked bars 顯示 open exit、coast contact、bed deposition、forcing start、max age、data gap、numerical failure 的比例及 raw count。
- 另畫 data-gap／numerical-failure density，使空間集中失敗可被辨識。
- 按 domain、receptor、material、arrival strata 分層；任何一層的成功率顯著較低時，主要 footprint 必須附 reliability 警示。

### F11 物理敏感度與不確定性

- no-Stokes、deep-water、finite-depth、Kh/Kz、dt、domain 與 boundary policy 以 baseline map、difference map 及來源排名變化並列。
- difference map 使用以 0 為中心的 diverging palette；每個 panel 使用相同上／下限。
- 來源段權重以 point-range／forest plot 顯示 bootstrap interval；另報 top-k rank stability 或 HDR overlap。
- 若使用 forcing ensemble，加入 rank histogram 與 spread–error；隨機 member spread 不能被誤稱為所有模式誤差。

### F12 收斂與合成驗證

- member convergence：x 軸為 `M`，y 軸至少含 top-source weight、HDR area／overlap、median travel time、pathway distance；標出選定最小合格 M。
- dt convergence：比較 `dt`、`dt/2`、必要時 `dt/4` 的關鍵指標與計算成本。
- known-source synthetic：並列真實 source、forward footprint、backward footprint、HDR coverage 與位置／時間誤差。
- NumPy/Numba、checkpoint/restart、domain expansion 以簡潔表或差異圖呈現，不以「測試通過」文字取代定量誤差。

## 5. 正式統計表

| 表號 | 內容 | 必要欄位 |
|---|---|---|
| T01 | forcing 與研究範圍 inventory | domain、month、time span、resolution、units、coverage、gap、schema/checksum |
| T02 | 四區各 10×20×50 情境設計與 coverage | region、每區 receptor=20、全案 receptor=80、每區組合數 10,000、全案組合數 40,000、缺列／重列數、experiment cases、M |
| T03 | run 與停止結果摘要 | released/effective/completed、各 event raw count/percentage、wall time、particle steps、I/O、failure rate |
| T04 | 來源段／受體排名 | receptor、source segment、raw n、有效分母、relative weight、bootstrap CI、median travel time、rank stability |
| T05 | 物理敏感度 | case、參數、HDR overlap、排名變化、travel-time 差、bottom-contact 差、限制 |
| T06 | 數值與科學驗證 | analytic error、dt ratio、selected M、synthetic coverage/error、restart/checksum、NumPy/Numba difference |

所有百分比欄位須同時提供 raw count 與 denominator；四捨五入後總和不等於 100% 時，在表註說明。

## 6. 圖面設計與出版規格

1. 地圖在指定 metric CRS 計算，標示經緯度網格、比例尺、projection、海岸、水深與有效 domain；多 panel 比較使用相同 extent。
2. 採色盲友善色盤。backward age／travel time 使用 sequential palette；signed difference 使用以零為中心的 diverging palette；類別色只用於少量、固定語意的狀態。
3. 不使用彩虹色盤。若密度跨多個數量級，可使用 log normalization，但圖例必須明示轉換、零值與下限。
4. 核心比較圖使用共同 color limits；若為呈現局部細節而改變範圍，必須在 panel 與圖說明示，並另提供共同尺度版本。
5. 每張圖的 caption／sidecar 至少記錄：run ID、commit、config/input hash、資料期間、receptor/material/arrival strata、experiment case、M、有效 `n`、denominator、統計定義、格網／bandwidth、單位、CRS、CI/HDR 定義與已知限制。
6. 主成果輸出 PDF/SVG 與 300 dpi PNG；互動圖或 MP4 是補充產品，不能取代可列印的靜態圖與 underlying data。
7. 圖中使用可讀的繁體中文或一致英文學術標籤；數學符號、單位與方向慣例全案一致。
8. 所有 figure 都由資料產品與版本化腳本重製，不接受以繪圖軟體手動移動資料點或修改數值標籤。

## 7. 產出結構與可重現性

```text
figures/
├── figure_registry.json
├── main/
│   ├── F01_study_area.*
│   ├── F02_method_and_scenario_design.*
│   ├── F03_representative_3d_paths.*
│   ├── F04_conditional_source_footprint.*
│   ├── F05_pathway_and_first_passage_time.*
│   ├── F06_source_receptor_connectivity.*
│   ├── F07_travel_time_distribution.*
│   ├── F08_season_tide_small_multiples.*
│   ├── F09_vertical_and_bed_contact.*
│   ├── F10_outcomes_and_failures.*
│   ├── F11_sensitivity_and_uncertainty.*
│   └── F12_convergence_and_validation.*
├── supplement/
├── tables/
├── caption_sidecars/
└── data_sidecars/
```

`figure_registry.json` 為每張圖保存 figure ID、檔案、產製命令、輸入 aggregate checksum、caption sidecar 與狀態。每張主圖需有最小可重繪的 data sidecar，避免讀者必須重新掃描全部 ragged trajectories。

## 8. 驗收清單

正式報告送審前逐項確認：

- [ ] 每區 10,000、全案 40,000 是基礎情境數；`M`、experiment case、每區與全案總軌跡數分欄呈現。
- [ ] 沒有以全軌跡疊圖取代密度、連通、時間與不確定性分析。
- [ ] pathway visit fraction、residence time 與 first-passage time 的定義和單位分開。
- [ ] 所有比例／KDE/HDR 標明 raw `n`、有效分母、成功率與 bandwidth。
- [ ] 季節、潮況、物性與敏感度比較使用共同 extent 與 color limits。
- [ ] max-age、data-gap 與 numerical failure 沒有被當作低來源密度。
- [ ] 未具 prior/likelihood 時，不使用 posterior probability 或確定來源措辭。
- [ ] member、dt、domain 與已知來源合成驗證均有定量圖表。
- [ ] 每張圖具 registry、caption sidecar、data sidecar 與可重製命令。
- [ ] PDF/SVG、300 dpi PNG、表格與 release manifest checksum 全數驗收。
