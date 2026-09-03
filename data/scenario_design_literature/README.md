# 情境設計因子之文獻備查

## 使用目的與結論範圍

本資料夾保存「三、Lagrangian 系集逆向溯源」之情境設計所引用的開放取用原文，及其紅框標註副本。紅框只標示本專案實際採用的證據段落，原文未改動；`originals/` 是下載時的原始版本，`annotated/` 則是供審查時快速核對的副本。

本次查證針對的問題是：在沒有特定廢棄物粒徑、密度、生物附著或現地釋放紀錄時，是否有研究支持把下列條件明確地拆分為 Lagrangian 情境因子：

- 粒子垂向速度與垂向混合，特別是沉降速度敏感度；
- 到達（受體觀測）時間、季節性與漂浮／回溯時間；
- 水平受體位置；
- 垂向受體層位與三維邊界。

結論是肯定的。四篇研究均顯示這些條件會改變傳輸途徑、來源歸因或不確定性，因而應以可識別、可重現的情境條件處理。

**本資料夾不主張文獻規定本計畫書的 `10×20×50` 數量。** 該乘積是計畫書所定的樣本結構；文獻所支持的是將材料／行為、受體空間位置與到達時間分開處理的科學方法。專案最新基線依研究主持人裁決，將此原則具體化為 10 個只含負值的海廢材質／形狀代理與沉降格點、5 個水平位置 × 4 個垂向層位，以及 48+2 個分層到達時間；確切規則請參閱 [五站點情境與巢狀邊界設計基線](../../docs/08_design_baseline_and_derived_gates.md)。

## 紅框對照與本專案採用方式

| 設計因子 | 紅框原文與頁碼 | 可支持的研究判讀 | 本專案採用方式 |
|---|---|---|---|
| 沉降速度與垂向混合 | `van_der_molen_2021_north_sea_microplastics_marked.pdf`，第 5 頁三框 | 正／負浮力、垂向速度、垂向混合及水平擴散均會改變近表／近底分布；作者對沉降 PS 顆粒採 `-0.0015、-0.004、-0.006 m/s` 並比較範圍端點與中點。 | 最新基線只保留負值，將十個 `settling_velocity_mps` 視為未校準敏感度格點；文獻支持量級與敏感度方法，但不支持把某一速度直接指定給 iOcean 的大型海廢類別。 |
| 到達時間、季節性與漂浮時間 | `van_duinen_2022_beached_plastics_backtracking_marked.pdf`，第 1 頁兩框 | 來源具有顯著季節變異，且漂浮時間是逆向來源歸因的主要不確定性。 | 50 個到達時間不可只是任意等距時間戳；依兩年、季節、大小潮與潮內相位分層，另補高浪與強流事件。 |
| 水平受體位置與觀測時間 | `carlson_2017_adriatic_floating_debris_marked.pdf`，第 2 頁兩框 | 浮游廢棄物的觀測位置、時間與豐度用來設定軌跡初始條件；距模式邊界過近的觀測點被排除。 | 每站點以固定 seed 的 metric-space maximin 產生 5 個 persistent-wet 水平受體位置，並在受體核心／local domain 內保持可用空間緩衝。 |
| 垂向層位與三維受體條件 | `pierard_2024_abyssal_nanoplastics_backtracking_marked.pdf`，第 4 頁兩框 | 三維回溯使用明確 z 層、受體起始水平位置分布與指定初始深度；深度與海床／海面邊界是模型條件，不能視為二維表面問題。 | 每個水平位置配置 4 個由相對水深與最低有效 OCM 層位決定的垂向受體；實際 z 必須在所有 50 個時次可被 `zcor` 有效夾擠，不作垂向外插。 |

## 文件清單與追溯資訊

| 編號 | 文件 | 原始公開來源與 DOI | 授權／使用提醒 | SHA-256（原始／標註） |
|---|---|---|---|---|
| 1 | `originals/carlson_2017_adriatic_floating_debris.pdf`；`annotated/carlson_2017_adriatic_floating_debris_marked.pdf` | [Carlson & Aliani (2017), Frontiers in Marine Science](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2017.00078/full), [PDF](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2017.00078/pdf), DOI: [10.3389/fmars.2017.00078](https://doi.org/10.3389/fmars.2017.00078) | Frontiers 開放取用文章；標註副本僅增加紅色外框。 | `3137cd1d3e5612958b9040f731a8a840be8c6e2591130a2a59a36e3c95bdd76d` / `867097d8b11a3e4ab7daeba8c307984330d7304a037dd4be3d46a7b8d4c6506b` |
| 2 | `originals/van_der_molen_2021_north_sea_microplastics.pdf`；`annotated/van_der_molen_2021_north_sea_microplastics_marked.pdf` | [van der Molen et al. (2021), Frontiers in Marine Science](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2021.607203/full), [PDF](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2021.607203/pdf), DOI: [10.3389/fmars.2021.607203](https://doi.org/10.3389/fmars.2021.607203) | Frontiers 開放取用文章；標註副本僅增加紅色外框。 | `b9550f00d98a4f02eae20359d4712a5f3adcecf0f42da3579c59e21f8c6fa6ad` / `05b61cb575d1fe39c5a575249afa554ffbb3d88ca77eef18e40db95e2dc33b67` |
| 3 | `originals/van_duinen_2022_beached_plastics_backtracking.pdf`；`annotated/van_duinen_2022_beached_plastics_backtracking_marked.pdf` | [van Duinen, Kaandorp & van Sebille (2022), *Geophysical Research Letters*](https://doi.org/10.1029/2021GL097214), [開放 PDF（VLIZ）](https://www.vliz.be/imisdocs/publications/372490.pdf) | 原文首頁標示 CC BY-NC；保存於本機計畫備查，後續使用仍須保留作者、DOI 與授權條件。 | `f749859372dcacbff1fc43739c477e69dc13ac0e9c01b803b6dda24d3b3be29b` / `b6349d2fc3ed9e12df3abd22128afd6012639d72ec0697a0b72b48d23e6d31b7` |
| 4 | `originals/pierard_2024_abyssal_nanoplastics_backtracking.pdf`；`annotated/pierard_2024_abyssal_nanoplastics_backtracking_marked.pdf` | [Pierard, Meirer & van Sebille (2024), *Ocean and Coastal Research*](https://doi.org/10.1590/2675-2824072.24008), [Utrecht University Repository PDF](https://research-portal.uu.nl/ws/files/247227647/Identifying_the_origins_of_nanoplastics_in_the_abyssal_South_Atlantic_using_backtracking_Lagrangian_simulations_with_fragmentation.pdf) | 出版者版本由機構典藏公開；保存時保留原始來源及其再利用條款。 | `28edb67acbf10ab5dbc21b24437603ad922a7fa14540830ac79857036582f72b` / `b4a08abb1a879479bfde4584e9afd9fc9f80a1c164b7f17257bf7ef82f4c5d8e` |

## 完整參考文獻

1. Carlson, D. F., & Aliani, S. (2017). *Combining Litter Observations with a Regional Ocean Model to Identify Sources and Sinks of Floating Debris in a Semi-enclosed Basin: The Adriatic Sea*. **Frontiers in Marine Science, 4**, 78. https://doi.org/10.3389/fmars.2017.00078
2. van der Molen, J., van Leeuwen, S. M., Govers, L. L., van der Heide, T., & Olff, H. (2021). *Potential Micro-Plastics Dispersal and Accumulation in the North Sea, With Application to the MSC Zoe Incident*. **Frontiers in Marine Science, 8**, 607203. https://doi.org/10.3389/fmars.2021.607203
3. van Duinen, B., Kaandorp, M. L. A., & van Sebille, E. (2022). *Identifying Marine Sources of Beached Plastics Through a Bayesian Framework: Application to Southwest Netherlands*. **Geophysical Research Letters, 49**, e2021GL097214. https://doi.org/10.1029/2021GL097214
4. Pierard, C. M., Meirer, F., & van Sebille, E. (2024). *Identifying the origins of nanoplastics in the abyssal South Atlantic using backtracking Lagrangian simulations with fragmentation*. **Ocean and Coastal Research, 72**, e24043. https://doi.org/10.1590/2675-2824072.24008

## 完整性與再標註

- 原始與標註檔的 SHA-256 已列於上表。若重新下載、壓縮、修改紅框或更新論文版本，雜湊值必定改變，應同步更新本 README。
- 紅框對應的頁碼與段落語意已列於「紅框對照與本專案採用方式」。若日後改用不同版次的 PDF，必須重新以該頁的實際版面標註，不能沿用現有紅框位置。
- 標註 PDF 已以 Poppler 重新渲染並逐頁目視核對第 2、5、1、4 頁；框線未遮蔽正文，且輸出頁數分別為 16、19、9、17 頁。
