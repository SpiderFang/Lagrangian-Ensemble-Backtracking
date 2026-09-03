# Lagrangian 成果呈現與驗證之學術文獻備查

## 使用目的與結論範圍

本資料夾依既有文獻封存規格保存八篇與 Lagrangian 軌跡、來源足跡、到達時間、來源—受體連通、不確定性及數值誤差相關的學術文獻。`originals/` 保存下載時的完整原文；`annotated/` 保存只增加紅色外框的審查副本。紅框涵蓋作者原有的圖、表、圖說或必要說明，用來讓計畫主持人快速核對「文獻如何呈現成果」，不修改原文數值或作者結論。

這批文獻支持的是**成果呈現與驗證方法**，不是本專案正式 SERVER 運算已完成的證明。目前若尚未建立來源先驗、觀測似然與獨立驗證，逆向系集輸出只能稱為「條件式來源足跡」或「相對來源權重」，不可把一般粒子訪問比例寫成「後驗來源機率」，也不可宣稱因果歸因。正式成果仍須通過本專案既定的輸入 manifest、缺時、數值失敗、幾何、聚合與報告發布門檻。

## 由文獻歸納的成果呈現結構

| 成果問題 | 建議圖表組合 | 分母、語意與限制 | 對應本專案規劃 |
|---|---|---|---|
| 個別粒子如何由受體回溯 | 少量具代表性的軌跡，以時間或粒子年齡著色；不得用全部軌跡製造視覺密度假象 | 圖中需交代選樣規則、起點／受體、時間方向與停止狀態 | F03 代表性三維路徑 |
| 哪些區域經常被系集訪問 | 粒子位置計數圖或「至少訪問一次」的成員比例圖，並在圖說明確寫出分母 | 這是條件式訪問足跡；兩種分母不能混稱為同一種機率 | F04 條件式來源足跡與 F06 來源—受體連通 |
| 何時首次到達或經過某區域 | 空間訪問比例與平均／中位首次到達時間並列，另以直方圖或累積曲線呈現通行時間分布 | 未到達者不得填零；時間統計的母體須限於已到達成員，並另報未到達比例 | F05 首次通過時間與 F07 傳輸時間分布 |
| 季節、月份、來源類別是否不同 | 月份／季節 small multiples、長條圖、來源比例及誤差棒 | 必須同時報樣本數、分母與不確定性，不以單一月份個案代表全年 | F08 季節／潮相與 F09 材質比較 |
| 模式系集是否包絡觀測 | 觀測熱圖疊加系集等值線或信賴橢圓，搭配 rank histogram、spread–error 或命中率 | 只有具獨立觀測與相同配對定義時才能執行；不存在觀測時不得虛構驗證 | F11 敏感度與不確定性 |
| 逆向結果是否受數值方向偏差影響 | 正向—逆向配對分離誤差隨時間曲線、空間誤差圖、時間步與插值敏感度 | 即使積分器可重現，非零散度與邊界處理仍可能形成穩定性偏差 | F12 合成驗證與數值誤差診斷 |

## 紅框頁面與專案採用方式

下表的「第 n 頁」一律指 PDF 閱讀器顯示頁碼，不用期刊印刷頁碼取代；期刊頁碼另置於括號，避免不同版次對頁時發生偏移。

| 文獻 | 紅框頁面 | 文獻呈現方式 | 本專案可採用的方式與必要限制 |
|---|---|---|---|
| Carlson et al. (2017) | 第 4 頁（期刊頁 4）Table 2、Figure 2；第 6 頁（期刊頁 6）Figure 4 | 以少量時間著色的逆向軌跡說明路徑；以受體 transect × 海岸段矩陣表達來源—受體連通；表格分開列出擱淺、仍漂浮、開放邊界、平均時間與 bootstrap 95% 信賴區間。 | 代表軌跡用於 F03、連通矩陣用於 F06、狀態與信賴區間格式用於 F10/F11；本案還必須把乾點、域外、資料缺口與數值失敗分開，不可合併為「未到達」。 |
| Cedarholm et al. (2019) | 第 6 頁（期刊頁 6826）Figure 4 | 將平均到達日期與穿越網格機率配成左右欄，並並列成功逆向、南／北側正向試驗。 | F05 應把訪問比例與首次到達時間並列；若沒有先驗與似然，本案圖名使用「visit fraction／條件式來源足跡」，不沿用後驗機率語意。 |
| de Aguiar et al. (2023) | 第 8 頁 Figure 4；第 12 頁 Figure 7 | 觀測油膜熱圖與模式系集輪廓／邊際分布同圖比較，再用二維 rank histogram 判讀過度離散、一致或低度離散。 | 若本案取得可配對的現地或遙測觀測，可將此組合納入 F11；若沒有觀測，只能報系集內部 spread、敏感度與收斂，不能稱為模式驗證。 |
| Ko et al. (2018) | 第 6 頁（期刊頁 5）Figure 2；第 10 頁（期刊頁 9）Figure 5 | 年／月重量與件數以長條圖呈現，月平均附 ±1 SE；來源國比例以月份分組並附誤差棒，讓季節差異與來源組成同時可讀。 | F08/F09 可採用同一月份軸與誤差棒的比較形式；本案來源類別必須是已定義的邊界段或來源區，不得把海廢清除占比直接當作來源先驗。 |
| Pierard et al. (2022) | 第 6 頁 Figure 3、Figure 4 | 在明確來源先驗與 Lagrangian likelihood 下產生後驗來源圖；另以粒子年齡曲線、標準差陰影與有效粒子數同圖表達時間依賴及資料支撐度。 | 這是本案未來若完成正式貝氏層後可採用的 F04/F07/F11 格式；在此之前不得把條件式訪問足跡稱為 posterior probability。 |
| Reijnders et al. (2026) | 第 15 頁（正文 14/24）Figure 6；第 19 頁（正文 18/24）Figure 9 與結論段 | 用平均配對分離誤差與時間方向擴散率比較積分／插值方法，再以空間圖顯示正—逆向偏差集中位置。 | F12 應納入 forward–backward closure、時間步、速度場更新間隔、二維／三維及邊界處理敏感度；這是誤差診斷要求，不是對所有逆向方法的全面否定。 |
| Rypina et al. (2014) | 第 5 頁（期刊頁 8181）Figure 3 | 對同一組軌跡並列空間訪問機率圖與對應 travel-time 圖，並說明網格大小、首次訪問計數與樣本數對統計穩定度的影響。 | F04/F05 應使用相同空間格網並列訪問分率與首次通過時間；網格解析度與最小有效成員數必須寫入圖說與報告資料契約。 |
| van Sebille et al. (2018) | 第 14 頁（期刊頁 62）Figure 4；第 15 頁（期刊頁 63）Figure 5 | 明確區分「所有粒子位置計數後正規化」與「曾至少訪問一次的不同粒子比例」兩種 probability map，並以粒子年齡直方圖與累積曲線表示 transit-time distribution。 | 直接支援本案聚合分母政策、F04 來源足跡及 F07 傳輸時間分布；每張圖都必須把粒子、時間步或成功到達成員的分母寫清楚。 |

## 文件清單、來源與完整性

| 編號 | 原始／紅框副本 | 公開來源與 DOI | 授權／使用提醒 | SHA-256（原始／標註） |
|---|---|---|---|---|
| 1 | `originals/carlson_2017_adriatic_floating_debris.pdf`；`annotated/carlson_2017_adriatic_floating_debris_marked.pdf` | [Frontiers 全文](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2017.00078/full)、[PDF](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2017.00078/pdf)、DOI [10.3389/fmars.2017.00078](https://doi.org/10.3389/fmars.2017.00078) | CC BY；使用時仍須標示完整作者與原始出版資訊。 | `3137cd1d3e5612958b9040f731a8a840be8c6e2591130a2a59a36e3c95bdd76d` / `5708dace2bcc87d5c3f5d484f653dc31a884ea6082654b56162d5d06624e258c` |
| 2 | `originals/cedarholm_2019_subsurface_fukushima.pdf`；`annotated/cedarholm_2019_subsurface_fukushima_marked.pdf` | [AGU 全文](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019GL082500)、[PDF](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019GL082500)、DOI [10.1029/2019GL082500](https://doi.org/10.1029/2019GL082500) | CC BY-NC-ND；紅框副本僅供本機非商業審查備查，不得直接對外散布修改版。 | `995fb89b6a7f024f5f99d06b30a4eea9787fe7878e124c3b53ad608b8c2f37da` / `5dc3e97b6321cad7ba86433177f59d1f8cd7bdec0567a2b99c5b2756cf62d629` |
| 3 | `originals/de_aguiar_2023_ensemble_drift_oil_slicks.pdf`；`annotated/de_aguiar_2023_ensemble_drift_oil_slicks_marked.pdf` | [Frontiers 全文](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1122192/full)、[PDF](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1122192/pdf)、DOI [10.3389/fmars.2023.1122192](https://doi.org/10.3389/fmars.2023.1122192) | CC BY；重用圖表或改作時須保留作者、出版資訊及授權。 | `220f3e3364fe67d380d1d86ce102ab32abedd90f6bc127fac74e5896e06f5c3c` / `28808e06aad255eab55f154ae5dd8be147eb1fd3642894e50e7a651dcd428c67` |
| 4 | `originals/ko_2018_macro_ocean_litter.pdf`；`annotated/ko_2018_macro_ocean_litter_marked.pdf` | [國立臺灣大學公開 PDF](https://homepage.ntu.edu.tw/~cyko235/papers/2018_Monitoring%20multi-year%20macro%20ocean%20litter%20dynamics.pdf)、DOI [10.1088/1748-9326/aaaf21](https://doi.org/10.1088/1748-9326/aaaf21) | CC BY 3.0；任何再利用須維持引用與授權說明。 | `3618d262010c54465946c1b233d4e46139a202706c609ce4bb65f070d7f3bb7d` / `896dfc42b794cb91d8098fa6414ecc0d11b5cf1c0d4dad8583ed66cb4517797a` |
| 5 | `originals/pierard_2022_bayesian_plastic_sources.pdf`；`annotated/pierard_2022_bayesian_plastic_sources_marked.pdf` | [Frontiers 全文](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.925437/full)、[PDF](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.925437/pdf)、DOI [10.3389/fmars.2022.925437](https://doi.org/10.3389/fmars.2022.925437) | CC BY；後驗機率語意只能在相同的先驗與 likelihood 契約成立時引用。 | `de089bb6ac9d9df9190dda57ba509f13cf0ab746716562eb93fbfaa5105aefda` / `e55ba2c35de9fb1a69350dc372c8113866f29e4c9e02ab501854ce00999e7207` |
| 6 | `originals/reijnders_2026_stability_bias.pdf`；`annotated/reijnders_2026_stability_bias_marked.pdf` | [Utrecht University Repository](https://dspace.library.uu.nl/handle/1874/480130)、[AGU 全文](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025MS005470)、DOI [10.1029/2025MS005470](https://doi.org/10.1029/2025MS005470) | 出版者版本標示 CC BY；引用時須保留文章版本與 DOI。 | `5b89b74d43cb8b60450c9b2bb7ef8a26c48a243faf0c3d0c9c9d300abeadf95f` / `92547aeed53f7caafd7563a360b3e4046c019c5898725fd32d46c5ec8e8f110d` |
| 7 | `originals/rypina_2014_fukushima_dispersal.pdf`；`annotated/rypina_2014_fukushima_dispersal_marked.pdf` | [WHOI 公開 PDF](https://cafethorium.whoi.edu/wp-content/uploads/sites/9/2019/06/Rypina-et-al-drifter-based-estimates-JGR-Oceans-2014-jgrc20996.pdf)、[AGU 全文](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014JC010306)、DOI [10.1002/2014JC010306](https://doi.org/10.1002/2014JC010306) | 原文標示 All Rights Reserved；全文與紅框副本僅供本機研究備查，對外應提供 DOI／來源連結，不散布標註全文。 | `73016b5bbfc6cbb21e3357cf8dffe43597e9e157a51006751bae43dca2ba44de` / `788ed93f7f930a73da6aafccdbf3542341358e5422fbfb1d645b9d210fe86bb1` |
| 8 | `originals/van_sebille_2018_lagrangian_ocean_analysis.pdf`；`annotated/van_sebille_2018_lagrangian_ocean_analysis_marked.pdf` | [NOAA Repository PDF](https://repository.library.noaa.gov/view/noaa/32404/noaa_32404_DS1.pdf)、[ScienceDirect 文章頁](https://www.sciencedirect.com/science/article/pii/S1463500317301853)、DOI [10.1016/j.ocemod.2017.11.008](https://doi.org/10.1016/j.ocemod.2017.11.008) | CC BY 4.0；任何再利用仍須標示作者、期刊、DOI 與改作說明。 | `86b23deeaa2a4b5128c0e1fd2b826c8abfd8590cd92d675ab7681f028c90a3c1` / `50527689ee175c0638f504f963aec98f4f4875fd66564b729565bbd8c9f7b51a` |

來源網址可能因出版平台調整而變動，DOI 是優先的永久識別碼。PDF 及紅框副本依根目錄 `.gitignore` 留在本機 `data/`，只有本索引可納入版本控制；這既避免大型全文進入程式庫，也降低誤散布受限版本的風險。

## 完整參考文獻

1. Carlson, D. F., Suaria, G., Aliani, S., Fredj, E., Fortibuoni, T., Griffa, A., Russo, A., & Melli, V. (2017). *Combining Litter Observations with a Regional Ocean Model to Identify Sources and Sinks of Floating Debris in a Semi-enclosed Basin: The Adriatic Sea*. **Frontiers in Marine Science, 4**, 78. https://doi.org/10.3389/fmars.2017.00078
2. Cedarholm, E. R., Rypina, I. I., Macdonald, A. M., & Yoshida, S. (2019). *Investigating Subsurface Pathways of Fukushima Cesium in the Northwest Pacific*. **Geophysical Research Letters, 46**, 6821–6829. https://doi.org/10.1029/2019GL082500
3. de Aguiar, V., Röhrs, J., Johansson, A. M., & Eltoft, T. (2023). *Assessing ocean ensemble drift predictions by comparison with observed oil slicks*. **Frontiers in Marine Science, 10**, 1122192. https://doi.org/10.3389/fmars.2023.1122192
4. Ko, C.-Y., Hsin, Y.-C., Yu, T.-L., Liu, K.-L., Shiah, F.-K., & Jeng, M.-S. (2018). *Monitoring multi-year macro ocean litter dynamics and backward-tracking simulation of litter origins on a remote island in the South China Sea*. **Environmental Research Letters, 13**, 044021. https://doi.org/10.1088/1748-9326/aaaf21
5. Pierard, C. M., Bassotto, D., Meirer, F., & van Sebille, E. (2022). *Attribution of Plastic Sources Using Bayesian Inference: Application to River-Sourced Floating Plastic in the South Atlantic Ocean*. **Frontiers in Marine Science, 9**, 925437. https://doi.org/10.3389/fmars.2022.925437
6. Reijnders, D., Denes, M. C., Rühs, S., Breivik, Ø., Nordam, T., & van Sebille, E. (2026). *Stability Bias in Lagrangian (Back)tracking in Divergent Flows*. **Journal of Advances in Modeling Earth Systems, 18**, e2025MS005470. https://doi.org/10.1029/2025MS005470
7. Rypina, I. I., Jayne, S. R., Yoshida, S., Macdonald, A. M., & Buesseler, K. (2014). *Drifter-based estimate of the 5 year dispersal of Fukushima-derived radionuclides*. **Journal of Geophysical Research: Oceans, 119**, 8177–8193. https://doi.org/10.1002/2014JC010306
8. van Sebille, E., Griffies, S. M., Abernathey, R., Adams, T. P., Berloff, P., Biastoch, A., Blanke, B., Chassignet, E. P., Cheng, Y., Cotter, C. J., Deleersnijder, E., Döös, K., Drake, H. F., Drijfhout, S., Gary, S. F., Heemink, A. W., Kjellsson, J., Koszalka, I. M., Lange, M., ... Zika, J. D. (2018). *Lagrangian ocean analysis: Fundamentals and practices*. **Ocean Modelling, 121**, 49–75. https://doi.org/10.1016/j.ocemod.2017.11.008

## 完整性、視覺 QA 與再標註規則

- 八份原始 PDF 與八份標註 PDF 的 SHA-256 均列於上表。重新下載、線性化、壓縮或修改紅框都會改變雜湊值，屆時必須同步更新本索引。
- 原始／標註頁數已以 Poppler `pdfinfo` 逐對核對，分別保持 16、9、15、14、9、25、17、27 頁，沒有因合併紅框而遺失頁面。
- 所有紅框頁已用 Poppler `pdftoppm` 重新渲染並人工目視檢查；框線未遮蔽原圖、座標軸、圖例、數值或圖說。QA PNG 位於本機暫存目錄，不屬正式研究資料，也不納入 Git。
- 若日後改用接受稿、補充資料或重新排版的出版者版本，必須依新 PDF 的實際顯示頁碼重新定位；不得直接沿用本版紅框座標。
- 任何簡報、報告或論文重製圖表前，仍須依各文章授權判斷是否可改作與散布；本機存在紅框副本不等同取得額外授權。
