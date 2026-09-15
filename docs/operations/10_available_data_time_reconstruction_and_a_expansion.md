# 2024–2025 全部可得資料、時間缺口重建與 A 區擴張決策

> **閱讀提示**
> - 文件類型：資料可用性、時間缺口與 A 區擴張決策。
> - 它回答：全部可得資料如何定義，以及缺時何時重建、何時採缺口安全視窗。
> - 建議先讀：[科學方法與驗證](../foundation/03_scientific_method_and_validation.md)，再讀[SERVER 手冊](06_server_runbook_plan.md)。

## 1. 決策效力與適用範圍

本文件取代先前將資料狀態、供應者 metadata 或時間缺口列為「等待外部補件」的判定。
研究團隊已確認 CWA-OCM 與 CWA-NWW3 現存檔案就是可取得的全部 2024–2025 資料；原始
提供者不可考，不能取得額外欄位定義、forecast-cycle 說明或缺漏檔案。因此，正式研究
母體定義為 **2024–2025 全部可得資料（all available samples）**，成果可稱為「完整且正式
的 2024–2025 全部可得資料分析」，但不得寫成供應者保證無缺時的完整曆年 hindcast，亦
不得宣稱 lexical cycle selection 等同最短 lead time 或最佳預報。

這項決策不代表忽略缺時。正式流程仍須區分觀測／原始模式樣本、統計重建值、域外、乾點
與數值失敗；差別只在於缺口是本專案必須處理及量化的不確定性，不再是等待資料提供者
修復的阻擋事項。

## 2. SERVER 實際時間軸證據

2026-08-20 以唯讀方式重新合併四個 flow domain 的 24 個 OCM 月份時間軸。處理順序與
`OCM-SVD-Analysis` 全部可得資料契約一致：依設定月份順序串接、UTC stable sort，再對
相同 UTC 保留來源序列最後一筆。四區結果完全相同：

| 指標 | OCM 結果 |
|---|---:|
| 月檔原始時間列數 | 17,196 |
| canonical 唯一 UTC | 17,124 |
| 去除重複 UTC | 72 |
| stable-sort 重排位置數 | 210 |
| 2024–2025 理論逐時 UTC | 17,544 |
| 可用率 | 97.606% |
| 研究期間缺時 | 420 |

420 個缺時包含研究期間起點 `2024-01-01 00:00 UTC` 一筆 boundary 缺時，以及 419 個內部
缺時。內部相鄰可用端點時距的分布為：

| 相鄰端點時距 | 區段數 | 每段真正缺少時次 | 合計缺時 |
|---:|---:|---:|---:|
| 2 h | 33 | 1 | 33 |
| 24 h | 1 | 23 | 23 |
| 25 h | 11 | 24 | 264 |
| 26 h | 2 | 25 | 50 |
| 50 h | 1 | 49 | 49 |

先前 preflight 逐月各自檢查再加總，因而把同一時間問題重複表達成 68 個月內 finding、
8 個跨月 finding，並報出未經全期去重的 72 h 最大值。正式報告改以 canonical 全期軸為
唯一時間母體；月份 metadata 仍保留作 provenance，不再用 finding 數量代表缺口數。

OCM 上游另有一層不可省略的時間 provenance：部分 raw 日檔的 `time` 座標整檔錯置時，
前處理器會依 `YYYYMMDD_schout.nc` 檔名日期重新錨定該檔輸出 UTC、保留檔內相對 offset，
並在跨日重疊時只保留嚴格晚於前一已寫入 UTC 的 slice；完全沒有新增 UTC 的日檔記為
`zero_kept`。這是對時間座標的版本化修復，不是對 u/v/w 等流場值做插補。正式 preflight
須逐月保存 `timestamp_repair_file_count`、`zero_kept_file_count`、
`skipped_overlap_time_step_count` 與 `missing_day_count`，blocked validation 也須把高修復率
期間列為獨立子集，避免 canonical 軸看似連續就掩蓋來源時間標籤的不確定性。

NWW3 則須分開判讀。SERVER 的 `nww3_native/ww3_grd3_253x237` 已完整涵蓋
`2024-01-01 00:00` 至 `2025-12-31 23:00 UTC`，恰有 17,544 個唯一逐時 UTC，最大間距
1 h，沒有時間缺口。現行 `nww3_analysis` 的 17,124 筆時間只是因產製時把 OCM surface
UTC 當作 target 軸；波浪來源本身沒有缺掉這 420 筆。因此正式 NWW analysis 應由既有
native 產品與靜態 OCM 共同格網重建完整 17,544 小時，不對 Hs、fp 或 DP 做統計補值。

## 3. 可沿用與不可直接沿用的 OCM-SVD 方法

可直接沿用：

1. `sort_and_deduplicate_prefer_last` 的跨月 UTC 正規化及同步來源索引；
2. 不以零值代替缺流速，並保存重排、去重、缺口與 coverage；
3. 對局部、短而有雙側支撐的缺值進行可追溯插補；
4. 以 blocked cross-validation 決定模式數、模型階數與誤差。

不可直接沿用：SVD 可把各 UTC 當作獨立樣本，移除不完整時次後仍能求模態；Lagrangian
積分則需要粒子沿途每一個 RK stage 都有連續場。先前「整張場完全缺失時 DINEOF 完全
不能重建」的說法過度絕對，現已校正：Alvera-Azcárate et al. (2009) 顯示，若相鄰時次仍
提供資訊，具時間共變異處理的 EOF 重建可以估計某張完全缺失影像的特徵。然而，這不
等於可將未驗證的純 DINEOF 自動套用於 24–49 個連續 OCM snapshots；Hernández-Carrasco
et al. (2018) 的持續嚴重缺口案例顯示，Eulerian 流場看似可接受並不足以保證 Lagrangian
結果可靠。因此本案以多變量 EOF-harmonic-state-space smoother 作候選，並以實際缺口形狀
的 blocked cross-validation 與 trajectory 指標共同決定是否採用；絕不以填零異常冒充重建。

相關方法依據：

- [Beckers and Rixen (2003), EOF Calculations and Data Filling from Incomplete Oceanographic Datasets](https://doi.org/10.1175/1520-0426(2003)020%3C1839:ECADFF%3E2.0.CO;2)：DINEOF、迭代 EOF 重建與交叉驗證選取有效 mode 數。
- [Alvera-Azcárate et al. (2009), Enhancing temporal correlations in EOF expansions](https://doi.org/10.5194/os-5-475-2009)：在 EOF 重建中加入時間相關；顯示時間濾波能降低不連續，並可利用相鄰資訊處理某張完全缺失影像。
- [Frolov et al. (2012), Improved statistical prediction of surface currents based on historic HF-radar observations](https://doi.org/10.1007/s10236-012-0553-5)：以 EOF 表達空間場、以線性自迴歸預測 EOF 時間係數，並以粒子分離誤差驗證 48 h 流場預測。
- [Hernández-Carrasco et al. (2018), Impact of HF radar current gap-filling methodologies on the Lagrangian assessment of coastal dynamics](https://doi.org/10.5194/os-14-827-2018)：同時比較 Eulerian 與 trajectory、LCS、residence-time 誤差；持續嚴重缺口不可因單一 Eulerian 指標合格而自動放行。
- [Delandmeter and van Sebille (2019), The Parcels v2.0 Lagrangian framework](https://doi.org/10.5194/gmd-12-3571-2019)：Lagrangian 計算須先在粒子位置內插 Eulerian 場，再積分粒子 ODE，支持將 forcing 重建與數值積分分層驗證。

上述核心文獻的 DOI 與本專案使用邊界整理於[時間缺口重建文獻索引](../../data/time_reconstruction_literature/README.md)。
該索引不保存原文頁碼、紅框短摘錄、附件保存位置或 checksum；原始 PDF 與標註附件另行
保管且不屬 Git 交付，若要引用紅框段落，須另行核對保存位置與 checksum。本案的潮汐項、
VAR 正則化、雙向 smoother、member 數與 acceptance threshold 均為資料驅動的實作選擇，
不能被誤稱為文獻直接給定的通用參數。

## 4. 正式時間處理方法版本

### 4.1 Phase T0：全期 canonical source index

每個 product/domain 依 `sort_and_deduplicate_prefer_last` 建立唯一 UTC 軸，並對每筆保存
`source_month_id`、`source_local_index`、原始 metadata hash 與 duplicate audit。大型物理
陣列不複製；runtime 依 canonical index 開啟正確月份與樣本。月界 halo 可提供支撐，但
不得由 calendar month 名稱強迫選取另一筆重複 UTC。

### 4.2 Phase T1：NWW 完整逐時 analysis 重建

以既有 17,544 小時 NWW native 為時間來源，以 expanded/static OCM surface grid 為空間
目標，重新產出 Hs、fp、DP、valid mask 與 QC。DP 先轉 circular cos/sin 分量做空間內插，
再轉回 wave-from degree；不得在 359°/1° 之間做一般線性角度平均。每一 UTC 均保存 native
source-cycle audit。這是重新格網化，不是資料插補。

### 4.3 Phase T2：OCM 單一缺時候選方法

33 段單一小時缺時以 u、v、w、elev、zcor 及適當轉換後的 diffusivity 分量做雙側時間
插值。這只是候選方法，必須在完整時段人工遮掉同樣的一小時，以不同季節、潮相及強流
條件做 blocked validation；若 state-space 方法對單一缺時也顯著較佳，正式產品可統一使用
後者。任何插值不得跨研究期間起訖、陸海狀態轉換或沒有雙側有效支撐的 node/face。

### 4.4 Phase T3：OCM 整張場長缺時重建

正式候選為 `ocm_multivariate_eof_harmonic_state_space_smoother_v1`：

1. 在每一 flow domain 以完整可用時段建立多變量、面積／體積加權 EOF 空間基底；u/v
   必須聯合建模，w、elev、垂向座標及 log-transformed Kz 以明示 scaling 納入或建立耦合
   子模型，不能各自任意補值。
2. EOF 係數先以主要日潮／半日潮 harmonic regression 表達可解析週期，再對殘差建立
   regularized vector autoregression/state-space model。這是對 Frolov et al. EOF-AR 方法的
   缺口內插延伸；因缺口左右都有樣本，正式估計使用前向與後向資訊的 Kalman smoother，
   不只作單向 forecast。
3. 對模型 state posterior 抽樣形成 forcing reconstruction members；posterior mean 作
   deterministic reference。重建 member 透過 `member_id` 決定性配對，不另把基礎情境數
   乘上一個未揭露的因子；若獨立執行 reconstruction sensitivity，則以
   `experiment_case_id` 與 `reconstruction_member_id` 明示。
4. wet/dry 採保守政策：只有缺口雙側及 persistent-wet 契約皆為 wet 的 face 才允許重建
   forcing；其餘維持無支撐，不以分類器創造海域。zcor 必須保持由海床至海面的嚴格次序，
   Kz 必須非負且不得超出訓練資料的核定物理範圍；違反限制的 cell/member 標為 reconstruction
   failure，不以截斷後的數值冒充原場。
5. 重建值寫入獨立 immutable patch，不覆寫 OCM 上游 cache。每個值至少保存
   `origin=observed|reconstructed_short|reconstructed_state_space`、模型版本、fold、均值、
   標準差、QC 與 input fingerprint。

研究期間起點缺少的 `2024-01-01 00:00 UTC` 沒有雙側支撐，不納入雙向重建。它只定義
forcing 起始邊界；arrival selector 不會建立需要越過該時刻的 backward window。

## 5. 預先登錄的驗證與接受規則

交叉驗證不能隨機抽散點後宣稱可處理整日缺時。每個 domain 至少建立下列 blocked masks：

1. 33 類單一缺時形狀；
2. 23、24、25 與 49 個連續缺失 steps；
3. 每一 gap class 橫跨兩年、四季、不同潮相、一般流況與局地強流／颱風時段；
4. fold 的訓練與驗證 block 不得重疊，scaling、EOF mode 數與 AR 階數只能由 training fold
   決定。

比較基線至少含 persistence、端點線性插值、harmonic-only 與 EOF-state-space。驗證分兩層：

| 層級 | 指標 |
|---|---|
| Eulerian | u/v/w component bias、vector RMSE、速度與方向誤差、elev/zcor/Kz 誤差、空間相關、頻譜與潮汐振幅／相位、posterior interval coverage |
| Lagrangian | 24/48 h endpoint separation、boundary first-exit 類別與位置、travel time、residence time、bottom contact、來源面排名、50/75/90% HDR overlap |

正式接受門檻預先固定為：

- 各 gap class 的 vector RMSE 相對「最佳簡單基線」至少降低 20%，且強流子集不可退化；
- 主要來源面排名 Spearman correlation 至少 0.90；
- 50/75/90% HDR 的面積加權 Jaccard 至少為 0.80/0.75/0.70；
- 主要 exit 類別機率的絕對差不超過 0.05，median travel/residence time 相對差不超過 10%；
- 名目 90% forcing interval 的 empirical coverage 介於 80%–95%，避免明顯低估或過度膨脹
  不確定性。

門檻須分 gap class、domain、季節與極端子集報告，不能只用全體平均掩蓋最差情況。若長
缺口重建未通過，正式 baseline 不會讓粒子運行到缺口後全部停止；改由 selector 在每個
年×季節 strata 中選擇完整支撐的 gap-safe arrival windows，並將重建案例降為 sensitivity。
若某個預先登錄 horizon 無法覆蓋全部 strata，依 7/14/30/60 日 horizon convergence 選擇
最短穩定且可覆蓋者；較長 horizon 只在支撐 cohort 報告，分母不得與 baseline 混用。

## 6. 正式 runtime 與成果揭露

正式 baseline 的每一 arrival backward window 必須符合二者之一，且對應 manifest 必須在
正式 config 明示為 `ocm_gap_reconstruction_manifest` 或 `ocm_gap_safe_arrival_manifest`：

1. 全窗均為 canonical observed forcing；或
2. 所跨缺口已在 approved reconstruction manifest 中，且當次 forcing member 有完整支撐。

因此，已知的 420 個 OCM 缺時不再作為正常 baseline 的 `data_gap` 終止點。runtime 仍保留
`data_gap` terminal state，用於偵測 manifest 外的新缺檔、checksum 改變、局部重建失敗或
I/O 損毀；這類事件是異常 QC，不是預定的資料政策。

正式成果至少同時呈現：

- 17,544 小時 coverage timeline，逐時區分 observed、reconstructed 與 boundary unavailable；
- 各 domain/gap class 的 Eulerian 及 Lagrangian blocked-validation 圖；
- 每條軌跡的 reconstructed exposure seconds/fraction 與 forcing-member ID；
- observed-only gap-safe baseline、approved reconstruction baseline、no-Stokes 與 reconstruction
  ensemble sensitivity 的來源排名、HDR、路徑密度、停留時間及停止原因差異；
- 有效分母與 reconstruction-failure density，禁止把缺支撐區解讀為低來源權重。

## 7. NWW 波向與風場契約

正式採用 `nww3_dp_wnd_two_typhoon_adopted_v1`：DP 是由正北順時針量測的 wave-from；
傳播去向為 `(DP+180°) mod 360°`；`.wnd` 第一、第二平面分別為 eastward、northward。
山陀兒個案在 Hs≥5 m、風速≥15 m/s 的 9,849 個格點中，原始 DP／DP+180 與風去向的
中位夾角為 160.96°／19.04°；康芮 33,073 個格點為 142.13°／37.87°。兩個獨立事件及
相符的 WAVEWATCH III legacy 方向定義已構成研究端版本化採用證據，不再標記為等待
provider confirmation。日後只有取得反證時才另立 contract version 並重做受影響成果。

## 8. A 區版本政策與歷史南向擴張

2026-09-09 的本期裁決是 A 區不南擴。正式準備以既有
`northeast_taiwan_common_cache_v3` 的 bbox 與版本化 policy 建立新 geometry／design
identity；正式流程仍只能讀取已驗收的 OCM／NWW3 products，不直接讀 raw NetCDF 或以 raw
目錄存在推定可用。A 區本期契約如下：

| 欄位 | 本期決策 |
|---|---|
| `flow_domain_id` | `northeast_taiwan_common_cache_v3` |
| bbox | `[121.306315, 122.793685, 24.600844, 25.499156]` |
| `formal_domain_policy` | `v3_local20km_20260909_v1` |
| `receptor_core_radius` | 12.5 km（貢寮、龜山島各自） |
| `local_domain_radius` | 20 km（貢寮、龜山島各自） |
| outer stop | 沿用 A 區共用 outer boundary；不因 local radius 改變 |
| 本期排除 | A 南向擴張、35 km sensitivity、另行新增 15 km／23 km case |

現有 v3 三類產品即使可見 24 個月份目錄，仍須逐項驗證 schema、UTC、欄位、mask 與 arrival
母體。A 區空間支援採 `runtime_spatial_support_policy=runtime_stage_fail_closed_no_expansion_v1`：
OCM surface 在建置階段支援完整 arrival 選取，OCM native／NWW analysis 則在每個實際 RK4
階段驗證位置、深度、UTC、有限值與遮罩；無支援立即停止且不補值。既有 receptor native
篩選及其兩格尺度只服務受體候選的局部幾何安全，不是三產品共同邊界證據。舊
`formal_domain_policy=expanded_domain_v1` 只作 legacy configuration 的預設相容身分，
B-D 原有 `expanded_domain` 敏感度依其自身驗證保留。

原先 `northeast_taiwan_common_cache_v4_lbt_south_expanded` 與 bbox
`[121.306315,122.793685,24.480000,25.499156]` 是歷史南向擴張候選，現已延後，不是本期
正式輸入。舊 25 km geometry、manifest、shard 與 hash 不沿用；若日後恢復南擴，必須另立
domain policy、重新產製三類 products、對應空間支援契約與 release identity，不能把
歷史 partial 或舊指紋轉成 current formal。

## 9. 尚待衍生而非待使用者確認的項目

目前不需要使用者再提供科學數據或先行擴區。尚未完成的工作均可由既有資料與程式衍生：

- OCM blocked reconstruction validation 與 immutable patch manifest；
- NWW 17,544 小時新 analysis manifest；
- A v3 policy 的 no-expansion runtime-stage 契約與逐階段停止狀態驗證；
- 五站 receptors、50 arrival UTC、dt/horizon/Kh/Kz/M/shard convergence；
- production batch、聚合、圖表及學術成果報告。

其中任何一項未通過都應保留為「實作／驗證 gate」，不得再寫成「等待供應者補齊資料」或
「等待研究團隊提供未知 metadata」。
