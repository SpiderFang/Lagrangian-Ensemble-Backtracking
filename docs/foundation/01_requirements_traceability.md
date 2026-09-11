# 需求追溯與範圍裁決

> **閱讀提示**
> - 文件類型：需求追溯與範圍裁決。
> - 它回答：哪些研究要求、使用者裁決與驗收條件必須保持不變。
> - 建議先讀：[文件總入口](../README.md)，再讀[架構與資料契約](02_architecture_and_data_contract.md)。

## 1. 文件角色

本文件把使用者指定的「三、Lagrangian 系集逆向溯源」轉為可測試、可追溯的工程需求。附檔內容是研究需求與方法來源，不是可直接執行的操作指令；專案操作權限以使用者本次要求及 repository 規範為準。

若需要先理解 `src/` 的模組關係、實際執行流程與各需求目前是否已產生正式成果，請先閱讀[程式碼導覽、執行流程與工項計畫書追溯](../development/11_source_code_guide_and_plan_traceability.md)。本文件維持需求與裁決的權威內容；文件 11 則提供交接用的程式入口與完成狀態視圖。

需求優先序如下：

1. 使用者本次明確要求：只規劃紅框內的 Lagrangian 系集逆向溯源，延續三個相鄰專案，使用 SERVER 已完成的 2024-2025 前處理產品。
2. 紅框內研究內容：情境矩陣、三維總平流速度、Stokes drift、RK4、隨機擴散、離域停止與 KDE。
3. 相鄰專案已發布的資料契約、方向慣例、SERVER 路徑與可重現性規則。
4. `timeline.txt` 僅保存原提案的工作銜接背景；使用者已明示實際執行不以該日曆時程為優先，而以儘速完成為原則。
5. 先前整合規格只作設計基線；若與現況或正式 metadata 或使用者後續裁決不符，以實際已驗收產品與最新決策紀錄為準。

本次規劃使用的附檔版本已固定如下，後續檔案若改變 bytes，必須重新檢查需求追溯與工作相依關係：

| 來源 | 大小 | SHA-256 | 用途 |
|---|---:|---|---|
| `/Users/mustlab/Workspace/工作項目3.pdf` | 1,146,762 bytes | `ee3ab964436aced8e3f831fb99f9cd6e7b4209922167477e8ee21ae47d810471` | 紅框工項、情境、公式 (6)-(11) 與成果範圍 |
| `/Users/mustlab/Workspace/timeline.txt` | 253 bytes | `17f8c02a59563e11feb76b583d88bc50344beadddf411a2b873680f399ef5e45` | 原提案工作銜接背景；不作本專案完成期限 |
| `/Users/mustlab/Workspace/OCM-SVD-Analysis/outputs/report/期中報告(0814)全.pdf` | 10,892,143 bytes | `d3e9d931e4df93c3dc9eecffd1a3f47b5ee74e303150c22de4de07d6e819e21d` | 表 2-9、圖 2-17 的四個分析海域定義及五個調查位置對應 |
| `/Users/mustlab/Workspace/OCM-Data-Preprocessing/configs/ocm_flow_domains.json` | 依 Git 版本 | `b8db61c38138d5690d203bf1b3785c6b2e581572d08f097573772c554ef373b3` | 四個 flow domain 的正式 bbox、中心與研究區對應 |
| `/Users/mustlab/Workspace/OCM-SVD-Analysis/configs/guishan_gongliao_northeast_taiwan_flow_domain_water_column_svd_available_2024_2025.json` | 依 Git 版本 | `daeb9f876eb8a62996b2f7b762e5cee0e03298adf010c802f03261527bdf67e0` | 證明貢寮／龜山島共用 A 區完整水柱 forcing，而非合併情境單元 |
| [海洋保育署 iOcean 海洋廢棄物管理頁](https://iocean.oca.gov.tw/OCA_OceanConservation/PUBLIC/Marine_Litter_v2.aspx) | 動態網頁；2026-08-27 查閱 | 不適用 | 只採用查詢介面的十個海廢項目名稱；網站清除重量／件數不作沉降校準或來源先驗 |
| 使用者提供、未納入版本控制的簡報照片 | 287,376 bytes | `bff3187f875a8e312aa9744ae5488401fc7cfb5110e0675a0f21c23d901693ed` | 沉底漁業用具廢棄物的報告優先層之定性依據；照片與長官口頭關注均待正式文件確認，不作沉降速度、來源先驗、發生頻率或 repeated-contact 機制的定量證據 |

## 2. 原始需求到實作的映射

| ID | 原始要求 | 實作解讀 | 驗收證據 |
|---|---|---|---|
| REQ-001 | 完全沉沒於三維水體，含懸浮與底床沉積 | 粒子狀態使用 `z_m_positive_up`；依 2026-08-27 最新裁決，正式基線只含 `sinking` 與 `near_bed`，不含 windage、中性或上浮物性速度 | 垂向取樣、沉降、海面與海床解析測試 |
| REQ-002 | 原提案列 10 種沉降／上升速度；研究主持人最新要求取消上升並對應海廢材質與形狀 | 由版本化 material manifest 提供 10 個嚴格負值 `settling_velocity_mps`，一對一連結 iOcean 十個項目與代表材質／形狀代理；速度是未校準敏感度格點 | schema 驗證、`w_b < 0` 約束、十個官方分類唯一覆蓋、材質／形狀欄位完整性與沉降解析解 |
| REQ-003 | 每一海域設定 20 個到達地點，可在任意懸浮深度 | 五個獨立站點各 20 個、全案 100 個 receptors；每站採 5 個 metric maximin 水平位置 × 4 個有效垂向層位，並綁定 `study_site_id` | receptor manifest、五站點幾何圖、每站 20／全案 100 個 ID coverage 表 |
| REQ-004 | 每一海域 50 個到達時間，涵蓋四季與大／小潮 | 五站點各有 50 個條件；固定採 48 個年份×季節×大／小潮×潮內相位，加 2 個局地高波／強流事件。確切 UTC 由資料決定 | 每站 50 時次 coverage matrix、forcing availability 與可重現選取紀錄 |
| REQ-005 | 每一海域敘述寫高達 1000 組，但矩陣明列 10×20×50 | 依使用者裁決，每站點採完整交叉 10,000 個基礎情境；五站點合計 50,000，1,000 視為計畫書誤植。每情境 stochastic members `M` 另由收斂測試決定 | 決策 D004、每站 10,000／全案 50,000 列的 coverage 表、member-convergence 曲線與 seed 表 |
| REQ-006 | 離開關注區域即停止 | 各站離開自身 local domain 時記錄 primary first-exit；貢寮／龜山島本期依 `formal_domain_policy=v3_local20km_20260909_v1` 使用 20 km local domain，事件後繼續至共用 A 區 outer boundary，B-D 因 local 與 flow domain 重合而於同一 crossing 停止。貢寮／龜山島穿越對方 local domain 僅記錄非終止的 cross-site diagnostic event。另設海岸、海床、海面 regime、資料起點、最大回溯期、缺口及數值失敗事件 | own-local／foreign-local／outer event table、20 km 邊界的 OCM native／surface／NWW 共同 margin（至少兩個共同有效格點）驗證、重合邊界去重與步內 crossing 測試、跨站事件不改變粒子狀態測試、停止原因覆蓋 |
| REQ-007 | `v_total = v_current + v_stokes + v_falling` | 定期及終止的既有 `Observation` 均可保存速度，但只有在同一 UTC／`x_m,y_m,z_m` 與步首 sample 完全對齊時才附帶；保存固定九欄 m/s：`total_u_mps`、`total_v_mps`、`total_w_mps`、`ocm_u_mps`、`ocm_v_mps`、`ocm_w_mps`、`stokes_u_mps`、`stokes_v_mps`、`settling_w_mps`；三個 total 分別以明定容差核對 OCM、Stokes 水平與向上為正的沉降（本專案沉降為負）合成。`CombinedMonthForcing` 有效樣本提供完整分項；明確關閉 Stokes 或自然海況計算結果可為有效 0，缺少 NWW3 波浪資料則為缺值／非零 QC；`VelocitySampleStatus` 與 `velocity_qc_flags` 獨立保存 `not_sampled`、`total_only`、缺值及無效狀態。速度不是位移平均值、不是 RK4 stage；逆向不另改符號，舊成果不自動補新證據。 | 關閉單項敏感度、單位 gate；`test_mesh_forcing.py` 的完整／no-Stokes／沉降合成與同值檢查、`test_velocity_recording.py` 的 total-only、逆向、缺值／非有限／bool／sum mismatch、同點終止與 no-reference 測試；正式 writer／reader 另依版本化輸出契約驗證 |
| REQ-008 | 由 `Hs/Tp/θ/L` 計算 Stokes drift | `Tp=1/fp`，解有限水深 dispersion 得 k/L；波向由 wave-from 轉 propagation-to；深水極限回復附檔式 (7) | 深水／淺水極限、cardinal direction、no-Stokes 對照 |
| REQ-009 | 四階  Runge-Kutta  進行軌跡積分 | RK4 只積分確定性 drift；signed time step 處理 backward，不在 caller 與 velocity 內重複取負號 | 常流、旋轉、剪切、正反向 closure 與四階收斂 |
| REQ-010 | 隨機漫步擴散 | 使用獨立 stochastic split。常數 K 先通過 `Var(Δx)=2KΔt`；空變 K 加入必要的 diffusivity-gradient drift 並驗證 well-mixed 性質 | 均值／方差、seed、障壁、空變 K 統計測試 |
| REQ-011 | Smagorinsky 水平渦動擴散 | 在公尺投影中由局地速度梯度計算，明定 `Cs`、`Δ`、上下限與梯度修正；與常數 Kh 對照 | 解析剪切場、旋轉不變性、上下限及敏感度 |
| REQ-012 | 邊界穿越點 KDE | 主產品同時保存原始 exit points、沿邊界弧長的 1D density、投影平面 2D KDE 與 50/75/90% HDR；至少三種 bandwidth | 質量正規化、boundary segment、bandwidth 與 bootstrap CI |
| REQ-013 | 視覺化主要潛在來源路徑 | 依相關學術研究採「代表軌跡 + 條件式足跡／密度 + 來源—受體矩陣 + 旅行時間分布 + 不確定性／敏感度」的組合；三維結果使用平面圖搭配深度—時間剖面，避免只用易遮蔽的透視 3D 圖 | `docs/results/07_results_visualization_plan.md`、figure registry、caption sidecar、固定比較尺度與圖表驗收清單 |
| REQ-014 | 依新增簡報照片與長官口頭意見，將沉底漁業用具列為報告優先層 | 保持 `design_baseline_v2_non_rising_oca_proxy` 既有 10 類、數值、情境 ID 與每站 `10×20×50` 不變；正文主要切片優先呈現 `material_id=oca_fishinggear_open_mesh_bundle` × `vertical_id=near_bed`，其他材質／水層完整保留。照片與口頭關注只屬定性、待正式確認，不能推導沉降速度、來源先驗、發生頻率或 repeated-contact 機制 | 來源表之簡報照片、§2.1 邊界說明、`docs/results/07_results_visualization_plan.md`、`docs/foundation/08_design_baseline_and_derived_gates.md`、`report_material_statistics.py` 的 member-level count/fraction 與去重測試 |

### 2.1 沉底漁業用具的新增定性證據與實作邊界

合作團隊目前只提供一頁簡報照片（來源、大小與 SHA-256 詳見第 1 節來源表），並有主管口頭表示特別關注「沉底的漁業用具廢棄物」。
這兩項訊息是研究優先順序的定性證據，不是可用來推導物性或母體權重的定量調查：

| 證據 | 目前可確認內容 | 不可推導的內容 | 實作裁決 |
|---|---|---|---|
| 使用者提供、未納入版本控制的簡報照片（來源表所列） | 簡報文字指出「海底廢棄物中以漁業用具類廢棄物為最大宗」，並以掌握覆網與海廢分布為調查目的 | 件數／重量／面積比例、材質分布、沉降速度、來源先驗與不確定性 | 將 `oca_fishinggear_open_mesh_bundle` × `near_bed` 列為正文主要分析層；保留全十類與四個垂向層位 |
| 主管口頭意見（待正式文件確認） | 沉底漁業用具廢棄物為特別關注對象 | 「特別關注」不等於發生率、速度或因果來源權重 | 提高成果呈現優先序，不修改 v2 速度、情境 ID 或 `10×20×50` 矩陣 |

因此，材質統計只對有效 member 計算首次海床接觸與沉積的 raw count／fraction；資料缺口與數值
失敗 member 不進有效分母。`BED_CONTACT` 可重複出現但以 member 去重，`DEPOSITED` 可獨立作為
終止沉積證據；基線 `deposit_on_first_contact_and_stop` 不要求 repeated-contact 欄位。

## 3. 需求矛盾與正式裁決

### 3.1 每區 1,000 與 10,000：已裁決每區採 10,000

附檔先寫「針對每一處開放海域」執行高達 1,000 組情境，緊接著在同一工項下定義 10 種速度、20 個 receptor locations 與 50 個 arrival times。其自然作用域是每一獨立研究站點，因此 `10 × 20 × 50 = 10,000` 是每站點矩陣。使用者另裁決貢寮與龜山島雖共用 forcing，情境須各自完整建立，正式契約如下：

- 貢寮、龜山島、新竹、後灣與連江各有 10 種非上浮海廢材質／形狀代理、20 個受體與 50 個到達時間的完整交叉，每站點 `scenario_count` 恰為 10,000。
- 五站點共 100 個 receptors，A 區兩站合計 20,000 個基礎情境，全案合計 `5 × 10,000 = 50,000`；「四區合計 20 個、每區 5 個」及「A 區兩站共用 20 個」均已撤銷。
- 第 `s` 個情境的獨立隨機實現數記為 `M_s`，因此全案單一 experiment case 的總軌跡數為 `sum(M_s)`；只有所有情境採相同 `M` 時，才等於 `50,000 × M`，每站點則為 `10,000 × M`。
- `M` 並非附檔指定值。確定性案例為 `M=1`；隨機擴散或 forcing／初始條件擾動時，正式 `M` 由 exit ranking、HDR、travel-time 等統計量的收斂曲線決定。
- no-Stokes、Kh/Kz、domain 與邊界等敏感度以獨立 `experiment_case_id` 管理，不加入基礎情境數，但會增加實際總運算量。

代表性 benchmark 的用途改為決定 `M`、shard 大小、並行度、RAM、scratch、輸出量與 checkpoint 策略；不得再用 benchmark 把任一站點的完整交叉靜默降為 1,000，或把五站點合併為單一 10,000 矩陣。

### 3.2 「SDE 採 RK4」需拆成兩個數值步驟

傳統 RK4適用於確定性 ODE，不能直接把每一個隨機位移放入四個 RK stage。基線採：

1. RK4 積分 OCM + Stokes + 向下沉降的確定性 drift。
2. Euler-Maruyama 或經驗證的 Milstein/operator splitting 加入 stochastic diffusion。

此拆分保留附檔指定的 RK4 與 random walk，同時避免混淆兩種數值問題。

### 3.3 「反擴散」不等於負擴散係數

負擴散係數會造成病態問題，不納入基線。backward ensemble 以逆時間 advection 加正變異擴散建立 conditional source footprint。嚴格 time-reversed stochastic process 的轉移機率需要額外的密度或平穩性假設，只能作獨立研究敏感度，不能與基線混稱。

### 3.4 只用離域條件可能永不停止

封閉或回流軌跡可能長期留在 domain，且 forcing 只涵蓋 2024-2025。因此基線另設：

- 抵達 forcing 起始時間；
- 達到核定 `max_backtrack_days`；
- 遇到 manifest 外缺檔、checksum/I/O 損毀、空間必要欄位無效或重建失敗；已知 OCM
  整時缺口須在 run 前由 approved reconstruction 或 gap-safe arrival/horizon 處理；
- 進入無法物理解釋的海陸／海床以下狀態；
- 超過步數、NaN 或定位失敗等數值保護條件。

所有停止原因分欄保存，不能混成 `exited`。

### 3.5 四個 forcing domains 與五個情境站點分層保存

期中報告表 2-9 與圖 2-17 將 5 個調查位置依水動力機制對應至 4 個分析流場；`OCM-SVD-Analysis` 亦讓貢寮與龜山島共用完整 A 區水柱矩陣，避免重複計算相同 forcing。這只能決定 forcing 層，不能取消兩個站點各自的受體與完整情境。後續程式與圖表以 `study_site_id` 為第一層、`analysis_region_id` 為次要彙整層。

| region | 經度範圍（°E） | 緯度範圍（°N） | 獨立站點 | forcing domain |
|---|---:|---:|---|---|
| A 東北角海域 | 121.306315–122.793685 | 本期沿用 v3 南界 24.600844 | 貢寮、龜山島 | `northeast_taiwan_common_cache_v3`；本期 `formal_domain_policy=v3_local20km_20260909_v1`，12.5 km receptor core、20 km local domain；南向擴張為歷史候選 |
| B 新竹外海 | 119.70-121.19 | 24.30-25.19 | 新竹 | `hsinchu_cache_v3` |
| C 後灣海域 | 120.16-121.62 | 21.55-22.44 | 後灣海生館周邊 | `houwan_nmmba_cache_v3` |
| D 連江海域 | 119.19-120.70 | 25.75-26.64 | 分析範圍整合南竿、北竿 | `lienchiang_common_cache_v3` |

四個 forcing domains 供五站點使用：A 區包含貢寮與龜山島兩套各 20 個 receptors，其餘 B-D 各一套 20 個，全案共 100 個。2026-09-09 裁決本期 A 區不南擴，兩站以 anchor-centered 12.5 km receptor core 與 20 km local domain 建立新 geometry／design identity；兩站共用同一 A 區 forcing cache 與 outer boundary，以保留相近海域的共同水動力影響。自身 local crossing 是主要入口事件，穿越另一站 local domain 則為不終止的方向性連通診斷，所有 ID、狀態與分母仍按原始站點分開。

本期不再登錄 25 km baseline、35 km sensitivity、15 km 或 23 km A 區 case。原先南向擴張候選 `northeast_taiwan_common_cache_v4_lbt_south_expanded` 與其 bbox `[121.306315,122.793685,24.480000,25.499156]` 僅保留為歷史規劃；舊 25 km geometry、manifest、shard 與 hash 不沿用，必須依本期 policy 重建。現有 v3 三類產品即使具備月份目錄，也必須由 OCM native、OCM surface 與 NWW analysis 實際證明 20 km local boundary 至 outer boundary 的共同 margin；目前 receptor native 篩選 gate 不等於此三產品、兩共同有效格點驗證。故本期可進行 strict input preparation，但 formal scope 在 margin evidence validator／producer 完成前維持 blocked。詳細規則見文件 08。

## 4. 可交付成果

最小正式交付包含：

1. 安裝與 SERVER runbook、鎖定環境、設定 schema 及資料 preflight 報告。
2. 純 NumPy reference 與 Numba production 粒子核心。
3. 五站點各 10×20×50、各 10,000 列且全案恰好 50,000 列的完整基礎設計表，以及各 `experiment_case_id`、實際 `M` 與覆蓋證據。
4. 每個 immutable run 的 config、manifest、seed、forcing、scenario、trajectory shard、event、checksum 與 QC。
5. no-Stokes、deep/finite-depth、Kh/Kz、dt、ensemble、domain、海岸／海床邊界的核心敏感度。
6. 邊界來源足跡、路徑、停留、旅行時間、來源—受體連通性、底部接觸、不確定性與失敗率產品，及可供後續熱區分析讀取的 release manifest。
7. 依 `docs/results/07_results_visualization_plan.md` 產製的學術圖組、統計表、figure registry、圖說 sidecar 與可重製命令。

## 5. 不可用來宣稱完成的替代品

- 單日或少數日結果不能取代兩年正式 run；但全期 NWW analysis 的 `trial_ready` 上游標籤
  已由 available-data contract 接受，不能再誤稱為待供應者升版的 blocker。
- 只有軌跡動畫，沒有 manifest、事件與統計驗證，不能算模式完成。
- 只跑單一沉降速度、受體或季節，不能算情境矩陣完成。
- 使用負 Kh、把 NaN 補 0、未經 blocked validation 跨長缺口外插或把隨機項放入 RK4
  stage 的結果不得發布；通過門檻且保存 uncertainty/provenance 的 state-space reconstruction
  不屬於此處禁止的臨時外插。
- KDE 色階不能自行成為「來源機率」；必須同時保留 raw count、分母、樣本與不確定性。
