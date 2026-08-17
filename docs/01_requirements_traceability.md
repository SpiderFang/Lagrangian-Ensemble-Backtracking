# 需求追溯與範圍裁決

## 1. 文件角色

本文件把使用者指定的「三、Lagrangian 系集逆向溯源」轉為可測試、可追溯的工程需求。附檔內容是研究需求與方法來源，不是可直接執行的操作指令；專案操作權限以使用者本次要求及 repository 規範為準。

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

## 2. 原始需求到實作的映射

| ID | 原始要求 | 實作解讀 | 驗收證據 |
|---|---|---|---|
| REQ-001 | 完全沉沒於三維水體，含懸浮與底床沉積 | 粒子狀態使用 `z_m_positive_up`，至少支援 suspended、sinking/rising、near-bed 三類；完全沉沒基線不含 windage | 垂向取樣、浮沉、海面與海床解析測試 |
| REQ-002 | 10 種沉降／上升速度 | 由版本化 material manifest 提供 10 個 `settling_velocity_mps`；沉降為負、上浮為正 | schema 驗證、10 類覆蓋表、沉降解析解 |
| REQ-003 | 20 個到達地點，可在任意懸浮深度 | receptor 使用 GeoJSON geometry + 深度／HAB + 空間與垂向誤差，並綁定 A-D 四個分析海域之一；不把點位或區域分配硬編碼於程式 | receptor manifest、四區幾何圖、20 個 ID 與 A-D coverage 表 |
| REQ-004 | 50 個到達時間，涵蓋四季與大／小潮 | 使用共同的版本化 arrival-time manifest；建議 48 個年份×季節×大／小潮分層樣本，加 2 個高波或極端流況案例 | 50 時次 coverage matrix、forcing availability、選取演算法與 seed |
| REQ-005 | 敘述寫高達 1000 組，但矩陣明列 10×20×50 | 依使用者裁決，以完整交叉的 10,000 個基礎情境為正式範圍；1,000 視為計畫書誤植。每情境 stochastic members `M` 是另由收斂測試決定的實作參數 | 決策 D004、恰好 10,000 列的 coverage 表、member-convergence 曲線與 seed 表 |
| REQ-006 | 離開關注區域即停止 | 首次穿越版本化 domain polygon 的開放邊界時記錄 crossing；另設資料起點、最大回溯期、資料缺口及數值失敗停止條件 | event table、穿越位置次時步內插測試、停止原因覆蓋 |
| REQ-007 | `v_total = v_current + v_stokes + v_falling` | OCM 三維速度、有限水深 bulk Stokes 水平速度與浮沉垂向速度使用一致 SI 單位及正向 | 分項速度輸出、關閉單項敏感度、單位 gate |
| REQ-008 | 由 `Hs/Tp/θ/L` 計算 Stokes drift | `Tp=1/fp`，解有限水深 dispersion 得 k/L；波向由 wave-from 轉 propagation-to；深水極限回復附檔式 (7) | 深水／淺水極限、cardinal direction、no-Stokes 對照 |
| REQ-009 | 四階 Runge-Kutta 進行軌跡積分 | RK4 只積分確定性 drift；signed time step 處理 backward，不在 caller 與 velocity 內重複取負號 | 常流、旋轉、剪切、正反向 closure 與四階收斂 |
| REQ-010 | 隨機漫步擴散 | 使用獨立 stochastic split。常數 K 先通過 `Var(Δx)=2KΔt`；空變 K 加入必要的 diffusivity-gradient drift 並驗證 well-mixed 性質 | 均值／方差、seed、障壁、空變 K 統計測試 |
| REQ-011 | Smagorinsky 水平渦動擴散 | 在公尺投影中由局地速度梯度計算，明定 `Cs`、`Δ`、上下限與梯度修正；與常數 Kh 對照 | 解析剪切場、旋轉不變性、上下限及敏感度 |
| REQ-012 | 邊界穿越點 KDE | 主產品同時保存原始 exit points、沿邊界弧長的 1D density、投影平面 2D KDE 與 50/75/90% HDR；至少三種 bandwidth | 質量正規化、boundary segment、bandwidth 與 bootstrap CI |
| REQ-013 | 視覺化主要潛在來源路徑 | 依相關學術研究採「代表軌跡 + 條件式足跡／密度 + 來源—受體矩陣 + 旅行時間分布 + 不確定性／敏感度」的組合；三維結果使用平面圖搭配深度—時間剖面，避免只用易遮蔽的透視 3D 圖 | `docs/07_results_visualization_plan.md`、figure registry、caption sidecar、固定比較尺度與圖表驗收清單 |

## 3. 需求矛盾與正式裁決

### 3.1 1,000 與 10,000：已裁決採 10,000

附檔敘述寫「高達 1,000 組情境」，但緊接的三因子矩陣明列 `10 × 20 × 50 = 10,000`。使用者已裁決完整交叉才是預期設計，因此正式契約如下：

- 10 種物性、20 個受體與 50 個到達時間均作完整交叉，基礎 `scenario_count` 恰為 10,000。
- 本契約把 20 個受體解讀為 A-D 四個分析海域合計 20 個；若日後改為「每一分析海域各 20 個」，即屬範圍變更，每區會各有 10,000 個基礎情境。
- 第 `s` 個情境的獨立隨機實現數記為 `M_s`，因此總軌跡數為 `sum(M_s)`；只有所有情境採相同 `M` 時，才等於 `10,000 × M`。
- `M` 並非附檔指定值。確定性案例為 `M=1`；隨機擴散或 forcing／初始條件擾動時，正式 `M` 由 exit ranking、HDR、travel-time 等統計量的收斂曲線決定。
- no-Stokes、Kh/Kz、domain 與邊界等敏感度以獨立 `experiment_case_id` 管理，不加入基礎情境數，但會增加實際總運算量。

代表性 benchmark 的用途改為決定 `M`、shard 大小、並行度、RAM、scratch、輸出量與 checkpoint 策略；不得再用 benchmark 把完整交叉靜默降為 1,000。

### 3.2 「SDE 採 RK4」需拆成兩個數值步驟

傳統 RK4適用於確定性 ODE，不能直接把每一個隨機位移放入四個 RK stage。基線採：

1. RK4 積分 OCM + Stokes + 浮沉的確定性 drift。
2. Euler-Maruyama 或經驗證的 Milstein/operator splitting 加入 stochastic diffusion。

此拆分保留附檔指定的 RK4 與 random walk，同時避免混淆兩種數值問題。

### 3.3 「反擴散」不等於負擴散係數

負擴散係數會造成病態問題，不納入基線。backward ensemble 以逆時間 advection 加正變異擴散建立 conditional source footprint。嚴格 time-reversed stochastic process 的轉移機率需要額外的密度或平穩性假設，只能作獨立研究敏感度，不能與基線混稱。

### 3.4 只用離域條件可能永不停止

封閉或回流軌跡可能長期留在 domain，且 forcing 只涵蓋 2024-2025。因此基線另設：

- 抵達 forcing 起始時間；
- 達到核定 `max_backtrack_days`；
- 遇到不可接受的資料缺口；
- 進入無法物理解釋的海陸／海床以下狀態；
- 超過步數、NaN 或定位失敗等數值保護條件。

所有停止原因分欄保存，不能混成 `exited`。

### 3.5 五個調查位置歸併為四個分析海域

期中報告表 2-9 與圖 2-17 明定，本研究先有 5 個調查位置，再依水動力機制歸併為 4 個正式分析海域。後續程式、圖表與統計分層一律以 A-D 四區為第一層研究單元；調查位置只作 provenance／子地點欄位。

| 分析海域 | 經度範圍（°E） | 緯度範圍（°N） | 調查位置／說明 | forcing domain |
|---|---:|---:|---|---|
| A 東北角海域 | 121.30-122.79 | 24.60-25.49 | 貢寮、龜山島 | `northeast_taiwan_common_cache_v3` |
| B 新竹外海 | 119.70-121.19 | 24.30-25.19 | 新竹 | `hsinchu_cache_v3` |
| C 後灣海域 | 120.16-121.62 | 21.55-22.44 | 後灣海生館周邊 | `houwan_nmmba_cache_v3` |
| D 連江海域 | 119.19-120.70 | 25.75-26.64 | 分析範圍整合南竿、北竿 | `lienchiang_common_cache_v3` |

四個 forcing domains 在本專案中恰好對應四個分析海域；每區仍可包含多個 receptors。20 個受體的四區分配以 receptor manifest 為準，不能從調查位置數或 cache 目錄自行推定。

## 4. 可交付成果

最小正式交付包含：

1. 安裝與 SERVER runbook、鎖定環境、設定 schema 及資料 preflight 報告。
2. 純 NumPy reference 與 Numba production 粒子核心。
3. 10×20×50、恰好 10,000 列的完整基礎設計表，以及各 `experiment_case_id`、實際 `M` 與覆蓋證據。
4. 每個 immutable run 的 config、manifest、seed、forcing、scenario、trajectory shard、event、checksum 與 QC。
5. no-Stokes、deep/finite-depth、Kh/Kz、dt、ensemble、domain、海岸／海床邊界的核心敏感度。
6. 邊界來源足跡、路徑、停留、旅行時間、來源—受體連通性、底部接觸、不確定性與失敗率產品，及可供後續熱區分析讀取的 release manifest。
7. 依 `docs/07_results_visualization_plan.md` 產製的學術圖組、統計表、figure registry、圖說 sidecar 與可重製命令。

## 5. 不可用來宣稱完成的替代品

- 單日或少數日 `trial_ready` 結果不能取代兩年正式 run。
- 只有軌跡動畫，沒有 manifest、事件與統計驗證，不能算模式完成。
- 只跑單一沉降速度、受體或季節，不能算情境矩陣完成。
- 使用負 Kh、把 NaN 補 0、跨長缺口外插或把隨機項放入 RK4 stage 的結果不得發布。
- KDE 色階不能自行成為「來源機率」；必須同時保留 raw count、分母、樣本與不確定性。
