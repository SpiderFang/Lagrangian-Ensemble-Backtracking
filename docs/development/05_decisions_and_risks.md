# 決策與風險登錄

> **閱讀提示**
> - 文件類型：設計決策、狀態、風險與限制登錄。
> - 它回答：每項決定的依據、目前狀態、必要閘門與未決影響是什麼。
> - 建議先讀：[實作狀態](../implementation_status.md)，再讀[需求追溯](../foundation/01_requirements_traceability.md)。

## 1. 狀態定義

- `decided`：已有範圍、依據、版本與影響；修改需新 decision record。
- `derived_pending`：演算法與驗收標準已決定，尚待 SERVER 資料、網格或 pilot 計算實際值；不需使用者再選方案。
- `provisional`：可供程式與 pilot 使用，但正式兩年 run 前仍需核定。
- `open`：不得以任意預設啟動受影響的正式 run。
- `blocked`：缺少權限、資料或外部決策，且沒有安全替代路徑。
- `superseded`：被新版決策取代，歷史仍保留。

## 2. 決策登錄

| ID | 狀態 | 必須完成的閘門 | 問題與目前處理 | 未決時的限制 | owner |
|---|---|---|---|---|---|
| D000 | decided | 範圍基線 | 專案只實作紅框「三、Lagrangian 系集逆向溯源」，前處理、SVD、TRAP 不在本 repo 重作 | 範圍擴充需新決策 | 研究團隊／開發 |
| D001 | decided | G0 | 現有 OCM/NWW 定義為本研究「2024–2025 全部可得資料」的完整正式母體；原始提供者、額外 metadata 與補件均不可取得，未知欄位照實保存，不再等待外部確認 | 不得宣稱 provider-confirmed best forecast cycle 或虛構 metadata；專案內仍須完成 canonicalization、QC 與衍生產品 | 研究團隊／自動 preflight |
| D002 | derived_pending | G0/G1 | SCHISM 參考文件支持 hvel/w 為 m/s、diffusivity 為 m²/s、z positive-up；`wetdry_elem` 0/1 由 metadata、實值 snapshot 與測試確認 | 未通過欄位不得被靜默轉換；不需人工任選語意 | 自動 preflight／海洋數值審查 |
| D003 | decided | G1 | 採 `nww3_dp_wnd_two_typhoon_adopted_v1`：由山陀兒與康芮兩個獨立事件判定 `DP` 為自正北順時針 wave-from、`+180°` 轉 propagation-to，`.wnd` planes 1/2 為東／北向風分量；後續產品做 cardinal/vector QC | 只有新實證直接反駁時才建立新版契約並重跑；供應者不可考不是未決項 | 研究團隊／海洋數值審查 |
| D004 | decided | 情境基線 | `10×20×50=10,000` 套用於貢寮、龜山島、新竹、南灣、連江每一獨立站點；A 區 20,000、全案 50,000，1,000 為誤植。`M` 由 member convergence 衍生 | 不得把任一站 baseline 縮為 1,000、把 A 區兩站合併或把全案縮為 10,000 | 研究團隊／數值／系統 |
| D005 | decided | G0/G3 | 每站 5 個 seeded random persistent-wet 水平 face × 每面 4 個 seeded random `(0,1)` normalized vertical draws，共 20；全案 100。實際座標與每個 arrival 的 z 由 accepted OCM mesh／zcor 生成，random rank 不代表物理層位 | manifest 未通過前可用合成 receptor 測試；current formal 必須保存 seed、draw order、fraction 與 bracket evidence | geometry／受體 selector／數值審查 |
| D006 | decided | G0/G3 | 依研究主持人 2026-08-27 裁決，取消中性與所有上浮情境；十個速度固定為 `-0.0001,-0.0002,-0.0005,-0.001,-0.002,-0.005,-0.010,-0.020,-0.050,-0.100 m/s`，一對一連結 iOcean 十類及代表材質／形狀條件 | 所有速度必須嚴格小於 0；iOcean 只供分類名稱，數值仍是 `provisional_proxy`，不得宣稱為官方量測或類別平均 | 研究主持人／研究方法基線 |
| D007 | derived_pending | G0/G3 | 每站 50 個 arrival times 採 48 個年份×季節×大／小潮×三潮位相位 proxy，加局地高波與強流各 1；確切 UTC 由 observed/reconstructed forcing selector 產生，重建未過門檻時限於 gap-safe windows | 不得刪除 strata；若較長 horizon 無法支撐全部 strata，採最短已收斂且可完整覆蓋者並揭露限制 | deterministic selector／統計審查 |
| D008 | derived_pending | G2/G4 | local first exit 記錄後續跑、flow-domain exit/coast/deposition 停止；sinking 代理因擴散越過海面時反射並記錄；`max_backtrack_days` 比較 7/14/30/60 日取最小穩定值 | horizon 未收斂前只可跑 pilot；正式設定若含零速或正值須在情境建表前拒絕 | 數值 pilot／方法審查 |
| D009 | derived_pending | G2 | 常數 Kh/Kz 作 reference，實際值由 Brownian/well-mixed 與 pilot 決定；Smagorinsky + gradient drift 通過 PDE/well-mixed 測試後才可升為基準 | 未通過時不得把空變 K 當正式基準 | 數值驗證／統計 |
| D010 | provisional | G3 | 軌跡採 ragged NumPy columns、事件／scenario 採 Parquet、immutable shards | benchmark 若顯示 I/O 不合適，需新 schema minor/major 決策 | 開發／系統 |
| D011 | derived_pending | Pilot/G4 | 正式 output root、local scratch、NFS publish、備份與配額由容量／檔案系統 preflight 決定 | 未通過不得啟動大批次或把半成品寫入正式路徑 | 系統 preflight／管理者 |
| D012 | provisional | G5 | 結果措辭限定 conditional footprint／relative source weight；absolute probability 需 prior/likelihood/observations | 未補證據不得升級措辭 | 研究團隊 |
| D013 | decided | G5 | 成果採文獻支持的代表軌跡、條件式密度／HDR、來源—受體矩陣、旅行時間分布、季節／潮況小多圖與不確定性／敏感度圖組；規格見文件 07 | 核心圖與統計表不得以單一動畫或全軌跡疊圖取代 | 研究團隊／開發 |
| D014 | superseded | G0/G5（歷史） | 2026-08-27 的歷史基線：貢寮／龜山島舊候選 bbox 不作 local domain；兩站各以 anchor 半徑 12.5 km receptor core、25 km local domain，並比較 20/35 km。local domains 允許重疊且不作 Voronoi 切割；兩站共用同一 A 區 forcing 與 outer boundary。軌跡只以 own local domain 定義主要入口，foreign-local crossing 非終止且只作連通診斷 | 本期由 D019 取代；舊 25 km geometry、manifest、shard 與 hash 不得沿用 | 使用者／研究方法基線 |
| D015 | superseded | G0/G1/G4（歷史） | 2026-08-27 的歷史南向擴張候選：`northeast_taiwan_common_cache_v4_lbt_south_expanded`，bbox `[121.306315,122.793685,24.480000,25.499156]`，原規劃以 35 km geodesic margin 與 25/35 km local boundary 驗證 OCM/NWW 共同 mask | 本期由 D019 延後；不得把歷史 v4 候選、舊 margin 或 native source margin 當成本期 formal evidence | geometry／forcing preflight |
| D016 | decided | G0/G1 | OCM 24 月先 stable sort／`prefer_last` 建 canonical 軸；33 個單一缺時比較短缺口內插與 state-space，23–49-step 長缺口採多變量 EOF-harmonic state-space bidirectional smoother 與 posterior forcing members，並以實際缺口形狀做 Eulerian/Lagrangian blocked validation。pure DINEOF 不得單獨補整個缺失 snapshot | 方法未過門檻時 baseline 改用 gap-safe 分層 arrival windows；已知缺口不作正常 runtime terminal | 研究方法／數值驗證 |
| D017 | decided | G0/G1 | NWW native 2024–2025 實測恰有 17,544 個連續逐時 UTC；正式 analysis 從 native 重採樣到 OCM 靜態格網，方向用單位向量作圓形內插，時間軸涵蓋 observed/reconstructed OCM UTC | 不對波浪時間作統計補值，也不沿用舊 gappy OCM target-time 軸 | forcing 產製／QC |
| D018 | decided（定性優先項） | G5 | 依合作團隊一頁簡報與主管口頭意見，將 `oca_fishinggear_open_mesh_bundle` 列為正文主要分析材質；current formal 垂向結果依 `random_vertical_draw_0..3` identity 分層，不把 random rank 命名為 `near_bed` | 不得由「最大宗」或「特別關注」推導件數／重量先驗、沉降速度或因果來源；十類、速度、情境 ID 與 10×20×50 不變 | 研究團隊／報告統計 |
| D019 | decided（2026-09-09；空間支援條款由 D020 更新） | G0/G1/G4 | 本期 A 區不南擴，沿用 `northeast_taiwan_common_cache_v3` bbox `[121.306315,122.793685,24.600844,25.499156]`；以 `formal_domain_policy=v3_local20km_20260909_v1` 建立貢寮／龜山島各自 12.5 km receptor core 與 20 km local domain，共用原 A 區 outer boundary。`design_baseline_v3_non_rising_a_v3_local20_20260909` 形成新的 geometry／design identity；原 35 km sensitivity 移出本期，不自行加入 15 km 或 23 km。舊 `formal_domain_policy=expanded_domain_v1` 只保留 legacy configuration 相容身分；B-D 原有 `expanded_domain` 敏感度不因本決定刪除 | A 區範圍、半徑、outer stop 與不擴張決策維持不變；原列的共同兩格 margin 前置條件由 D020 的逐 RK4 階段封閉失敗契約取代 | 研究團隊／geometry／forcing preflight |
| D020 | decided（2026-09-15） | G0/G1/G4 | A 區正式空間支援沿用 24 小時工程 DEMO 的核心機制，但不沿用 DEMO 母體或成果：`runtime_spatial_support_policy=runtime_stage_fail_closed_no_expansion_v1`、`formal_release_domain_status=no_expansion_runtime_stage_fail_closed`。OCM surface 在 input build 支援完整 arrival 母體；OCM native／NWW analysis 於每個實際 RK4 階段依位置、深度、UTC、有限值與 mask 嚴格取樣，任一必要 forcing 無支援立即停止並保存 `data_gap`／對應品質狀態 | 不建立 A v4、不南擴、不新增半徑；禁止零值、最近值、未登錄跨月／域外外插及 current-only 降級。站點欄位 `minimum_flow_domain_margin_local_grid_scales=2` 只作受體候選的局部幾何安全篩選，不得再稱為三產品共同 margin。既有 1 天、20 情境只是 `engineering_only` DEMO；30 天 formal 必須重建並驗證四區五站、100 receptors、250 arrivals、5,000 pairs 與 50,000 基礎 scenarios | 使用者／研究團隊／runtime／forcing preflight |

## 3. 決策紀錄模板

```text
Decision ID / version:
Status / decided_at:
Owner / reviewers:
Question:
Options considered:
Decision and rationale:
Evidence and version:
Affected configs, schemas, products and runs:
Migration or rerun plan:
Known limitations:
Supersedes:
```

會改變單位、正向、時間、domain/receptor、情境數、速度項、diffusion generator、邊界、seed 或結論措辭的變更，都必須使用此模板並使舊 run 可被辨識為 `superseded`。

## 4. 風險矩陣

評分使用可能性 L/M/H 與影響 L/M/H。H/H 與 H/M 每週審查。

| ID | 風險 | L/I | 預防與緩解 | 觸發與應變 |
|---|---|---|---|---|
| R001 | SERVER 路徑、認證或資料權限阻塞 G0 | M/H | 只要求資料管理者在已登入終端跑唯讀 preflight；不透過聊天傳密碼；輸出 inventory 可離線審查 | inventory 未取得時維持本機合成開發，正式 run 標 blocked |
| R002 | OCM partial month、重複時次與完整 snapshot 缺時被誤當成連續 forcing | M/H | 逐 domain/month 盤點，stable sort/prefer-last canonicalization；實際 gap shapes 做 blocked reconstruction validation；保留 origin 與 exposure | 重建未過門檻即採 gap-safe 分層 arrival windows，不在 runtime 最近值填補，也不等待供應者補件 |
| R003 | OCM wetdry、w 或 Kz 語意錯誤 | M/H | 參考文件、實值分布、海岸 snapshot、合成與局地診斷交叉檢查 | 無法確認則基準不用 wetdry/Kz 的物理解釋，相關 case 降級 |
| R004 | 已核定的 NWW wave-from/to 契約在後續實作中被反向套用 | L/H | 固定 `nww3_dp_wnd_two_typhoon_adopted_v1`、cardinal unit tests、傳播箭頭 QC、DP+180 對照與 manifest contract ID | 新實證衝突或程式違反契約時立即 major method version，所有受影響 Stokes run 重跑 |
| R005 | native face triangulation 在 quad、洞或海岸跨越錯誤 | M/H | 只用原生 connectivity、退化／orientation QC、coast mask、真實圖面抽查 | 任一跨陸地案例為 G1 blocker，禁止用全域 Delaunay 迴避 |
| R006 | 垂向內插在陡坡或海床下製造有效速度 | H/H | 每 node 包夾、三 node 完整支撐、surface/bed event、invalid reason | data-gap/bed-contact 異常集中時回查 sampler，不以最近層填補 |
| R007 | backward diffusion 被誤當真實歷史或機率 | H/H | conditional-footprint 命名、正向-逆向 synthetic、分母與 prior 欄位、措辭審查 | 未通過合成驗證禁止 probability／source attribution 字樣 |
| R008 | 空變 K 缺 gradient drift，造成人工聚集或穿障壁 | H/H | 常數 K reference、PDE/well-mixed/障壁測試、Milstein sensitivity | 測試失敗：Smagorinsky 不進 baseline，只保留工程診斷 |
| R009 | bulk Stokes 在近岸、淺水或混合風浪下偏差大 | H/H | finite/deep/no-Stokes、kh/steepness QC、方向與 mean wavelength 對照 | ranking 對 formulation 高敏感：列主要不確定性，不給單一結論 |
| R010 | OCM 已含波流耦合效應，額外 Stokes 可能重複 | M/H | 查 OCM 產品說明與模式設定；將 no-Stokes 設為核心對照 | 無法確認：報告明載可能 double counting，不作精確量值歸因 |
| R011 | local/flow domain 太小，人工邊界主導 entry/exit KDE | M/H | 本期 A 採 v3 原 bbox、12.5 km receptor core 與 20 km local domain；兩站共用 outer stop，local domains 可重疊。逐 RK4 階段嚴格驗證 OCM native／NWW analysis 空間支援，並把合法 outer exit、`data_gap` 與 numerical failure 分開統計；舊 v4 南擴與 35 km sensitivity 僅作歷史背景 | 若短時同邊退出或空間無支援高度集中、主要指標不穩定，正式結果降級或阻擋科學發布；不得用補值、current-only 或擴域偷偷改變已核定範圍 |
| R012 | domain 擴大或 `50,000×M×experiment cases` 導致計算／儲存爆量 | H/H | particle-step benchmark、最小收斂 M、向量化/Numba、shard/checkpoint、流式聚合與容量緩衝 | 超出資源時先增加合理並行、降低已驗證的儲存頻率並延後非核心案例；不得把任一站 baseline 靜默縮為 1,000 |
| R013 | 每站 20／全案 100 receptors 或每站 50 times 的衍生 manifest 尚未產出，trial 資料被誤當正式成果 | H/H | schema 與 synthetic test 可先行；正式 config 驗證 derived gate 與 `status=approved` | 未通過不得啟動 G4，輸出必須標 TRIAL |
| R014 | 粒子停留不出界，模擬無限延長 | M/H | forcing start、max age、step limit與 manifest 外 data gap 停止；各原因分開統計 | max-age 比例過高：調整研究問題/domain/horizon，不把它當 exit；不得依賴已知缺口代替科學停止條件 |
| R015 | dt 過大漏掉窄通道、海岸或 crossing | M/H | advective/vertical/diffusive CFL、step-interpolated crossing、dt halving | ranking/HDR 未收斂：縮 dt 並重跑受影響 cases |
| R016 | checkpoint、worker 或 shard 改變 seed | M/H | hash-derived seed、seed table、restart/merge 等價測試 | checksum/ID 不一致為 G3 blocker，禁止只重跑「看起來失敗」的 member |
| R017 | KDE bandwidth 與分母選擇主導結論 | H/M | raw exits、1D boundary + 2D map、三 bandwidth、HDR、bootstrap、分母 sidecar | ranking 翻轉：不得給單一來源排序，改報範圍與不穩定性 |
| R018 | observed/reconstructed/unsupported 區被混合，低 coverage 誤讀為低來源或低路徑密度 | M/H | origin mask、每軌跡 reconstructed exposure、failure density、有效分母、forcing coverage 與成功率地圖；並列 observed-only/gap-safe、reconstructed 與 forcing-member sensitivity | reconstruction exposure 主導或空間失敗集中時降級結論、改採 gap-safe baseline 或改善模型；不發布未校正密度 |
| R019 | 大型資料或私密 SERVER path 誤提交 Git | L/H | `.gitignore`、root token、secret/path scan、只提交 schema 與小 fixture | 發現即停止發布，移除敏感歷史並重建乾淨 release |
| R020 | 中文註解、README、schema 與程式行為不同步 | M/M | PR checklist、docstring/README/tests 同任務更新、每週文件檢查 | 行為已變但文件未更新：不得合併或通過 gate |
| R021 | local domains 重疊造成 site 歸屬、事件或分母混用 | M/H | own-local first-exit 與 foreign-local diagnostic 使用不同 event types；`study_site_id` immutable；跨站比例按原站有效 members 正規化 | foreign crossing 改變 scenario/site/seed/停止狀態或進入主要入口分母時，視為 G2/G5 blocker |
| R022 | 含 Stokes 軌跡超出 NWW analysis 或 OCM native 的實際支撐 | M/H | A 本期由 OCM surface 驗證完整 arrival 母體；runtime 再於每個 RK4 階段分別驗證 OCM native、NWW analysis、UTC、有限值與逐時 mask，並保存失敗位置及狀態。B-D 保留原 `expanded_domain` 敏感度，仍依各自支援契約驗證 | 任一必要 forcing 在實際 stage 無效即停止該粒子並列入 `data_gap`／對應品質分母；不以 current-only、最近值或零值結果冒充 baseline |
| R023 | iOcean 清除分類或文獻量級被誤讀為單體沉降校準 | H/H | manifest 分開保存 `classification_source`、`velocity_source`、代表形狀、適用條件、`calibration_status` 與證據等級；YAML 與程式測試拒絕零速、正值及缺欄位 | 未取得樣本密度、尺寸、含氣／附著狀態與終端速度前，只能報告代理敏感度；不得以清除重量作來源先驗或把暫定值稱為實測 |
| R024 | 主管關注沉底漁具被誤讀成統計權重，或 repeated contact 被當成基線必要資料 | M/H | 報告層固定以 `study_site_id × material_id` member 分母計算；首次接觸／沉積各自保存 raw count 與比例，`BED_CONTACT` member-level 去重 | 只有正式樣本或再懸浮敏感度才能加入覆網掛附、拖曳、再移動與 repeated-contact 動力；目前不調整基線速度與情境矩陣 |

## 5. 資源不足時的裁決順序

完成速度以工程並行與效率提升處理，不以日期驅動刪減科學範圍。若資源仍不足，由研究團隊依下列順序裁決並記錄：

1. 先增加合理的 shard 並行度、向量化／Numba、local scratch 與流式聚合，並以 benchmark 選擇最小收斂 `M`。
2. 在不改變積分精度與事件偵測的前提下，降低經驗證的軌跡儲存頻率；raw events、聚合量、失敗資訊與可重現 manifest 不得刪除。
3. 延後非核心動畫、額外探索圖面與未列入核心矩陣的物理案例；每站 10,000／全案 50,000 baseline、核心敏感度與驗證仍保留。
4. 若需改變每站 10,000 或全案 50,000 個基礎情境，必須建立明確的新版範圍決策；不得自行恢復每站 1,000、A 區兩站共用 20 receptors 的設計，也不得把代表期間結果命名為 2024-2025 全期成果。
5. 任何情況均不得省略 schema/單位/方向 gate、合成測試、dt/member 收斂、失敗率、seed/checksum、有效分母或限制措辭。
