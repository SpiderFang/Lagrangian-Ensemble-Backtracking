# 決策與風險登錄

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
| D001 | derived_pending | G0 | 使用者確認 2024-2025 OCM/NWW 已前處理完成；實際 SERVER inventory 依唯讀 preflight 產生 | 可寫程式與本機 trial；inventory 未通過前正式 run blocked | 資料管理／自動 preflight |
| D002 | derived_pending | G0/G1 | SCHISM 參考文件支持 hvel/w 為 m/s、diffusivity 為 m²/s、z positive-up；`wetdry_elem` 0/1 由 metadata、實值 snapshot 與測試確認 | 未通過欄位不得被靜默轉換；不需人工任選語意 | 自動 preflight／海洋數值審查 |
| D003 | derived_pending | G1 | NWW 相鄰專案採 `DP` wave-from；config 固定 `+180°` 轉 propagation-to，並以 cardinal arrow QC 驗證 | 若 metadata 衝突則停止並升版；不靜默猜測 | 自動 preflight／海洋數值審查 |
| D004 | decided | 情境基線 | `10×20×50=10,000` 套用於貢寮、龜山島、新竹、後灣、連江每一獨立站點；A 區 20,000、全案 50,000，1,000 為誤植。`M` 由 member convergence 衍生 | 不得把任一站 baseline 縮為 1,000、把 A 區兩站合併或把全案縮為 10,000 | 研究團隊／數值／系統 |
| D005 | derived_pending | G0/G3 | 每站 5 個 deterministic maximin 水平位置 × `0.10H/0.40H/0.70H/near-bed` 四層，共 20；全案 100。實際座標由 OCM persistent-wet mesh 生成 | manifest 未通過前可用合成 receptor 測試，不得凍結正式 scenario table | geometry selector／數值審查 |
| D006 | decided | G0/G3 | 十個垂向行為速度固定為 `-0.100,-0.030,-0.010,-0.003,-0.001,0,+0.001,+0.003,+0.010,+0.030 m/s`，作未校準行為敏感度類別 | 不得把類別名稱寫成有量測支持的特定材質 | 研究方法基線 |
| D007 | derived_pending | G0/G3 | 每站 50 個 arrival times 採 48 個年份×季節×大／小潮×三潮位相位 proxy，加局地高波與強流各 1；確切 UTC 由 forcing selector 產生 | 任一站未滿 50 或 coverage 不足時正式矩陣 blocked | deterministic selector／統計審查 |
| D008 | derived_pending | G2/G4 | local first exit 記錄後續跑、flow-domain exit/coast/deposition/surface-regime 停止；`max_backtrack_days` 比較 7/14/30/60 日取最小穩定值 | horizon 未收斂前只可跑 pilot；邊界政策不再任意選擇 | 數值 pilot／方法審查 |
| D009 | derived_pending | G2 | 常數 Kh/Kz 作 reference，實際值由 Brownian/well-mixed 與 pilot 決定；Smagorinsky + gradient drift 通過 PDE/well-mixed 測試後才可升為基準 | 未通過時不得把空變 K 當正式基準 | 數值驗證／統計 |
| D010 | provisional | G3 | 軌跡採 ragged NumPy columns、事件／scenario 採 Parquet、immutable shards | benchmark 若顯示 I/O 不合適，需新 schema minor/major 決策 | 開發／系統 |
| D011 | derived_pending | Pilot/G4 | 正式 output root、local scratch、NFS publish、備份與配額由容量／檔案系統 preflight 決定 | 未通過不得啟動大批次或把半成品寫入正式路徑 | 系統 preflight／管理者 |
| D012 | provisional | G5 | 結果措辭限定 conditional footprint／relative source weight；absolute probability 需 prior/likelihood/observations | 未補證據不得升級措辭 | 研究團隊 |
| D013 | decided | G5 | 成果採文獻支持的代表軌跡、條件式密度／HDR、來源—受體矩陣、旅行時間分布、季節／潮況小多圖與不確定性／敏感度圖組；規格見文件 07 | 核心圖與統計表不得以單一動畫或全軌跡疊圖取代 | 研究團隊／開發 |
| D014 | decided | G0/G5 | 貢寮／龜山島舊候選 bbox 過小，不作 local domain；兩站各以 anchor 半徑 12.5 km receptor core、25 km local domain，並比較 20/35 km。local domains 允許重疊且不作 Voronoi 切割；35 km 需 expanded forcing domain | 重疊不合併 scenario 或 denominator；25 km 需保留至少兩格 outer margin，尺度敏感時報告範圍 | 使用者／研究方法基線 |

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
| R002 | 「已前處理完成」仍含 partial month、時間缺口或 schema 差異 | M/H | 逐 domain/month 讀 metadata/time/QC，不以目錄存在判定 ready | 任一差異分版；arrival times 排除缺口或由研究團隊核准限制 |
| R003 | OCM wetdry、w 或 Kz 語意錯誤 | M/H | 參考文件、實值分布、海岸 snapshot、合成與局地診斷交叉檢查 | 無法確認則基準不用 wetdry/Kz 的物理解釋，相關 case 降級 |
| R004 | NWW wave-from/to 解讀反向 | M/H | cardinal unit tests、事件慣例版本、傳播箭頭 QC、DP+180 對照 | 發現衝突立即 major method version，所有 Stokes run 重跑 |
| R005 | native face triangulation 在 quad、洞或海岸跨越錯誤 | M/H | 只用原生 connectivity、退化／orientation QC、coast mask、真實圖面抽查 | 任一跨陸地案例為 G1 blocker，禁止用全域 Delaunay 迴避 |
| R006 | 垂向內插在陡坡或海床下製造有效速度 | H/H | 每 node 包夾、三 node 完整支撐、surface/bed event、invalid reason | data-gap/bed-contact 異常集中時回查 sampler，不以最近層填補 |
| R007 | backward diffusion 被誤當真實歷史或機率 | H/H | conditional-footprint 命名、正向-逆向 synthetic、分母與 prior 欄位、措辭審查 | 未通過合成驗證禁止 probability／source attribution 字樣 |
| R008 | 空變 K 缺 gradient drift，造成人工聚集或穿障壁 | H/H | 常數 K reference、PDE/well-mixed/障壁測試、Milstein sensitivity | 測試失敗：Smagorinsky 不進 baseline，只保留工程診斷 |
| R009 | bulk Stokes 在近岸、淺水或混合風浪下偏差大 | H/H | finite/deep/no-Stokes、kh/steepness QC、方向與 mean wavelength 對照 | ranking 對 formulation 高敏感：列主要不確定性，不給單一結論 |
| R010 | OCM 已含波流耦合效應，額外 Stokes 可能重複 | M/H | 查 OCM 產品說明與模式設定；將 no-Stokes 設為核心對照 | 無法確認：報告明載可能 double counting，不作精確量值歸因 |
| R011 | local/flow domain 太小，人工邊界主導 entry/exit KDE | M/H | 舊候選 bbox 已撤銷；貢寮／龜山島比較 20/25/35 km，另作 flow-domain expansion、早期 exit、HDR/ranking 比較 | 主要指標變化 >10% 或大量短時同邊退出：報告尺度依賴性並建立新版 domain |
| R012 | domain 擴大或 `50,000×M×experiment cases` 導致計算／儲存爆量 | H/H | particle-step benchmark、最小收斂 M、向量化/Numba、shard/checkpoint、流式聚合與容量緩衝 | 超出資源時先增加合理並行、降低已驗證的儲存頻率並延後非核心案例；不得把任一站 baseline 靜默縮為 1,000 |
| R013 | 每站 20／全案 100 receptors 或每站 50 times 的衍生 manifest 尚未產出，trial 資料被誤當正式成果 | H/H | schema 與 synthetic test 可先行；正式 config 驗證 derived gate 與 `status=approved` | 未通過不得啟動 G4，輸出必須標 TRIAL |
| R014 | 粒子停留不出界，模擬無限延長 | M/H | forcing start、max age、step limit、data gap 停止；各原因分開統計 | max-age 比例過高：調整研究問題/domain/horizon，不把它當 exit |
| R015 | dt 過大漏掉窄通道、海岸或 crossing | M/H | advective/vertical/diffusive CFL、step-interpolated crossing、dt halving | ranking/HDR 未收斂：縮 dt 並重跑受影響 cases |
| R016 | checkpoint、worker 或 shard 改變 seed | M/H | hash-derived seed、seed table、restart/merge 等價測試 | checksum/ID 不一致為 G3 blocker，禁止只重跑「看起來失敗」的 member |
| R017 | KDE bandwidth 與分母選擇主導結論 | H/M | raw exits、1D boundary + 2D map、三 bandwidth、HDR、bootstrap、分母 sidecar | ranking 翻轉：不得給單一來源排序，改報範圍與不穩定性 |
| R018 | 失敗／缺值區被誤讀為低來源或低路徑密度 | M/H | failure density、有效分母、forcing coverage 與成功率地圖 | 空間失敗集中時先修資料／sampler；不發布未校正密度 |
| R019 | 大型資料或私密 SERVER path 誤提交 Git | L/H | `.gitignore`、root token、secret/path scan、只提交 schema 與小 fixture | 發現即停止發布，移除敏感歷史並重建乾淨 release |
| R020 | 中文註解、README、schema 與程式行為不同步 | M/M | PR checklist、docstring/README/tests 同任務更新、每週文件檢查 | 行為已變但文件未更新：不得合併或通過 gate |

## 5. 資源不足時的裁決順序

完成速度以工程並行與效率提升處理，不以日期驅動刪減科學範圍。若資源仍不足，由研究團隊依下列順序裁決並記錄：

1. 先增加合理的 shard 並行度、向量化／Numba、local scratch 與流式聚合，並以 benchmark 選擇最小收斂 `M`。
2. 在不改變積分精度與事件偵測的前提下，降低經驗證的軌跡儲存頻率；raw events、聚合量、失敗資訊與可重現 manifest 不得刪除。
3. 延後非核心動畫、額外探索圖面與未列入核心矩陣的物理案例；每站 10,000／全案 50,000 baseline、核心敏感度與驗證仍保留。
4. 若需改變每站 10,000 或全案 50,000 個基礎情境，必須建立明確的新版範圍決策；不得自行恢復每站 1,000、A 區兩站共用 20 receptors 的設計，也不得把代表期間結果命名為 2024-2025 全期成果。
5. 任何情況均不得省略 schema/單位/方向 gate、合成測試、dt/member 收斂、失敗率、seed/checksum、有效分母或限制措辭。
