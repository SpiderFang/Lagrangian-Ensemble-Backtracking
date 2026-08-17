# 決策與風險登錄

## 1. 狀態定義

- `decided`：已有範圍、依據、版本與影響；修改需新 decision record。
- `provisional`：可供程式與 pilot 使用，但正式兩年 run 前仍需核定。
- `open`：不得以任意預設啟動受影響的正式 run。
- `blocked`：缺少權限、資料或外部決策，且沒有安全替代路徑。
- `superseded`：被新版決策取代，歷史仍保留。

## 2. 決策登錄

| ID | 狀態 | 必須完成的閘門 | 問題與目前處理 | 未決時的限制 | owner |
|---|---|---|---|---|---|
| D000 | decided | 範圍基線 | 專案只實作紅框「三、Lagrangian 系集逆向溯源」，前處理、SVD、TRAP 不在本 repo 重作 | 範圍擴充需新決策 | 研究團隊／開發 |
| D001 | provisional | G0 | 使用者確認 2024-2025 OCM/NWW 已前處理完成；路徑基線來自相鄰 runbook。本執行環境唯讀 SSH 因無可用認證未成功 | 可寫程式與本機 trial；正式 inventory、run blocked | 資料管理者 |
| D002 | provisional | G0/G1 | SCHISM 參考文件支持 hvel/w 為 m/s、diffusivity 為 m²/s、z positive-up；`wetdry_elem` 0/1 與 SERVER metadata 仍需確認 | 未確認欄位不得被靜默轉換；濕乾事件正式 run blocked | 海洋數值人員 |
| D003 | provisional | G1 | NWW 相鄰專案採 `DP` wave-from、第一／第二 wind component east/north 的事件推定慣例；本工項只需 DP，仍須在 config 明示 | 可做有標記的 pilot；正式報告需保留 inferred 限制 | 海洋數值人員 |
| D004 | decided | 情境基線 | 使用者裁決矩陣的 `10×20×50=10,000` 為正式完整交叉，敘述中的 1,000 為誤植。`M` 是每情境獨立隨機實現數，由 member convergence 另定 | 不得把 baseline 分層縮為 1,000；未決 `M` 時只能跑 deterministic／pilot | 研究團隊／數值／系統 |
| D005 | open | G0/G3 | 20 個 receptor 在 A-D 四個分析海域的分配、geometry、z/HAB、空間／垂向／時間誤差與現場依據 | 可用合成 receptor 測試；不得凍結正式 scenario matrix | 研究／現場團隊 |
| D006 | open | G0/G3 | 10 個浮沉速度／物性類別、符號、分布與文獻依據 | 可用解析測試值；不得產科學結果 | 研究團隊 |
| D007 | provisional | G0/G3 | 50 arrival times 建議 48 個年份×季節×大／小潮×3，加 2 個事件；潮汐分類與事件門檻待核定 | 可實作 selector；正式 50 時次 blocked | 研究／統計 |
| D008 | open | G2/G4 | `max_backtrack_days`、open boundary、coast/bed/surface policy | 可用合成預設；正式 travel-time 統計 blocked | 研究／海洋數值 |
| D009 | provisional | G2 | 常數 Kh/Kz 作 reference；Smagorinsky + gradient drift 通過 PDE/well-mixed 測試後升為基準或敏感度 | 未通過時不得把空變 K 當正式基準 | 海洋數值／統計 |
| D010 | provisional | G3 | 軌跡採 ragged NumPy columns、事件／scenario 採 Parquet、immutable shards | benchmark 若顯示 I/O 不合適，需新 schema minor/major 決策 | 開發／系統 |
| D011 | open | Pilot/G4 | 正式 output root、local scratch、NFS publish、備份與配額 | 不得啟動大批次或把半成品寫入正式路徑 | 系統管理者 |
| D012 | provisional | G5 | 結果措辭限定 conditional footprint／relative source weight；absolute probability 需 prior/likelihood/observations | 未補證據不得升級措辭 | 研究團隊 |
| D013 | decided | G5 | 成果採文獻支持的代表軌跡、條件式密度／HDR、來源—受體矩陣、旅行時間分布、季節／潮況小多圖與不確定性／敏感度圖組；規格見文件 07 | 核心圖與統計表不得以單一動畫或全軌跡疊圖取代 | 研究團隊／開發 |

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
| R011 | domain 太小，人工邊界主導 exit KDE | M/H | 7-14 日擴域 pilot、早期 exit 比例、HDR/ranking 比較 | 主要指標變化 >10% 或大量短時同邊退出：擴域或限制結論 |
| R012 | domain 擴大或 `10,000×M×experiment cases` 導致計算／儲存爆量 | H/H | particle-step benchmark、最小收斂 M、向量化/Numba、shard/checkpoint、流式聚合與容量緩衝 | 超出資源時先增加合理並行、降低已驗證的儲存頻率並延後非核心案例；不得把 baseline 靜默縮為 1,000 |
| R013 | 20 receptor／50 times 尚未核定，假資料產生看似正式成果 | H/H | schema 與 synthetic test 可先行；正式 config 驗證 `decision_status=approved` | 未核定不得啟動 G4，輸出必須標 TRIAL |
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
3. 延後非核心動畫、額外探索圖面與未列入核心矩陣的物理案例；baseline 10,000 情境、核心敏感度與驗證仍保留。
4. 若需改變 10,000 個基礎情境，必須由使用者／研究團隊建立明確的新版範圍決策；不得自行恢復 1,000 分層設計，也不得把代表期間結果命名為 2024-2025 全期成果。
5. 任何情況均不得省略 schema/單位/方向 gate、合成測試、dt/member 收斂、失敗率、seed/checksum、有效分母或限制措辭。
