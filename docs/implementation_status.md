# 實作狀態與正式驗證邊界

本文件是目前程式狀態的短版權威索引；歷史 Phase／Slice 文件保留設計脈絡，不能取代
本文件、目前程式碼與當次驗證結果。這裡的「已實作」是指有程式契約與工程測試，不等於
已完成 2024–2025 全期真資料批次、正式科學 evidence 或學術結論。

## 目前已實作的工程能力

| 領域 | 目前可由程式與測試確認的內容 | 正式性界線 |
|---|---|---|
| 設定與輸入 | 設定、來源 manifest、OCM／NWW3 schema、UTC 軸、幾何與 dynamic pair 的 strict loader／validator | 仍須以當次 accepted-product inventory 與正式 release manifest 通過 gate |
| forcing | OCM native 三維 current、OCM surface selector、NWW3 analysis、有限水深 Stokes、月份視窗與 cache | 只讀已驗收產品；缺值不補零、未知跨月／跨網格外插拒絕 |
| engine | 正向物理速度的 signed-time RK4、獨立隨機擴散、沉降、巢狀邊界與事件；CPU／NumPy reference orchestration | synthetic 與 unit／integration tests 不構成真資料科學驗證 |
| 情境與執行 | 五站點 scenario builder、固定排序、pilot／formal workspace、shard／chunk、progress／lock、reconcile／validate | formal 仍須 approved config、完整 inventory、gap-safe／full-product evidence |
| checkpoint | execution checkpoint writer schema `3.0.0`；immutable history segment + compact current state、SHA-256 chain、cursor／identity／binding／RNG continuation 驗證；讀取 schema `2.0.0`、`2.1.0`、`2.2.0` 舊拓撲 | 舊 2.x 目錄唯讀且不原地升級；v3 synthetic round-trip／故障拒絕測試不構成真資料 30 天正式成果證據 |
| trajectory | 新 writer schema `3.0.0`，含環境與速度 payload；reader／validator 可讀 `1.0.0`、`2.0.0`、`3.0.0` 的固定拓撲 | v1 不得作正式垂向證據；v2 保留既有位置／環境報告用途；同一正式 run 不混用版本 |
| 跨區 pilot 矩陣 | `pilot-matrix-validate` 只讀各 run 的 `run_plan.json` 與 `normalized_config.json`，比較 M／seed、experiment、積分／邊界／Stokes、材質、scalar snapshot 與 deployment provenance；明示區域 identity、輸入／幾何 binding 與 Kh／Kz／Smagorinsky cap 可不同 | 只證明工程設定是否可比較，不驗 trajectory／forcing 內容、不判定 run 完成，也不構成正式科學 evidence |
| 聚合與報告基礎 | aggregate／report statistics、trajectory stream、source binding、release I/O、共同 staging／格式／checksum 基礎 | caller 必須提供已驗證 products；不能由 typed facade 或 synthetic release 推導科學完成 |

## A 區本期政策與尚缺的實測門檻

本期 A 區不南向擴域，現行 `northeast_taiwan_common_cache_v3` bbox 為
`[121.306315, 122.793685, 24.600844, 25.499156]`。本期 geometry／design 身分採
`design_baseline_v3_non_rising_a_v3_local20_20260909`，範圍政策為
`formal_domain_policy=v3_local20km_20260909_v1`：貢寮與龜山島各有 12.5 km receptor
core、20 km local domain，共用原 outer stop。原 material v2 non-rising 物理契約仍承接，
但舊 25 km geometry、南擴 v4 候選與 35 km A 敏感度只作歷史／延期依據，不是本期 formal
產製目標；不新增 15／23 km。B–D 原 `expanded_domain` 敏感度仍保留，與 A 政策分開
驗證；legacy `formal_domain_policy=expanded_domain_v1` 僅供舊設定相容讀取。

新政策可供 strict preparation，但尚未解除 formal gate。仍缺現行 v3 實際 OCM native、OCM
surface 與 NWW analysis 的共同 forcing 邊界餘裕證據（至少兩個共同有效格點），以及完整
field、mask、UTC、arrivals 與正式輸入 manifest 驗收；目前 existing gate 只做 receptor
native 篩選，不是 20 km 邊界的三 forcing／兩共同格點驗證。正式 gate 未通過前不稱 20 km
已被科學證實足夠，也不以準備工作節省承諾端到端加速或日期提升。

## ABCD 第一次工程試跑稽核

目前已取得 A 區貢寮與龜山島兩站各 20 粒子（合計 40）、B／C／D 各 20 粒子，均為 24 小時、
4 shards、`M=1` 的第一次工程試跑紀錄；
明示輸入 registry 固定為 `2024-01-02T01:00:00Z` 到前一日同時刻、inclusive 25 個逐時節點。
A 區必須以貢寮與龜山島 exact pair 建立，B／C／D 各自以單站建立。完整數值與限制見
[ABCD 第一次試跑稽核](results/15_four_region_first_pilot_audit.md)。

稽核已確認既有結果不能直接稱為「相同設定」：A 使用 `no_stokes`，B／C／D 使用
`finite_depth_stokes`。各區因資料與校準可有不同的 Kh、Kz 與 Smagorinsky cap，這些是
`pilot-matrix-validate` 白名單允許的區域差異；Stokes 實驗、M／seed、積分／邊界、材質與
部署 provenance 則必須 exact match。A 需以 `finite_depth_stokes` 和共同乾淨 snapshot 重跑，
四區也要統一程式 tree 與環境後，才可重新判讀跨區比較。

第一次稽核的工程診斷是：貢寮 `forcing_start=19` 且有 1 個 `numerical_failure`；龜山島
`forcing_start=11`、`max_age=1`、`flow_domain_open_exit=6`、`coast_contact=2`；B
有 5,780 筆 observation 與 825,070 steps；C 有 14 個 `data_gap`、5 個 `max_age`、
1 個 `numerical_failure` 與 3,097 筆 observation；D 有 12 個 `data_gap`、8 個 `max_age`、
4,397 筆 observation 與 824,794 steps。C／D 的 `data_gap` 帶 `QC=32`（NWW 波浪空間支援
不足）；C 的失敗階段為 `step_start/k2/k4=1/8/5`，D 為 `4/4/4`。這些是待解釋的工程診斷，
不能以零值或圖面完整性掩蓋。

## NFS preview／figure 發布標記的語意

`pilot_preview` 與海岸圖面的 NFS `nfs_completion_marker_v1` 會在同一父目錄鎖、完整 staging
自我驗證、逐檔 durable move、manifest／程式指紋／儲存閘門綁定後，最後建立 `.complete`。
`.complete` 只代表 preview／figure artifact 的 inventory、bytes 與 provenance 可供 reader
讀取；它不是 run lifecycle 的完成旗標。run 是否完成仍必須由 `run_progress.json`、
`run-reconcile` 及 `validate-run --require-complete` 判定。A 早期圖面因 NFS 不支援
`renameat2(RENAME_NOREPLACE)` 未發布，但其 run／reconcile／benchmark／checksum 檢查已通過；
B／C／D 圖面可讀不會改變其 pilot-only 性質。

## 逐點速度紀錄契約

定期及終止的既有 `Observation` 均可附帶速度；只有當其 `x_m`、`y_m`、`z_m` 與世界協調
時間（UTC）和已取得的步首參考取樣完全相同時才附上。沒有同點參考就保留
`not_sampled`／缺值，不借用其他位置或四階 Runge-Kutta 法（RK4）的 stage／internal step；
它不是位移除以時間的平均速度，逆向積分仍保存正向物理方向，不另行反號。

九個速度欄位的單位都是 m/s：

`total_u_mps`、`total_v_mps`、`total_w_mps`、`ocm_u_mps`、`ocm_v_mps`、
`ocm_w_mps`、`stokes_u_mps`、`stokes_v_mps`、`settling_w_mps`。

其中 `u/v/w` 分別是東、北、向上的方向；總速度滿足
`total_u = ocm_u + stokes_u`、`total_v = ocm_v + stokes_v`、
`total_w = ocm_w + settling_w`，使用明定容差驗證。`settling_w_mps` 是以向上為正的
垂向沉降速度；本專案沉降值為負。本模型只納入水平 Stokes 與垂直沉降；Stokes 垂向與
沉降水平不是本模型分項，缺少波浪資料時必須保留缺值／非零 QC，不得當作 0。

`CombinedMonthForcing` 的正式有效樣本提供完整分項；明確關閉 Stokes 可記有效的 0，自然
海況計算結果也可能是有效的 0，但沒有 NWW3 波浪資料則是缺值並帶非零 QC，不能記成 0。
只有總速度的 synthetic callback 標為 `total_only`，不能當作 OCM／NWW3 來源證據。`not_sampled`、`missing`、`invalid`、
`nonfinite` 與 `sum_mismatch` 各自保留狀態；缺資料在資料契約中是 `None`／NaN sentinel，
不可與 0 混淆。

## 報告與 evidence 狀態

### 已完成的工程邊界

- `report_trajectory_stream.py` 可由完整且已驗證的 trajectory iterator 建立 pathway、環境完整性、材質與代表軌跡統計；失敗 member 不進有效分母，raw numerator／denominator 與零分母狀態保留。
- `report_statistics.py` 提供停止結果、方向性跨站、source–receptor、pathway、travel age、KDE/HDR 與材質產品的 renderer-facing facade；它不負責圖表或自行讀取 forcing。
- `report_release.py` 提供 report-v1 的 source binding、固定 F01–F12／T01–T06 registry、exact inventory、checksum 與 atomic sibling writer／reader／validator。
- `report_render.py` 的共同 staging、固定格式、canonical sidecar、bytes／SHA-256 與 immutable staging view 已有契約，但不是完整報告建置器。
- `report_validation_evidence.py` 已有 schema `1.0.0` 的 immutable `ValidationMetric`／`ValidationEvidence` 及 strict canonical JSON I/O；schema／I/O 通過不表示正式測試已執行。
- `report_pipeline.py` 的建置前唯讀 gate 會驗證 complete run、aggregate／spec binding、MPLCONFIGDIR、output／evidence policy 與 formal trajectory schema。正式可消費的 trajectory 版本是 v2 或 v3，會回報實際 manifest version；v1 明確拒絕正式垂向證據，v2／v3 混用也拒絕。
- `source_pathway_release.py` 提供獨立 `source-pathway-v1` build／validate：由已驗證 aggregate
  與 ReportSpec 建立向下沉降情境的每站六面板圖、三張 Parquet sidecar、caption 與 manifest。
  它沿用 `build_report_statistics` 的訪格／first-passage／停留／KDE／停止統計，保存
  `bed_first_contact_count`／`bed_repeated_contact_count` 的底床接觸診斷，並以
  `aggregate_manifest_sha256` 綁定來源 release；這是條件式來源足跡工程產品，不加入完整
  F01–F12／T01–T06 registry，也不宣稱沉積質量、絕對來源機率或正式科學 evidence。

### 尚待正式驗證或發布

主線 `report-build`、F01–F12／T01–T06 專屬 artifact adapters、
`report_comparison_statistics.py` 的 comparison release I/O 的
完整 pipeline／render 整合與正式 SERVER 科學發布尚未完成；目前 CLI 也沒有 `report-build`
子命令，因此不能把 preflight 稱為完整 pipeline。F11/T05 的 exact compatibility 與核心差值純計算已存在，但真實 sensitivity cases、
材質來源核對與 release 整合仍缺，因此不能稱 F11/T05 科學成果完成。

F12/T06 的正式 evidence 尚未產生：解析解 `analytic_solution`、dt／M 收斂
`timestep_convergence`／`member_convergence`、known-source `known_source_synthetic`、
restart `checkpoint_restart`、NumPy／Numba `numpy_numba_consistency` 及
forward-validation `forward_validation` 都仍需以正式規格、真實或核准的驗證案例建立。
不能將 schema／I/O 通過、unit tests、synthetic evidence 或 pilot 圖表解讀為正式 OCM／NWW3
科學驗證，也不能因此宣稱全案完成。

## 驗證層級與 evidence 管理

工程測試用來鎖定資料契約、數值不變性、writer／reader round-trip、RNG continuation 與
consumer 相容性；測試通過不等於正式科學 evidence，也不把 pilot 或 synthetic 變成正式結果。
正式 evidence 必須由版本化設定、accepted input manifest、可追溯 run、獨立數值／敏感度／
觀測檢查建立，並保存 source binding、有效分母、QC、缺值狀態與 checksum。各次測試數字以
pytest／CI 報告、run record 與正式 evidence manifest 為準，本文件不固定單次執行計數。

目前不宣稱 SERVER 狀態；若需確認部署、資料權限、容量、lock 或 run lifecycle，應以當次
deployment／run records 與唯讀 preflight 為準，不以歷史 README 敘述推定現況。
