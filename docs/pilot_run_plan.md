# 單站沉降先導執行計畫

## 範圍與科學限制

此入口只供完整已驗收來源上的工程先導，不修改正式五站 50,000 個基礎情境、十類嚴格
負值沉降代理、OCM／NWW 契約或校準設定。來源必須先通過既有設定、清單、動態初始條件、
幾何及輸入盤點檢查，並驗證五站完整情境覆蓋；不是把裁剪清單當成完整母體。
不在程式內選定實際到達時間，也不依成功率、停止類型或圖面效果挑選案例。

`study_site_id` 是單一研究站點；`arrival_id` 對應來源 `arrival_time_id`，不是 UTC 字串或
列索引；`material_id` 是來源材質識別碼。三者都由操作端明示。選中組合必須對該站每個
來源受體恰好有一筆情境，不缺、不增、不重複。現行 5 水平×4 垂向形成 20 個情境，
**粒子數為情境數×M**；來源集合檢查不以寫死的 20 代替受體識別碼比對。

## 公開入口及持久化

- 純選擇函式：`pilot_selection.select_exact_pilot_scenarios(scenarios, receptors,
  expected_source_scenario_count, *, study_site_id, arrival_id, material_id, run_kind)`。
- 執行建立器：`runtime.initialize_run`／`initialize_pilot_run` 新增三個可選參數
  `pilot_study_site_id`、`pilot_arrival_id`、`pilot_material_id`。全省略保留舊行為；
  部分指定、重複命令列選項或與 `pilot_scenarios_per_stratum` 混用均拒絕。
- 識別碼只持久化於 `run_plan.json` 的 `scenario_selection`；不新增 config 欄位，
  不改 `pilot_execution_binding` 或 calibration artifact，不改來源清單。
- 根計畫維持 `2.1.0`；舊完整／分層選擇繫結仍為 `1.0.0`，舊 `2.0.0` 根計畫仍按
  原完整模式讀取。精確選擇使用獨立繫結版本 `2.0.0`、`mode=pilot_exact`，僅 pilot 可用。

精確繫結固定保存：版本與政策、站點／到達／材質 ID、完整來源與選中情境數及 ID 集合
SHA-256、完整來源情境／受體記錄 SHA-256、本站來源受體數及 ID 集合 SHA-256。
來源記錄依識別碼排序後以固定 JSON 計算內容指紋；包含未選中的記錄、UTC 奈秒、沉降
公尺／秒及受體模板公尺深度，不只驗 ID。動態實際初始深度仍由既有 component hash
綁定，不拿模板深度取代。計畫不保存 SERVER 私有路徑或憑證。

`load_validated_run_static_inputs` 在 run-shard／重開／resume 前重新驗證完整來源並套用
`apply_scenario_selection`；識別碼、內容指紋、數量、受體垂向對應或欄位遭改動即拒絕。
情境原物件與 ID 不變；粒子 seed 仍只取決於 master seed、scenario ID、experiment case
與 member ID，因此較小 M 的成員 seed 是較大 M 的前綴集合。此性質不保證不同 M 的
統計已收斂，也不表示任何後續執行軌跡必然逐點相同。

## 回溯天數的表示精度

設定仍保存 `max_backtrack_days` 浮點天數，一天固定 86,400 秒、一秒十億奈秒。
runtime 的 formal gap-safe 檢查與 `RuntimeRequestFactory` 共用同一個轉換函式：

1. 若既有 `Decimal(str(days))×86400×1e9` 已為整數奈秒，保留原值。
2. 否則先按 builder 相同的浮點 `days×86400` 得秒數，再用其最短十進位表示計算
   奈秒候選。距最近整數的誤差必須嚴格小於 0.5 ns，且不超過該秒數兩個相鄰
   浮點間距（ULP）；候選轉回秒、再除 86400，必須與原天數 float 完全相同。
3. 未滿足全部條件即拒絕，不以無條件 round、floor 或 ceil 放行。有效長度須為
   正有號 64 位奈秒；最早 UTC 以 `arrival_ns−horizon_ns` 整數運算，亦不得溢位。

兩個浮點間距涵蓋乘法及十進位表示的誤差，不能代替往返核對。`1/24` 天可還原為
3600 秒及 3,600,000,000,000 ns，`1/48` 天為1800秒；由 10 ns 換算的可還原天數
亦接受。`1e-12` 天＝86.4 ns、10.5 ns 及可分辨但不能精確往返的鄰近輸入仍拒絕。
浮點數本來無法區分的更細時間不能靠本函式恢復，不宣稱任意設定都有奈秒準確度。
這不改 builder YAML、run-plan schema、來源選擇、arrival、沉降、步長、seed 或 RNG；
不合法時間在建立 forcing manager 前停止，也不代表已修復任何積分或海洋取樣失敗。

## 重建命令

下列 task-specific 變數必須由操作端先明示；文件不提供或猜測部署位置與實際 arrival。
PILOT_CONFIG 應是先前通過校準／輸入驗證的執行設定；M、dt、horizon 沿用該設定。

```bash
uv run lbt run-create \
  --config "$PILOT_CONFIG" --input-inventory "$PILOT_INVENTORY" \
  --destination "$PILOT_RUN_ROOT" --run-id "$PILOT_RUN_ID" \
  --run-kind pilot --experiment-case "$PILOT_EXPERIMENT_CASE" \
  --pilot-study-site-id "$PILOT_SITE_ID" \
  --pilot-arrival-id "$PILOT_ARRIVAL_ID" \
  --pilot-material-id "$PILOT_MATERIAL_ID"
```

後續依計畫列出的**全部 shards**逐一執行既有 `run-shard`；不能只挑一個分片完成就
宣稱整個先導完成。續跑保留原 config、seed 與 checkpoint root，使用原 `--resume`
契約；全部完成後執行 `validate-run --require-complete`。部署、真資料取樣與結果審查
是獨立步驟；本機合成測試只驗工程契約，不是 PI 真資料成果。

負沉降速度表示正向物理時間的沉降；逆向時間曲線變淺不代表材料具有上浮物性。
單站／單到達／單材質工程試跑未經 M、dt 收斂及獨立觀測驗證，不宣稱全期代表性、
絕對來源機率或因果來源歸因。

## 獨立工程預覽

公開函式為 `pilot_preview.build_pilot_preview(run, *, config_path, output,
checkpoint_root=None, font_path=None, max_particles=2000, max_observations=250000,
max_curves_per_vertical=20)`。它不擴充正式報告的固定拓撲，也不新增主 CLI 命令。

```bash
uv run python scripts/build_pilot_preview.py \
  --run "$PILOT_RUN" --config "$PILOT_CONFIG" --output "$PILOT_PREVIEW" \
  --checkpoint-root "$PILOT_CHECKPOINT_ROOT"
```

外置 checkpoint root 只在原執行使用時提供；預設在 run 內。輸出應為新的
`<runid>.pilot-preview-v1` sibling 或操作端明示的新目錄，父目錄須事先存在；不得在
來源 run 內建立、覆寫任何節點或經符號連結。`MPLCONFIGDIR` 必須由操作端設定為
專用、已存在的普通可寫目錄。可另傳 `--font-path`；字碼不足時圖用英文，繁中說明
仍保留，不下載字型、地圖或任何外部素材。

入口先以小型 `load_run_plan`／分片 manifest 宣告數量做「早期拒絕」，並不視為驗收。
粒子／觀測預設上限為 2,000／250,000；操作端至多提高至 10,000／2,000,000。
完整靜態來源仍由 `runtime.load_validated_run_static_inputs(expected_run_kind='pilot',
require_complete=True)` 驗證，再由 `iter_complete_run_trajectory_shards` 逐片讀取及
重新核對身分／計數。僅接受單站、同到達、同材質、`pilot_exact` 及軌跡 schema `2.0.0`；
legacy 因環境欄位不可用而拒絕此預覽，不改一般讀取器的舊版相容性。

輸出固定為三張 PNG、`particles.csv`、`observations.csv`、`summary.json`、繁中
`README.md` 及 `manifest.json`。水平圖以來源 AEQD 等距方位投影公尺座標呈現總覽與
每個水平受體局部面板；不推測海岸、不放大位移。深度圖完整保留四垂向，near_bed
優先，海面 eta、海床 bed、粒子 z 取既存實值、公尺正向上；缺值留空。
觀測列保留 UTC 奈秒及回溯秒數，不重新取樣。全部停止類別（含零計數）、失敗
`numerical_failure`／`data_gap` 及全部 M 成員均保留；終止位置樣本 n、情境 n 與
粒子 N 分開列出，有限位置可包含失敗，不等於科學有效樣本。

畫線依 member／receptor／scenario ID 固定順序，預設每垂向至多 20、硬上限 100；
不能低於該垂向水平受體數，以便每個局部面板至少保留一條各層曲線。面板明列畫出／
全部數量，與完整統計分母不同，不能用停止類型或位移長短挑選。輸出清單逐檔保存
SHA-256（清單本身不循環自我雜湊），summary 保存計畫、設定、來源 component／geometry、
軌跡清單及程式指紋、OCM／NWW 來源契約、沉降速度、Kh/Kz 設定、dt、horizon、M、seed。

診斷使用實際失敗事件 `failure_reason`／`failure_stage`／`qc_flags`、
`diagnostic_version`、`attempted_dt_seconds`、步數／下限累計／上限與
`sample_x_m/y_m/z_m/time_utc_ns/eta_m/bed_z_m`。來源 availability 布林值原樣保留；
省略欄位在預覽表格為空／null，不推定為零，也不拿終止位置代替失敗查詢位置。
發布前再核對來源文件指紋，使用同父目錄原子拒覆寫，僅清理自有暫存目錄；不支援
排他改名時停止，改名後 fsync 失敗則保留 final 並回報耐久性未確認。
