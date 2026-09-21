# WP-7：native 逆向批次 orchestrator 操作契約

本文件說明 `src/lagrangian_backtracking/native_runner/orchestrator.py` 提供的工作單元
規劃、排程、平行執行、續跑與到達時刻缺口安全檢查，以及對應的 `lbt native-plan`／
`lbt native-run`／`lbt native-status` 子命令與 `scripts/run_native_backtrack_server.sh`
伺服器入口。這是 WP-7 的交付範圍；它只負責「如何有效率、可續跑、可稽核地把工作單元
分給 worker 並執行」，不改變任何物理核心、時間積分或亂數契約，也不宣稱已完成波浪或
後處理整合（見下方「已知限制」）。

## 1. 工作單元與排程鍵值

每個工作單元對應一次可直接餵給 `backward_driver.run_backward_batch` 的粒子列位置集合，
鍵值是：

```
(analysis_region_id, arrival_time_utc_ns, settling_velocity_mps, study_site_id)
```

之所以保留 `study_site_id`，是因為貢寮與龜山島雖然共用 A 區流場資料，但各自有獨立的
`interest_region_polygon`（own local domain），而 `run_backward_batch` 要求同一次呼叫
內只能有一個 interest region；把 `study_site_id` 拆開可以避免把兩站粒子混進同一次呼叫
而破壞其中一站的局部邊界語意。

排程／forcing 重用鍵值則刻意更粗：

```
(flow_domain_id, arrival_time_utc_ns)
```

`plan_worker_schedule` 用這個較粗的鍵值把同一流場區域、同一到達時刻的工作單元盡量排在
同一個 worker 上連續執行，藉此讓作業系統頁面快取與 `forcing_bridge` 既有的月份滑動快取
發揮效果，不必每個工作單元都重新讀取相同的月份陣列。

`plan_native_work_units` 用 `np.lexsort` 依 `(analysis_region_id, arrival_time_utc_ns,
settling_velocity_mps, study_site_id)` 排序後切分，並重用既有 `validate_batch_partition`
（來自 `particle_input.py`）確認切分結果對來源粒子身分表無遺漏、無重疊。每個工作單元的
`unit_id` 由區域、站點、到達時刻、沉降速度與輸入內容雜湊的前 12 碼組成，具決定性；
`input_hash` 涵蓋來源身分表內容雜湊、切分鍵值與實際列位置，供續跑判斷輸入是否改變。

### 正式規模估算

`estimate_formal_native_work_plan` 依 README「五站四區的沉降情境」登錄的正式設計常數
（5 站、每站 20 個受體、50 個到達時刻，10,000 個 base 情境／站）分析式估算：

| 成員數 M | 工作單元數 | 每單元粒子數 | 粒子總數 |
|---|---|---|---|
| 1 | 2,500 | 20×M = 20 | 50,000 |
| 10 | 2,500 | 200 | 500,000 |

工作單元數本身不隨 `M` 改變（因為排程鍵值不含 member），只有每個單元的粒子數隨 `M`
線性增加；這與 `tests/test_native_orchestrator.py` 的
`test_estimate_formal_native_work_plan_matches_50000_times_m_design` 一致。

## 2. Worker 排程與記憶體策略

`plan_worker_schedule` 使用 LPT（Longest Processing Time first）貪婪 bin-packing：先把
工作單元依前述粗鍵值分組、以組內粒子數加總作為成本代理，再由大到小依序指派給目前負載
最輕的 worker；組內單元再依細鍵值排序，讓同一 worker 內同一流場、同一到達時刻的單元
仍然連續執行。這是啟發式平衡，不保證嚴格最佳解，但避免了舊架構「每個 shard 各自啟動一次
Python 程序」的重複匯入與 JIT 重新編譯成本。

執行引擎（`run_native_plan`）用 `ProcessPoolExecutor`（預設 `spawn` context）啟動固定數量
的 persistent worker，每個 worker 只呼叫一次 `--resource-provider` 指定的
`build_worker_resources`，之後在同一個子程序內連續處理被指派的整串工作單元，藉此讓
BayTrace 模組載入、網格建構與 Numba JIT 編譯只發生一次。`NUMBA_CACHE_DIR` 需在匯入
加速模組前設定為通過儲存檢查的絕對路徑（沿用既有安全路徑契約），才能讓同一部機器上的
重複啟動命中磁碟編譯快取。

Worker 數量由 `estimate_worker_count` 決定，輸入是**實際量測**的可用記憶體
（`read_available_memory_bytes` 讀取 `/proc/meminfo` 的 `MemAvailable`）與呼叫端提供的
單一 worker 預估用量，並可疊加 CPU 數上限、保留記憶體與人工上限；本模組刻意不像
`dynamic_execution.py` 那樣寫死 30 GiB 門檻——量測與預估用量都由呼叫端／CLI 參數提供，
在不同伺服器規格上都可重新設定。`lbt native-run` 也支援直接以 `--worker-count` 手動指定，
略過自動估算。

## 3. 續跑與失敗隔離

每個工作單元完成後，`_persist_outcome` 會先把結果摘要與完成標記寫到暫存檔，再以
`os.replace` 原子改名成 `<unit_id>.result.json`／`<unit_id>.done.json`；`.done.json` 內含
該次使用的 `input_hash`。`filter_pending_units` 在下一次呼叫 `run_native_plan` 時，會先讀
既有 `.done.json`，只有 `input_hash` 相同的單元才視為「已完成、可跳過」，`input_hash`
改變（例如來源身分表內容不同）則視為需要重算，不會誤用舊結果。若一批全部命中續跑，
`run_native_plan` 會直接回傳空的 `worker_reports`，不啟動任何子程序。

單一工作單元失敗（例如底層 `run_unit` 拋出例外）只會被該 worker 記錄下來、繼續處理下一個
被指派的單元，不會讓整批工作停止；`NativeRunSummary.all_succeeded` 為 `False` 時，
`lbt native-run` 會回傳結束碼 2，並在輸出中列出失敗單元清單與例外訊息，供人工檢視。

## 4. 進度回報

執行期間，`run_native_plan` 會定期（依已回報訊息數與時間）把目前累積的完成、失敗、跳過
單元數，以及依平均每單元耗時推算的粒子步數與估計剩餘時間，原子性寫成一個 JSON 快照；
`lbt native-status`／`read_progress_snapshot` 讀取同一份快照，並額外掃描輸出目錄下的
`*.done.json` 統計目前完成狀態分布，可在 `native-run` 執行期間隨時查詢，不影響正在跑的
worker。

## 5. 到達時刻缺口安全機制（重用既有選取器，非重新實作）

本模組**沒有**重新實作到達時刻選取邏輯。既有 `arrival_times.select_arrival_times` 已經
負責季節、潮汐分層與 `backward_window_available` 遮罩；WP-7 只新增一個橋接函式
`derive_backward_window_available`，把 `forcing_bridge.inventory_forcing_gaps` 的既有缺口
盤點結果，轉成 `select_arrival_times` 已經在用的 `backward_window_available` 布林遮罩
形狀，再原封不動呼叫 `select_arrival_times`：

```
inventory_forcing_gaps()  →  ForcingGapInventoryReport
                                    │
                    derive_backward_window_available()  (WP-7 新增的純函式橋接)
                                    │
                                    ▼
                    backward_window_available  →  select_arrival_times()  (既有，未改)
```

`check_arrival_gap_safety` 則是 `lbt native-plan` 用來做**唯讀前置檢查**的函式：對粒子
身分表內每個唯一 `(study_site_id, arrival_time_utc_ns)`，判斷其 30 天（可調）回溯窗是否
完全落在對應流場區域的已盤點缺口之外，回傳每一筆的 `is_gap_safe`、重疊的缺口 ID 與原因
（`ok`／`backward_window_overlaps_unregistered_gap`／`backward_window_exceeds_inventoried_range`／
`no_gap_report_for_region`）。`lbt native-plan` 會把所有不安全的到達時刻連同缺口起訖一併
寫進輸出報告與計畫檔的 `is_gap_safe_ready`／`unsafe_arrivals` 欄位；`lbt native-run` 預設
在計畫檔標記不安全時直接拒絕啟動（fail-closed），只有明確加上 `--allow-unsafe-arrivals`
才會覆寫。

若需要**重新選取**受影響的到達時刻，`rebuild_gap_safe_arrival_times` 會用最新缺口盤點
重新推導 `backward_window_available`，再呼叫**同一個**未修改的 `select_arrival_times`，
因此重選仍然套用既有的季節／潮汐分層規則與 48+2 分層邏輯，不是另一套規則。重選後應該用
`diff_stratum_coverage`／`summarize_stratum_coverage` 比較替換前後 `(season, tide_class)`
的分層涵蓋計數；如果某個分層在重選後從「有」變成「無」，`StratumCoverageDiff.is_degraded`
會標成 `True`，呼叫端必須明確在報告中呈現這個劣化，不能因為 `select_arrival_times` 本身
沒有拋例外就視為已通過驗收。

`tests/test_native_orchestrator.py` 內以下測試涵蓋這條路徑：
`test_check_arrival_gap_safety_finds_overlap_and_reports`、
`test_check_arrival_gap_safety_reports_missing_region_report`、
`test_derive_backward_window_available_excludes_only_windows_touching_gap`、
`test_rebuild_gap_safe_arrival_times_reuses_selector_and_flags_degradation`、
`test_rebuild_gap_safe_arrival_times_raises_when_too_few_candidates_remain`、
`test_diff_stratum_coverage_flags_stratum_that_disappears`。

## 6. Manifest 與 provenance

`build_native_run_manifest` 重用既有 `provenance.collect_code_provenance` 收集 Git commit
與 dirty flag，並記錄設定雜湊、seed policy、`master_seed`，以及（若提供）BayTrace 來源根
目錄與 `BayTraceModuleHandles.dependency_sha256`。`lbt native-plan`／`lbt native-run` 都會
把各自呼叫時組出的 manifest 寫入輸出，讓一次執行可以回答「用的是哪個程式版本、哪份設定、
哪個 seed policy、哪個 BayTrace 來源」。

## 7. CLI 用法

```bash
# 只規劃＋缺口檢查，不執行任何 kernel；秒級完成
uv run lbt native-plan \
  --identity-table /data/LBT/inputs/particle_identity_table.npz \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --max-backtrack-days 30 \
  --output /data/LBT/native-run/native-plan.json

# 執行／續跑；--resource-provider 指向呼叫端自備的
# (build_worker_resources, run_unit) 模組層級 factory
uv run lbt native-run /data/LBT/native-run/native-plan.json \
  --output /data/LBT/native-run \
  --resource-provider my_site_package.native_resources:make_resources \
  --auto-worker-count --per-worker-memory-bytes 2147483648 \
  --numba-cache-dir "$NUMBA_CACHE_DIR" \
  --baytrace-root "$LBT_BAYTRACE_ROOT"

# 查詢進度
uv run lbt native-status --output /data/LBT/native-run
```

伺服器入口 `scripts/run_native_backtrack_server.sh` 把上面三個子命令包成
`plan`／`start`／`status` 三個動作，`start` 會先重跑一次 `plan`、確認
`is_gap_safe_ready` 後才用 `nohup` 背景啟動 `native-run`，並印出 log 路徑與背景 PID；
完整環境變數與範例見腳本檔頭註解。

## 8. 已知限制（誠實揭露）

- **生產環境的 `build_worker_resources`／`run_unit` 接線刻意不內建**：實際網格建構、
  每站 `interest_region_polygon`、forcing 月份目錄解析與波浪橋接依賴
  `config.py`／`mesh.py`／`geometry.py`，以及 `wave_bridge.py`／`postprocess.py`；後兩者
  在 WP-7 交付時可能尚未完成。與其提供一個可能不正確或不完整的預設接線，`lbt native-run`
  透過 `--resource-provider` 讓呼叫端自行提供、自行對其正確性負責。
- **尚未整合波浪與後處理**：目前的 orchestrator 只負責 `run_backward_batch` 這一層的
  排程與執行；`wave_bridge`／`postprocess` 完成後，需要由另一個資源 provider 或後續工作
  把它們接進 `run_unit`，本模組不會自動變化。
- **排程是啟發式，不是最佳解**：LPT bin-packing 只保證合理平衡，不保證嚴格最短總時間；
  精確估算仍需以實際單元耗時校正。
- **記憶體估算依賴呼叫端提供的單一 worker 用量**：`estimate_worker_count` 本身不會自動
  量測某個特定 `run_unit` 實作的實際記憶體用量，需要呼叫端自行量測後傳入
  `--per-worker-memory-bytes`。
