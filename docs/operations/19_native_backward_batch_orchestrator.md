# WP-7／WP-9：native 逆向批次 orchestrator 與小時主軸排程操作契約

本文件說明 `src/lagrangian_backtracking/native_runner/orchestrator.py` 提供的工作單元
規劃、排程、平行執行、續跑與到達時刻缺口安全檢查，以及對應的 `lbt native-plan`／
`lbt native-run`／`lbt native-status` 子命令與 `scripts/run_native_backtrack_server.sh`
伺服器入口。這是 WP-7 的交付範圍；它只負責「如何有效率、可續跑、可稽核地把工作單元
分給 worker 並執行」，不改變任何物理核心、時間積分或亂數契約，也不宣稱已完成波浪或
後處理整合（見「已知限制」）。第 9 節另外說明 WP-9 在這之上新增的「小時主軸」排程
（`native_runner/hour_major.py`），這是把同一個流場區域內多個到達時刻／沉降速度的
工作單元合併成單一主迴圈、共用每小時固定開銷的效能優化，不是取代第 1-8 節描述的
逐工作單元路徑（後者仍是保留的、經過完整位元等價驗證的參考實作）。

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

## 9. WP-9：小時主軸排程（`native_runner/hour_major.py`）

### 9.1 問題與設計動機

第 1-8 節描述的路徑（下稱「逐工作單元路徑」）每個工作單元各自呼叫一次
`run_backward_batch`：同一個流場區域、同一個到達時刻、不同沉降速度的多個工作單元，
即使彼此的回溯窗完全重疊，也各自從頭重算「乾濕判定＋Stokes 全網格計算」
（`compose_current_and_stokes_fields`）、CFL 子步數估計（`estimate_substep_count`）與
（啟用擴散時）Smagorinsky 擴散場。`run_backward_batch` 本身已經支援用呼叫端傳入的
`composed_base_cache`／`smagorinsky_cache` 在**同一到達時刻**的不同沉降速度批次間
共用這些結果（見第 3 節「任務 3」提到的優化），但**不同到達時刻**之間、以及粒子的
「實際到達時刻」與「小時邊界」對齊時的重複計算仍然存在——正式規模（5 站 × 50 個到達
時刻 × 10 個沉降速度）下，同一個流場區域一個月內有大量到達時刻彼此的 30 天回溯窗
高度重疊，這部分的固定開銷是可觀的攤提空間。

小時主軸排程的核心想法：把「以工作單元為外迴圈、小時為內迴圈」反過來，變成「以小時為
外迴圈（沿一個流場區域的絕對 UTC 時鐘往回走）、以沉降速度分組為內迴圈」，讓同一個
流場區域內、回溯窗重疊的多個到達時刻與多個沉降速度分組，在同一個小時只需要：

- 一次 `forcing_resolver` 查詢、一次乾濕缺口判定；
- 一次 `compose_current_and_stokes_fields`（乾濕＋Stokes，與沉降速度無關）；
- 一次 `estimate_substep_count`（只讀純海流場，見下方「支撐事實」，與沉降速度無關）；
- 每個沉降速度分組各自一次（而不是每個到達時刻＋沉降速度的組合各自一次）
  `apply_settling_velocity` ＋粒子移動 kernel 呼叫。

尚未到達自己 `arrival_time_utc_ns` 的粒子，以及已經終止的粒子，都跟「延續中作用中」
的粒子共存在同一個合併陣列裡，靠 BayTrace 原生 `batch_*` kernel 對
`status != 1` 的列一律略過（見支撐事實 1）來保證安全共存，不需要每小時重新配置陣列。

### 9.2 支撐事實（決定正確性的四個關鍵觀察，逐一在原始碼中確認過）

1. `batch_move_rk4`／`batch_check_boundaries`／`batch_clamp_z` 這幾個決定粒子軌跡的
   kernel，一率以 `if p_status[ip] != 1: continue` 開頭，對 `status != 1` 的列**完全
   不讀寫**，可以安全和作用中粒子共存在同一個合併陣列裡。
2. **但 `batch_interpolate_velocity`／`batch_interpolate_diffusivity` 不是這樣**：這兩個
   kernel 對 `status != 1` 的列會**明確寫入預設值**（速度歸零；擴散係數寫
   `hdc`／垂向 `3.0e-4`、梯度寫 `0.0`），不是單純跳過（見
   `baytrace-root/baytrace/batch_interp.py` 第 76-80、457-463 行）。這與支撐事實 1
   的模式不同，是本次開發過程中實際踩到、造成粒子速度被錯誤歸零的臭蟲根源（見 9.4）。
3. `apply_settling_velocity` 只疊加進垂向場（`w_field1`／`w_field2`），完全不動水平場
   （`u_field1`／`v_field1`／`u_field2`／`v_field2`），因此 `estimate_substep_count`
   （只讀水平場）算出的子步數與子步秒數只跟「流場區域＋小時」有關、與沉降速度無關，
   可以在同一小時內跨沉降速度分組共用同一次計算結果。
4. `ParticleTracker.find_initial_elements`（`locate_initial_elements` 的底層實作）
   **不是**依 `status` 篩選的——它對傳入陣列裡的每一列都做 KD-tree 搜尋並無條件覆寫
   `element`／`status`／`z`／`u/v/w`／`x_prev`／`y_prev` 等欄位。因此**絕不能**對整個
   合併陣列直接呼叫這個函式；`_release_new_arrivals` 改用一個只裝新到達列的獨立暫時
   `ParticleArrays`，定位完成後才把結果的特定欄位散寫回合併陣列對應的列。

### 9.3 Smagorinsky 擴散場的小時鏈結（跨到達時刻合併時的正確性陷阱）

`run_backward_batch` 的算子分裂擴散用 `hf1_field`（這一小時開始時的場）／`hf2_field`
（這一小時結束時的場）夾出一個小時內的線性內插；`hf1_now` 的定義是「上一次迴圈迭代的
`hf2`（如果有）否則等於這次的 `hf2`」——這個鏈結是**依批次的迭代次序**定義的，不是
依絕對 UTC 小時定義的。合併多個到達時刻到同一個沉降速度分組後，同一個絕對小時可能同時
出現「剛到達、需要 `hf1=hf2=這小時的值`」與「延續中、需要 `hf1=上一小時的值`」兩種
粒子，兩者要的 `hf1_field` 不同，不能用同一次 kernel 呼叫服務。

解法：只有在（啟用擴散 **且** 同一小時同時有新到達與延續中作用中粒子）時，才把這個
分組這一小時拆成兩次 `mirror_rk4_substep_loop` 呼叫（分別暫時把另一批粒子的 `status`
設成 0），其餘情況維持一次呼叫。拆成兩次呼叫時，兩次呼叫都會各自觸發支撐事實 2 描述的
「`batch_interpolate_velocity`／`batch_interpolate_diffusivity` 對暫時停用的列寫入預設值」
副作用，因此每次暫時停用前後都必須額外保存／還原被停用列的 `u/v/w`（與啟用擴散時的
`dhfx/dhfy/dhfz/grdx/grdy/grdz`），見 `hour_major._snapshot_fields`／`_restore_fields`。

另外，`prev_hf2` 本身在「分組曾經整個小時完全沒有作用中粒子」（例如兩個到達時刻之間
有空檔）之後會過期；`hour_major.py` 用 `prev_hf2_utc_ns` 額外記錄它是哪一個絕對小時算
出來的，只有等於「這一小時的下一小時」才信任它是新鮮的，否則退回當下小時重算的
`hf2_now`，避免把好幾個小時前的擴散場誤用在間隔後才恢復活動的粒子上。

### 9.4 開發過程中實際發現並修正的臭蟲（誠實記錄，供之後維護參考）

以下四個問題全部是在**跑通測試的過程中**發現的，不是設計階段就想到的，記錄下來是為了
提醒之後修改本模組的人這些地方特別容易出錯：

1. **`batch_interpolate_velocity` 對暫時停用列的歸零副作用**（見支撐事實 2、9.3）：
   `_prime_new_arrivals`（對剛到達列重新內插起點速度）與 `_run_group_hour` 的擴散拆分
   路徑，都用「暫時把其他列的 `status` 設成 0」的方式限制某次 kernel 呼叫的作用範圍；
   一開始沒注意到這兩個 kernel 會把 `status != 1` 的列寫成預設值（而非略過），導致
   跟其他到達時刻共用同一個沉降速度分組的延續中粒子，其速度／擴散狀態在下一次合併呼叫
   前被錯誤歸零，造成移動量少了一個 RK4 子步的量——用單一單元跑 vs. 併入多到達時刻的
   分組跑，結果會出現看似隨機大小的位置偏差。修正是在每次這種暫時停用前後，額外保存／
   還原被停用列的 `u/v/w`（與擴散狀態）。
2. **軌跡取樣少一筆、且其餘各筆整體錯位一小時**：`run_backward_batch` 在
   `locate_initial_elements` 之後、任何本小時移動之前，就把「到達當下」的原始位置存成
   軌跡的第一筆（`trajectory_utc = [arrival_utc]`）；小時主軸排程一開始漏了這一步，
   只在小時邊界之後才取樣，導致每個單元的軌跡都少了到達當下那一筆、其餘各筆對應到的
   實際小時也因此整體錯開一小時。修正是在 `_release_new_arrivals` 之後，對這一小時
   剛到達的每個單元立刻補一筆「到達當下」的取樣。
3. **回溯上限／缺值缺口觸發時多出一筆重複軌跡、`hours_run` 多算**：`run_backward_batch`
   的回溯上限與缺值缺口判定都在小時迴圈本體最前面、任何移動與軌跡取樣之前就 `break`，
   因此觸發那一小時**不算進** `hours_run`、也不會多取樣一筆軌跡；小時主軸排程原本的
   `hours_run`／軌跡取樣邏輯用「這個單元是不是剛好到達」當代理條件，沒有精確對應「這一
   小時是不是真的跑過移動 kernel」，導致這兩種情況下 `hours_run` 多算一次、軌跡多出一筆
   與前一筆重複的取樣。修正是明確追蹤每個分組「這一小時實際被排進移動 kernel 的列遮罩」
   （`attempted_by_group`），`hours_run` 遞增與軌跡取樣都以這個遮罩為準，不再用「是否
   剛到達」這種較粗的代理條件。
4. **`final_status`／`final_x_m` 不能從軌跡陣列的最後一筆取**：延續上一點，回溯上限／
   缺值缺口觸發的那一小時既然不取樣軌跡，`trajectory_status[-1]`／`trajectory_x[-1]`
   在粒子剛好於那一小時終止時，仍然停在終止**前一小時**的 `ACTIVE` 狀態與位置，不是
   終止當下的正確值——這與 `run_backward_batch` 直接用迴圈結束當下的 `native_status`／
   `pa.x`（而非 `trajectory_status[-1]`／`trajectory_x[-1]`）組最終結果的作法不同。
   修正是讓 `_finalize_unit_result` 直接讀分組目前的 `native_status`／`pa.x/y/z`，不
   透過軌跡陣列——這樣讀是安全的，因為粒子一旦終止（`status` 歸零），後續任何 kernel
   呼叫都不會再寫它的列（支撐事實 1）。
5. **（測試基礎設施本身的臭蟲，非 `hour_major.py`）**：效能測試最初直接沿用小型合成網格
   測試共用的 `_NVRT = 2` 常數去建構「真實規模」網格（20 層）用的合成流場陣列，導致
   BayTrace 的內插 kernel 讀到陣列外的記憶體，輕則算出天文數字的速度值，重則直接
   segfault。修正是讓測試的欄位陣列建構函式改讀 `driver_grid.schism_grid.nvrt`，不要
   沿用模組層級寫死的層數常數。

第 1、3、4 點若沒有 `tests/test_native_hour_major.py` 的位元等價驗收（尤其是「擴散啟用
且跨到達時刻合併」與「回溯上限精確觸發」兩個測試）幾乎不可能靠人工審查發現——差異都在
公尺等級以下、或是只在特定的批次組成下才出現，這也是為什麼驗收測試要求「同一顆粒子
單獨跑 vs. 併入更大批次跑必須位元相同」而不是「大致相符」。

### 9.5 分批粒度與 `NativeWorkUnit` 的關係、checkpoint

`run_flow_domain_hour_major` 一次處理一個 `HourMajorDomainBatch`（`build_hour_major_
domain_batch` 從一組 `NativeWorkUnit` 建出，內部依沉降速度分組、每組再依到達時刻／
測站合併成連續陣列），輸出是 `{unit_id: BackwardDriverResult}`——**每個原始
`NativeWorkUnit` 各自一份**、欄位與 `run_backward_batch` 的輸出完全相同的
`BackwardDriverResult`，因此下游 `postprocess.PostprocessAccumulator` 不需要知道粒子
是用哪條路徑算出來的。已知的相容性落差：`BackwardDriverResult.diagnostics`／
`overhead_diagnostics` 在小時主軸路徑下固定回傳歸零的 `SubstepDiagnostics()`／
`HourlyOverheadDiagnostics()`（改成回傳 `HourMajorRunResult.domain_overhead_diagnostics`
這個**分組層級的彙總**），不逐工作單元拆分；需要逐工作單元診斷數據的下游用途，目前
只能退回逐工作單元路徑。

Checkpoint（`save_hour_major_checkpoint`／`load_hour_major_checkpoint`）以「整個流場
區域」為單位，每個沉降速度分組各自一個 `.npz`（完整粒子陣列狀態、擴散種子與步數、
`prev_hf2`＋其所屬絕對小時、每個單元已取樣的軌跡）＋一份 `meta.json`（原子性寫出，
先寫暫存檔再 `os.replace`）；下次以同一個 `checkpoint_dir` 呼叫時會先讀回、驗證分組數
／沉降速度／單元順序與目前輸入完全一致（任何不一致都直接拒絕續跑），再從中斷處繼續。
`tests/test_native_hour_major.py::test_hour_major_checkpoint_resume_bit_identical` 驗證
「中斷再續跑」與「不中斷一次跑完」得到位元相同的結果。

### 9.6 平行化設計（一個決策：目前只做「每流場區域一個行程」，暫不做共享記憶體 prefetch）

工作書要求的平行化設計是「一個流場區域一個行程（至多 4 個並行）＋可選的共享記憶體
prefetch worker」。本次交付**做了前者、刻意沒有實作後者**，原因與取捨如下：

- 小時主軸排程本身已經把「同一流場區域、同一小時」的固定開銷（forcing 查詢、乾濕缺口
  判定、Stokes 全網格計算、CFL 子步估計、Smagorinsky 場）從「每個到達時刻×沉降速度各
  一次」降到「每小時一次」，這是本次優化的主要收益來源（見 9.7 實測數據）；一個流場
  區域內部已經沒有「同一份流場資料被重複讀取」這件事需要另一層 prefetch worker 解決。
- 共享記憶體 prefetch（提前非同步讀取下一小時的流場、讓計算與 I/O 重疊）在流場讀取
  是**I/O bound**（例如冷快取、網路檔案系統）時才有明顯收益；本次沒有在正式規模、正式
  儲存後端（`ocm_native` schema 3 的月份 memmap）上量測過 I/O 是否真的是瓶頸，缺乏
  數據支持「值得為此新增一層跨行程共享記憶體與同步機制」的複雜度。
- 4 個流場區域、一個行程一個區域，在 32/64 核心機器上會讓每個行程平均分到 8-16 核心，
  但小時主軸排程**本身是單執行緒**（沒有對同一小時內的沉降速度分組做行程內平行化）；
  這代表在核心數遠多於流場區域數的機器上，會有大量核心閒置。**這是本次交付已知、
  刻意標記為「待正式規模量測後重新評估」的取捨**，不是遺漏——修正方向有兩種：(a) 在
  `run_flow_domain_hour_major` 內對「同一小時內彼此獨立的沉降速度分組」做行程內平行
  （例如 `ProcessPoolExecutor` 或執行緒池，需要先確認 BayTrace 的 `@njit(cache=True)`
  kernel 在多執行緒下是否真的因為 GIL 而無法平行，見模組文件關於 `nogil` 的觀察）；
  (b) 讓一個「流場區域」的定義更細（例如同一大流場區域內再切成多個獨立子網格區塊），
  用更多行程吃滿核心數。兩者都需要正式規模的 profiling 數據才能判斷該選哪一個，本次
  交付沒有做，是誠實列在這裡的已知限制。

### 9.7 效能實測與外插（誠實揭露：縮小規模量測，非正式規模逐字重跑）

`tests/test_native_hour_major.py::test_hour_major_performance_vs_per_unit_path` 用縮小
過的規模（8,000 節點、20 層、3 個相隔 2 小時的到達時刻、4 個沉降速度、回溯上限 4 小時）
量測小時主軸路徑 vs. 逐工作單元路徑（無任何快取共用，作為舊路徑的代理）的壁鐘時間比：

| M（每情境粒子數／2） | 單元數 | 每單元粒子數 | 逐工作單元耗時 | 小時主軸耗時 | 加速比 | kernel 呼叫數 |
|---|---|---|---|---|---|---|
| 1 | 12 | 2 | 1.57s | 0.68s | **2.29×** | 32（vs. 逐單元理論上限約 12×5=60） |
| 10 | 12 | 20 | 2.13s | 1.23s | **1.73×** | 32 |

型態與既有 `test_realistic_grid_scale_hourly_overhead_benchmark`（60,000 節點、單一
小時、3 個沉降速度分組共用一次乾濕＋Stokes 計算）量測到的「固定開銷佔比隨粒子數增加而
下降、因此加速比隨粒子數增加而下降」的方向一致（M 越大，每小時的粒子步計算本身佔比
越高，固定開銷攤提的相對收益越小）。

**這不是正式規模（30,000 節點、20 層、2 個流場區域、每區域 5 個到達時刻 × 10 個沉降
速度 × 20/200 顆粒子、30 天回溯窗）的逐字重跑**，原因是正式規模單次量測本身可能需要
數分鐘到數十分鐘（`test_realistic_grid_scale_hourly_overhead_benchmark` 在 60,000 節點、
單一小時下就要約 110 秒），在測試套件的時間預算內重複兩條路徑、多個 M 值不可行。外插到
正式規模時的推論鏈與明確假設：

1. 固定開銷攤提的收益主要來自「同一流場區域、同一小時」被重複計算的次數，這個次數約等於
   `n_settling × n_arrivals_overlapping_this_hour`；正式規模是 10 個沉降速度、50 個到達
   時刻（30 天回溯窗高度重疊，多數小時會有全部或大部分到達時刻同時作用中），遠高於本次
   量測的 4×3=12，預期加速比**不會比本次量測到的 1.7-2.3× 低**，且隨到達時刻數增加，
   固定開銷攤提的分母變大，方向上應該更接近或超過已知的 `test_realistic_grid_scale_
   hourly_overhead_benchmark` 量測到的、乾濕＋Stokes 計算本身在 60,000 節點下的相對成本
   數量級。
2. 這個外插**沒有**把 9.6 節提到的「單一流場區域內部是單執行緒」這個限制的影響量化
   進去——在 32/64 核心機器上，若 4 個流場區域各自一個行程、其餘核心閒置，總體（跨全部
   流場區域）的牆鐘時間加速比會**遠小於**單一流場區域內部的 1.7-2.3×，因為沒有利用到
   閒置核心；這正是 9.6 節列為待重新評估項目的原因。
3. 以上兩點合起來，誠實的結論是：**單一流場區域內部、小時主軸排程對固定開銷的攤提效果
   已經用縮小規模量測與既有 60,000 節點量測共同佐證，方向與量級合理；但「正式規模、
   32/64 核心、4 個流場區域同時執行」的總體吞吐量改善幅度，本次交付沒有實測數據，
   需要在正式規模環境上另外量測才能給出可信的數字**，不應該直接把 1.7-2.3× 當成正式
   規模的預期總體加速比。

### 9.8 CLI／`orchestrator.py` 整合現況（WP-11 已完成接線，本節保留原始規劃紀錄）

**WP-11 更新**：本節原本描述的「尚未接線、留給 `production_wiring.py` 負責的部分」已經由
`native_runner/hour_major_provider.py` 完成（見該模組 docstring 與第 9.9 節）；本節文字保留
原樣（含歷史用詞），作為當初的設計紀錄，不再代表目前的待辦事項。

`orchestrator.plan_native_hour_major_work_units(units)` 把 `plan_native_work_units` 產生
的逐工作單元依 `flow_domain_id` 摺疊成「每個流場區域一個」排程單元（`unit_id` 帶
`hourmajor::` 前綴，其餘四個鍵值填 sentinel 值、呼叫端不應該讀取），回傳
`(domain_units, members)`；`members[domain_unit.unit_id]` 是摺疊前的原始工作單元，
依 `unit_id` 排序。這樣設計是為了讓摺疊後的排程單元可以**原封不動**交給既有
`run_native_plan` 執行引擎（worker 指派、原子性完成證據、續跑判斷、進度快照全部
不需要修改就能運作，因為這套引擎完全是靠 `NativeWorkUnit` 的欄位運作，沒有任何一處
假設「一個 `NativeWorkUnit` 只對應一次 `run_backward_batch` 呼叫」）。

`lbt native-plan` 新增 `--schedule {per-unit,hour-major}`（預設 `per-unit`，行為完全
不變）；選 `hour-major` 時，到達時刻缺口安全檢查仍以摺疊前的逐工作單元粒度進行（摺疊
後的排程單元沒有單一 `arrival_time_utc_ns` 可以做這個檢查），只有寫進計畫檔給
`native-run` 排程執行的 `units` 欄位改成摺疊後的結果，並額外寫入 `hour_major_members`
欄位（`write_native_plan_file`／`read_native_plan_file`／`read_native_hour_major_members`
三者支援這個欄位的寫入與重建）。

**尚未接線、留給 `production_wiring.py` 負責的部分**（不在本次交付範圍，因為需要 import
`config.py`／`mesh.py`／`geometry.py`／`wave_bridge.py`／`postprocess.py`，違反本模組
「不依賴重量級模組」的設計原則，見第 8 節第一條已知限制的同一個理由）：

- `--resource-provider` 指定的 factory 函式，其回傳的 `run_unit(resources, spec)` 需要
  能辨識 `spec.unit_id` 是否帶 `hourmajor::` 前綴；若是，應該：
  1. 呼叫 `native_orchestrator.read_native_plan_file(plan_path)`（`plan_path` 在
     factory 建構時就該從 CLI 參數取得並閉包住）取回 payload，再用
     `native_orchestrator.read_native_hour_major_members(payload)` 重建 `members`
     對照表，查出 `spec.unit_id` 底下的原始工作單元清單；
  2. 對每個原始工作單元，從 `ParticleIdentityTable` 取出對應列的 `x0_m`／`y0_m`／
     `z0_m`／`seed_hex`，組成 `hour_major.HourMajorUnitInput`；
  3. 依各測站 `study_site_id` 取得對應的 `interest_region_polygon`（`geometry.py`／
     `config.py` 職責），組成 `interest_regions: Mapping[str, Polygon]`；
  4. 呼叫 `hour_major.build_hour_major_domain_batch` ＋
     `hour_major.run_flow_domain_hour_major`（帶入 `forcing_resolver`／
     `wave_field_resolver`／`checkpoint_dir` 等），取得
     `HourMajorRunResult.unit_results`；
  5. 把 `unit_results[原始 unit_id]`（一個 `BackwardDriverResult`）轉成 `run_unit`
     該回傳的摘要字典（與逐工作單元路徑的 `run_unit` 回傳格式一致，供
     `postprocess.PostprocessAccumulator` 消費）。
- 上述第 5 步的「一個 `hourmajor::` 排程單元的完成證據，實際上代表底下一整批原始工作
  單元都跑完了」這件事，目前的 `.done.json`／`.result.json` 只以摺疊後的 `unit_id`
  為檔名——若下游（例如既有的 `postprocess`／報表流程）預期看到逐工作單元的完成證據檔，
  需要在上述 `run_unit` 實作裡額外把每個原始工作單元的結果也各自寫一份，或改造下游
  改讀摺疊後的檔案並自行拆分，這個決定留給 `production_wiring.py`。
- `--resource-provider` 的呼叫慣例（factory 函式呼叫時不帶參數）本次**刻意沒有改動**
  ——這是與波浪／後處理整合（WP-8，另一個並行開發中的子代理）共用的邊界，任何簽章變更
  都應該協調後再做，不應該由 WP-9 單方面破壞。

## 9.9 WP-11：切段平行與伺服器指令鏈

### 9.9.1 目標與動機

WP-9 的小時主軸排程把「每個流場區域」摺成一個排程單元，全區域內部單執行緒；伺服器
（56 核心、125 GB）只有 4 個正式流場區域，若照 9.8 節的原始設計，`native-run` 最多只能
同時跑 4 個 worker，用不滿 56 核。WP-11 在「摺疊」的基礎上再加一層「切段」：把每個流場
區域依到達時刻由舊到新切成 `K` 段，每段各自摺成一個 `hourmajor::` 排程單元，可以分派給
不同 worker 平行執行；`K` 依核心數自動決定（也可以手動指定），讓總排程單元數約等於
worker 數。

### 9.9.2 切段機制（`orchestrator.plan_native_hour_major_segments`）

對每個流場區域，把該區域內全部相異 `arrival_time_utc_ns` 由舊到新排序後，切成 `K` 個
盡量等長（相差至多 1 個到達時刻）的連續區段（`_split_sorted_into_k_chunks`）；每個
`NativeWorkUnit` 依自己的到達時刻歸入對應區段，摺成一個 `unit_id` 帶
`__seg{i}of{K}__` 的排程單元，`members[domain_unit_id]` 是該段底下的原始工作單元（與
9.8 節「摺疊」機制完全相同的資料結構，只是分組鍵值從「全區域」變成「區域＋段」）。

**正確性保證**：亂數是計數器型、與批次組成無關（見第 9.2 節「支撐事實」）；切段只改變
「哪些 `NativeWorkUnit` 被摺進同一次 `run_flow_domain_hour_major` 呼叫」，不改變任何粒子
的起點、seed 或亂數串流。`K=1` 與 `K>1` 對同一顆粒子必須得到位元相同的最終狀態，這由
`tests/test_native_hour_major_provider.py::test_native_run_k1_bit_identical_to_k2` 端對端
驗證（透過真正的 CLI 流程，不只是呼叫 `hour_major.run_flow_domain_hour_major` 本身）。

段與段之間只在各自回溯窗重疊處重複計算（例如兩段的到達時刻只差幾小時，各自往回推
`max_backtrack_seconds` 天，重疊區間會被兩段各自合成一次底流場），不會有「同一到達時刻
被拆進兩段」這種需要額外去重邏輯的情況。額外成本估計由
`orchestrator.estimate_segment_extra_cost_fraction` 計算（`HourMajorSegmentPlanReport`
的一部分，`lbt native-plan --schedule hour-major-segmented` 會把它印在報告裡），`K` 太大
時重疊成本會顯著推高總計算量，見 `recommend_hour_major_segment_counts` 對這個 trade-off
的自動選擇邏輯與第 9.9.5 節的效能外推。

worker 數上限同時受記憶體限制：`orchestrator.estimate_worker_count`／
`read_available_memory_bytes` 依 `--per-worker-memory-bytes` 與可用記憶體算出上限，
`lbt native-run --auto-worker-count` 會用這個估算值，不會假設 56 個 worker 一定放得下。

### 9.9.3 正式資源入口（`native_runner/hour_major_provider.py`）

9.8 節列的「尚未接線」清單，由本模組完成：`make_hour_major_resource_provider()` 是
`lbt native-run --resource-provider` 可直接指定的模組層級 factory，內部靠環境變數
（`LBT_NATIVE_CONFIG`／`LBT_NATIVE_PLAN`／`LBT_NATIVE_IDENTITY_TABLE`／
`LBT_NATIVE_RESULTS_DIR` 等，完整清單見該模組 docstring「環境變數契約」）取得設定，串接
`production_wiring.make_domain_resources_factory`（WP-8）取得每個流場區域的網格／
interest region／forcing／波浪解析器，與 `hour_major.run_flow_domain_hour_major`（WP-9）
執行一段。每個原始工作單元的完整 `BackwardDriverResult` 序列化成
`<unit_id>.driver_result.npz`（`save_backward_driver_result_npz`／
`load_backward_driver_result_npz`），供 `lbt native-postprocess` 讀回。

**WP-11 修正的既有缺陷（誠實揭露）**：最初的實作把 `spec.flow_domain_id`（真正的 OCM
產品目錄名稱）直接當成 `production_wiring.build_domain_resources` 的 `domain_id` 參數傳入，
但該參數依其既有契約必須是 `analysis_region_id`；兩者只有在 `native-plan` 省略
`--region-to-flow-domain` 時才會剛好相同，正式部署一定會提供這個參數，兩者不同。已修正為
從排程單元底下原始工作單元的 `analysis_region_id` 欄位取得正確值，並在
`tests/test_native_hour_major_provider.py` 裡明確提供 `--region-to-flow-domain`（讓
`analysis_region_id` 與 `flow_domain_id` 刻意不同）驗證修正後的路徑，避免同一個缺陷再次被
「兩者剛好相同」的測試環境掩蓋。詳見 `hour_major_provider.py` 模組文件「已知限制」第 5 點。

### 9.9.4 CLI 指令鏈與伺服器腳本

`lbt native-preflight`／`lbt native-gap-fill`（延遲匯入另一個工作包提供的
`native_runner.gap_fill`）／`lbt native-plan --schedule hour-major-segmented`／
`lbt native-run --resource-provider lagrangian_backtracking.native_runner.
hour_major_provider:make_hour_major_resource_provider`／`lbt native-status`／
`lbt native-postprocess` 六個子命令合起來就是完整流程；`scripts/run_native_backtrack_
server.sh {preflight|gap-fill|plan|run|status|postprocess|all}` 把它們包成一條可在伺服器
上背景執行的指令鏈（`run` 用 `nohup` 背景執行、log 落地；已完成的步驟依既有輸出／標記檔
略過，見該腳本檔頭）。`scripts/deploy_native_to_server.sh` 負責把目前分支用 `git bundle`
部署到運算主機（`bundle`／`deploy` 兩個動作），完成後印出下一步指令。

### 9.9.5 效能外推（誠實揭露：外推，非正式規模逐字重跑）

見 `reports/WP-11.md`「效能外推」一節的完整假設與數字；本節只記錄外推方法本身：以
`tests/test_native_hour_major.py::test_hour_major_performance_vs_per_unit_path` 同一套
合成網格（可調節規模）量測「單小時底場合成成本」與「每粒子每小時步進成本」，用正式規模
（4 區域、每站 50 個到達時刻、10 沉降速度、20 受體、`M=1`／`M=10`）與
`estimate_segment_extra_cost_fraction` 的重疊成本模型，外推 56 核心下的總牆鐘時間與建議
`K`。

## 9.10 WP-13：動態拉取佇列、正確成本模型與資料就緒串接

完整設計理由、量測數字與誠實揭露的限制見 `reports/WP-13.md`；本節只記錄操作介面。

### 9.10.1 動態拉取佇列（取代 9.9 之前的靜態 LPT 指派）

`run_native_plan` 不再用 `plan_worker_schedule`（LPT bin-packing）把整串工作單元一次性
靜態指派給固定 worker——舊版的問題是「某個 worker 分到的整串單元做完就沒事做、即使其他
worker 還有一大串沒做完，也不能互相支援」，在「1 個很貴的區段＋多個很便宜的區段」這種
尾端負載不均的組合下，會讓總牆鐘時間被那個最慢的 worker 卡住。WP-13 改成對**每一個**
待執行單元各自呼叫一次 `pool.submit(_run_single_unit_task, ...)`：`ProcessPoolExecutor`
內建的呼叫佇列本來就是「任何一個常駐子程序處理完手上的任務，立刻去佇列拿下一個」的行為，
不需要另外實作佇列協定；每個 worker 只在第一次真正拿到任務時呼叫一次
`build_worker_resources`，之後所有指派給它的單元共用同一份快取資源，與舊版的資源重用
行為相同。`plan_worker_schedule` 本身仍保留在原始碼裡（供既有呼叫端相容），但
`run_native_plan` 不再呼叫它。

### 9.10.2 正確成本模型（取代粒子數代理）

舊版排程用「工作單元底下的粒子數」當成本代理，這對小時主軸排程是**系統性錯誤**的：小時
主軸單元真正的成本主要來自「要合成幾個流場小時的底場（乾濕＋Stokes＋Smagorinsky）」，
這由該單元涵蓋的到達時刻跨度＋回溯窗決定，跟粒子數幾乎無關（`apply_settling_velocity`
＋粒子移動 kernel 每粒子每小時的成本遠小於底場合成成本，見 `reports/WP-13.md` 的分解
量測）。`orchestrator.estimate_hour_major_unit_cost_seconds` 用

```
成本（秒）≈ 要合成的流場小時數 × 節點數 × 層數 × seconds_per_node_layer_hour
           + 粒子數 × 平均每小時子步數 × seconds_per_particle_substep
```

估算單一排程單元的成本，第一項在正式規模下幾乎主導總成本（第二項的校準常數刻意保留但
在目前量測到的規模下貢獻可忽略，見該函式 docstring）。`seconds_per_node_layer_hour`
預設值 `7.0e-6`（`orchestrator._DEFAULT_SECONDS_PER_NODE_LAYER_HOUR`）是在本次開發用的
沙箱環境（2 核心）以 29,929／67,600 節點×48 層合成網格量測、確認線性模型在 2.3 倍網格
規模差距下誤差在 0.3% 以內後校準出來的；**這不是正式 56 核心伺服器的量測值**，正式
部署前應該用 `--seconds-per-node-layer-hour` 覆寫成伺服器上實測的常數（方法：跑一次
`hour_major.run_flow_domain_hour_major`，記錄 `HourlyOverheadDiagnostics.
time_compose_base_s` 除以節點數×層數×小時數）。

`orchestrator.plan_native_hour_major_segments`（透過 `lbt native-plan --schedule
hour-major-segmented`）新增 `--node-counts`／`--layer-counts`（各自一個 JSON 檔
`{flow_domain_id: 數量}`）：**兩者都提供時**才啟用這個成本模型（`recommend_hour_major_
segment_counts_by_workload`）自動選段數，否則 fail-closed 退回舊版純粒子數啟發式
（`recommend_hour_major_segment_counts`），不會用臆測的網格規模頂替。`--target-worker-
count` 在兩種模式下都用來當自動選段數的目標平行度上限。

### 9.10.3 自動選段數與 `native-plan` 報表新增欄位

`choose_hour_major_segment_count` 對每個流場區域，在 `1..distinct_arrival_count` 之間
搜尋讓下式最小的段數 `K`：

```
total_wall_clock_seconds(K) ≈ max(total_cost_seconds(K) / min(K, target_worker_count),
                                    max_segment_cost_seconds(K))
```

（`total_cost_seconds(K)` 隨 `K` 增加而增加，因為段與段之間的回溯窗重疊會被重複合成；
`max_segment_cost_seconds(K)` 隨 `K` 增加而下降；两者的極大值在中間某個 `K` 取最小。）
`--node-counts`／`--layer-counts`／`--worker-count-for-estimate`（省略時退回
`--target-worker-count`）都提供時，`native-plan` 的 JSON 報表會多出一個 `cost_model`
物件：

```json
{
  "cost_model": {
    "seconds_per_node_layer_hour": 7e-06,
    "worker_count_for_estimate": 56,
    "estimated_wall_clock_seconds": 1234.5,
    "max_single_segment_cost_seconds": 210.0,
    "total_cost_seconds": 9876.5,
    "repeat_cost_fraction_vs_no_split": 0.18,
    "segment_cost_seconds": {"northeast_taiwan_common_cache_v3::seg0of8__": 180.2, "...": "..."}
  }
}
```

`repeat_cost_fraction_vs_no_split` 是「切段後的總成本」相對「完全不切段（K=1）的總成本」
多出的比例，用來判斷切段的重疊代價是否划算。`native-plan` 同時把 `segment_cost_seconds`
寫進計畫檔 `manifest.extra.unit_costs`；`native-run` 讀回計畫檔時會自動取用這份成本表
（見下一節），呼叫端不需要在 `native-run` 階段重新提供節點／層數。仍然可以用
`--segments`（單一整數或逐區域 JSON）手動覆寫段數，跳過自動選段。

### 9.10.4 `native-run --wait-for-staged`：資料就緒後才把整區工作加入佇列

伺服器的原始資料是逐流場區域分批到位的（工作書記載的到站順序：連江 → 後灣 → 新竹 →
東北，見 `reports/WP-13.md`），舊版必須等**全部**區域的資料都就緒才能啟動
`native-run`，即使先到的區域其實已經可以開始算。`lbt native-run --wait-for-staged
<目錄>` 讓 `run_native_plan` 用「哪些流場區域已經出現在計畫檔的 `units` 裡」自動算出
需要等待的 `flow_domain_id` 集合，在 `<目錄>/<flow_domain_id>.done` 標記檔出現前，
該區域的工作單元先保留在「尚未就緒」清單裡、**不**提交進 `ProcessPoolExecutor`
的佇列；已就緒區域的單元立刻正常提交。主迴圈在「還有進行中的 future、或還有保留住的
單元、或還有尚未就緒的區域」任一條件成立時都不會結束，因此空閒 worker 在資料還沒到齊
時只是繼續等待，不會誤判「佇列暫時空了」就提前結束或跑出不完整的結果；標記檔出現後，
下一輪輪詢立刻把該區域的單元補進佇列。全部區域都就緒、且佇列清空後才正常結束、寫出
最終的 `NativeRunSummary`。省略這個參數時（預設）行為與 WP-13 之前完全相同（視為全部
區域已就緒）。

驗收：`tests/test_native_orchestrator.py::
test_run_native_plan_waits_for_staged_domains_before_finishing`（背景執行緒＋輪詢，模擬
標記檔延遲出現，確認先就緒區域的單元先完成、後就緒區域的單元直到標記出現才被提交）、
`tests/test_native_hour_major_provider.py::
test_native_run_wait_for_staged_processes_domains_as_markers_appear`（同一件事透過真正
的 CLI 端對端驗證）。

### 9.10.5 小時內平行化（任務 2）：已驗證位元相同，但刻意未接線到 `native-run`

`backward_driver.compose_current_and_stokes_fields`／`hour_major.run_flow_domain_hour_
major` 都新增了 `stokes_pool`／`stokes_worker_count` 參數：史托克斯漂移（Stokes drift）
逐點求解彼此獨立、沒有跨節點化簡，因此可以切塊分派到呼叫端自備的
`concurrent.futures.ProcessPoolExecutor` 平行計算；已用**不巢狀**在任何其他行程池 worker
之內的直接呼叫，驗證輸出與逐點單一行程執行位元相同（`tests/test_native_hour_major.py::
test_hour_major_stokes_pool_matches_sequential_bit_identical`）。

**誠實記錄未完成的部分**：原本規劃是讓 `native_runner/hour_major_provider.py`（`native-
run --resource-provider` 的正式接線）依環境變數在每個 worker 子程序內自動建立這個子行程
池，這樣才能真正利用「排程單元數少於核心數」時的閒置核心。但 `native-run` 的每個 worker
子程序本身已經是 `orchestrator.run_native_plan` 用 `ProcessPoolExecutor`（spawn）開出來
的子程序；實測在「子程序裡面再建一個 `ProcessPoolExecutor`」（巢狀行程池）的組態下，
即使兩層都明確指定 `spawn` context，仍然會不穩定地重現兩種失敗模式：整個 worker 卡死
（CPU 時間數分鐘內不再增加，符合死結特徵）、或孫行程在 bootstrap 階段丟出
`FileNotFoundError`（號誌檔案 rebuild 失敗，屬於競態，不是每次都會發生）。受限於本工作
包的時間範圍未能查明根本原因，因此**沒有**把這個旋鈕接上正式 `native-run` 管線——正式
管線的可靠性優先於這項優化的效能收益。詳細重現腳本、程序樹觀察與分析見
`reports/WP-13.md`「任務 2」一節；`stokes_pool` 參數本身保留在
`backward_driver.py`／`hour_major.py`，供未來有更安全的方式建立/取得子行程池（例如頂層
建立、不巢狀在 worker 之內）時直接使用，不需要重新設計介面。
