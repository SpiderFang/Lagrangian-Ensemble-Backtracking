# 新竹 24 小時 Numba 與 checkpoint 間隔平行測速紀錄

> 文件狀態：2026-09-14 SERVER 工程試跑紀錄。本文只描述 B 區新竹、單一到達時刻、20
> 個情境與 `M=1` 的工程測速，不是 30 天工期承諾、五站正式成果或科學驗收。本文所稱
> 來源結果仍只能解讀為條件式來源足跡或相對來源權重。

## 1. 測試目的與比較邊界

本輪用同一份已建置且通過檢查的輸入、同一個到達時刻、同一份 scenario／seed table
與同一個共同亂數流，分別比較：

1. 既有純 NumPy 參考核心，以四個分片依序執行的刻意串行 baseline；
2. `a7bd697` 的 Numba OCM 插值後端（`numba_ocm_v1`），以四個獨立 worker 同時執行，
   並把 `checkpoint_interval_sweeps` 從 `1000` 調為 `10000`。

舊版串行測試是為了取得每一分片的可重現基準，不表示系統沒有分片平行能力。既有執行
模型是「每個 worker 內部依序處理一個分片，分片之間由外部 worker 平行」；本輪特別把
這兩個因素分開記錄，避免把平行化、Numba 與 checkpoint 間隔的效果誤歸因於單一改動。

固定條件如下：

| 項目 | 設定 |
|---|---|
| 研究區與站點 | B 區、新竹（`hsinchu`） |
| 到達時刻與回溯 | `2024-01-02T01:00:00Z`，向前 24 小時 |
| 情境與分片 | 20 scenarios，`4 shards × 5 scenarios` |
| 成員數 | `M=1`，共 20 粒子 |
| 共同亂數流 | `abcd-hsinchu-20240102t0100z-m1-v1` |
| active chunk | `5` |
| 舊版後端 | 純 NumPy 參考核心，checkpoint 每 `1000` sweeps |
| 新版後端 | `numba_ocm_v1`，checkpoint 每 `10000` sweeps |
| 新版 checkout | `/home/mustlab/Workspace/Lagrangian-Ensemble-Backtracking-a7bd697`，Git `a7bd697faf5325be73c650f7fe0933e384883138`，`dirty=false` |
| 新版部署樹 | `cdd1e5034b16eff148bb29caa2a73b6c8870304eab52f7041b197b8d23c234e6` |

兩組物理案例各自保留獨立的 run、checkpoint、輸出與 profile：

- `no_stokes`：只有流；
- `finite_depth_stokes`：流加有限水深 Stokes 波浪。

## 2. 只有流（`no_stokes`）

### 2.1 舊版 NumPy 串行 baseline

下表的 wall／CPU／checkpoint bytes 取自每一個已完成分片的 `run_progress.json`。四片
依序執行的 wall 總和為 `2810.614961 s`，CPU 總和為 `2681.483410 s`。

| 分片 | 舊版 wall (s) | 舊版 CPU (s) | 舊版 checkpoint (bytes) |
|---|---:|---:|---:|
| 0：`00000000-00000005_shd_1e1a145ae32e2c3be549` | 683.392676 | 650.357565 | 329,635,532 |
| 1：`00000005-00000010_shd_66c456e2c16fba23e794` | 732.863314 | 698.349797 | 369,888,325 |
| 2：`00000010-00000015_shd_1dfab9241a1d607d3c4b` | 754.803080 | 722.964716 | 329,815,705 |
| 3：`00000015-00000020_shd_b340262a6d60d172354a` | 639.555891 | 609.811331 | 306,474,209 |

### 2.2 Numba、四 worker 與 `checkpoint_interval_sweeps=10000`

四 worker 的外層實際 elapsed 為 `347.843958 s`（約 5 分 47.844 秒）；四個 worker
均正常 exit 0。下表的 worker wall／CPU／RSS 取自各自的 `/usr/bin/time -v` log；
checkpoint 與 output bytes 取自完成後的 run progress。

| 分片 | 新版 worker wall (s) | 新版 CPU (s) | checkpoint (bytes) | output (bytes) | 最大 RSS (bytes) | 相對舊版 wall |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 323.936677 | 320.299457 | 37,676,587 | 909,593 | 442,343,424 | 2.110× |
| 1 | 327.944327 | 324.263280 | 38,281,817 | 958,903 | 446,365,696 | 2.235× |
| 2 | 345.789641 | 342.131317 | 37,966,936 | 962,953 | 430,366,720 | 2.183× |
| 3 | 315.122449 | 312.429061 | 26,777,159 | 877,718 | 439,582,720 | 2.030× |

四片總步數為 `839,288`；新 worker wall 總和為 `1312.793094 s`，但實際四 worker
elapsed 只需 `347.843958 s`。以舊版四片串行 wall 總和對照，整批 elapsed 為約
`8.080×` 的改善；此數字同時包含 Numba、四 worker 與較低 checkpoint 寫入頻率，不能
單獨宣稱全數由 Numba 造成。

## 3. 流加波浪（`finite_depth_stokes`）

### 3.1 舊版 NumPy 串行 baseline

四片依序執行的 wall 總和為 `3642.866338 s`，CPU 總和為 `3486.713956 s`。

| 分片 | 舊版 wall (s) | 舊版 CPU (s) | 舊版 checkpoint (bytes) |
|---|---:|---:|---:|
| 0：`00000000-00000005_shd_86302ec6851e6e2d99ad` | 966.382489 | 914.683777 | 411,131,749 |
| 1：`00000005-00000010_shd_ce6a581919af7487dc01` | 935.608930 | 898.782258 | 345,838,839 |
| 2：`00000010-00000015_shd_953b7bdf6b7dff22fe6f` | 928.508803 | 892.738725 | 350,997,522 |
| 3：`00000015-00000020_shd_78860cd29e5745f02c5d` | 812.366116 | 780.509196 | 324,843,836 |

### 3.2 Numba、四 worker 與 `checkpoint_interval_sweeps=10000`

四 worker 的外層實際 elapsed 為 `512.887357 s`（約 8 分 32.887 秒）；四個 worker
均正常 exit 0。

| 分片 | 新版 worker wall (s) | 新版 CPU (s) | checkpoint (bytes) | output (bytes) | 最大 RSS (bytes) | 相對舊版 wall |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 510.267261 | 506.420978 | 37,994,726 | 945,539 | 457,674,752 | 1.894× |
| 1 | 510.790918 | 507.052162 | 38,128,833 | 943,560 | 456,474,624 | 1.832× |
| 2 | 498.333026 | 494.284387 | 38,953,868 | 954,522 | 450,031,616 | 1.863× |
| 3 | 466.294765 | 462.794137 | 36,593,285 | 870,385 | 457,322,496 | 1.742× |

四片總步數為 `841,766`；新 worker wall 總和為 `1985.685970 s`，實際四 worker
elapsed 為 `512.887357 s`。以舊版四片串行 wall 總和對照，整批 elapsed 為約
`7.103×` 的改善；同樣不能把這個綜合值視為單獨的 Numba speedup。

## 4. checkpoint 空間與 NFS 鎖競爭

checkpoint 對 30 天回溯仍有必要：SERVER 中斷或需要分段重啟時，必須保留粒子狀態、
觀測／事件游標與亂數狀態，才能從合法邊界恢復並維持亂數連續性。完全關閉 checkpoint
會把一次中斷的重算範圍擴大到整個 run，也失去可驗證的 recovery 證據。因此本輪的結論
是保留 checkpoint，但調整寫入頻率與拓撲；不是把 checkpoint 拿掉。

目前 schema 2 checkpoint 每代會把累積的 observation／event history 再寫出一次。當
`checkpoint_interval_sweeps=1000` 且四 worker 同時運作時，四個程序會頻繁競爭 NFS 上的
`progress.lock`；診斷中可見程序長時間停在 NFS lock wait（`nlmclnt_block`），因此
該次 `cp1000` 四 worker 測試被停止，return code `143`，不列為成果 run。

將間隔改為 `10000` 後，四個 worker 均約以 100% CPU 運算，未再出現該次 lock storm。
checkpoint tree 的 `du -sb` 空間如下；這裡的總量包含各代 checkpoint、`latest.json`
與必要 metadata，和單一分片 progress 中的 logical `checkpoint_bytes` 不完全相同：

| 物理案例 | 舊版 `cp1000` checkpoint tree | 新版 `cp10000` checkpoint tree | 空間減少 |
|---|---:|---:|---:|
| `no_stokes` | 1.335844065 GB | 140.707819 MB | 約 89.5% |
| `finite_depth_stokes` | 1.432843344 GB | 151.676130 MB | 約 89.5% |

這組數字直接回答 NFS I/O 與空間疑慮：`10000` 間隔在本次 24 小時、20 情境測試中，
將作用中 checkpoint tree 從約 1.34／1.43 GB 降到約 141／152 MB，同時解除本輪觀察到
的 progress lock 競爭。正式 30 天設定仍應以可接受的重算窗口反推 checkpoint 間隔，並
在固定設定中保存該選擇；不能只因本輪可行就把 `10000` 視為所有站點與所有回溯長度的
普遍最終值。

schema 3 的 immutable history segments 與 compact state 已另有一組 SERVER engineering
baseline，詳見下一節。它預期把每代新增 history 以追加段保存，降低長回溯的重複寫入；
該 baseline 仍須完成正式 validator、cold resume、故障、同步與容量審查，不能直接作為
正式 30 天執行依據。

## 5. schema 3 `cbfe58e4` SERVER engineering baseline

本節補記一組以 schema 3 checkpoint 執行的 B 區新竹 24 小時 engineering baseline。
它與前兩節的 `a7bd697` schema 2 `cp10000` 數據分開保存；本節只測
`no_stokes`（只有流），沒有把 schema 3 的結果延伸宣稱至 `finite_depth_stokes`、30
天或五站正式矩陣。

### 5.1 版本、輸入與執行範圍

| 項目 | 實際值 |
|---|---|
| checkout | `/home/mustlab/Workspace/Lagrangian-Ensemble-Backtracking-cbfe58e` |
| Git commit | `cbfe58e4144a6f6f3f5ba418f2c38a16d7bc5240` |
| deployment tree | `e916c91dec056109b800ea02097b8d846bc0d597` |
| deployment bundle | `/data/LBT/packages/lbt-cbfe58e.bundle` |
| bundle SHA-256 | `48a732cf25be528aa603be712793e810dcf89d2a7bce466287a34328e57eebf0` |
| engineering root | `/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914` |
| input artifact | `/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/artifacts/B-hsinchu-24h-cbfe58e-v1` |
| run ID | `b-hsinchu-no-stokes-cbfe58e-schema3-4w-v1` |
| 物理案例 | `no_stokes`，只有流 |
| workload | B 區新竹、24 小時回溯、20 scenarios、`M=1`、`4 shards × 5` |
| random stream | `abcd-hsinchu-20240102t0100z-m1-v1` |
| checkpoint | schema 3，immutable history segment + compact state |

### 5.2 刻意 pause、resume 與完整性

為驗證 schema 3 的可恢復路徑，先只執行 shard 0，刻意在 `20,000 sweeps` 停止；當時
完成 `100,000 particle steps`、產生 2 代 schema 3 checkpoint。該階段 elapsed 為
`153.91 s`，最大 RSS 為 `394,324 KiB`，pause 後 checkpoint tree 的 `du -sb` 為
`5,329,302 bytes`。接著以相同 run plan、input binding、random stream、shard ID
與 checkpoint root resume，最終 `4/4 shards COMPLETE`、20 scenarios、20 particles，
`validate-run --require-complete` 的紀錄為 `valid=true`、`errors=[]`、exit `0`。

pause 計時來自 `/usr/bin/time -v`；它證明中斷點確實寫出可恢復 generation，不代表
正常完整 run 的批次 elapsed。

### 5.3 完成後 controller 資源與 checkpoint 指標

下表的 wall／CPU／RSS／`checkpoint_active_bytes` 是四個完成 shard 的 controller
指標。`checkpoint_active_bytes` 是該 shard 目前保留的 schema 3 checkpoint tree
容量；`checkpoint_bytes` 則是 benchmark report 的 logical checkpoint bytes-written
指標，兩者語意不同，不應互相加總後再當成磁碟使用量。

| 分片 | controller wall (s) | controller CPU (s) | 最大 RSS (bytes) | checkpoint active (bytes) |
|---|---:|---:|---:|---:|
| 0：`00000000-00000005_shd_1e1a145ae32e2c3be549` | 325.071858 | 323.661282 | 403,787,776 | 10,378,905 |
| 1：`00000005-00000010_shd_66c456e2c16fba23e794` | 332.854940 | 331.566838 | 424,435,712 | 10,857,319 |
| 2：`00000010-00000015_shd_1dfab9241a1d607d3c4b` | 339.991760 | 338.646759 | 411,013,120 | 11,356,576 |
| 3：`00000015-00000020_shd_b340262a6d60d172354a` | 286.554583 | 285.533785 | 424,480,768 | 9,135,734 |

`benchmark-report.json` 的四片彙總為：wall `1286.128858 s`、CPU `1280.444640 s`、
最大 RSS `445,562,880 bytes`、`particle_steps=839288`、
`checkpoint_active_bytes=41,728,534`、`checkpoint_bytes=41,726,398`、
`output_bytes=3,709,340`，且 `completed_shard_count=4`、`scenario_count_completed=20`。

完成後四 worker 外層腳本的計時欄位誤報 `0.0`，因此不引用該欄位作批次 elapsed。
四個程序是近同時啟動；最慢 worker 的 `/usr/bin/time` elapsed 為 `342.73 s`，只能
作正常四 worker 批次 elapsed 的近似上界。前一組沒有 pause 的 `a7bd697` schema 2
`cp10000` `no_stokes` 批次為 `347.844 s`，可作鄰近 workload 參照；本次 pause 加
resume 的端到端 `153.91 + 342.73 = 496.64 s` 不是正常完整 run 耗時。

### 5.4 schema 3 空間效果與輸出對照

完成後 schema 3 checkpoint tree 實際 `du -sb` 為 `41,731,694 bytes`。同一 workload
的舊 schema 2 `cp10000` checkpoint tree 為 `140,707,819 bytes`，因此 schema 3 約減少
`70.34%`，容量約為舊版的 `1/3.37`。新 run workspace（不含 checkpoint root）的
`du -sb` 為 `3,782,098 bytes`；這是 run metadata／輸出工作區容量，不能與 checkpoint
tree 混為同一指標。

新舊 shard 以 `diff -qr --exclude=manifest.json` 比較，return code `0`；所有 NPY 與
Parquet payload 逐位元一致。唯一差異是各自的 `manifest.json`，因為 commit、artifact
hash、run ID 與 resource metrics 必然不同。這證明 schema 3 的 checkpoint 拓撲變更沒有
改寫本次粒子輸出，不等於已完成正式科學驗收。

### 5.5 chain load 與壓縮容量候選

最大分片的 schema 3 chain warm load 讀取第 5 代、5 particles、1,442 observations
與 11,339 events，elapsed `1.90 s`、最大 RSS `251,740 KiB`。這是 warm-cache 測量，
不是 cold-cache resume；不能用來承諾 SERVER 重啟後的首次載入時間。

以現有 38 個 `compact_state.json`／`history_segment.json` 做 gzip level 1 的 dry
estimate（沒有寫出壓縮檔），原始 `41,701,362 bytes` 可降至 `3,937,076 bytes`，減少
`90.56%`；此估算 elapsed `0.29 s`、user CPU `0.23 s`、最大 RSS `14,596 KiB`。
這只是 schema 3.1 的容量候選證據，壓縮格式會引入另一個設定變因，不能回填成本次
未壓縮 schema 3 正式數字。

### 5.6 工程限制與正式採用條件

本 baseline 建立了 schema 3 的 pause／resume、四分片完整收束、容量與 payload parity
證據，但仍屬 engineering-only：

- SERVER NFS 當時剩餘約 `9.8 TB`、使用率約 `89%`；未壓縮 schema 3 的容量數字不可直接
  當成正式長期配置。
- 本輪沒有 `fsync` 故障注入、generation retention、真正 cold resume 或完整 NFS
  斷線恢復測試；`progress.lock` 仍存在，不能宣稱所有 NFS lock 競爭已消失。
- gzip level 1 只做 dry estimate，尚未測試壓縮寫入、解壓 resume、故障原子性與正式
  validator，不能與未壓縮 run 混用。
- 這是 `no_stokes` 的 24 小時工程 baseline，不是 `finite_depth_stokes`、30 天回溯、
  四區五站或正式成果的工期／科學驗收證據。正式採用前須完成 cold／failure recovery、
  NFS I/O 與 retention 策略、版本化 schema／容差及正式 release gates。

### 5.7 SERVER SIGINT fault／resume 小測

另以同一 schema 3 artifact 建立 `b-hsinchu-no-stokes-cbfe58e-sigint-v1`，只驗證
operator 以 `SIGINT` 中斷時，最近一個已發布 checkpoint 是否保留，以及後續 resume
是否維持執行狀態。這是操作員中斷與安全邊界測試，不是掉電、NFS client crash 或
`fsync` durability 測試。

測試順序與觀察如下：

1. 先以 `--sweep-budget 10000` 執行 shard 0，產生 schema 3 `checkpoint-00000001`
   後進入 `PAUSED`；此階段完成 `10,000 sweeps`、`50,000 particle steps`，elapsed
   `78.42 s`，最大 RSS `382,776 KiB`。
2. 從該 checkpoint 以 GNU `timeout` 執行 30 秒，送出 `SIGINT`；程序 return code
   `130`。中斷後 progress 仍為 `RUNNING`、`sequence=1`、`sweeps=10000`、
   `particle_steps=50000`，目錄只保留 `checkpoint-00000001` 與 `latest.json`，沒有
   `.partial` 目錄，也沒有殘留 process。這表示訊號落在下一個可安全恢復邊界前時，
   不會把未發布的半代誤認為可恢復 checkpoint。
3. 再以相同 binding resume、`--sweep-budget 10000`，產生
   `checkpoint-00000002` 並進入 `PAUSED`；此階段完成 `sweeps=20000`、
   `particle_steps=100000`，elapsed `78.62 s`。
4. 將中斷／恢復 run 的 generation 2 與未中斷 control run 的 generation 2 交給正式
   loader 比較：`identities_equal`、`executions_equal`、`rng_states_equal`、
   `triangle_hints_equal`、`observation_cursors_equal` 與 `event_cursors_equal` 均為
   `true`，兩邊 sequence 均為 `[2, 2]`。最後 `validate-run` 為 `valid=true`，
   lifecycle 為 `PAUSED`。

此測試支持「checkpoint 對 operator 中斷與續跑是必要的」：SIGINT 後能回到最後合法
generation，並保留亂數、粒子提示與 history 游標的一致性。它仍不能推論掉電或 NFS
寫入途中故障的耐久性；正式 release 必須另做 fsync、斷線、cold resume、partial
generation 與 retention 測試。

證據均位於同一 SERVER engineering root：

```text
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/checkpoints/b-hsinchu-no-stokes-cbfe58e-sigint-v1
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/runs/b-hsinchu-no-stokes-cbfe58e-sigint-v1
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/logs/sigint-initial-pause.log
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/logs/sigint-mid-advance.log
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/logs/sigint-resume-to-gen2.log
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/logs/sigint-validate-paused.json
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/profiles/sigint-mid-advance-status.txt
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/profiles/sigint-control-checkpoint-comparison.json
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/scripts/compare_checkpoint_cbfe58e.py
```

### 5.8 schema 3.1 `c67ad89` 正式 SERVER 工程比較

在 schema 3.0 baseline 與 SIGINT 小測後，另以 schema 3.1 進行同一 B 區新竹 24 小時、
20 scenarios、`M=1`、`4 shards × 5` 的 SERVER 工程比較。這裡的「正式」只表示依
固定版本、artifact、run ID 與完整驗證流程執行的正式工程測次；它仍不是正式 30 天
研究成果或五站科學驗收。

#### 5.8.1 版本與初始化流程

| 項目 | 實際值 |
|---|---|
| checkout | `/home/mustlab/Workspace/Lagrangian-Ensemble-Backtracking-c67ad89` |
| Git commit | `c67ad893e5d2527be5c00bf87f72b0a802069297` |
| deployment tree | `1223b13bb6ef6f61b4575e455f7ec566fe84272a` |
| deployment bundle | `/data/LBT/packages/lbt-c67ad89.bundle` |
| bundle SHA-256 | `99cac85ff3241931aef2e0be2458e307b056aef5e5dd693a7c6a73781e0df752` |
| engineering root | `/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914` |
| input artifact | `/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/artifacts/B-hsinchu-24h-c67ad89-v1` |
| run ID | `b-hsinchu-no-stokes-c67ad89-schema31-4w-v2` |
| 亂數流 | `abcd-hsinchu-20240102t0100z-m1-v1` |
| checkpoint payload | 所有 generation 使用 schema `3.1.0`，payload 為 `.json.gz` |

第一個 v1 嘗試在四個 worker 同時建立 workspace；只有 shard 3 成功，耗時 `290.47 s`，
其餘三片安全以 `FileExistsError` 拒絕，外層計時 `290.487542890 s`、return code `1`。
這次不是效能成果，而是確認必須先由 controller 完成 `run-create`／workspace 初始化，
再派發 resume worker。

v2 先由 shard 0 以 budget 1 初始化 workspace，耗時 `5.37 s`，完成 5 particle steps
並建立 generation 1 `PAUSED` checkpoint；之後才同時啟動四個 resume worker。四 worker
外層 elapsed 為 `372.216224447 s`、return code `0`，最後 `4/4 COMPLETE`，20
scenarios／particles，`validate-run --require-complete` 為 `valid=true`、`errors=[]`、
exit `0`。

#### 5.8.2 schema 3.1 controller 資源與容量

下表為四片完成後的 controller 指標；shard 0 的時間與資源含前述初始化累積。

| 分片 | controller wall (s) | controller CPU (s) | 最大 RSS (bytes) | checkpoint active (bytes) |
|---|---:|---:|---:|---:|
| 0：`00000000-00000005_shd_1e1a145ae32e2c3be549` | 307.313707 | 306.936861 | 425,422,848 | 994,645 |
| 1：`00000005-00000010_shd_66c456e2c16fba23e794` | 349.786126 | 349.364385 | 422,600,704 | 1,029,341 |
| 2：`00000010-00000015_shd_1dfab9241a1d607d3c4b` | 369.764398 | 369.384170 | 412,131,328 | 1,072,465 |
| 3：`00000015-00000020_shd_b340262a6d60d172354a` | 294.425198 | 294.093251 | 420,847,616 | 876,057 |

`v2-benchmark-report.json` 的彙總為：wall `1322.845371 s`、CPU `1320.775614 s`、
最大 RSS `449,146,880 bytes`、`particle_steps=839288`、
`checkpoint_active_bytes=3,972,508`、`checkpoint_bytes=3,970,364`、
`output_bytes=3,709,329`，且 `completed_shard_count=4`、`scenario_count_completed=20`。
實際 `du -sb` 為 checkpoint tree `3,976,052 bytes`、run workspace `3,782,083 bytes`。

checkpoint tree 的跨版本比較如下：

| checkpoint 版本 | 實際 tree bytes | 對 schema 3.1 容量倍率 | 相對 schema 3.1 減少 |
|---|---:|---:|---:|
| schema 2 `cp10000` | 140,707,819 | 35.39× | 97.17% |
| schema 3.0 | 41,731,694 | 10.50× | 90.47% |
| schema 3.1 | 3,976,052 | 1.00× | — |

schema 3.1 的外層 elapsed `372.216224447 s` 比前一組無 pause 的 schema 2 `cp10000`
批次 `347.843958 s` 慢 `7.01%`；相對 schema 3.0 只有最慢 worker 的 `342.73 s`
近似上界，慢 `8.60%`，不能視為嚴格的 outer-to-outer 比較。schema 3.1 的四片 CPU
總和相對 schema 3.0 的 `1280.444640 s` 增加 `3.15%`。以上只是一輪 SERVER 測量，
會受當時共享主機負載與檔案快取影響。

#### 5.8.3 payload parity 與 warm load

所有 schema 3.1 generation 均使用 `.json.gz` payload。新 run 與 schema 2、schema 3.0
對應 shard 以 `diff -qr --exclude=manifest.json` 比較均為 return code `0`；所有 NPY
與 Parquet payload 逐位元一致。差異只存在於各 run 的 manifest，原因是 commit、artifact
hash、run ID 與資源欄位不同。

最大分片 schema 3.1 warm load 讀取 generation 5、5 particles、1,442 observations、
11,339 events，elapsed `1.97 s`、最大 RSS `255,824 KiB`。schema 3.0 對應 warm load
為 `1.90 s`、`251,740 KiB`；兩者都是 warm-cache 測量，不能當成 cold resume 數字。

#### 5.8.4 正式限制

- 本輪沒有 `fsync` 故障注入、generation retention 或 cold resume；舊祖先 payload 的
  讀取依 immutable 假設，若更舊 payload 已被破壞，full loader 必須 fail-closed。
- `progress.lock` 仍存在；schema 3.1 的壓縮與追加 payload 降低了容量，但沒有證明
  所有 NFS 鎖競爭已消失。
- 當時 `/data` NFS 約剩 `9.8 TB`、使用率約 `89%`。schema 3.1 雖已成功完成工程寫入，
  正式 30 天仍需通過容量閘門、同步／故障驗證與安全回收（retention）設計。
- 這是 `no_stokes` 24 小時的 schema 3.1 工程比較，不包含 `finite_depth_stokes`，
  也不構成 30 天、四區五站、M 收斂或科學成果驗收。

證據絕對路徑如下：

```text
/home/mustlab/Workspace/Lagrangian-Ensemble-Backtracking-c67ad89
/data/LBT/packages/lbt-c67ad89.bundle
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/artifacts/B-hsinchu-24h-c67ad89-v1
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/runs/b-hsinchu-no-stokes-c67ad89-schema31-4w-v1
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/runs/b-hsinchu-no-stokes-c67ad89-schema31-4w-v2
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/checkpoints/b-hsinchu-no-stokes-c67ad89-schema31-4w-v1
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/checkpoints/b-hsinchu-no-stokes-c67ad89-schema31-4w-v2
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/logs/no-stokes-shard-0.log
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/logs/no-stokes-shard-1.log
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/logs/no-stokes-shard-2.log
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/logs/no-stokes-shard-3.log
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/logs/no-stokes-v2-init-shard0.log
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/logs/no-stokes-v2-shard-0.log
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/logs/no-stokes-v2-shard-1.log
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/logs/no-stokes-v2-shard-2.log
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/logs/no-stokes-v2-shard-3.log
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/logs/v2-reconcile.json
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/logs/v2-validate-run.json
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/profiles/no-stokes-4w-outer.txt
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/profiles/no-stokes-v2-4w-outer.txt
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/profiles/v2-benchmark-report.json
/data/LBT/benchmarks/checkpoint-schema31-c67ad89-20260914/profiles/v2-largest-shard-warm-load.log
```

## 6. 輸出完整性與 NumPy 對照

兩組新版 run 都以 `validate-run --require-complete --checkpoint-root ...` 完成只讀
檢查，結果均為 `valid=true`、`errors=[]`、exit `0`，並確認 `4/4` shards、20
scenarios、20 particles 與 `run_lifecycle=COMPLETE`。

與同一亂數流的 NumPy 串行輸出逐分片比對後，分類欄位、筆數與 schema 均一致：

| 物理案例 | `particle_table` rows | `events` rows | schema／分類結果 | 最大浮點差異 | event time 差異 |
|---|---:|---:|---|---:|---|
| `no_stokes` | 20 | 42,450 | schema 相同；particle identity、status、step count、事件分類與識別欄位一致 | `events.fraction`：`3.894459754683055e-11` | 1 筆，差 1 ns |
| `finite_depth_stokes` | 20 | 42,549 | schema 相同；particle identity、status、step count、事件分類與識別欄位一致 | `events.fraction`：`3.0095592684631356e-11` | 2 筆，各差 1 ns |

兩組的 `events.fraction` 都超過預先登錄的 `1e-12` 容差；因此本輪只能說明輸出結構、
分類、筆數與主要數值行為相符，不能直接宣告 Numba 已達正式數值等價。必須先把正式
浮點與事件時間容差版本化、補上相應的科學／工程驗收規則，再決定是否納入正式 release。

## 7. SERVER 證據索引

以下路徑均為本輪實測的絕對路徑：

```text
# 新版程式與 artifact
/home/mustlab/Workspace/Lagrangian-Ensemble-Backtracking-a7bd697
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/artifacts/B-hsinchu-24h-numba-cp10000-v1
/data/LBT/benchmarks/abcd-five-site-paired-e13569d-task-01a09e01-20260914T035400Z/configs/B-hsinchu-24h-numba-a7bd697-cp10000-source.yaml

# 舊版 NumPy baseline run
/data/LBT/benchmarks/abcd-five-site-paired-e13569d-task-01a09e01-20260914T035400Z/runs/b-hsinchu-current-only-24h-m1-eba4f49-v1
/data/LBT/benchmarks/abcd-five-site-paired-e13569d-task-01a09e01-20260914T035400Z/runs/b-hsinchu-current-wave-24h-m1-eba4f49-v1

# 新版四 worker run
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/runs/b-hsinchu-no-stokes-a7bd697-cp10000-4w-v1
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/runs/b-hsinchu-wave-a7bd697-cp10000-4w-v1

# 新版 checkpoint tree
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/checkpoints/b-hsinchu-no-stokes-a7bd697-cp10000-4w-v1
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/checkpoints/b-hsinchu-wave-a7bd697-cp10000-4w-v1

# 新版外層四 worker elapsed 與每 worker /usr/bin/time -v
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/profiles/no-stokes-cp10000-4w-v1-summary.json
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/profiles/wave-cp10000-4w-v1-summary.json
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/logs/no-stokes-cp10000-4w-v1/worker-0.time.log
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/logs/no-stokes-cp10000-4w-v1/worker-1.time.log
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/logs/no-stokes-cp10000-4w-v1/worker-2.time.log
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/logs/no-stokes-cp10000-4w-v1/worker-3.time.log
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/logs/wave-cp10000-4w-v1/worker-0.time.log
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/logs/wave-cp10000-4w-v1/worker-1.time.log
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/logs/wave-cp10000-4w-v1/worker-2.time.log
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/logs/wave-cp10000-4w-v1/worker-3.time.log

# 逐分片 NumPy／Numba 輸出比較
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/profiles/no-stokes-cp10000-4w-v1-shard0-parity.json
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/profiles/no-stokes-cp10000-4w-v1-shard1-parity.json
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/profiles/no-stokes-cp10000-4w-v1-shard2-parity.json
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/profiles/no-stokes-cp10000-4w-v1-shard3-parity.json
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/profiles/wave-cp10000-4w-v1-shard0-parity.json
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/profiles/wave-cp10000-4w-v1-shard1-parity.json
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/profiles/wave-cp10000-4w-v1-shard2-parity.json
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/profiles/wave-cp10000-4w-v1-shard3-parity.json

# NFS lock 競爭診斷（cp1000，刻意停止，非成果 run）
/data/LBT/benchmarks/ocm-runtime-a7bd697-20260914/profiles/no-stokes-4w-v1-summary.json

# schema 3 cbfe58e4 engineering baseline
/home/mustlab/Workspace/Lagrangian-Ensemble-Backtracking-cbfe58e
/data/LBT/packages/lbt-cbfe58e.bundle
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/artifacts/B-hsinchu-24h-cbfe58e-v1
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/runs/b-hsinchu-no-stokes-cbfe58e-schema3-4w-v1
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/checkpoints/b-hsinchu-no-stokes-cbfe58e-schema3-4w-v1
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/logs/no-stokes-shard0-pause.log
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/logs/no-stokes-shard-0-resume.log
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/logs/no-stokes-shard-1-resume.log
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/logs/no-stokes-shard-2-resume.log
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/logs/no-stokes-shard-3-resume.log
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/logs/reconcile.json
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/logs/validate-run.json
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/profiles/benchmark-report.json
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/profiles/largest-shard-warm-load.log
/data/LBT/benchmarks/checkpoint-schema3-cbfe58e-20260914/profiles/schema3-gzip1-dry-estimate.log
```

## 8. 研究解讀界線

本輪證明的是：在一個 24 小時新竹工程 workload 上，既有四分片平行入口可以配合
Numba OCM 後端與較低 checkpoint 頻率完成；相較刻意串行 NumPy baseline，實際外層
elapsed 明顯下降，且 checkpoint tree 大幅縮小。它沒有證明 30 天工期可以按 24 小時
比例線性外推，也沒有證明五站、四區、全部正式情境的 NFS throughput、記憶體峰值、
收斂性或科學有效性。正式採用前仍需完成長回溯容量／中斷恢復、版本化容差、完整正式
輸入 gate、M 收斂與研究報告 gate。
