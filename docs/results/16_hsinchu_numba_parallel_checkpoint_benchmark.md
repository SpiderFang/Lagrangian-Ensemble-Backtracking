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

schema 3 的 immutable history segments 與 compact state 仍在審查，未納入本節兩次
SERVER 實測數字。它預期把每代新增 history 以追加段保存，降低長回溯的重複寫入；在
validator、resume、故障與容量測試完成前，不把它當作正式執行依據。

## 5. 輸出完整性與 NumPy 對照

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

## 6. SERVER 證據索引

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
```

## 7. 研究解讀界線

本輪證明的是：在一個 24 小時新竹工程 workload 上，既有四分片平行入口可以配合
Numba OCM 後端與較低 checkpoint 頻率完成；相較刻意串行 NumPy baseline，實際外層
elapsed 明顯下降，且 checkpoint tree 大幅縮小。它沒有證明 30 天工期可以按 24 小時
比例線性外推，也沒有證明五站、四區、全部正式情境的 NFS throughput、記憶體峰值、
收斂性或科學有效性。正式採用前仍需完成長回溯容量／中斷恢復、版本化容差、完整正式
輸入 gate、M 收斂與研究報告 gate。
