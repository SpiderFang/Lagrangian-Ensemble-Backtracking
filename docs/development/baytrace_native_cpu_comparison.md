# Q2 BayTrace 原始 CPU 與 LBT 對照

本工具是本機工程 benchmark，不是 full tracker、SERVER throughput 或正式科學驗證。
它在可控的公尺制單一三角形、固定水深、平面 sigma、靜態非均勻節點速度場上，直接執行
PI BayTrace 原始 `batch_interpolate_velocity`／`batch_move_rk4`，並和 Q1 thin adapter、
既有 LBT `numba_ocm_v1` + `numba_cpu_v1` 的 direct passive RK4 對照。

## 直接執行

每次請指定新的輸出路徑；工具拒絕覆寫既有 JSON 或 profile：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/mustlab/Workspace/Lagrangian-Ensemble-Backtracking/.venv/bin/python \
scripts/benchmark_baytrace_native_cpu.py \
  --baytrace-root /Users/mustlab/Workspace/Lagrangian-Ensemble-Backtracking/BayTrace/Baytech_BayTrace_v4.7.2 \
  --output /private/tmp/q2-baytrace-native-$(date +%Y%m%d-%H%M%S).json \
  --particles 128 --steps 16 --repeats 3 --dt-seconds 0.25 --profile
```

`--profile` 只對已暖機的 backward `ProductionBatch` lifecycle 產生獨立
`.production.cprofile`；instrumented wall 不進任何 speed ratio。預設值是
`128 × 16 × 3`，可用 `--particles`、`--steps`、`--repeats` 調整。

## 可比較的內容

- raw、adapter 與 LBT direct 都各自重置 initial arrays；每個粒子每步重新取一次 k1，
  共計 `N × steps` 次 k1 與四階 RK4 stage evaluation。raw 的 `x0/y0/z0` copy 計入
  steady wall，adapter 的等價 copy 在 `step_native` 內計入。
- forward 與 backward 都使用相同正 `dt`（目前 `.25` 或 `1` 秒），方向由 `ibf=+1/-1`
  表示；兩個 forcing time rows 完全相同，固定 `trat=0.5`，避免 alpha 變化造成假等價。
- 位置單位是 m、速度是 m/s；每條路徑均記錄個別 repeat、median、particle-steps/s、
  finite、status、source hash、Numba signatures 與最大絕對位置誤差。stationary interior
  case 的明示數值通過容差是 `1e-9 m`。
- `ProductionBatch` 是逆向引擎，只在 backward 表中量測並標記
  `actual_direction=backward`；它的 construction/request identity、active compaction／
  scatter、alignment/QC/boundary checks、observations/events、`numba_cpu_v1` RK4 與
  零係數 operator-split diffusion/RNG lifecycle 都是額外成本。它不是 BayTrace 等量，
  不列入 forward 倍率或核心 equivalence pass。
- LBT synthetic provider 具備 `sample(..., triangle_hint=None)`、`__call__` 與
  `step_start_sample_reuse_safe=True`，只轉交同一個 `OCMNativeMonth.sample`。每顆核心
  粒子使用既有 `HintTrackingVelocityProvider(triangle_hint=0)`，並把步首 sample 明示傳給
  RK4；ProductionBatch 也使用同一 provider。JSON 會記錄每個 repeat 的實際 sample 次數、
  每粒子每步次數與非空 hint 次數；Production 不把額外查詢次數預設成 4 或 5。

## 不具備的比較能力

BayTrace direct path 沒有 full tracker 的 SEARCH、clamp、輸出與完整事件生命週期；LBT
direct path 也刻意排除 boundary、events、observations、diffusion 與 ProductionBatch
管理成本。這個 synthetic case 不代表真實 forcing、完整 tracker、SERVER 排程或吞吐量。

若改成 time-varying rows，BayTrace 原始 RK4 會以同一 `trat` 用於所有 stage，而 LBT
`OCMNativeMonth.sample` 可依 stage UTC 使用不同 alpha；兩者差異是已知演算法語意，不是
本工具要修正的 bug，也不可把該 case 當作 stationary equivalence。

## 驗證與 evidence

```bash
LBT_BAYTRACE_ROOT=/Users/mustlab/Workspace/Lagrangian-Ensemble-Backtracking/BayTrace/Baytech_BayTrace_v4.7.2 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/mustlab/Workspace/Lagrangian-Ensemble-Backtracking/.venv/bin/python \
-m pytest -q tests/test_baytrace_native_benchmark.py
```

測試驗證 OCM `(time,node,layer,component)` 與 BayTrace `(layer,node)` 的逐節點／逐層
對齊、非均勻場 k1 一致性與既有 evidence 防覆寫；沒有 wall-time hard threshold。JSON
中的 `comparison.pass=false` 會使 CLI 寫出失敗 evidence 後以非零狀態結束，不會把不一致
誤報為通過。

本輪公平性修正版 evidence：
`q2-benchmark-128x16x3-fairness-final3-20260919.json` 與
`q2-benchmark-128x16x3-fairness-final3-20260919.json.production.cprofile`；兩者位於本機
輸出目錄 `/private/tmp/lbt-baytrace-native-evidence-20260919/`，不是版本追蹤文件。

## Q3 synthetic ProductionBatch 生命週期 benchmark（最新 review）

Q3 的 cache review 與 9 項實源測試已完成。最新 review JSON 檔名為
`q3_baytrace_production_benchmark_review.json`，本機輸出目錄為
`/private/tmp/lbt-baytrace-production-evidence-20260919/`；它記錄 synthetic OCM current
加定值沉降、no-Stokes 路徑的完整 `ProductionBatch` 生命週期。先前的
`q3_baytrace_production_benchmark_v2.json` 保留為歷史 evidence，不代表目前結果。
這不是完整 SERVER I/O 或真實 forcing；結果只說明現有 LBT lifecycle 接上原版 BayTrace
CPU batcher 後，在此 synthetic case 的可重現性與 wall time。

| 規模 | legacy (s) | native (s) | 最大位置誤差 (m) | frame builds | cache hits | fallback 粒子步數 | comparison.pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128×16×3 | 1.500 | 0.578 | 0 | 1 | 2,047 | 0 | true |
| 512×16×3 | 6.194 | 2.341 | 8.88e-16 | 1 | 8,191 | 0 | true |

兩組 wall time 的算術比值約為 2.6×。歷史 Q2 direct-core 的 139× 只代表核心對照，
不能外推成完整 `ProductionBatch`、完整 SERVER I/O 或 full-run throughput；本結果也不
代表已完成 SERVER／真實 forcing 驗證。該歷史版本曾允許 NWW Stokes 與未支援形狀
退回舊核心並計數；這不是下述目前嚴格原版接線的行為，舊測速不能當作波浪證據。

## Q4 runtime 接線與解讀邊界

Q4 只在 `execution.advection_backend: baytrace_native_cpu_v1` 明示時建立 Q2
`NativeBayTraceBatchAdvector` 與 Q3 `OCMPreparedFrameResolver`；舊設定省略欄位時仍
使用既有 LBT stage RK4，並保留舊 canonical hash。`LBT_BAYTRACE_ROOT` 是唯一原始
BayTrace source root，`NUMBA_CACHE_DIR` 是外部 cache；未設定 cache 時由 runtime 建立
source tree 之外的 temporary directory。實際 source SHA-256、環境值、native／fallback
粒子步與 batch 計數，以及 resolver report 會寫入
`run_metadata.advection_execution`，並標註單次 shard invocation scope。

目前此後端對純流與有限水深波浪均使用 `require_native=True`。波浪政策
`finite_depth_node_fields_v1` 在 OCM 的兩端UTC、有效濕節點及垂向層上，以既有
有限水深公式計算 NWW3 Stokes 水平速度，先組成總 u/v，再交原版內插及批次 RK4；
沉降只加一次。這與舊版每個粒子／RK4階段取樣波浪參數的離散方法不同，不能宣稱
軌跡完全等價；上述純流合成測速也不是此波浪方法的誤差或實機吞吐估計。

工程不支援直接拋出可恢復例外，不改用舊核、不偽裝成粒子數值停止。真實乾點、域外、
時間等品質狀態仍逐項保存。512MiB場快取上限不是程序RSS上限，月份資料與建場暫存
另占記憶體。真資料小試使用 `scripts/run_native_wave_trial.py --help`，只對新正式
H30/M1完整計畫的一個明示分片做一次掃描並保存檢查點，不縮短科學回溯窗；
控制器仍完成靜態繫結與逐片驗證，外層保留單執行緒、30GiB可用記憶體與SIGINT停等。
小試紀錄分開保存設定／來源、程式與操作腳本SHA、準備及步進含檢查點時間、原版完成／
補算計數和群組大小。首次編譯與步進尚未分離時不把整段耗時稱為純核心速度。
小試通過不等於完整H30計算、波浪科學驗收或獨立驗證完成。

## Q4 入口與繁中註解檔案

| 用途 | 檔案 |
|---|---|
| 設定 token 與來源建立 | `src/lagrangian_backtracking/config.py`、`runtime.py` |
| 分片／續跑 metadata 與 counter delta | `src/lagrangian_backtracking/run_control.py` |
| 確定性 task 與既有生命週期 | `advection.py`、`engine.py`、`production.py` |
| 原版 CPU、OCM frame 與批次轉接 | `baytrace_native_cpu.py`、`baytrace_ocm_adapter.py`、`baytrace_batch_advection.py` |
