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

## Q4 展示子集入口與場快取容量

固定五站純流100顆展示使用 `scripts/run_current100_subset.py --help`：先由既有
`run-create --run-kind formal --experiment-case no_stokes` 建立新的完整計畫，再把
新／來源工作區、設定、forcing、checkpoint、NFS scratch／log／cache、儲存驗證快照及
`--expected-commit` 交給腳本。預設僅準備；經核准加 `--execute` 才以五個持續 `run-worker`
各跑兩片。腳本固定選取 gongliao、guishan、hsinchu、lienchiang、nanwan 各站最早指定鋁材
的全部20受體，各站到達UTC與ID記入新log session的 `demo-manifest.json`；不同站不宣稱同時
間配對，不改原計畫／seed／dt／H30／沉降／擴散。五項thread限制均為1，MemAvailable低於
30GiB或不可觀測時停止，對本次子程序SIGINT等待300秒再有界收尾。`events.jsonl` 記錄PID、
CPU累積秒、RSS、步數、checkpoint序列及逐站原版完成證據；工程例外使子集失敗，真實資料QC
原樣保留。整機elapsed直接使用單一操作程序時鐘，不能相加worker時間。僅100選取子集可完成，
完整50,000情境run仍未完成，不呼叫全量aggregate或以圖宣稱正式成果。

同一100顆子集續跑時，沿用上述參數及原 `--workspace`／`--checkpoint-root`，明示加
`--resume --execute`。預設仍只接受全未啟動的新工作區；續跑僅放行固定十片的合法狀態，
其餘4,990片必須維持 `PLANNED`／零步數。原run計畫、seed與檢查點繫結不改寫，新程式commit、
實際來源、初始十片進度與revision另存新log session。既有CLI `--resume` 負責片鎖下的檢查點
還原與完成輸出驗證，不重算已驗證完成片、不先執行全run reconcile。此選項**不略過**既有
CLI／runtime的全run歷史驗證，因此啟動仍可能耗時；不能將setup當成粒子推進。切換前必須確認
舊operator與workers均已退出，不能同片雙寫；本腳本不代替操作者停止其他invocation。

純流逐站出圖：`build_completed_subset_preview.py --current100-site SITE` 配合該站兩個
`--shard`，在該站20顆完成後輸出四張PNG，沿用完整單片輸出驗證；站名取目前設定，五站到達UTC
分別標示。尚未完成時可用 `build_current100_checkpoint_preview.py --run RUN --checkpoint-root
CHECKPOINT --config CONFIG --project-root CODE --site SITE --output NEW_OUTPUT` 讀取該站目前
已發布checkpoint，先提供水平／垂向未完成軌跡圖與位置快照。只畫已保存粒子，分母明示已保存
數／站20／總選取100；不補線、不縮短H30、不將當前末端或資料QC當成來源認定。圖檔與快照須放
在新的成果目錄，不改運算checkout或原run。

已審兩站岸界40顆入口 `run_native_coast40_subset.py` 的完整命令與限制見
[已審模型岸界](reviewed_native_coast.md)。原版場快取的runtime容量可成對明示
`LBT_BAYTRACE_FRAME_CACHE_MAX_FRAMES` 與 `LBT_BAYTRACE_FRAME_CACHE_MAX_BYTES`；兩者皆未
設定仍為2場／512MiB。值須為ASCII十進位正整數，最高64場／4GiB；只設其一、空值、負數、
小數或超限皆拒絕。這只是執行記憶體配置，不更改物理設定、原版RK4、粒子順序、批次分組、
seed、dt或run plan。固定100入口 `run_current100_subset.py` 與已審兩站岸界40顆入口
`run_native_coast40_subset.py` 明示覆寫20場／4294967296 bytes，將實際環境與容量記入新
session `demo-manifest.json` 的 `frame_cache_limits`；resolver report沿用實際上限與建場、
淘汰計數。五worker場快取合計上限20GiB，但常駐場快取上限**不是程序RSS上限**，建場暫存、
forcing與檢查點另占記憶體；仍須保留MemAvailable30GiB安全停止及五項單執行緒限制。不得無
條件將這個配置套到56worker或50k；其他入口未設上述環境即維持原容量。

## 有界唯讀步進診斷

OCM原版轉接建場採節點／層批次遮罩與向量化拓樸檢核，保留原sigma算式與
NumPy弱純量dtype捨入，不更動PI核心、RK4、步長、種子或停止條件。底部未使用層
仍重複首個有效層，所有有限值／單調／範圍／重建／索引／方向／面積gate保留，
四邊形第二三角形獨立檢核；不同錯誤同時發生仍回報原節點／元素順序的首個理由。
`tests/baytrace_scalar_reference.py`保存向量化前逐節點／元素參考，
由混合float32／float64逐陣列bitwise與拒絕理由測試核對，不能據此宣稱已排除全部效能瓶頸。

`scripts/diagnose_failure_points_readonly.py` 僅用於一至五個既有失敗點的唯讀比對：
依原事件座標／UTC及固定代提示，呼叫原速度取樣器、帶／不帶提示的定位，以及
全部三角形包圍盒候選的既有精確定位；只量測投影與邊界距離，不自行分類或外插。
以 `--spec` 指定已核對的乾淨模型版本與原run／事件／狀態路徑，`--output` 必須是
獨立NFS操作目錄；只建一個診斷子程序，180秒硬限及30GiB可用記憶體門檻。
不建立執行批次、不推進、不消耗亂數、不寫原進度，結果不是正式成果。
規格JSON的新欄位 `checkpoint_dir`、`expected_sequence`、`expected_point_count`
必須成套指定；序號為正整數，點數為1..5整數，布林值、文字與浮點數皆拒絕。
舊規格僅含 `checkpoint62` 時仍固定第62代／五點；新舊欄位混用拒絕。
原十粒子分片、描述檔序號／分片、compact狀態SHA、事件粒子唯一且屬於request與
狀態、位置與UTC精確相等的檢核均保留；全面候選精驗容差仍為1e-10。
結果分開保存模型版本與 `diagnostic_script_commit`／腳本SHA，不能將診斷工具
更新誤認為現行數值模型版本切換，也不以水平取樣不支援自行決定停止分類。

`scripts/diagnose_batch_readonly.py` 只在新的獨立程序記憶體中重現指定純流分片，
不是正式計算或成果。必須明示 `--project-root`、`--expected-commit`、`--workspace`、
`--config`、`--checkpoint-root`、`--ocm-native-root`、`--ocm-reconstruction-root`、
`--shard-id` 與唯一 `--output`（位於 NFS operations 的診斷目錄）；
有檢查點時另外明示 `--checkpoint`。使用原版本環境的 Python 啟動獨立腳本，
不將腳本複製進計算 checkout。原 BayTrace 根目錄由既有 `LBT_BAYTRACE_ROOT` 保留。

另明示獨立工具環境的 `--py-spy` 絕對路徑。最多 `--max-sweeps 1000`，
`--timeout-seconds` 限120–180秒；py-spy直接啟動新診斷Python，以祖先權限每秒5次
非停止式預設採樣60秒並保存 `profile.raw`；`--profiling-seconds` 可指定30–150整數秒，
且須不超過 `--timeout-seconds` 減20秒，預留採樣落盤及清理時間。較長單次重現可用
`--setup-validation startup --profiling-seconds 150 --timeout-seconds 180 --max-sweeps 1000`，
仍不能超過180秒硬上限；命令與來源紀錄保存實際採樣秒數。
不附加既有worker、不sudo或改系統權限，
也不混用faulthandler/cProfile。每10次掃描彙總已完成步數與年齡。保留原種子、檢查點亂數、
步長、物理及批次平流器，不呼叫控制器執行／恢復／調和流程，也不發布軌跡或新檢查點。
`--setup-validation startup`（原預設）使用既有私有啟動驗證能力，保留不可變繫結與
路徑檢核，不掃全批歷史；明示檢查點的單片鏈仍由既有載入器驗證。
`--setup-validation full` 則不傳私有能力，完整走公開控制器預設的static歷史驗證；
這是驗證路徑的有界比較，不另外重複CLI handler的前置掃描，也不能據此先宣告停滯原因。
兩模式皆記錄setup起訖、建構／恢復、每10次步進與模組來源，硬時限及唯讀限制相同。
來源紀錄分開保存模型commit／目錄與腳本路徑／SHA；startup明示
`diagnostic_startup_bindings_only=true`、`full_history_validation_not_performed=true`。
full啟動時只標記已要求完整驗證，是否完成必須以`setup_complete`判讀，不能由採樣器
退出碼推斷。所有模式均`formal_result=false`，不改原worker驗證政策或PI BayTrace原始碼。
所有暫存與快取另置診斷目錄、五種執行緒限制均為1；採樣完成或硬逾時只清理診斷自建群組。
`provenance.json`、`pid.json`、`numerical-pid.json`、`stdout.jsonl`、`stacks.stderr`、
`profile.raw`、`result.json` 是診斷證據；採樣失敗不得冒稱已定位卡點。
每10次掃描的摘要另保存cache hits/misses、frame builds/evictions、cached bytes及
native completed/fallback/group-size程序累計統計；快取容量不因向量化而調整。
原進度可能由既有worker正常更新，不能把它當成診斷寫入。
