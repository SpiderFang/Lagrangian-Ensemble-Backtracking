# Execution checkpoint schema 3.1 gzip 工程候選操作契約

本文件說明尚未正式發布的 execution checkpoint schema `3.1.0` gzip 工程候選。它延續
schema `3.0.0` 的 immutable segment 契約，並解決長時間回溯中
每代重寫完整 observation／event history 造成的寫入量平方增長；它只保存可恢復的工程
狀態，不能取代 trajectory、輸入 manifest、科學驗證或正式成果報告。

SERVER 目前仍使用 schema 2.x；本機先前產生的 schema 3 draft 不是已發布格式，沒有持久化
相容承諾，也不得交給 final schema 3.1 loader／resume。schema 3.0 draft 與本候選均未部署；
只有以本文件契約建立新 3.1 chain root、完成 SERVER fault／resume 與資源驗證後，才能另行
決定正式部署。writer 固定發布 3.1.0，loader 仍可讀 3.0.0 舊拓撲。

## 操作順序

1. 先確認同一 run 的 checkout、run plan、input binding、seed policy 與 checkpoint root
   沒有變更；SERVER 的 checkpoint root 必須位於已通過儲存閘門的 `/data/LBT` 子目錄。
2. 由 `run-shard` 在完整 sweep／macro boundary 呼叫 `ProductionBatch.write_checkpoint`。
   不要手動複製 JSON、刪除中間 generation，或把 partial directory 改名成正式 generation。
3. 每代發布後檢查 `checkpoint.json`、`compact_state.json.gz`、`history_segment.json.gz` 的
   壓縮檔 `st_size`／SHA-256／解壓後大小，以及 `latest.json` 與 progress 的 sequence／counter
   交叉連結。
4. 中斷時保留已發布 generations；以原 run plan、同一 input binding、seed、shard ID 與
   checkpoint root 執行 `run-shard --resume`。controller 會先掃描 generation，再由最高代
   完整驗證 segment chain，通過後才建立 request factory 與載入 forcing。
5. 若 validator 回報 chain 缺失、checksum、cursor、identity 或 binding 錯誤，立即停止該
   shard，保存錯誤與目錄清單，不能退回較舊 generation 產生看似完整的結果。

目前尚未啟用 generation retention，因此同一 shard 的已發布 generation 序號必須完整
連續（`1..highest`）。schema 可以全程維持 `2.0.0`／`2.1.0`／`2.2.0`，也可以由
2.x 遷移一次後全程使用 `3.1.0`；既有 `3.0.0` 可接續升級到 3.1.0，但一旦出現任一
schema 3.x，後續插入 2.x 會被 scanner 拒絕，也不允許 3.1.0 回降 3.0.0。
若 `RUNNING` progress 尚停在較舊代，crash window 只允許採認恰好下一代的 orphan；
高出一代以上表示中間 generation 遺失或 progress 被回退，必須保留現場並停止恢復。

## 目錄與欄位

每個不可覆寫的 generation 位於固定路徑：

```text
$LBT_CHECKPOINT_ROOT/<run_id>/<shard_id>/
├── checkpoint-00000001/
│   ├── checkpoint.json
│   ├── compact_state.json.gz
│   └── history_segment.json.gz
├── checkpoint-00000002/
│   └── ...
└── latest.json
```

`checkpoint.json` 是 generation manifest，保存 schema、sequence、粒子／觀測／事件總數、
完整 binding、固定 particle order、兩個 payload 的檔案大小／SHA-256、固定的
`chain_root_sequence` 與 optional `legacy_source`。3.1 manifest 對每個 `.json.gz` payload
另保存 `content_encoding="gzip"` 與 `uncompressed_size_bytes`；`size_bytes` 永遠是 gzip
完成關閉後的壓縮檔 `st_size`。`segment.previous_checkpoint_json_sha256` 以及
`history_segment.json.gz` 解壓內容內的同名欄位會指向上一代完整 `checkpoint.json`；因此上一代
compact、RNG、identity、binding、history payload checksum 與遷移來源都被同一條 hash chain
綁定，不能跳代、重排或改寫後再繼續。

`compact_state.json.gz` 每代只保存目前的粒子狀態與恢復控制資料。每個 particle record
包含 RunUnit identity、`ParticleState`、步數、最小步長夾制次數、下一個輸出 age cursor、
observation／event cursor、PCG64DXSM RNG state、triangle hint 與最後一筆尚可能被 engine
更新 context 的 pending observation。最後一筆觀測暫放 compact，是因為同一時間／位置的
engine context 可能在下一次取樣前被替換；跨代判定沿用 engine 的同點契約：particle 與
UTC time 必須相同，age 允許 `np.isclose(rtol=0, atol=1e-12)` 的浮點容差；它不代表目前
state 必定與觀測落在同一時間。
若某粒子的 `ParticleState.status` 已不是 `ACTIVE`，後續 generation 必須逐欄保留該粒子的
current state、計數器、輸出與 history cursor、pending observation、RNG、triangle hint 及
既有 history 長度；writer 會用前代 compact 做 O(P) 比對並拒絕任何終止後追加或修改，避免
公開 API 產生看似可讀但生命週期已分歧的 checkpoint。

`history_segment.json.gz` 每粒子只保存上一代 cursor 之後新增的 immutable rows。每筆 record
包含 identity、觀測與事件的起訖 cursor、row count、觀測 rows 與事件 rows。loader 會由
chain root 依固定 particle order 逐段附加資料，再把 compact 的 pending observation 接在
末端；任何缺失、重複、跳號、cursor 回退、identity／binding 改變或 checksum 不符都會
fail-closed。
writer 另在本代新增範圍及其 pending 邊界檢查相鄰 observation 的 engine key：同一 particle
若 UTC 相同且 age 差距不超過 `1e-12` 秒，只能保留一列。這會攔截以 `append`、`extend`、
`insert`、空 slice insertion、`+=` 或 `*=` 造成的重複假列；合法的 pending context 更新
仍是替換同一列，並依 particle／UTC／age 核心契約通過。檢查只掃本代增量與一個邊界列，
不會因 generation 數增加而重掃完整已發布 history。
writer 在建立 partial 前也會把本代 compact state、觀測（含 pending）與新增事件以 loader
同一套 strict semantic decoder 預驗，並核對 observation／event 與 RunUnit 的 identity；
因此欄位 primitive、fraction、UTC 型別或粒子歸屬錯誤會在 atomic publish 前拒絕，不會留下
只能發布卻無法立即 resume 的 generation。這項預驗只涵蓋本代 `Θ(P + ΔH)` 資料；已發布
的更早 history 仍由既有 chain checksum／loader 驗證。
同一個 partial 建立前也會嚴格驗證每個 RunUnit identity 的非空文字欄位、member／seed 的
非負原生整數與 arrival UTC 的原生整數型別；所有 RunUnit 的 `particle_id` 必須全域唯一。
因此不同 member 或 seed 仍不能共用同一 `particle_id`，避免 loader 的 particle order 無法
唯一還原。這些欄位驗證使用與 loader 相同的 canonical identity，成功發布即不會留下之後才
被 loader 拒絕的 identity payload。
同一個 preflight 也會驗證 checkpoint binding 的六個固定欄位都是非空原生字串；若有
`random_stream_id`，則必須是非空白原生字串。writer 與 loader 共用此 parser，禁止 tuple、
list、bool 或空值經由 JSON 正規化後才造成 binding 比對失敗；schema 2.x 舊檔也沿用相同
欄位驗證，保留缺少 optional stream 欄位的相容語意。

controller 的 generation scan 仍會讀取並 JSON parse 每代 compact／history payload；3.1 會先
驗證 gzip 壓縮檔 `st_size`／SHA-256，再解壓核對 raw 大小與 JSON 欄位拓撲、cursor、row count，
但只在最高代完整 restore 時建立全部
`Observation`／`BoundaryEvent` 物件。若保留很多代，冷快取或 NFS 連線下的 scan／resume
可能多次線性讀取整條 chain；這部分必須以 SERVER fault／resume 實測評估，不能用本機 page
cache 命中推論 NFS 實際讀取量或完成時間。現行 loader 會先保留完整 chain 的 JSON dict，
再建立還原物件，因此 resume 的記憶體峰值約為 `Θ(GP + H)` 的 JSON／history 內容，可能
是檔案長度的數倍；本輪不以 streaming loader 取代既有契約，需先以 SERVER fault／resume
benchmark 量測 RSS 與時間後，才能評估正式部署。

## 寫入與容量語意

writer 的發布順序是：在同一父目錄建立 `.partial-*` → 以 gzip level 1、mtime=0、空 filename
串流寫 compact 與 segment → 關閉 gzip trailer → 計算壓縮 payload checksum／大小與 generation
manifest → `os.replace` 發布 generation。controller 隨後原子更新
`latest.json`，最後更新 progress。partial 目錄沒有完整 manifest 時不具備可恢復資格；若
Python 例外或 `KeyboardInterrupt` 發生在 writer 的清理範圍內，`BaseException` cleanup
會嘗試移除該 partial；NFS 權限／I/O 錯誤仍可能讓暫存目錄殘留。這不涵蓋 `SIGKILL`、主機掉電或
NFS client crash，因為這些情況沒有機會執行清理。scanner 對殘留 partial／未知項目會 fail-closed；遇到這類現場應先停止 worker、
保存程序與檔案系統證據，再依 SERVER runbook 由人工處理，不能把 partial 改名或當成成功
generation。若 generation 已發布而 latest／progress 更新失敗，controller 會以完整 loader
驗證與固定序號採認 orphan，並由下一次 reconcile 重建 latest，不會以新的序號重寫同一狀態。

`resource_usage.checkpoint_bytes` 是每次新 generation 的 lifetime logical bytes-written，
即使未來實作 retention 也不應拿它當成目前磁碟使用量。`checkpoint_active_bytes` 是目前
shard checkpoint tree 的已發布普通檔案 `st_size` 邏輯長度加總，包含已保留 generations、
`latest.json` 與 checkpoint payload。它不包含目錄 block、block rounding、metadata、NFS
replication／snapshot 或 partial／暫存檔，不能取代 `du`、`df` 與 SERVER 儲存閘門。run
report 可把各 shard 的 active logical file bytes 各計一次得到當下 run tree 的檔案長度，
但不能把同一 shard 的歷代 gauge 再累加。

目前 writer 不刪除 generation，因此 hash chain 的歷史與 active logical file bytes 都可直接回查；
`checkpoint_bytes` 在 orphan recovery 也會由所有已發布 generation 檔案實際大小重建，
`checkpoint_active_bytes` 則另含 latest pointer。
若以 `G` 表示 generation 數、`P` 表示粒子數、`H` 表示整條執行期間新增且發布的觀測／
事件列數，v3 全鏈的容量與寫入量可拆成 compact／manifest 的 `Θ(GP)` 加上 history
segment 資料的 `Θ(H)`（另有每代固定的 JSON／checksum 開銷），即為 `Θ(GP + H)`。
相較之下，舊格式每代重寫累積 history 時為 `Θ(GP + GH)`；若用每代平均新增
`ΔH` 表示，則可寫成 `Θ(GP + G²ΔH)`。因此 v3 不再因每代重寫全部歷史而平方放大。
現行未啟用 retention，所以 active logical file bytes 也會保留整條 chain，隨同一組已發布
generations 線性增加。
若未來要加入 retention，必須先證明被刪除的 generation 不再被任何保留 segment 引用，並
同步更新 validator、恢復測試與操作契約；在此之前不得手動清除整棵 checkpoint 或中間
generation。`latest.json` 很小且可由已發布 generation 重建，不能把刪除 latest 當成主要
回收方式；只有確認沒有 writer、保存現場證據並依 runbook 處置的殘留 `.partial-*`，才可由
人工流程清理。

這一版沒有額外呼叫 file／directory `fsync`；同父目錄的 atomic rename 只保證名稱發布順序，
不等同於主機掉電或 NFS durability。若主機掉電、NFS client crash 或儲存系統在 flush 前
遺失資料，下一次 hash／checksum 驗證會偵測截斷並 fail-closed，不會自動退回較舊代。這項
SERVER fault test 與成本評估完成前，schema 3 只能作工程候選，不能直接作正式部署契約。

schema 3 消除歷史列的重複寫入，但不會自動消除 controller 對共享 `progress.lock` 的
同步競爭；每次 checkpoint 仍需依既有 run-level progress 契約更新該鎖保護的 JSON。若多
worker 在 NFS 上因鎖等待成為瓶頸，必須另行調整 checkpoint interval、worker 併發或
progress 發布策略，不能把 schema 3 的線性 history 寫入誤稱為鎖競爭已解決。

`ProductionBatch.advance` 必須先正常回傳完整 sweep 的結果，controller 才會發布下一代
checkpoint。若 Ctrl-C／例外發生在 advance 內，當前 execution 可能只完成部分粒子，
controller 會保留上一個已發布 generation 與原有 `RUNNING` progress；下一次 `--resume`
會由該代重新執行整個 interval。這個安全邊界避免使用舊 sweep counter 為半個 sweep 建立
看似可恢復的檔案，也避免在沒有 phase cursor 時重複或遺失粒子步進。

## 舊 checkpoint 與遷移界線

loader 仍可唯讀讀取 schema `2.0.0`、`2.1.0` 與 `2.2.0` 的舊三檔拓撲；舊目錄不會被原地
轉換。從舊 checkpoint 續跑時，第一代 v3 generation 會把舊完整 history 一次寫成新的
chain root，並以 `legacy_source` 保存舊 schema、sequence、同 parent 相對目錄與舊
`checkpoint.json` SHA-256。舊 schema 合法的 sequence `0` 也可遷移到 v3 sequence `1`；
此一次性成本與後續每代追加成本要分開記錄，不能把 v3 chain root 的建置誤稱為每代重寫。
目前 loader 會在每次驗證時重新讀取並核對 `legacy_source` 所指的舊目錄，因此該目錄
必須留在同一 parent 的 canonical 路徑；在不改版 migration／provenance 契約前不可回收
或移動舊 schema 檔案。

## 驗證與故障定位

本機 synthetic 契約可用下列命令驗證；它不下載真實 forcing，也不代表 SERVER 科學成果：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python3 -m pytest \
  tests/test_checkpoint_segments.py tests/test_checkpoint_execution.py -q
PYTHONDONTWRITEBYTECODE=1 uv run ruff check \
  src/lagrangian_backtracking/checkpoint.py \
  src/lagrangian_backtracking/production.py \
  src/lagrangian_backtracking/run_control.py \
  src/lagrangian_backtracking/run_validation.py
```

遇到故障時先保存 `run_progress.json`、`latest.json`、所有 generation 的
`checkpoint.json` 與外層程序／檔案系統記錄，再依錯誤類型區分：partial 是未發布寫入、
checksum 是檔案內容或傳輸損壞、chain missing／cursor 是歷史拓撲不完整、identity／binding
是錯誤 run 或設定混用。這些情況都不能以零值、最近值、重新派 seed 或跳過歷史列補救。

## 研究性限制

schema 3 只證明工程狀態可以逐段保存與恢復；恢復後仍須通過 trajectory validator、輸入
coverage／gap-safe gate、資源與 provenance 檢查。synthetic 的逐欄等價、segment 線性成長
與故障拒絕測試，不構成 OCM／NWW3 的物理準確性、30 天可行性或五站正式成果證據。
