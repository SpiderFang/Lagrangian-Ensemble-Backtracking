# WP-51-V 獨立驗證報告（per_member 逐成員沉底年齡＋BayTrace 擴散對齊）

驗證對象：分支 `wp51-per-member`（e8f30a7..2f3d871，4 個提交：6c2fa36 A、df6160c A.5、b4ee5cc D、2f3d871 B（一））。
驗證者：WP-51 子任務 V（只審查、測試、量測；未修改功能程式）。
環境：伺服器 .203（`~/Workspace/lbt-wp51/verify/`，沿用 `lbt-repo/.venv`；Python 3.12、numpy 2.5.2、numba 0.67.0）；
真實資料為 `lbt-inputs/formal/v3-newsettling/20260924_final-sites_h30-h60-h90_common_m10_v3` 與 `lbt-local-data`、`run04-h90-m10-v3`（皆唯讀）。
驗證腳本都在本 worktree 的 `scripts/wp51_verify/`（可重跑）。

## 0. 結論（先看這裡）

1. **阻擋項（嚴重）**：`sample_bed_residence_member_ages_conditioned_on_availability` 在**真實 OCM 缺口**下對 M=10 **一定 fail-closed**。
   22 個種子（含設定檔種子 20260916）全部丟 `ValueError`（「缺口條件式抽樣無可用候選」）；M=5 為 0/20、M=2 為 2/20、M=1 為 15/20。
   原因：OCM 每年有約 15 段約 24 小時的整日缺口，而 per_member 的分層只有 4–5 個整點寬（舊的 per_observation 分層有 43 個整點寬，從不會整層無候選）。
   所以目前分支**不能實際建出正式 per_member 母體**；單元測試用的是無缺口合成資料，所以測不出來。修正建議見 §5 第 1 項（我用原型驗證可行）。
2. 其餘設計項（逐成員起點深度、ID／種子、per_observation 位元相同、BayTrace 擴散對齊、z 夾限、可重現性、Numba 速度）**皆確認無誤**（逐項見 §1）。
3. 全測試套件：4270 項，修正環境因素後 **4227 通過、4 失敗、39 略過**。4 項失敗中 2 項為本分支造成（架構地圖未收錄 `deposition_labels` 模組），2 項在 e8f30a7 同樣失敗（環境因素）。
4. 流場小時數：per_member 相對現行 +9%（H90，每區 42 段）；H30 +25%、H60 +13%。垂向擴散對 Numba 迴圈**沒有可量到的減速**（融合 kernel 0.98×，真實流場 1.00×）。

## 1. A：程式審查（逐項）

圖例：OK＝確認無誤；問題＝需要處理；備註＝不確定或需決策。行號為 `wp51-per-member` 分支（2f3d871）。

### 1.1 沉底年齡改逐成員（要求 1）

| 項目 | 結果 | 依據 |
|---|---|---|
| 抽樣層數：0..90 日切成 50M=500 層 | OK | `bed_residence.py:1324-1345`：`slot_count = sample_count*member_count`，`_stratum_bounds(sample_count=slot_count)`；分層寬 4–5 h（`total_hours=2161`）。並有 `slot_count > total_hours` 防呆（:1313）。 |
| 排列是否真的打亂 | OK（抽樣器本身） | `:1328` 以 PCG64DXSM `permutation(slot_count)`。無缺口、30 個種子：Spearman(年齡, 觀測 rank) 平均 −0.007、標準差 0.036、最大 |ρ|=0.074（500 點理論標準差 0.045），舊缺陷為 1.0。逐成員 50 點的 Spearman 也在隨機範圍。腳本 `a1_sampler_properties.py`。 |
| 產生器種子與 per_observation 獨立 | OK | `:1252-1264` 以角色標籤＋種子 SHA-256 衍生，不與舊方法共用起點。 |
| 缺口拒絕是否針對配對後的實際觀測 | OK | `:1346` `rank_observation = times[rank]`，對**該槽位 rank 的各站觀測**逐站檢查 `obs - age*3600e9 ∈ available`（:1351-1355）。實測：人為拿掉 nanwan 在 rank 7 某槽位的沉底整點，稽核只在該槽位出現 `rejected=(848,) by_site=(('nanwan',),)`，新年齡 849；其餘 499 個槽位年齡不變（每槽位固定消耗一次層內 permutation，拒絕不會位移後續亂數流）。 |
| 五站是否同槽位同年齡 | OK | 單一年齡矩陣 `[rank][member]`，五站共用（`:1272-`；`input_derivation.py:7075-7102` 同一個 `sampling` 套到各站；`manifests.py` 驗證器 2124 起逐站核對 `age == matrix[rank][member]`）。 |
| 缺口條件下是否仍可成功建立母體 | **問題（阻擋）** | 見 §3。 |
| 層內無候選時 fail closed、不外插 | OK（行為正確，但導致上一項） | `:1363-1367`。 |
| 輸入列順序不影響結果 | OK | 實測：五站觀測時刻反序輸入，年齡矩陣與排列完全相同。`apply_*` 以 `(time_utc_ns, arrival_time_id)` 排序，與抽樣器 `_sorted_observation_rank_times` 一致，且後者拒絕同站重複 UTC。 |
| 唯一性與範圍自檢 | OK | `:1381-1385`（500 個年齡互異且在 0..2160）。 |

### 1.2 粒子身分表與逐成員起點深度（要求 2）

| 項目 | 結果 | 依據 |
|---|---|---|
| `arrival_time_utc_ns` 逐成員解析 | OK | `production_wiring.py:976-979` `resolve_arrival_time_utc_ns` 取 `(觀測錨點 ID, member_id)` 對應的**沉底 arrival** 的 `time_utc_ns`；找不到即 `WiringError`，不以觀測時刻頂替。`particle_input.py` 的 resolver 要求整數。排程單元（`plan_native_work_units`）依身分表的逐粒子到達時刻分組（`orchestrator.py:343-348`），所以逐成員時刻會正確進入 `hour_major` 的釋放時序。 |
| 起點深度＝受體深度比例×該成員沉底時刻的水柱 | OK | `production_wiring.py:956-965`：初始條件以「成員自己的沉底 arrival ID」為鍵；`input_derivation.py:5215-5560`（`_dynamic_initial_payload`）對每個 arrival 以 `arrival.time_utc_ns`（=沉底 UTC）查 OCM `zcor/elev`，並以同一組隨機垂向 draw 的比例重建 `z`。深度比例不隨成員變。 |
| 真實資料乾 face 風險 | OK | 以真實受體 face（每站 5 個）× 原型沉底時刻 2500 個查 OCM `wetdry_elem`：0 個乾 face 配對（現行 per_observation 也是 0）。`e_dry_face.py`。 |
| 種子／ID 不受影響（per_observation） | OK | 以真實 v3 設定重建 M=10 身分表（500,000 粒子）：`content_sha256` = `07cfbe7c…bff7ff`，與 run04 已存 `identity_table.npz` **完全相同**（e8f30a7 也能跑出同值；`a3_identity_bitwise.py`）。 |
| 種子／ID（per_member） | 備註 | per_member 的情境 ID 由**觀測錨點**建立（`manifests.py` `load_scenario_inputs`），粒子種子由情境＋成員決定、與沉底時刻無關，這是正確的；但因此與 run04（情境 ID 取自 per_observation 的沉底 arrival ID）的粒子種子**不同**，不能與 run04 逐粒子對照。 |
| `labels` 欄位只在有標籤時進雜湊 | OK | `particle_input.py` `_compute_identity_table_hash`；上列 sha 相同即為證明。 |

### 1.3 per_observation 與 e8f30a7 位元相同（要求 3）

| 項目 | 結果 | 依據 |
|---|---|---|
| 抽樣與 manifest | OK | `input_derivation.py` 的 per_observation 區塊僅被縮排進 `if scope != per_member`（內容未變）；`bed_residence.py` 的 per_observation 函式只新增選填稽核欄位且不輸出（`as_metadata` 以 None 守門）。v3 的 `arrival.json`、`initial_conditions.json` 經分支的 `load_scenario_inputs(formal=True)` 驗證通過（身分表可建）。 |
| 身分表 | OK | 見 §1.2，位元相同。 |
| 數值核心（函式庫預設） | OK | `HourMajorConfig` 預設 `gaussian/off`，`mirror_rk4_substep_loop_numba` 預設亦同；測試 `test_hour_major_library_default_is_legacy_scheme`、`test_vertical_off_leaves_z_untouched_and_default_is_legacy` 通過。注意：**正式 native-run 的預設已改為 uniform/constant（要求 5），所以數值結果相對 run04 本來就不同**。 |
| 舊 per_particle／固定日曆窗路徑 | OK（fail closed） | `input_derivation.py:6955-6970`、`manifests.py` 對 per_member 要求 `full_horizon_from_deposition`＋條件式抽樣。 |

### 1.4 BayTrace 對齊（要求 4）

BayTrace 參考碼核對結果（`baytrace/batch_interp.py`、`physics.py`、`tracker.py`、`main.py`）：

- passive 模式（`mod_part=0`）的 `batch_move_passive`（1086–1119）**只有平流，完全沒有隨機漫步**；隨機漫步只存在於 oil 模式 `batch_move_oil`（1397–1549），而 `physics.py:62-63` **明文禁止 oil 模式逆向**（`mod_part=1` 必須 `ibf=1`）。所以「BayTrace 被動模式逆向的 imixing／Visser 預設」嚴格說沒有對應參考行為；能對齊的只有 oil 模式的公式。
- `imixing` 預設 0（`tracker.py:89`、`main.py` `--mixing` 預設 0），0＝簡單隨機漫步、1＝Visser (1997)。我們的預設 `visser_correction=False` 與之**一致**（`hour_major.py:~336`、`hour_major_provider.py` `LBT_NATIVE_VISSER` 預設 0）。
- BayTrace 的垂向 Kv 預設 `ikvmodel=0`（SCHISM `diffusivity` 場，且只在 `mod_part=1` 時讀入）；我們 OCM 無此場，故用 `ikvmodel=3`（使用者常數，`kv_const` 預設 3.0e-4）——與要求 4 的「Kv 本次用常數 3.0e-4」一致（`substep_numba.py:576`，`batch_interpolate_diffusivity(..., ikvmodel, 1.2e-5, kz_const, 30.0)` 的引數順序對應其簽章 `ikvmodel, kv_background, kv_const, mld`）。

| 項目 | 結果 | 依據 |
|---|---|---|
| 水平與垂向皆為均勻亂數 `rnd=2u−1∈(−1,1)`、位移 `rnd*sqrt(6*K*dtb)` | OK | `substep_numba.py:254-295`（`baytrace_uniform_diffusion_displacement`）逐式對應 `batch_interp.py:1510-1532`。`u` 取自 `(bits>>11 + 1)/(2^53+1)`，嚴格落在 (0,1)。 |
| 擴散係數時間內插 `trat/2` | OK（未改） | kernel 仍呼叫 BayTrace 的 `batch_interpolate_diffusivity`，`trat` 的定義未動（`substep_numba.py:580`）。 |
| 亂數逐粒子可重現、與批次組成無關 | OK | 三個均勻亂數來自 `(種子, 計數器)`（`:229-251`），每顆作用中粒子每子步消耗一個計數器（`:615-622`）；前兩個與舊 Box-Muller 的 `u1,u2` 位元相同，第三個用獨立鹽值。測試 `test_uniform_triple_*`、`test_hour_major_vertical_diffusion_batch_composition_invariance` 通過。 |
| 垂向位移後仍做 z 夾限 | OK | 順序：擴散位移 → `batch_check_boundaries` → `batch_clamp_z`（`substep_numba.py:~640-670`），與 BayTrace `tracker.py` 同序（移動後 `_batch_clamp_z`）。夾限語意是 BayTrace 的「碰底改放到第二層 `zlev[kbpl+1]`、碰面改放到表層」，不是反射。實測：1 小時後 z 標準差 1.470 m = `sqrt(2*3e-4*3600)`=1.470，範圍落在水柱內。 |
| 逆向時符號 | OK（常數 Kv）／備註（ocm_field） | 隨機項對稱，與 `ibf` 無關。梯度漂移 `+dKv/dz*dtb`（`substep_numba.py:293`）不隨 `ibf` 翻號；常數 Kv 時為 0，沒有影響。BayTrace 的 oil 模式從未逆向執行，沒有對照；就均勻密度的時間反轉擴散而言，不翻號是合理的，但 `ocm_field` 尚未接線（`hour_major.py:356-360` 明確 fail closed），接線時請再獨立檢核（我不確定）。 |
| 水平 Kh：BayTrace 另含水平梯度漂移 `grdx/grdy` | 備註 | `batch_move_oil` 的平流含 `grad_x/grad_y`（Kh 的水平梯度）；LBT 的舊 gaussian 路徑與新 uniform 路徑都沒有水平梯度漂移（舊行為，非本次引入）。若要「完全對齊」才需處理。 |

### 1.5 正式 native-run 實際採用 uniform＋constant、速度（要求 5）

| 項目 | 結果 | 依據 |
|---|---|---|
| 預設是否生效 | OK | `hour_major_provider.py:522-531`：`LBT_NATIVE_DIFFUSION_NOISE` 預設 `uniform`、`LBT_NATIVE_VERTICAL_DIFFUSION` 預設 `constant`、`LBT_NATIVE_KZ_CONSTANT_M2PS` 預設 3.0e-4；`_resolve_params` 結尾 `validate_diffusion_scheme`（:538）使 `ocm_field` 等無效組合在啟動時就失敗。實測 `_resolve_params(os.environ)`（不設任何擴散環境變數）得到 `uniform constant 0.0003 False`。每個排程單元結果摘要都寫入實際組態（`_diffusion_scheme_provenance`）。 |
| Numba 融合 kernel 速度 | OK | 20,000 粒子×150 子步，每次 = 1 流場小時，5 次中位數（`b_kernel_bench.py`）：e8f30a7 legacy 14.91 s；分支 legacy 15.08 s（+1%）、uniform/off 14.77 s（0.98×）、uniform/constant 14.77 s（0.98×）、+Visser 14.59 s（0.97×）。無減速。 |
| 真實流場（lienchiang、真實身分表與 run04 計畫）每流場小時耗時 | OK | 同一批 4,000 粒子、單一到達時刻，單核，交替量測 3 次（`d2_bench_diffusion.py ... alt`）：見 §4.3，比值 1.001（見 §4.4）。每小時約 13 秒是底場合成固定開銷，粒子推進不是瓶頸。 |

### 1.6 其他發現（依嚴重度，詳見 §5）

- 檢查點（checkpoint）與續跑不綁擴散組態（`hour_major.py` `save_hour_major_checkpoint` meta 沒有 scheme），而正式預設從 gaussian/off 改成 uniform/constant：用新程式續跑舊 checkpoint 會**靜默混用兩種方案**。
- 正式設定檔 `physics.vertical_diffusion.constant_kz_m2ps: 0.00135262…` 與實際採用 3.0e-4 不一致（`hour_major_provider` 不讀該欄位；b4ee5cc 的「設定檔數值等於實際行為」沒有涵蓋這一項，實作者已在註解中揭露）。
- `deposition_labels.py`（2f3d871）只是模組＋單元測試，**尚未接進 input-build**：`ArrivalTime.tide_class/phase_or_event` 與身分表的 `observation_*` 仍是觀測時刻標籤；身分表只有 `deposition_season` 一項沉底標籤（`production_wiring.py:996-1020`，其餘 `deposition_*` 因 metadata 沒有而整組省略）。若下游用大小潮／潮相分組，仍是舊的錯位標籤，直到 B（二）接線。
- 架構地圖測試未納入新模組（見 §2）。
- `output_interval_seconds` 300→3600：我推演 `hour_major.py:1352-1358` 在整點對齊下兩者皆每小時取樣一次、結果相同（未實測）。
- per_member 下 `cli.py` 的 `arrivals_by_time_utc_ns = {time: record}`（:1171、:2198）以 UTC 為鍵，同站不同成員、或兩站可能同一沉底整點而覆蓋；目前只在 `fixed_calendar_window`（per_member 不支援）使用，所以無影響，但若日後想由 per_member 結果導出 fixed_calendar_window（`native-derive-horizon`）會得到錯誤的年齡。

## 2. B：伺服器完整測試套件

做法：分支 `git archive` 解壓到 `~/Workspace/lbt-wp51/verify/wp51`，e8f30a7 同樣解壓到 `.../base`；8 個程序按檔案輪流分片並行（`runtests.sh`）。
第一輪發現環境問題：`ulimit -n`=1024（`mmap` 開檔過多，造成 15+ 項 `Too many open files`）、解壓目錄不是 git repo（`git rev-parse HEAD` 失敗）。我在兩份目錄補 `git init` 並以 `ulimit -n 65536` **只重跑失敗的 28 項**（分支與 e8f30a7 各跑一次）。

| | 項數 | 說明 |
|---|---|---|
| 測試總數 | 4270 | 153 個測試檔 |
| 第一輪通過 | 4203 | |
| 第一輪失敗／錯誤 | 24／4 | 全為上述環境因素＋下列 |
| 略過 | 39 | 與資料／來源可得性有關（例如「未找到指定 PI 原版來源」） |
| 重跑 28 項（分支） | 24 通過、4 失敗 | |
| **最終** | **4227 通過、4 失敗、39 略過** | |

4 項失敗：

1. `tests/test_native_postprocess.py::test_manifest_contains_provenance_and_matches_output_checksums`：`project_root 必須是非 symlink 的目錄`，**e8f30a7 同樣失敗**，與解壓目錄的路徑環境有關，非本分支問題（在正式 repo 路徑下應可通過，我沒有驗證）。
2. `tests/test_source_code_architecture_map.py::test_committed_html_is_source_generated`：**e8f30a7 同樣失敗**（我的快照 repo 沒有原 git 歷史，疑與此有關，未深查）。分支上此項可能同時疊加第 3、4 項的原因。
3. `...::test_module_catalog_covers_package_exactly`：**本分支造成**。`AssertionError: Extra items in the right set: 'deposition_labels'`，新模組沒有登錄到架構地圖。
4. `...::test_catalog_groups_and_flow_edges_are_closed`：**本分支造成**，同上。

其餘（含 `test_bed_residence_per_member`、`test_per_member_pipeline`、`test_deposition_labels`、`test_baytrace_diffusion_alignment`、`test_horizon_suite`、`test_input_derivation`、`test_native_hour_major*`）全數通過。
但請注意：**per_member 的測試用的是無缺口合成資料，沒有任何測試涵蓋「真實缺口導致整層無候選」**（§3）。

## 3. C：真實資料的 per_member 抽樣

設定：5 站×50 個觀測（取自 v3 `arrival.json` 的 `observation_time_utc_ns`，觀測範圍 2025-01-10..2025-12-30）、M=10、最大年齡 90 日、種子 20260916（設定檔 `effective_sampling_seed`）、五站可用軸取 `forcing_inventory.json` 的 `ocm_native` canonical time（重建後點數與 `canonical_time_count=17124` 相符）。

### 3.1 分支抽樣器直接呼叫：失敗

`c_real_sampling.py` → `ValueError: bed residence per_member 缺口條件式抽樣無可用候選：slot=41, observation_rank=4, member_id=1, stratum=248, candidate_count=5`。

可行性分析（`c2_feasibility.py`、`seedfail.py`）：

- 每層候選 4–5 個整點（平均 4.32）。25,000 個（rank, 層）組合中有 **1,935 個（7.74%）整層無五站皆可用的候選**；每個 rank 有 9–96 層不可行，每層有 0–13 個 rank 不可行。
- 原因：四個區的 OCM 缺口結構相同，共 48 段、缺 419 h；其中 15 段是約 24 h（23/24/25/49 h）的整日缺口。5 站觀測時刻不同，所以任一站落入缺口就使整層無候選。
- 202 個種子（1..200、20260916、20261007）做完隨機排列後，整層無候選的槽位數：最小 18、中位 39、最大 52，**失敗數為 0 的種子 0 個**；設定檔種子 39 個、20261007 為 37 個。
- 分支抽樣器實際呼叫：22/22 個種子（1..20、20260916、20261007）皆丟 `ValueError`；M=1：15/20 成功，M=2：2/20，M=5：0/20。
- 存在全可行的完全配對：以匈牙利法對「槽位×層」可行性矩陣求最小代價配對，不可行配對數為 0。所以問題在「純隨機排列後只能失敗」，不是資料本身不允許。

### 3.2 用原型修補後的數字（非分支程式，只供評估）

原型 `c3_repair_prototype.py`：先照分支做隨機排列；對每個不可行槽位，依種子固定順序找另一槽位使兩邊交換後都可行（單次交換）；然後層內隨機順序取第一個五站皆可用的整點（與分支相同）。種子 20260916：

- 39 個不可行槽位，**39 次單次交換全部修補成功**（受影響槽位約 78/500）；層內拒絕 10 次。
- 年齡互異 500/500；Spearman(年齡, 觀測 rank)＝**+0.046**（舊缺陷＝1.0）；逐成員（50 點）Spearman 絕對值最大 0.283（10 個成員取最大，屬隨機範圍）。
- 年齡直方圖（每 10 日一格，共 9 格）：[55, 56, 56, 55, 55, 57, 55, 55, 56]，均勻。
- 沉底時刻月份分布（五站合計 2,500）：2024-10:5、2024-11:107、2024-12:128、2025-01:210、2025-02:218、2025-03:207、2025-04:264、2025-05:216、2025-06:202、2025-07:189、2025-08:253、2025-09:245、2025-10:177、2025-11:63、2025-12:16。
  **涵蓋 10–12 月（2024-10 起至 2025-12）**；10–12 月（任一年）共 496/2500＝19.8%。各站沉底最早 2024-10（舊缺陷是只到 10 月初）。
- 各站沉底整點在其 OCM 可用軸內：不可用 0 筆。
- 無缺口時分支抽樣器本身：拒絕 0 次，Spearman 如 §1.1。

## 4. D：流場小時數與每流場小時耗時

### 4.1 分支內 `estimate_per_member_flow_hours.py`

腳本用合成觀測（2025-01-15..10-15 均勻、無缺口）與「舊配對以線性分層近似」。結果（K＝每區段數；摘錄 H90）：per_observation K=1 25,711 h、K=36 316,385 h；per_member K=1 42,209 h、K=36 342,593 h；即 K=1 時 +64%、K=36 時 +8%，且單元數從 2,470 增至 24,300。該腳本的 per_observation 年齡近似與觀測範圍和真實母體差很多（見 4.2），僅供方向性參考。另量得單元數 2,500→250 的「每組每小時 np.any 迴圈」成本 0.67 ms→5.6 ms、`_record_departures` 線性搜尋 28→271 µs/列（單元數 10 倍時線性放大，相對每小時約 7–13 秒的底場合成可忽略）。

### 4.2 真實母體估計（我的 `d_flow_hours.py`）

基線＝v3 真實 per_observation 到達時刻（`arrival.json` 的沉底時刻，2,460 個工作單元＝與 run04 計畫 `unit_count: 2460` 一致；每區 K=42 段＝run04 `seg0of42`）；per_member＝真實觀測＋§3.2 原型年齡（24,350 單元、1,773 相異到達時刻、跨距 425 日，基線 194 個相異到達時刻、跨距 268 日）。呼叫分支的 `plan_native_hour_major_segments`：

| H | per_observation 總流場小時 | per_member 總流場小時 | 比值 | 最長單段 |
|---|---|---|---|---|
| 30 | 125,989 | 158,085 | 1.25× | 1,567 h → 1,523 h |
| 60 | 246,949 | 279,045 | 1.13× | 2,287 h → 2,243 h |
| **90** | 367,909 | 400,005 | **1.09×** | 3,007 h → 2,963 h |

每區：H90 約 91.5–92.5k → 99.6–100.3k h。最長單段不變，所以 42 個 worker 的長尾也不變。粒子總數仍 500,000；單元數約 10 倍（2,460→24,350），每單元約 20 粒子。

### 4.3 每流場小時耗時（真實 lienchiang 流場、run04 計畫與身分表，單核）

`d2_bench_diffusion.py hourmajor::lienchiang_common_cache_v3__seg10of 12 10`（4,000 粒子、單一到達時刻；per_member 樣式＝同樣粒子列拆成 100 個 20 粒子單元、到達時刻散在 0..23 小時）：

| 樣式 | gaussian/off | uniform/off | uniform/constant |
|---|---|---|---|
| per_observation 樣式（走過 13 h）s/h | 12.44 | 13.05 | 13.99 |
| per_member 樣式（走過 35 h）s/h | 6.00 | 5.53 | 5.46 |

第一輪 per_observation 樣式差 +12% 與 per_member 樣式 −9% 方向相反，因為當時我同時在跑另一個量測（單機有其他工作），屬負載雜訊。改以交替量測重做（`... 6 10 alt`）：見 §4.4，比值 1.001。
兩種樣式不能直接比 s/h（per_member 樣式因粒子逐步釋放，前期作用中粒子少且走過小時數多 22 h）；總流場小時數才是主要成本，見 §4.2。底場合成是每小時約 7 秒的固定開銷（暖機 3 小時 21–27 s 中，JIT 只佔一次），粒子推進不是主因。

### 4.4 交替量測結果

同一批 4,000 粒子、單一到達時刻、6 小時、單核、gaussian/off 與 uniform/constant 交替各 3 次（暖機後）：gaussian/off 12.813／12.875／12.839 s/h（中位 12.839）；uniform/constant 12.983／12.849／12.747 s/h（中位 12.849）；**比值 1.001**。因此真實流場下新預設（uniform＋constant 垂向擴散）與舊方案每流場小時耗時無差別，第一輪的 +12% 是雜訊。

## 5. 建議修正清單（依嚴重度）

1. **【阻擋】per_member 抽樣器在真實缺口下必定失敗（§3）。** 在隨機排列之後、逐槽位挑年齡之前，加入「可行性修補」：預先算出 `feasible(rank, stratum)`（該層是否存在使五站皆可用的整點），對不可行槽位以種子固定的順序與其他槽位交換層（兩邊交換後都可行），仍保持「500 層各用一次」；必要時退回匈牙利／增廣路徑匹配。稽核要記錄交換（新增欄位並把 method id 升版，例如 `..._v2`），`manifests.py` 驗證器目前只要求「年齡在 `slot_permutation[slot]` 的分層內」，不需改動核心檢查，但要能驗證修補後排列仍是 0..499 的排列。補測試時請用**含整日缺口**的合成資料（例如每站 24 h 缺口 15 段、層寬 4–5 h），並以一個「純隨機排列必失敗、修補後成功」的案例當回歸測試。我的原型證明單次交換即足夠（39/39），且 Spearman＝+0.046、年齡均勻、涵蓋 2024-10..2025-12。
2. **【高】架構地圖未收錄 `deposition_labels`（2 項本分支造成的測試失敗，§2）。** 補 `docs`／`source_code_architecture_map` 的模組目錄並重新產生 committed HTML。
3. **【中】擴散組態沒有綁進 checkpoint／計畫。** `save_hour_major_checkpoint` 的 meta 與 native-plan 都不記錄 `diffusion_noise`／`vertical_diffusion`／`kz`／`visser`；正式預設已從 gaussian/off 翻成 uniform/constant，續跑舊 checkpoint 會靜默混用。建議把這四項寫進 checkpoint meta 並在載入時比對，不一致就拒絕續跑；同時寫進計畫檔的 `physics_case`／`extra` 供事後核對。
4. **【中】YAML 與實際行為仍有不一致：** `physics.vertical_diffusion.constant_kz_m2ps: 0.00135262…` 對 3.0e-4。要嘛把正式 YAML 改成 3.0e-4（並註明來源為 BayTrace `kv_const` 預設），要嘛讓 provider 讀取該欄位；否則論文／報告引用的 YAML 數值與實際執行不一致。
5. **【中】B（二）標籤接線未完成：** `deposition_labels` 沒有被 input-build 呼叫，沉底時刻的大小潮／潮相／事件旗標在正式路徑不存在，頂層 `tide_class`／`phase_or_event` 仍是觀測標籤。在接線前，不要用這兩欄對 per_member 結果分組。
6. **【低】`cli.py:1171／2198` 的 `{time_utc_ns: record}` 在 per_member 會因同一沉底整點而覆蓋**，目前只用在 `fixed_calendar_window`（per_member 不支援），建議加 per_member 守門（`native-derive-horizon` 導出 fixed_calendar_window 時 fail closed）。
7. **【低／備註】** 水平梯度漂移（BayTrace `grdx/grdy`）未實作（舊行為）；`ocm_field` 的梯度漂移符號待接線時重新檢核；`output_interval_seconds` 300→3600 我只推演、未實測位元相同。

## 6. 我不確定或沒做的事

- 沒有實際跑 per_member 的完整 `input-build`／`native-run`（抽樣器在真實資料下會失敗，且正式母體需要伺服器上的完整 OCM 產品；我只以真實 arrival／forcing_inventory 清單、真實 OCM wetdry 與身分表驗證）。
- 兩個在 e8f30a7 同樣失敗的測試，我歸因於解壓目錄環境（非 symlink 檢查、沒有原 git 歷史），沒有在原 repo 路徑重跑。
- §4.3 的耗時只在單核、機器上有其他工作的情況下量到；結論用的是融合 kernel 微基準（可重現）與交替量測，絕對秒數僅供參考。
- 預期的「M=10 正式母體 per_member」的總機時不只取決於流場小時數（見 §4.2）；底場合成與 worker 分攤的實際影響，建議等阻擋項修好、能建出真實 per_member 計畫後，用 `lbt native-plan` 的估計值與 `probe` 再確認。
