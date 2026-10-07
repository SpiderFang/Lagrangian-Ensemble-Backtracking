# WP-51 整合報告（per_member 可行性修補、擴散組態綁定、龜山島新錨點、v4 部署準備）

整合分支：`wp51-per-member`（worktree `work/wp51`），起點 2f3d871。三個待併分支（`wp51-post` B2、`wp51-verify` V、`wp51-guishan` C）已依序合併，
之後在其上新增 5 個整合提交。未 push、未動 main、未啟動任何正式運算、未建正式母體。

## 0. 結論（先看這裡）

| 任務 | 狀態 | 要點 |
|---|---|---|
| 1 合併三分支 | 完成 | 依序 B2 → V → C，無衝突（V 只新增報告與驗證腳本；B2 與 C 動到的檔案互不重疊）。 |
| 2 阻擋項：真實 OCM 整日缺口下抽樣器必定失敗 | **完成** | 排列後可行性修補（單次交換，後備增廣路徑匹配），method id 升版，稽核寫進 provenance 並由 manifest 載入器重算驗證。真實 v3 母體資料上 202 個種子、M=10 **202/202 成功**（舊版 0/22）。 |
| 3 架構地圖登錄兩個新模組 | 完成 | `test_source_code_architecture_map` 全數通過（HTML 在伺服器 Linux 環境產製）。 |
| 4 擴散組態綁進 checkpoint 與計畫檔 | 完成 | 續跑組態不同明確報錯，不再靜默混用。 |
| 5 正式 YAML Kz 設定＝實際 | 完成（需你確認數值） | 設定檔改為 3.0e-4，並成為 Kz 唯一來源。見 §10 待決 1。 |
| 6 龜山島錨點離陸地 ≥ 500 m | 完成 | 新錨點 (121.939148, 24.850326)；距陸地 578.8 m、距原陸地錨點 1,490 m。 |
| 7 伺服器乾淨 clone 全套測試 | 完成 | 4338 通過、1 失敗、39 略過；唯一失敗在 e8f30a7 同環境同樣失敗。 |
| 8 v4 範本與腳本（只準備） | 完成，未執行 | 範本已產生；`prep_v4.sh`／`run_formal_v4.sh` 語法與 `--check` 模式與樁測試驗證過。 |
| 9 本報告 | 本檔 | |

## 1. 合併內容

| 合併提交 | 分支 | 內容 |
|---|---|---|
| d5e07bf | `wp51-post`（B2） | `deposition_labels`／`deposition_label_arrivals`：沉底時刻的大小潮、潮相、大浪、強流標籤在 input-build 寫進 per_member arrival，manifest 載入器驗證；`native-postprocess` 依沉底時刻標籤輸出分組表（保留 `observation_*` 舊標籤作對照）；新增 `reports/WP-51-B2.md`。 |
| f078612 | `wp51-verify`（V） | 獨立驗證報告 `reports/WP-51-V.md` 與 `scripts/wp51_verify/`（可重跑的驗證腳本）。V 指出的阻擋項與修正清單即本次任務 2–5。 |
| d51b4e3 | `wp51-guishan`（C） | 龜山島錨點由陸地移到島西側海上（`anchor_policy`、`rank_anchor_snap_candidates`、`scripts/derive_guishan_anchor.py`、`tests/test_guishan_anchor.py`、`reports/WP-51-C.md`）。 |

整合提交：

| 提交 | 內容 |
|---|---|
| d6b7c72 | （一）per_member 抽樣器排列後可行性修補（阻擋項）＋真實缺口形狀回歸測試與 fixture |
| 48e69da | （二）架構地圖登錄 `deposition_labels`、`deposition_label_arrivals` 並重產 HTML |
| 506829f | （三）擴散組態綁進 checkpoint 與計畫檔；常數 Kz 以設定檔為唯一來源；正式 YAML Kz 改 3.0e-4 |
| 2e412ea | （四）龜山島錨點要求距校閱海岸陸地 ≥ 500 m，改為 (121.939148, 24.850326) |
| 0329dda | （五）v4 範本產生器與 v4 準備／執行腳本（只準備，不執行） |

## 2. 阻擋項：per_member 抽樣器的排列後可行性修補

### 2.1 問題

V 的結論：真實 OCM 每年約有 15 段約 24 小時的整日缺口，而 per_member 的分層只有 4–5 個整點寬（50 觀測 × M=10 = 500 個槽位，把 0..2160 小時切成 500 層）。
純隨機排列把某個觀測 rank 配到整層落在缺口內的分層時，該槽位「無可用候選」而 fail closed；M=10 時必定發生。單元測試用無缺口合成資料，所以沒測出來。

### 2.2 修法（`bed_residence.py`）

1. 初始排列不變：`PCG64DXSM` 隨機排列，把 500 個槽位配到 500 個分層（五站共用）。
2. 預先建立可行性矩陣 `feasible[rank][stratum]`：該層存在一個整點，使五站的沉底整點都落在各自 canonical available-time set 內。
3. 對每個不可行槽位，依固定順序與另一個槽位交換分層，且交換後兩邊都可行（單次交換）。交換順序用**獨立衍生**的產生器（角色標籤 `bed_residence_age_per_member_permutation_repair_v1`），不動原本的隨機流。
4. 找不到單次交換時，退回二分圖最大匹配的增廣路徑（BFS）；若仍無法完全配對才丟例外（訊息含「無法完全配對」，fail closed，不補值）。
5. 無需修補時結果與升版前完全相同（修補產生器不被消耗）。
6. method id 升版：`numpy_pcg64dxsm_per_member_stratified_shared_permutation_feasibility_repair_v1`；修補演算法 id `single_swap_then_augmenting_path_matching_v1`。

### 2.3 稽核與驗證

- 每個槽位的稽核新增 `initial_stratum_index`、`initially_infeasible`、`repair_steps`（`[{step, kind: swap|matching_path}]`）；root provenance 新增 `permutation_repair` 摘要
  （`algorithm_id`、`initial_slot_permutation_sha256`、`initially_infeasible_slot_count`、`changed_slot_count`、`repair_step_count`、`swap_step_count`、`matching_step_count`）。
- `manifests.py` 載入時**由種子重算修補前排列**，核對每個槽位的 `initial_stratum_index`；沒有 `repair_steps` 的槽位分層不得改變；修補前不可行的槽位必須被改動；`permutation_repair` 摘要必須等於重算結果。竄改任何一項都會被擋（測試涵蓋）。
- 新增 `tests/fixtures/wp51_real_ocm_gap_shape.json`（約 8 KB）：真實 OCM 四區共同缺口形狀（48 段、缺 419 h，其中 15 段約 24 h）與 v3 母體五站各 50 個觀測時刻；重建軸點數須為 17,124。
- 回歸測試（`tests/test_bed_residence_per_member.py` 24 項、`tests/test_per_member_pipeline.py` 25 項）：種子 20260916、1、2、3、7、20261007；M=1/2/5/10；Spearman 近 0；沉底月份涵蓋 10–12 月；增廣路徑案例；無解 fail closed；修補摘要；pipeline 以「每三天缺一整天」的 fixture 做端到端與四種竄改。
  舊版在同樣形狀上全數失敗，新版全數成功。

### 2.4 真實資料抽樣數字（重跑 V 的檢查）

資料：v3 母體 `20260924_final-sites_h30-h60-h90_common_m10_v3` 的 arrival／forcing_inventory（唯讀），腳本 `scripts/wp51_verify/c6_repair_real_seeds.py`。

| 項目 | M=10 | M=5 | M=2 | M=1 |
|---|---|---|---|---|
| 成功率（種子數） | **202/202**（1..200、20260916、20261007） | 22/22 | 22/22 | 22/22 |
| V 的原結果（舊版） | 0/22 | 0/20 | 2/20 | 15/20 |
| 修補前不可行槽位 | min 18／中位 39／max 52 | 中位 17 | — | — |
| 單次交換步數 | min 17／中位 37／max 51 | — | — | — |
| 增廣路徑步數 | **全為 0**（單次交換已足夠） | — | — | — |
| 被動到的槽位 | min 34／中位 72／max 99（共 500） | — | — | — |
| 層內拒絕次數 | min 1／中位 11／max 23 | — | — | — |
| Spearman(年齡, 觀測 rank) | 平均 +0.010、標準差 0.045、最大絕對值 0.117 | 標準差 0.063 | 標準差 0.093、最大 0.200 | 標準差 0.156、最大 0.338 |
| 10–12 月沉底比例 | 平均 0.209、最小 0.173 | — | — | — |
| 10、11、12 月皆有沉底 | 202/202 | — | — | — |

**設定檔種子 20260916、M=10（正式會用的那一組）**：

- 拒絕總次數 8、有拒絕的槽位 5；年齡直方圖（9 組，每組 240 h）[55, 56, 56, 55, 55, 57, 55, 55, 56]（近乎完全均勻）。
- 沉底整點不在可用軸上：0 筆；500 個槽位的年齡互異：500。
- 五站合計 2,500 個沉底時刻的月份分布：2024-10:5、11:85、12:140、2025-01:211、02:221、03:191、04:264、05:233、06:213、07:192、08:265、09:253、10:163、11:53、12:11；10–12 月 457/2,500 = 0.183。
- 逐成員 Spearman 絕對值最大 0.346（單一成員只有 50 個點，屬隨機範圍）。
- **需要如實說明**：此種子的整體 Spearman(年齡, 觀測 rank) = **+0.117**，是 202 個種子中最大的（約 2.6σ），修補前的初始排列是 +0.049，修補後變 +0.117。
  這仍在隨機範圍內（202 個種子取最大，期望的最大值本來就會落在 2.5–3σ），V 原型用不同亂數得 +0.046；若要避開，可換 `sampling_seed`（會改 config hash），但我不建議為了這個數字動種子。見 §10 待決 2。

## 3. 架構地圖

`scripts/render_source_code_architecture_map.py`：`deposition_labels`、`deposition_label_arrivals` 列入 data 群組並補 `MODULE_INFO`；`bed_residence` 補 per_member 入口與修補說明；`receptors` 補 `rank_anchor_snap_candidates`；
`FLOW_EDGES` 新增 `deposition_labels → deposition_label_arrivals → input_derivation／manifests`。HTML 是在伺服器（Linux）環境以 renderer 產製後放回；e8f30a7 提交的 HTML 與當時 renderer 的輸出本來就不一致（見 §7 基準對照，原因未深查），若在別的環境重產請重跑此測試。
`tests/test_source_code_architecture_map.py` 6 項全過（含「committed HTML 與 renderer 輸出位元相同」）。

## 4. 擴散組態綁定（V 的建議 3）

新增 `native_runner/physics_case.resolve_diffusion_scheme`，唯一解析 `diffusion_noise`（uniform／gaussian，預設 uniform）、`vertical_diffusion`（constant／off／ocm_field，預設 constant）、
常數 Kz、`visser_correction`（預設關），plan／run／checkpoint 三處共用。

| 位置 | 行為 |
|---|---|
| `native-plan` | 讀設定檔 Kz，把解析結果寫進計畫檔 `physics_case.diffusion_scheme`。 |
| `native-run` 啟動前 | 以 `LBT_NATIVE_CONFIG` 與環境變數解析，與計畫檔宣告比對；不一致丟 `PhysicsCaseError`（可由 `LBT_NATIVE_ACKNOWLEDGE_CONFIG_MODE_MISMATCH` 明示放行，與既有的回溯模式不符放行同一機制）。 |
| checkpoint | meta 寫入 `diffusion_scheme` 簽章；續跑時比對，不同就拒絕續跑（「拒絕續跑」訊息）。舊 checkpoint 沒有簽章時視為 gaussian／off／None／False（升級前的實際行為），因此用新預設續跑舊 checkpoint 會被擋下，不會靜默混用。 |

`hour_major_provider` 改用同一個解析函式。測試：`tests/test_diffusion_scheme_binding.py`（新）、`tests/test_native_hour_major_provider.py::test_native_plan_records_diffusion_scheme_and_run_rejects_mismatch`（新）。

## 5. Kz：設定＝實際（V 的建議 4）

- 規則：設定檔 `physics.vertical_diffusion.constant_kz_m2ps` > 環境變數 `LBT_NATIVE_KZ_CONSTANT_M2PS`（僅在設定檔沒有值時）> 預設 3.0e-4（BayTrace `kv_const`）。設定檔有值且環境變數不同 → fail closed。
- `configs/lagrangian_backtracking.formal_h30_h60_h90_m10.yaml`：`constant_kz_m2ps: 0.0003`（附說明註解）；`tests/test_horizon_suite.py` 期望值、`docs/operations/14_*` 表格同步（Kz、output_interval 3600）。
- 先前 YAML 的 0.0013526236792521886 是 Hsinchu 24 小時 pilot 的候選值，**從未被 native-run 採用**（v3 實際跑的是 3.0e-4）。所以這是把文件改成與實際相同，不改物理。
- 注意：伺服器上唯讀的 v3 範本仍寫 0.00135；若直接拿 v3 範本建母體，新程式會實際採用 0.00135（與 v3 成果的物理不同），所以 v4 範本必須改成 0.0003（§8 已做）。

## 6. 龜山島錨點：離陸地至少 500 m

`scripts/derive_guishan_anchor.py` 新增 `--min-land-distance-m`（預設 500，大於 0 必須提供 `--coastline`）與 `--max-examine`（預設 3000）。
伺服器真實 OCM 網格重推導（`--coastline ~/Workspace/lbt-inputs/coastline-taiwan.geojson --check-surface --min-land-distance-m 500`，候選 2,967 個）：

| 排名 | 距原錨點 | 全域面 | 距陸地 | 判定 |
|---|---|---|---|---|
| 0（前一版錨點） | 1,288 m | 486610 | 146 m | 排除（< 500 m） |
| 1–9 | 1,296–1,482 m | — | 84–303 m | 全部排除 |
| **10（採用）** | **1,490.5 m** | **493182（local 114356）** | **578.8 m** | 通過 |

新錨點：(121.939148, 24.850326)（面中心高精度值 121.93914794921875, 24.85032590230306）；距候選區邊界 86.1 m（需求 ≥ 60.3 m）；NWW 逐時四角支撐與 OCM surface 在 50 個到達時刻全部有效。
同步更新：`config.py` 常數與註解、正式與範例 YAML（規則第 6 條「距陸地 ≥ 500 m」、provenance 新增 `min_land_distance_m`、`distance_to_land_m`、`distance_to_candidate_region_boundary_m`、`land_distance_reference`）、`tests/test_guishan_anchor.py`（新增 provenance 與腳本參數測試）、README、`docs/foundation/08`。
安全距離不變：核心（12.5 km）外緣離 A 區 flow domain 邊界仍 > 14 km、local 圓外緣 > 7 km（測試斷言重新驗證）。

**未做**：沒有用新錨點在真實資料上重建完整 common-input（C 當時的重建要 48 分鐘，且屬「建母體」）。錨點本身已通過 persistent-wet、NWW、OCM surface 與離岸檢核，但龜山島 5 個隨機水平受體、random vertical 與 face support 要到 `prep_v4.sh` 第一步才第一次在真實資料上跑；見 §10 待決 3。

## 7. 測試結果（伺服器乾淨 git clone）

- 程式碼：`~/Workspace/lbt-repo-wp51-int`（`~/Workspace/lbt-repo-wp51` 已被他人佔用，故用此名）。由 `git bundle` 傳到伺服器後 `git clone -b wp51-per-member`，可追溯：HEAD 即本分支 HEAD，`git status` 乾淨。
- 環境：`~/Workspace/lbt-repo/.venv`（Python 3.12）＋`PYTHONPATH=<clone>/src`、`ulimit -n 65536`（預設 1024 會造成 `Too many open files`）、`LBT_BAYTRACE_ROOT=~/Workspace/lbt-runs/baytrace-v4.7.2`、私有 `--basetemp` 與 `NUMBA_CACHE_DIR`；8 個程序依檔案輪流分片（`~/Workspace/lbt-wp51/runtests_int.sh`，記錄在 `~/Workspace/lbt-wp51/testruns/final`）。
- **基準對照**：同一份歷史的 `e8f30a7`（本分支起點的祖先）另 clone 到 `~/Workspace/lbt-wp51/base-e8f30a7`，同環境、同方式跑全套（`.../testruns/base`）。

| | 通過 | 失敗 | 略過 |
|---|---|---|---|
| 整合分支（0329dda 的程式碼） | **4338** | **1** | 39 |
| 基準 e8f30a7 | 4170 | 2 | 39 |

- 整合分支唯一失敗：`tests/test_native_postprocess.py::test_manifest_contains_provenance_and_matches_output_checksums`（`project_root 必須是非 symlink 的目錄`：測試寫死 `project_root="/home/claude/lbt"`，伺服器上不存在）。**基準 e8f30a7 在同環境同樣失敗**，與本整合無關。
- 基準的第二個失敗 `test_source_code_architecture_map.py::test_committed_html_is_source_generated`：e8f30a7 提交的 HTML 與當時 renderer 輸出在 native_runner 面板文字上就不同（既有的陳舊），本次重產 HTML 後整合分支通過。
- 39 個略過兩邊相同（例如「未找到指定 PI 原版來源」「未找到指定 BayTrace root」：測試寫死 Mac 路徑）。
- 新增測試約 +168 項（修補、擴散綁定、龜山島、範本產生器、B2、C、V 帶入的測試）。
- 非 git 目錄會造成的假失敗（文件索引連結測試、provider 的 `git_commit None`）在真 git clone 中不再出現。
- 驗證 v4 腳本：`tests/test_make_formal_template_v4.py`（2 項）；兩支部署腳本以 `bash -n`、`--check` 與「假 `lbt` 樁」的整段流程測試（含失敗即停）驗證，測試用暫存目錄已刪除。

## 8. v4 部署準備（只準備，未執行）

### 8.1 範本

`~/Workspace/lbt-inputs/formal/template-v4-per-member.yaml`，由 repo 內 `scripts/make_formal_template_v4.py` 自伺服器 v3 範本（唯讀，未改動）產生，產生器會斷言差異路徑只在預期集合內，並用 `load_config` 載入驗證。
與 v3 範本的**全部差異**：

| 位置 | v3 | v4 | 原因 |
|---|---|---|---|
| `scenarios.bed_residence_time.age_sampling_scope` | 無（per_observation） | `per_member` | 本期目的 |
| 龜山島 `anchor_lonlat` | [121.951606, 24.843127]（陸地） | [121.939148, 24.850326] | 新錨點；另加 `anchor_policy` 與完整 `anchor_snap_provenance` |
| `physics.vertical_diffusion.constant_kz_m2ps` | 0.0013526236792521886 | 0.0003 | 設定＝實際（見 §5）；與 v3 實際物理相同 |
| `integration.output_interval_seconds` | 300 | 3600 | 設定＝實際（小時主軸每流場小時取樣一次；native 不讀、不改結果） |
| `integration.maximum_horizontal_cfl_fraction` | 0.25 | 0.5 | 設定＝實際（native 的 `cfl_number` 預設 0.5；native 不讀本欄位） |
| `integration.maximum_vertical_layer_fraction`、`maximum_diffusive_rms_fraction` | 0.25 | null | native 沒有對應機制，避免誤導 |

不變：M=10、arrival policy v2（25 小時潮汐強度時窗）、十檔沉降速度、受體與邊界設定、其餘全部欄位。

### 8.2 腳本

位置：伺服器 `~/Workspace/lbt-runs/tools/{prep_v4.sh, run_formal_v4.sh, v4_env.sh}`；repo 內同內容副本 `scripts/wp51_deploy/`。兩支都有 `--check`（只檢查、不建任何檔案）。

- 共同（`v4_env.sh`）：程式碼用乾淨 clone `$CODE`（預設 `~/Workspace/lbt-repo-wp51-int`，可用 `LBT_CODE` 覆寫）＋ `lbt-repo` 的 venv，`PYTHONPATH=$CODE/src`；開始前確認 git clone、工作樹乾淨、含 `feasibility_repair`、import 路徑正確；
  **明確 export** `LBT_NATIVE_DIFFUSION_NOISE=uniform`、`LBT_NATIVE_VERTICAL_DIFFUSION=constant`、`LBT_NATIVE_VISSER=0` 並 `unset LBT_NATIVE_KZ_CONSTANT_M2PS`；用 `resolve_diffusion_scheme` 解析並**核對**為 uniform／constant／Kz=0.0003（來源 config）／Visser 關，不符就中止；
  把 git commit、設定檔與範本 SHA-256、解析後的擴散組態寫進 `$B/logs/*.provenance.txt`。
- `prep_v4.sh`：建 v4 母體 `~/Workspace/lbt-inputs/formal/v4-per-member/<日期>_final-sites_h30-h60-h90_common_m10_v4`（`horizon-suite-create`，H30/60/90，讀 NAS `/data`）→ `prep_m10_probe.py` 建 M=10 身分表 → 兩案例 `native-plan`（`hour-major-segmented`、`--target-worker-count 42`）；
  目的地已存在即中止（不覆蓋）；最後印出工作單元數與計畫檔擴散組態。
- `run_formal_v4.sh`：輸出 `~/Workspace/lbt-runs/run05-h90-m10-v4`；worker 預設 42（`LBT_WORKERS`）；`no_stokes` 再 `finite_depth_stokes`；每案 `native-run` → `native-postprocess`（`--with-boundary-segments --with-footprint --footprint-land-geojson --bathymetry-grid-root --region-flow-domain-map`）→ `native-figures` full 與 none 各一次（`--bathymetry` 等深線、**`--config`**：後處理含足跡產品時出圖必須帶設定檔，v3 時期的 WP-38 腳本沒有，已補）；
  任一步離開碼非 0 即停；後處理與出圖的輸出目錄已存在即中止；`native-run` 可續跑（擴散組態與 checkpoint 簽章不符會拒絕）。
- 啟動流程：`prep_v4.sh --check` → `prep_v4.sh` → `run_formal_v4.sh --check` → `run_formal_v4.sh`。目前實測 `prep_v4.sh --check` 通過、`run_formal_v4.sh --check` 因母體尚未建立而依設計中止。

## 9. 逐成員起點水平位置隨機化（只估計，未實作）

**現況**：每個水平受體固定在 OCM 三角形網格面中心（5 個面／站），20 個成員（含 4 個垂向 draw）共用該點；起點深度已是逐成員的（`z` 依沉底時刻水柱重算）。

**容易改**：初始條件已有單一介面。`native_runner/production_wiring.py` 的 `build_formal_identity_table` 內部函式 `resolve_initial_position(unit)` 對每個工作單元回傳 `(x_m, y_m, z_m)`，
目前 `x_m, y_m = projection.project(receptor.lon, receptor.lat)`；`build_particle_identity_table`（`particle_input.py`）已把它當回呼。native-run、後處理、足跡產品都只讀身分表的起點，下游不用改。受體所屬網格面的編號已在 `Receptor.metadata`（`source_face_local_index`／`source_face_global_index`）。

**需要改的地方與工時（一人、含測試）**：

| 項目 | 內容 | 估計 |
|---|---|---|
| 三角形取點 | 以 (seed, 受體, 觀測, 成員) 衍生獨立亂數流（比照 `bed_residence` 的角色標籤做法），在面內以重心座標均勻抽點（`u,v` 反射法）；`build_formal_identity_table` 需讀 OCM 網格拿三角形頂點（`NativeMesh.from_directory`，每個流場區載入一次；`prep_m10_probe.py` 需多傳網格根目錄） | 0.5–1 天 |
| 守門 | 新點必須仍在同一個濕面內（重心座標保證），並檢查該點的 NWW 四角支撐、水深大於起點 `z` 的海床（以節點水深內插）；失敗 fail closed | 0.5 天 |
| 起點 z | **簡易版**：沿用面中心的 `z`（三角形邊長中位數約 30 m；面內水深差我沒有量測，預期不大，陡坡上要靠上面的海床守門）；**嚴謹版**：input-build 的初始條件改成逐成員點位的 eta／zcor 內插，需動 `input_derivation._dynamic_initial_payload` 與 manifest 格式 | 簡易 0 天（含於上）；嚴謹 3–4 天 |
| 稽核與測試 | 身分表記錄抽樣方法 id 與座標；manifest／provenance、重現性、均勻性（卡方）、per_observation 位元相同的回歸；`report`／figures 中「釋放範圍」圖確認 | 1 天 |
| 文件 | 設計文件與報告 | 0.5 天 |

合計**簡易版約 2–2.5 個工作天；嚴謹版約 5–6 天**。會改變粒子身分表與其雜湊（run 身分），所以必須在建 v4 母體／身分表之前決定；若要做，建議與本次 v4 一併上線。

## 10. 待主代理決定的事與風險

1. **Kz 3.0e-4 vs pilot 候選 0.00135**：我把設定改成實際使用的 3.0e-4（BayTrace `kv_const` 預設），與 v3 實際物理一致。若研究設計其實要 0.00135，需要的是**改物理**（設定檔改回 0.00135 即會生效，擴散組態綁定會讓它與 v3 checkpoint 不相容），請明確指示。`constant_kh_m2ps: 0.4429` 仍存在但 native 不讀（用 Smagorinsky），與此類似屬「設定非實際」，我沒有動。
2. **設定檔種子 20260916 的 Spearman +0.117**（§2.4）：屬隨機範圍但是 202 個種子中最大。接受即可；若在意可換種子（改 config hash，且五個站共用）。
3. **用真實資料驗證新龜山島受體與 B2 標籤**：B2 的標籤接線與新龜山島錨點下的隨機受體都沒有在真實 OCM／NWW 資料上跑過 input-build。`prep_v4.sh` 第一步（`horizon-suite-create`，C 當時約 48 分鐘）就是第一次真實驗證；若失敗（例如隨機受體的 face support、標籤序列首尾邊界、`unknown` 比例），需要修正後重跑。建議先跑 `prep_v4.sh` 並檢查 `manifest.config.label_group_summary` 再決定是否開跑。
4. **repo 內正式 YAML 與 v4 範本不同步**：repo 的 `formal_h30_h60_h90_m10.yaml` 仍是 arrival policy v1、未設 `per_member`（v3／v4 伺服器範本是 v2＋25 小時窗）。v4 範本由產生器從 v3 範本推得，不是從 repo YAML；若要 repo 也成為 v4 的單一來源，需另外決定（會改 repo 設定的 hash 與多個測試期望）。
5. **per_member 的文件已過時**：`docs/foundation/03_*` 仍寫 per_member「尚未完成」，我沒有改（涉及設計文字，建議你確認措辭後一併改）。
6. **`cli.py` 的 `{time_utc_ns: record}` 覆蓋風險**（V 建議 6）：per_member 時同一沉底整點會覆蓋；目前只用在 `fixed_calendar_window`（per_member 不支援），我沒有加守門，風險低但未消除。
7. **舊 release config 不相容**：C 已指出，新程式 `load_config` 舊母體的 release config（舊龜山島錨點）會報錯；run04 的龜山島成果對應舊受體，不可沿用；貢寮 release config 的 sha256 也因整份設定變動而不同。
8. **機時**：V 估計 per_member 相對現行 H90 總流場小時 +9%（42 段時最長單段不變）、單元數 2,460 → 24,350；實際值以 `prep_v4.sh` 印出的計畫檔摘要為準。
9. **/data 是 NAS**：C 的推導腳本在 NAS 上約 8 分鐘（IO 受限）；`horizon-suite-create` 會讀 NAS，耗時以小時計。
10. **共用伺服器注意**：整合期間的測試都用私有 `--basetemp`、只以 PID 結束自己的程序；未動 `run04-h90-m10-v3`、`v3-newsettling`、`lbt-repo`、`lbt-repo-wp51`、`lbt-repo-wp38`。

## 11. 重現方式

```bash
# 真實資料多種子抽樣統計
python scripts/wp51_verify/c6_repair_real_seeds.py <v3 母體目錄（含 common-input/）> [種子數=200]
# 龜山島錨點推導（伺服器）
python scripts/derive_guishan_anchor.py --config <release config> --arrival <arrival.json> --ocm-native-root ... \
  --ocm-surface-root ... --nww-root ... --coastline ~/Workspace/lbt-inputs/coastline-taiwan.geojson --check-surface --min-land-distance-m 500
# v4 範本
python scripts/make_formal_template_v4.py --base ~/Workspace/lbt-inputs/formal/template-v3-25h-newsettling.yaml \
  --repo-formal-config configs/lagrangian_backtracking.formal_h30_h60_h90_m10.yaml --output <新檔>
# 測試（伺服器）
ulimit -n 65536; PYTHONPATH=src LBT_BAYTRACE_ROOT=~/Workspace/lbt-runs/baytrace-v4.7.2 pytest tests -p no:cacheprovider --basetemp <私有目錄>
```

## 12. 增補（2026-10-07）：run05 prep_v4 失敗的診斷與修正、v4 母體建置結果

**失敗**：`prep_v4.sh` 在 `hsinchu/core 垂向支撐淘汰後候選不足：需要 5，實際 0`。診斷（`scripts/diagnose_receptor_face_support.py`，只跑受體池 face_support、五站、真實資料）：per_member 每站 500 個沉底整點，新竹在 2025-04-14T22:00Z 等少數整點於 NWW metric location 本身就無有效值（五站 metric location 無效整點：新竹 73、連江 60、南灣 9、龜山 1、貢寮 0，每站約 17,124 個 canonical 整點），對**所有面**都不可支撐（300 個面全 NWW 時刻無效）。v3（per_observation，只有 50 個沉底整點）對同一函式五站都通過（每站 5 面），排除 B2 標籤接線。屬**時刻層級**問題。

**修正（commit 875df09）**：沉底年齡可用性改為「OCM canonical ∩ NWW metric location 有效」（`restrict_available_times_to_supported`，政策 `nww_metric_location_exact_hour_support_v1`），由抽樣器在缺口拒絕時一併排除；排除整點清單與 SHA-256 寫入 `bed_residence_sampling.deposition_support_conditioning`，manifests 載入器核對並拒絕落在排除集合內的沉底整點。**沒有放寬任何受體檢核**；per_observation 不變。另加 `_summarize_face_support_failures`：失敗原因（NWW／OCM、最集中失敗整點）寫入候選不足的錯誤訊息與 `random_horizontal_receptors_by_site[*].face_support_failure_summary`。

**第二個問題（commit 9359da8）**：修正後 release 階段 validator 報 `source_unreadable:ocm_native:FileNotFoundError`——`v4_env.sh` 的 `OCM_NATIVE_ROOT` 等指到 `lbt-local-data`（只有 2024-07 起），inventory 卻涵蓋 2024-01 起。`prep_v4.sh` 只對 `horizon-suite-create` 取消這些環境變數（與 `build_v3.sh` 一致，母體建置只用 `--*-root` 指到 /data）。

**母體（68 分鐘，NAS 冷快取）**：`~/Workspace/lbt-inputs/formal/v4-per-member/20261007_final-sites_h30-h60-h90_common_m10_v4`。核對：十檔沉降速度 records 與 v3 完全相同；每站 500 筆（相異沉底整點 485–490）；年齡 0–2160 h；Spearman +0.117；沉底整點落在排除集合 0；10–12 月沉底每站 36–61（v3 為 3）；逐槽位拒絕總數每站 18；排列修補：初始不可行 39 槽、修補 37 步（swap 37、matching 0）、改動 72 槽；標籤 `unknown` 僅 `deposition_tide_phase` 0.36%，四季皆非 0（DJF 23.3%、MAM 27.5%、JJA 26.8%、SON 22.4%）；兩案各 500,000 顆、24,400 單元、168 段，pre_window 0、gap 檢查 2,440 筆不安全 0；擴散組態 uniform／constant／3.0e-4／Visser 關；龜山島錨點 (121.939148, 24.850326) 離島岸約 546 m，5 個受體離錨點 6.0–11.6 km（核心半徑 12.5 km 內）、離陸地 2.6–7.3 km，皆在海上。

**未通過的核對（需主代理決定）**：「其他站受體與 v3 相同」不成立。新竹水平位置同 v3 的 5 面（垂向 z 因 template 取自不同的第一個沉底整點而不同）；貢寮 2/5、南灣 3/5、連江 0/5 相同。原因是 v3 選到的面在 500 個沉底整點中有 1–2 個整點 NWW 四角無效（連江 5 面皆有；貢寮 3 面）或 OCM 垂向 zcor 無雙側支撐（南灣 2 面），依同一個種子化隨機順序被淘汰、改由後面的面補上（候選池指紋與 v3 相同）。這是 per_member 把受體檢核從 50 個整點擴大到 500 個的必然結果，不是接線錯誤。

## 13. 增補（2026-10-07）：縮短 v4 正式重算牆鐘（只改排程，不改物理）

**決策**：受體選 A（接受 v4 重抽的受體）。為縮短牆鐘，兩案同時跑（no_stokes 20＋finite_depth_stokes 24＝44 worker），並以 run04 實測成本決定動態佇列的提交順序（`native-run --unit-costs`）。

**為什麼 run04 每案約 20 小時**：計畫檔內建成本估計（流場小時×節點×層×全域常數）把新竹（實測僅 2.5 s／流場小時、全區 3 機時）排在南灣（10–14 s／流場小時、最長分段 10.7／13.5 小時）之前（v4 計畫估計：東北 8.5–11.4、新竹 3.4–4.6、南灣 3.1–4.2、連江 2.4–3.2 小時／段），最長的南灣分段排到佇列尾端；再加兩案依序執行。

**為什麼不把長尾切更細**：每段都必須走完回溯窗（90 天＝2,160 流場小時），單段成本下界 ≈ 2,160 × 單位成本（南灣含 Stokes 約 8.5 小時），無法降到 4–5 小時；切更多段只增加重複的回溯窗成本。故最長分段（含 Stokes 約 14.7 小時，v3 實測 13.5 ×1.09）是牆鐘下界。

**模擬**（run04 逐段實測耗時 ×1.09，兩案機時下界 316／392；實測耗時是下界，模擬牆鐘亦偏低，絕對值要乘約 1.2–1.5）：

| 配置（排序／worker） | 兩案牆鐘（模擬小時） |
|---|---|
| 依序、42 worker、計畫內建排序（run04 做法） | 30.4 |
| 依序、42、實測成本平滑排序 | 26.3 |
| 同時 21＋21、內建排序 | 23.9 |
| 同時 20＋24、內建排序 | 22.0 |
| 同時 24＋24、南灣優先 | 19.3 |
| **同時 20＋24、實測成本平滑排序（採用）** | **17.0** |
| 同時 24＋24、實測成本平滑排序 | 17.0 |
| 單一工作池 44／48、平滑排序（需改 orchestrator，未採用） | 16.5／15.1 |
| 同時 20＋24、成本位置偏移 ±2 段的擾動 | 17.0–19.3 |

worker 取 44 而非 48：含 Stokes 案受限於 392 機時／24 ≈ 16.3 小時，多給 4 個 worker 不縮短，且最壞情況（全是東北，約 2.4 GB／worker）44×2.4≈106 GB，48 則約 115 GB 超過 110 GB 預算。

**實作**：`native-run --unit-costs`（只覆寫提交順序；單元集合必須與計畫檔完全一致）、`make_unit_costs_v4.py`（由 run04 同案例同區域同時序位置的實測耗時，前後各 2 段平均，×1.09）、`run_formal_v4.sh` 的 `LBT_PARALLEL_CASES=1`（預設不變）。測試：成本檔載入／核對、parser 選項、成本檔產生函式；orchestrator＋CLI 共 166 passed。
