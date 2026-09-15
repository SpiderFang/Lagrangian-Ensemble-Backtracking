# Slice 1：SERVER v3 輸入衍生與 release contract

> **閱讀提示**
> - 文件類型：正式輸入衍生與發布契約。
> - 它回答：已驗收 OCM／NWW3 產品如何形成可重建、可核對的 runtime 輸入。
> - 建議先讀：[架構與資料契約](../foundation/02_architecture_and_data_contract.md)，再讀[SERVER 手冊](06_server_runbook_plan.md)。

本文件定義 `input_derivation.py` 的可重建資料流與發布邊界。它是正式 Lagrangian
runtime 的輸入前置契約，不是 SERVER 科學批次結果，也不取代 OCM／NWW3 上游產品的
原始驗收。正式流程只能讀已驗收的 OCM schema 3 `ocm_native`、OCM schema 3
`ocm_surface` 與 NWW3 schema 1 `nww3_analysis`；不得讀 raw NetCDF、transfer archive
或以最近值、零值填補缺時。

## 1. 輸入與路徑

CLI 參數優先使用明示的 root；未明示時，只讀 config 指定的環境變數。程式不猜測
`/data`、`/srv` 或任何 SERVER 絕對路徑，也不把絕對 root 寫入輸出 manifest。source
inventory 使用 `$OCM_NATIVE_ROOT/<flow_domain_id>/...`、`$OCM_SURFACE_ROOT/<flow_domain_id>/...`
與 `$NWW_ANALYSIS_ROOT/<flow_domain_id>/...` 等 lexical token。

下列命令中的 `$FORMAL_CONFIG_TEMPLATE` 必須指向實際存在且未綁定 release／pilot 的 YAML
template；不可改用已產生的 release config、`common-config.yaml` 或其他已有 derived path 的設定。

accepted-product fingerprint 對小型且直接影響資料契約的 `grid/metadata.json`、
`months/*/metadata.json` 與 `months/*/time_utc_ns.npy` 保存實際 SHA-256；大型 required
NPY 預設只保存檔案大小與 NPY header 的 shape/dtype structural fingerprint。validator
提供 root 時會重新檢查這些欄位，不會在正式建置中對大型 payload 逐 byte hash；相同大小與
header 的 payload 內容稽核必須另行明示 deep audit，不能把它誤稱為預設完整內容驗證。

每個 flow domain 必須有 `grid/metadata.json`、靜態 NPY 與 `months/YYYYMM/metadata.json`、
`time_utc_ns.npy` 及契約列出的 forcing arrays。OCM 的 schema major 必須是 3，NWW3
analysis 的 schema major 必須是 1；metadata 的 status、cache kind、domain ID 與月份也
必須符合 config。上游目錄、檔案及輸出父層不可經由 symbolic link 進入；僅放行 macOS
作業系統固定的 `/var -> /private/var` 與 `/tmp -> /private/tmp` 別名。

## 2. 輸出目錄與 immutable binding

`inputs-build` 先寫入同一檔案系統的 hidden partial directory，再以 `os.replace` 原子
發布到不存在的 final directory。final 目錄或其中任一 component 已存在時不覆寫，應以
新 release 目錄重建。每個 JSON 使用固定 key order／緊湊 separators 計算 canonical
SHA-256，同時保存 pretty JSON 的 raw SHA-256；相鄰 `.sha256` sidecar 綁定兩種 hash 與
byte size。`artifact_index.json` 綁定十個 component，`artifact_bindings.json` 再形成
包含 index 的目錄 closure。

固定檔名與 manifest kind 如下：

| kind | 檔名 | 主要內容 |
|---|---|---|
| `forcing_inventory` | `forcing_inventory.json` | 四域各自 OCM native／OCM surface／NWW3 月份、schema、時間軸、source files、metadata provenance 與 structural fingerprint |
| `ocm_gap_safe_arrival_horizon` | `ocm_gap_safe_arrival.json` | 每個 arrival 的回溯支援窗、缺時與跨缺口判定；舊政策為 7 日，新政策依設定的正整日上限 |
| `nww_full_hourly` | `nww_full_hourly.json` | 四域 NWW3 完整逐時 UTC 與 grid binding |
| `domain_geometry` | `domain.json` | 四個 flow-domain 外層幾何 |
| `local_geometry` | `local.json` | 五個 study-site local domain |
| `open_boundary` | `open.json` | flow/local boundary segment 與停止範圍 |
| `material` | `material.json` | 十類嚴格負值沉降材質／形狀代理 |
| `receptor` | `receptor.json` | 五站各 5 個水平位置 × 4 個垂向模板，共 100 筆 |
| `arrival` | `arrival.json` | 五站各 48 個 season×tide stratum 加 2 個事件，共 250 筆 |
| `initial_condition` | `initial_conditions.json` | 每個同站 receptor×arrival pair 一筆，共 5,000 筆 |

## 3. 時間與缺時政策

月份時間軸先經既有 `canonicalize_time_chunks` 正規化：排序、重複 UTC 使用 stable
prefer-last，並保存原始／canonical 計數與 gap 描述。正式研究期間為 2024-01-01
00:00 UTC 至 2025-12-31 23:00 UTC，共 17,544 個逐時步。

OCM gap 不可用最近值或零值穿越。每一個候選 arrival 都必須檢查 inclusive
`[arrival - support days, arrival]` 的每個整點是否存在（舊政策為 7 日）；缺任何一點就將 `crossed_gap` 設為
true，正式 validator 拒絕該目錄。只有已通過版本化重建與 blocked cross-validation 的
manifest，或逐 arrival 均 gap-safe 的 arrival/horizon manifest，才能成為正式 release
輸入。NWW3 完整逐時 analysis 不沿用 OCM 的缺時軸，也不進行統計填補；四域均須證明
17,544 小時才可將 `nww_full_hourly` 標為 `approved`。

NWW 月份 metadata 的 allowlist 同時保留既有
`ocm_analysis_grid_resample_from_nww3_native`，並接受目前 SERVER 使用的
`ocm_spatial_grid_resample_from_complete_available_nww3_archive_native_time_v1`；後者表示由完整
可取得的 NWW archive native 逐時軸重格網至 OCM spatial grid，不代表供應者 best-cycle。

`generated` 表示可供 synthetic／development 稽核的封裝，不表示已可啟動正式 run；
`approved` 只在 strict builder 或 formal validator 的所有條件通過後出現。

### 3.1 通用回溯支援與共同比較母體

新建共同輸入時，在既有完整設定中加入下列欄位。這是局部設定範例，不是可單獨執行的 YAML：

```yaml
inputs:
  backtrack_support_days: 30
boundaries:
  max_backtrack_days: 7
```

新版 `inputs-validate` 須提供 `--config`，以設定內的站點與區域對應驗證到達紀錄；到達紀錄沿用既有 schema，不新增區域欄位。缺少設定時會明確回報 `generic_horizon_config_required_for_site_region_binding`，不以待驗 gap 紀錄自己宣稱的區域充當獨立證據。

`backtrack_support_days` 是要求建置與驗證的正整日支援窗，不是已取得的驗收證明；任意正整數皆可使用，例如 21、37 或 60，並非固定為 7／30／60。`max_backtrack_days` 是這次粒子運算的上限，不得大於明示的輸入支援窗。新欄位未出現在舊設定時，保留舊政策與設定雜湊；既有 1 日工程試跑不能因新增欄位而升格為較長共同母體。

範例會以 **30 日** 支援條件選取五站各 50 筆到達時刻，建立 100 個受體及 5,000 筆動態初始配對；不是先按 7 日選樣再把標籤改為 30。A 兩站配對 UTC 也必須各自通過 30 日檢查。候選不足時回報站點及選樣失敗原因，不縮減原矩陣、不補缺值。逐時時間支援含首尾共 `24 × support_days + 1` 個節點；30 日即 721 節點。

驗證器核對 gap 紀錄與實際到達 ID、站點、流場及 UTC 一對一，檢查起訖、日數與節點計數，並依綁定來源的時間範圍與缺口重新計算缺時。不能只以 `missing_utc: []` 自述通過。即使只要求執行 7 日，共同 30 日母體在較早時段有缺口仍不能當作有效母體重用。來源 root 可讀時，除重查 metadata／時間檔雜湊，還會從各月的小型時間檔重建排序、去重後的時間軸，核對月份集合、時間範圍、節點數、缺口與時間軸雜湊；不為此重讀大型流速陣列。離線封裝核對不替代現場來源驗收。時間支援也不保證整條移動軌跡的空間或垂向有效性。

母體驗證後，`release-config-create --max-backtrack-days` 可分別產生 7 日、30 日設定，直接綁定同一批不可覆寫的輸入。兩者保留相同 `design_version`、到達／受體／初始條件／情境 ID 及輸入雜湊；回溯長度不同則執行設定雜湊不同，各自使用 run、checkpoint、輸出目錄與 runtime preflight inventory。原始建置設定雜湊仍作為來源證據，不改寫成後製執行設定的雜湊。

若日後要跑 60 日，原 30 日母體不足，需以參數另建 60 日輸入版本；程式不需改碼。新母體可能因缺時排除更多日期。若要嚴格比較 7／30／60 日，應重新從共同通過 60 日檢查的母體建立三份執行設定，不能把兩批不同日期當成只有回溯長度不同的比較。

#### 3.1.1 `horizon-suite` 一鍵建立契約

需要比較多個回溯長度時，`lbt horizon-suite-create` 將上述「先建立最長支援窗母體、再
衍生較短執行設定」固定成單一可稽核流程。命令的 `--backtrack-days` 接受任意數量的
正整日；例如 30、60、90 時，suite 先取最大值 90，從 caller 指定且符合上述條件的
`$FORMAL_CONFIG_TEMPLATE`（由 `--config-template` 傳入）產生 effective `common-config`，精確設定
`inputs.backtrack_support_days: 90`，並以同一組 accepted roots 僅執行一次
`inputs-build`。每個 arrival 必須通過 inclusive
`[arrival - 90 日, arrival]` 的逐時完整 gap-safe 支援，未通過者不能進共同母體；不能先
用 30 日或 60 日檢查選入，再將其標記為 90 日。真正缺時不得使用零值、最近值、跨缺口
內插或其他未登錄外插。

common-input 完成後，suite 只由這一份共同母體產生 30／60／90 的 release config，不再
重跑 inputs-build。每份 release config 的 `boundaries.max_backtrack_days` 等於該份
requested horizon，`boundaries.maximum_step_count` 則依 common config 的有限正值
`integration.dt_min_seconds` 設為
`ceil(days * 86400 / dt_min_seconds) + 1`；多出的 1 是包含起始節點的步數保留。這個
計算是 suite 的固定下限契約，不得因較短 horizon 省略步數，也不得以 `dt_max_seconds`
取代 `dt_min_seconds`。

suite 目的地必須是全新的不存在目錄，create 不覆寫 template、common-input 或既有
release。輸出拓撲固定包含：

```text
<suite>/
├── source-template.yaml       # 原始 template 的保存副本
├── common-config.yaml         # backtrack_support_days 等於最大 requested horizon
├── common-input/              # 唯一一次 inputs-build 的 immutable input artifacts
├── release-configs/           # 每個 requested horizon 的 release config
├── validations/               # common input 與各 release validator JSON
├── horizon-suite-manifest.json        # suite 模式、日數、拓撲、來源與 artifact hash
└── horizon-suite-manifest.json.sha256 # manifest 位元組的 SHA-256 binding
```

`source-template.yaml` 仍保存 caller 原始檔案；suite 只允許原範例所文件化的
`scenarios.receptor_arrival_initial_condition_manifest` placeholder（值為
`manifests/receptor_arrival_initial_condition.json`）進行精確改寫，再
補入共同支援窗與 suite 產物的相對路徑。template 若已綁定 release／pilot config，或
任何其他欄位已指向非預期 derived path，必須拒絕建置；suite 不猜測、遞迴替換或重用既有
binding。`--formal-release`／`--formal` 啟用正式 approved gate，且仍須通過既有 A 區
v3/local20 formal gate，suite 不得繞過；`--pilot` 只保留 generated／pilot 狀態，兩種
模式都必須完成 strict accepted-product、結構、gap-safe 與 SHA-256 檢查。任一步驟失敗時
不發布成功的 final suite；失敗時保留 `.partial-*` 現場供人工稽核，不自動遞迴刪除。共享
同帳號 SERVER 上，操作員清理前必須先確認相關 process、partial 的目錄擁有者、inode 與
final 狀態，不可採用先 `stat` 再 `unlink` 的競態方式；`.partial-*` 不得當作成功發布。

`lbt horizon-suite-validate` 以 suite root 為 positional path，固定讀取 suite 內的
`common-input/`，並可明示 OCM native、OCM surface、NWW3 analysis 三個 accepted roots。省略
三個 roots 時只驗 suite 內的 artifact closure，不代表重新核對 accepted source bytes 或
canonical UTC axis；正式或移機驗收必須明示三個 roots。不提供外部輸入目錄 override，以免
外部目錄破壞 release YAML 的 `../common-input/*` exact path binding。唯讀 validator 會核對 source-template／common-config 關係、90 日
gap-safe 根證據、一次 common-input 的完整性、所有 release 的日數與步數、共同
site／receptor／arrival／material／initial-condition／scenario 母體，以及各 artifact hash。
三個 release 的 config hash 可以不同，因為回溯長度與步數不同；共同母體與 artifact hash
必須完全一致。

這個流程保證的是可公平比較的設計母體與設定綁定，不是每粒子的固定存活時間。粒子仍可
因海岸、域外、資料缺口或數值狀態停止，不能把三份設定都通過解讀為每粒子實際走滿 90 日。
正式 accepted inputs 限 OCM schema 3 `ocm_native`、OCM schema 3 `ocm_surface` 與 NWW3
schema 1 `nww3_analysis`；raw NetCDF 與 transfer archive 不得進入 suite。

```bash
uv run lbt horizon-suite-create \
  --config-template "$FORMAL_CONFIG_TEMPLATE" \
  --backtrack-days 30 60 90 \
  --destination work/horizon-suite-2024-2025-h30-h60-h90-v1 \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --formal-release

uv run lbt horizon-suite-validate \
  work/horizon-suite-2024-2025-h30-h60-h90-v1 \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --formal-release
```

將兩個命令的最後 `--formal-release` 改成 `--pilot` 可建立／驗證 pilot suite；這不會
放寬 accepted product、gap-safe 或 hash binding，只改變正式 `approved` 狀態閘門。

## 4. 幾何、受體與 arrival

四個 flow domain 依 config bbox 建立 WGS84 GeoJSON，實際計算仍由既有
`DomainProjection` 投影到公尺座標。A 區本期依
`formal_domain_policy=v3_local20km_20260909_v1` 使用原 v3 bbox、12.5 km receptor core
與 20 km local domain；B–D local domain 等於各自 flow domain。幾何 manifest 保存實際
`flow_domain_id`、analysis region、source fingerprint、policy 與方法版本。

每站水平受體先從 OCM native mesh 建立既有 local／static-ocean 候選區；若站點同時明示
`anchor_lonlat` 與 `receptor_core_radius_m`，再以該 flow domain 的 AEQD 公尺投影建立核心
圓並與候選區求交。接著才以 persistent-wet、geometry、anchor-first maximin 選出 5 個
face，要求所有候選 arrival 均為 wet，並套用 config 的 flow-domain boundary margin；之後才逐一檢查候選 face 的全部 arrival 與四個垂向類別。垂向支撐必須
由同一 face 的每個 node 各自提供有限 `zcor <= target` 與 `zcor >= target`，不能先對
陡峭海床的淺／深 node 取中位數後掩蓋某一 node 缺層。任何候選 face 失敗都會在 wetdry
候選 copy 中 deterministic blacklist，重新執行同一 maximin；不搜尋最近有效 face、不
放寬 margin，也不以外插補足候選。候選不足時 fail closed。受體 manifest 的
`z_m_positive_up` 仍是第一個 arrival 的模板值，不能冒充全部 arrival 的實際水深。

設定檔的 `scenarios.other_site_receptor_candidate_domain` 以版本化政策
`site_explicit_core_intersect_local_else_local_or_flow_v1` 保存上述語意：明示核心的站點
使用「核心圓與既有 local/static-ocean 候選區」交集，未明示核心的站點維持 local；沒有
獨立 local 時才沿用 flow。這個政策只縮小受體候選 polygon，不把 B 區新竹的
`hsinchu_cache_v3` local domain 改成 12.5 km 核心，也不改五站各 20 個 receptor 的正式
計數契約。`StudySiteConfig` 在 config-load 時先要求兩個核心欄位成對，並拒絕非有限或非正
的 `receptor_core_radius_m`；實際 AEQD／ocean intersection 仍在 inputs-build 讀到 source
geometry 後再次 fail closed。

arrival selector 維持既有 48 個 season×tide strata 加 `high_wave_event`、
`strong_current_event` 兩筆事件。候選的 OCM elevation/current 必須來自 OCM surface
cache 的 `eta_m`、`u_surface_mps`、`v_surface_mps`、`surface_z`、`valid_mask_surface`、
`qc_flags` 與 UTC 軸；OCM native 的全域 `hvel` 不得被掃描來產生 arrival scalar。它們
與 NWW3 的 runtime-equivalent exact-hour spatial sample 及設定支援日數的 gap-safe window 一起
篩選候選。NWW input gate 只接受 runtime 可重現的一維、有限、嚴格遞增 lon／lat 規則格網；
站點座標必須在域內，四角 static mask 與該 UTC 的 `valid_mask_wave` 必須全為有效，四角
Hs、peak frequency、原始波向必須有限，並使用 runtime 相同的雙線性權重、Hs≥0、fp>0
與 sin/cos 波向向量非退化條件。這裡不搜尋最近有效格點、不補零、不重正規化，也不以
時間內插替代 exact-hour；任一 receptor face 的 50 個 arrival 只要有一個不支援，就由
本站 persistent blacklist 觸發 deterministic 重選。軌跡移動後仍由 runtime 每個 stage
的海陸、域外、時間與物理 QC 控制；初始 gate 不保證整條回溯路徑永遠有效。

Arrival 的 NWW metric proxy 使用版本化 policy
`anchor_first_nearest_runtime_supported_nww_cell_center_v1`：先嘗試站點 anchor，只有同一個
strict `48+2` selector 失敗才搜尋 local polygon 內的 bilinear cell center。候選距離由該站
anchor 附近實際 NWW 一維 lon／lat 軸投影後的局地代表格網尺度推導，最大 snap 固定為兩倍，
再按公尺距離、`y0`、`x0` 穩定排序；static 四角與完整 exact-hour dynamic series 仍逐一 gate，
不使用最近值、零值或時間外插。每筆 `ArrivalTime.metadata` 保存 `location_kind`、經緯度、
anchor distance、representative／maximum snap distance、四角 cell index 與 policy ID。這些
欄位只描述 arrival 分層與事件指標的 NWW metric proxy，不取代 receptor 實際位置，也不取代
runtime trajectory 的逐 stage forcing sample。A 區 paired UTC 以貢寮選出的 UTC 為 reference，
但 clone 前必須用龜山島自己的 OCM elevation/current、NWW metric series 與 gap-safe mask 逐 UTC
驗證，並重寫龜山島物理 metadata；CLI `inputs-build` 永遠 strict、fail-closed，不啟用 synthetic
fallback。

## 5. Dynamic initial condition

`initial_conditions.json` 對每一個同站 `receptor_id × arrival_time_id` 建立唯一一列，
不按十種 material 重複展開。每列保存：

- 到達 UTC、OCM 月份與 local time index、`ocm_time_origin=observed`；
- source face local/global index、`eta_m`、bed elevation、`wetdry_elem=0`，其中 0 是 wet；
- 由 OCM `zcor` 得到的實際 `z_m_positive_up`、上下 bracket 與 interpolation alpha；
- receptor／arrival／source flow-domain cross-reference 與 source fingerprint。

垂向 target／bracket 由 receptor template 與 dynamic pair 共用同一個 helper：先以既有
face median 定義代表性 bed／eta，再移除代表性 `bed <= zcor <= eta` 之外的 layer，沿用
10%、40%、70% 與 near-bed target。代表性 bracket 必須有嚴格正寬度；每個實際 pair
仍以當時 UTC 的三／四個 face node 逐一證明雙側有限支撐。禁止海床以下、海面以上、
單側外插、最近 layer 夾取、remainder renormalization 或把全 NaN 轉成零；若 template
gate 與 pair 實際 zcor 不一致，dynamic 建置直接失敗。這份 pair manifest 是 material
共用的初始條件；十種 material 只在 scenario layer 產生
`material × receptor × arrival` 的 50,000 個情境。

## 6. A 區公開標籤與 provenance

公開圖表的 A 區顯示文字固定為 `A 區分析域`。這只是 presentation label；任何內部
record、geometry、inventory、dynamic initial condition 與 provenance 都保留真正的
`flow_domain_id`、本期 `formal_domain_policy=v3_local20km_20260909_v1`、v3 bbox、schema
major、root token、source files、metadata/time SHA-256、NPY structural fingerprint 與
config hash。A v3 policy 建立 release config 時須將 12.5 km receptor core、20 km local
domain、共用 outer boundary、
`runtime_spatial_support_policy=runtime_stage_fail_closed_no_expansion_v1` 與
`formal_release_domain_status=no_expansion_runtime_stage_fail_closed` 同步綁定；不得以公開
標籤、舊 25 km geometry 或歷史 expanded bbox 代替實際來源空間。
因此不能把歷史 expanded source 重新命名成不存在的資料夾，也不能用公開標籤取代來源識別。

## 7. Release config 閘門

`release-config-create` 由指定 template 產生新的 YAML，不修改 template，也不覆寫已存在
的 output。它把十個 component 與 artifact index 的相對路徑寫入所有 runtime references，
並在 `release_binding` 保存 template hash、artifact index hash、每個 component 的
raw/canonical hash 與 exact path policy。

builder 在 `os.replace` 前會把完整 hidden partial directory 交給既有
`validate_input_derivatives`，並顯式傳入當次 config、formal flag 及 OCM native／OCM
surface／NWW analysis 三個 accepted roots。任一步 validator 或發布步驟失敗時，不自動遞迴
刪除 `.partial-*`，保留現場供人工稽核，且不建立成功的 final 目錄；不會在 final 路徑驗證、
降低 validator gate 或形成遞迴發布。共享同帳號 SERVER 上，人工清理前必須先確認 process、
partial 的 inode／owner marker 與 final 狀態，不可採用先 `stat` 再 `unlink` 的競態方式；
`.partial-*` 不能當作成功。只有下列
條件全部成立才可寫入 `config_status: approved`：component immutable binding
可讀且 hash 一致、四域／五站及 10／100／250／5,000 計數正確、NWW 四域均為 17,544
小時、每個 gap-safe horizon 不跨缺口、strict manifest loader 通過、A 當期 domain policy
與逐 RK4 階段封閉失敗空間契約通過、config formal gate 通過，以及所有 config reference
exact 指向同一批 artifact。其他情況仍可產生
`config_status: generated` 與 blocker，供人工稽核；`release-config-validate` 會再以唯讀
方式驗證，不會自動修復 binding。
CLI 的 `inputs-build` 即使是非正式 pilot 也固定以 strict 模式 fail-closed，不允許
synthetic constant-field fallback；`--formal-release` 只控制 approved status、完整時段與
gap-safe horizon 等正式發布閘門。

## 8. CLI 與驗證順序

```bash
uv run lbt inputs-build \
  --config "$FORMAL_CONFIG_TEMPLATE" \
  --destination work/input-release-2024-2025 \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --formal-release

uv run lbt inputs-validate work/input-release-2024-2025 \
  --config "$FORMAL_CONFIG_TEMPLATE" \
  --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-surface-root "$OCM_SURFACE_ROOT" \
  --nww-analysis-root "$NWW_ANALYSIS_ROOT" \
  --formal-release

uv run lbt release-config-create \
  --config-template "$FORMAL_CONFIG_TEMPLATE" \
  --input-directory work/input-release-2024-2025 \
  --output configs/release-2024-2025.yaml \
  --formal-release

uv run lbt release-config-validate configs/release-2024-2025.yaml \
  --input-directory work/input-release-2024-2025 \
  --formal-release
```

正式 SERVER validation 仍須由具權限的執行環境提供實際四域 root、A v3 policy domain ID、
OCM native／surface 與 NWW source metadata、磁碟／權限檢查、完整 NWW analysis 建置結果、
A 區 no-expansion runtime-stage 契約，以及必要的 OCM reconstruction 或逐 arrival gap-safe
證據。OCM surface 必須先支援完整母體的 arrival 選取；OCM native／NWW analysis 則由
runtime 依粒子的實際位置、深度、UTC 與 mask 逐階段判定，任一必要支援失敗立即停止且不補值。
本 slice 已提供重建與驗證入口，
但未登入 SERVER、未修改上游產品、未啟動正式軌跡或宣稱任何科學成果。

## 9. Slice 3A pilot calibration evidence

`lbt pilot-calibrate` 以本文件第 7 節的 release binding 為唯一輸入入口；它不直接回讀
raw NetCDF，也不從 config 猜測 OCM root。建置前會以 `validate_release_config` 驗證 input
artifact index、component path/hash 與 config reference，再由 `load_scenario_inputs` 及
`load_boundary_geometries` 取得 immutable actual-z pair 與公尺制 projection。每個 pair
的 `z_m_positive_up`、arrival UTC、OCM `ocm_month_yyyymm`／`ocm_source_time_index`、
source face、wet/dry 與 `ocm_time_origin` 必須原樣進入 evidence；`Receptor` 的模板 z
不得取代 pair actual z。

校準 output directory 固定只有：

| 檔名 | 契約 |
|---|---|
| `pair_samples.parquet` | 固定 Arrow schema；identity／UTC／actual-z／source provenance／QC 非空，velocity 與三個 Cs 的有效物理值可為 nullable，缺值不補零 |
| `calibration_report.json` | OCM-only 標記、stable SHA-256 per-site selection、QC counts、有效值 quantiles、Kh/Kz/floor/cap 與時間步長候選、input／geometry／code provenance、cache resource counters |
| `manifest.json` | exact root closure、兩個 payload 的 byte size／SHA-256／row count／schema fingerprint；manifest 自身不做自我 hash |

`_select_pairs` 先依 stable SHA-256 決定每站的 selected identity 集合；進入 OCM 取樣後，
`pair_samples.parquet` 再以 `(flow_domain_id, ocm_month_yyyymm, time_utc_ns, receptor_id,
arrival_time_id)` 的 deterministic locality order 寫出。後者只為讓同一 flow domain／月份
連續使用 memory-map、減少 NFS month reload 與 eviction，不改 pair set、seed、取樣係數、
統計公式或 evidence status。validator 依完整 identity、內容 checksum、schema 與可重算
統計驗證，不依賴 stable-hash selection 的列次序。

目前 builder 輸出的 calibration report schema 為 `1.1.0`。`time_limit_candidates` 的
diffusion products 分為 `horizontal_diffusion` 與 `vertical_diffusion`：前者固定使用
`(0.25*horizontal_scale_m)²/(2*constant_kh_m2ps)`，後者固定使用
`(0.25*vertical_scale_m)²/(2*constant_kz_m2ps)`；兩者各自保存 `formula`、
`statistics` 與 `unavailable_reason`。這個分軸契約來自各向異性 Brownian 位移的三軸
方差 `2*K_axis*dt`，禁止再以最小跨軸尺度搭配最大跨軸 K。Kh 或 Kz 缺少、非正或無效
時，只將對應 diffusion product 記為 unavailable，另一軸仍由 pair table 與 candidate
重算。唯讀 validator／reader 仍接受既有 `1.0.0` artifact，但依該版本明示重算舊的
`diffusion` combined formula；不會把 legacy report 靜默解讀為 1.1.0。

`calibration_report.json` 的 `evidence_class` 固定為
`server_real_data_pilot_candidate`；`source_status` 另保存 release config status，不能
用 generated/approved 字串把 evidence 改標成其他資料類別。本專案不以 synthetic accepted
product 產生此 artifact。`completion_status=complete` 的必要條件是 100 receptors、250
arrivals、5,000 unique pair records 與每站 1,000 筆；explicit per-site limit 小於 1,000
時固定為 `partial_engineering_sample`。這是資料閉包狀態，不是科學驗收狀態。實際 SERVER 建置完成後，
`lbt pilot-calibrate-validate` 可再帶同一 config/input directory 重驗外部 hash binding；
只有通過後才可把報告交給後續 well-mixed、PDE、收斂與 trajectory gate。

## 10. Slice 3B2b pilot scenario selection

`lbt run-create --run-kind pilot --pilot-scenarios-per-stratum 1` 只在完整且已驗證的
scenario/receptor manifests 上建立工程 sanity／benchmark 子集。selector 的 exact stratum
是 `(study_site_id, receptor.vertical_id)`；目前五站、四個垂向層位的完整資料預期選出
`5×4=20` 筆。這個 20 筆子集不代表正式結果，也不改變正式每站 10,000、全案 50,000
情境設計；formal run 禁止 selector 並維持完整 coverage。

run plan schema `2.1.0` 的 `scenario_selection` 保存版本化 ranking policy、完整 source
scenario count/hash、selected count/hash 與按 site／vertical 排序的 strata。`run-shard` 使用
static loader 時，會從目前完整已驗證 manifests 重算同一 selector，再對 immutable scenario
table、shard range 與 hash；因此 source count、scenario identity、receptor vertical mapping
或 selection metadata 改變都會 fail-closed。schema `2.0.0` 沒有此欄位時只按 full read-only
相容，不由舊 plan 猜測 pilot 子集。

## 11. Slice 3B2a calibration-bound pilot execution config

`lbt pilot-config-create` 將第 7 節的 source release config、第 9 節的 calibration
evidence 與明示的 pilot engineering scalar 組合成新的 generated YAML。source 只以
`load_config(..., formal_release=False)` 讀取，但仍須通過 `validate_release_config`；因此
四域、五站、50,000 情境設計與 release artifact binding 必須原樣保留。calibration 建立器
只接受 schema `1.1.0`、artifact kind `ocm_pilot_calibration_evidence`、evidence class
`server_real_data_pilot_candidate`、完整狀態與 recommendation
`candidate_pending_trajectory_convergence_and_scientific_validation`。legacy `1.0.0`
仍可由第 9 節 validator 唯讀驗證，但不能被此 builder 當作輸入。

四個 candidate 由 report 的 `candidates` 欄位直接帶入：
`constant_kh_m2ps`、`constant_kz_m2ps`、`floor_m2ps` 與 `cap_m2ps` 必須 available、有限、
非負，且 floor 不得大於 cap。它們分別寫入 horizontal constant Kh、vertical constant
Kz 與 Smagorinsky floor/cap 的既定 physics path；不得在 CLI 另填第二份候選值。所有
integration／boundary／scenario／execution scalar 都必須是 native finite numeric，且
`dt_min <= dt_max <= output_interval`、`maximum_step_count >=
ceil(max_backtrack_days*86400/dt_max)`。active chunk 的 `None` 只接受 CLI 明示的
`--active-chunk-size none`。

建立前會讀取固定檔名 `ocm_gap_safe_arrival.json`，要求 root 與每一筆 record 都支援所要求
的 max-backtrack horizon，`crossed_gap=false`、`missing_utc=[]`，且
`supported_step_count == expected_step_count`。建立器以 hidden temporary YAML、file fsync
與 atomic rename 發布；失敗不留下 final，既有 final 也不覆寫。target parent 改變時，
`ARTIFACT_FILENAMES` 登錄的 release artifact path 與 `_replace_manifest_references` 所有
runtime component reference 同步 rebase。

target root 的 `pilot_execution_binding` 精確採 schema `1.0.0`，保存 source semantic
config hash（由 calibration report 的 `input_binding.config_hash` 驗證）、input artifact
index hash、三個 calibration payload hash、四個 candidate value／field、
完整 scalar snapshot 與 `candidate_pending_dt_and_member_convergence`；binding 內禁止任意
絕對或相對 path。`lbt pilot-config-validate CONFIG --input-directory ... --calibration ...`
先驗證 target release/input binding，再以明示 input directory 驗證 calibration；target
不可被當成 calibration source config。validator 會重算 calibration payload SHA、比對
report input hash、release binding hash、candidate／scalar exact equality 與 target hash，
回傳不含 path 的 JSON-safe `valid/errors/summary`，invalid 使用 exit code `2`。通過不代表
`approved` 或科學驗收，後續仍須執行 trajectory、dt／member convergence 與物理驗證。
