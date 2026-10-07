# WP-51 C：龜山島西側站錨點由陸地移到海上

分支：`wp51-guishan`（自 `wp51-per-member` 分出）。未 push、未合併。

## 1. 結論

- 新錨點：**[121.939084°E, 24.845304°N]**（OCM 原生網格面中心，source face 全域編號 486610、local 編號 113232；高精度值 121.93908437093098, 24.845304489135742，寫入設定時四捨五入到小數 6 位，約 0.1 m）。
- 與原錨點 [121.951606, 24.843127] 的距離：**1,288 m**（AEQD 公尺座標；haversine 1.287 km）。原錨點在校閱海岸線的陸地內，且原 local domain polygon 不含它（`covers=False`）；新錨點不在陸地內，且新 local domain polygon 含它。
- 規則以「寫死座標＋離線推導腳本＋設定與受體 provenance 保存依據」落地，不新增執行期 snap（理由見 2.3）。新增 `anchor_policy` 值 `snap_to_persistent_wet_mesh_within_priority_polygon`。
- 12.5 km 受體核心與 20 km local domain 都以新錨點為圓心重建（已用伺服器真實資料完整重建驗證，見 5）。貢寮的錨點、local、受體、arrival 紀錄逐項與現行正式母體相同。

## 2. 錨點在程式中的用途（任務 1）

| 用途 | 位置 | 說明 |
|---|---|---|
| 設定與 schema | `config.py` `StudySiteConfig.anchor_lonlat` 與 `receptor_core_radius_m` 必須成對；`FORMAL_STUDY_SITE_ANCHORS_LON_LAT` 在 current 設計下逐站 exact 核對 | 新錨點改在這裡，並新增 `GUISHAN_ANCHOR_POLICY_ID` 與「錨點必須在走廊內」檢查 |
| local domain（20 km） | `input_derivation._geometry_payloads` → `geometry.build_anchor_local_domain` | 僅 A 區（貢寮、龜山島）：以錨點為圓心的 20 km 圓 ∩ 靜態海域 ∩ flow bbox，保留含錨點的連通分量；B–D 的 local 等於 flow，不用錨點 |
| 受體核心（12.5 km） | `_receptor_candidate_polygon_metric`；`_receptor_priority_polygon_metric` 再取走廊 ∩ 核心 | 核心圓 ∩ local 是水平受體候選區。current formal 用 seeded random，錨點只決定候選區範圍，不參與 random 排序 |
| arrival 指標位置 | `build_input_derivatives`：`_surface_series_for_location`（OCM surface 取錨點的潮位與流速）與 `_select_arrivals_with_nww_metric_location`（NWW 指標位置「錨點優先」，最大 snap 距離＝核心半徑） | 龜山島 arrival 的 UTC、潮汐類別與相位標籤由貢寮配對繼承（已驗證 UTC 序列不變）；但 `elevation_m`、`significant_wave_height_m`、`current_speed_mps` 與 `metric_location_*` 由龜山島自己的錨點重算，新錨點的 `metric_location_kind=anchor`、距離 0 |
| 圖面星號等 | `native_runner/figures.py` `_load_receptor_anchors` 直接讀設定 YAML 的 `study_sites[].anchor_lonlat` | 星號、`anchor_to_release_centroid_km`、`release_max_distance_from_anchor_km`、`anchor_on_sea` 自動使用新錨點，程式不需改 |
| 條件式來源足跡 | `footprint` 模組讀受體位置，不讀錨點 | 不受影響（星號才用錨點） |

### 2.1 persistent-wet 判定的資料來源

- 濕乾：OCM native 的 `wetdry_elem.npy`（每個 source face 每時次一個值，0＝濕），經 `_face_times` 在 50 個 arrival UTC 取出；`receptors.prepare_horizontal_receptor_candidates` 判定「50 個時次皆為有限且 ≈0」。
- 靜態海域：原生網格所有 face 的聯集（`_face_polygon_union`），local domain 與候選區都在此範圍內。
- 候選另需距候選區邊界不小於 `minimum_flow_domain_margin_local_grid_scales × OCM 三角形邊長中位數`（中位數 30.16 m，邊距 60.3 m）。
- 進入受體選樣後還要過 NWW exact-hour 四角支撐與 OCM 垂向 gate（`face_support`）。

### 2.2 其他站的 snap 政策

B、D、C 的 `anchor_policy: flow_domain_center_snap_to_persistent_wet_mesh` 目前只是設定檔標籤，**程式沒有任何地方讀取它**；那三站的錨點本來就是 flow-domain 中心座標，不是真的 snap 過的點，persistent-wet 只作用在受體候選池。因此沒有可重用的 snap 程式碼。

### 2.3 為何寫死座標

geometry 在 arrival 與濕乾資料載入之前就需要錨點（local domain 先建，濕乾與 arrival 後建），若改成執行期 snap，需要重排 input-build 的相依順序，並讓 NWW 指標、local domain、核心、星號都依賴一個執行期結果。寫死座標可讓這些都用同一個可審計的值。為保留可重現性：推導腳本進版控，設定與受體 manifest 都保存推導紀錄。

## 3. 新錨點規則（任務 2）

在 `receptor_priority_polygon`（島西側走廊）內，取離原錨點最近、且全部通過下列檢核的原生網格**面中心**（與受體候選池同一種點，因此錨點本身是受體候選池成員）：

1. 面中心在靜態海域與 A 區 flow domain 內，且在走廊內；
2. 50 個到達時刻皆濕（persistent-wet，重用 `receptors.prepare_horizontal_receptor_candidates`）；
3. 距候選區（走廊 ∩ 靜態海域 ∩ flow domain）邊界不小於 2 個網格尺度（60.3 m）；
4. NWW exact-hour 四角支撐與 OCM surface 的潮位／流速在 50 個到達時刻皆有限；
5. 不在校閱海岸線（`coastline-taiwan.geojson`，與 WP-38 `anchor_on_sea` 同一份）的陸地內。

實作：新增 `receptors.rank_anchor_snap_candidates`（共用 persistent-wet 判定，依距離、經度、緯度、全域面編號排序）與 `scripts/derive_guishan_anchor.py`（逐一套用外部資料檢核）。推導母體：`common_m10_v3` 的 `arrival.json`（sha256 `6bf8f6d4…`，50 個龜山島到達時刻）。

走廊內通過規則 1–3 的候選面共 2,967 個，前 5 名（距原錨點由近到遠）：

| 名次 | 經度 | 緯度 | 距原錨點 (m) | 距候選區邊界 (m) | 距陸地 (m) | 備註 |
|---|---|---|---|---|---|---|
| 0 | 121.939084 | 24.845304 | 1,288 | 93 | 146 | NWW、surface、海岸檢核皆通過，採用 |
| 1 | 121.939397 | 24.846703 | 1,296 | 61 | 282 | |
| 2 | 121.938680 | 24.840384 | 1,341 | 111 | 115 | |
| 3 | 121.937764 | 24.842107 | 1,404 | 132 | 152 | |
| 4 | 121.937673 | 24.843073 | 1,408 | 63 | 84 | |

## 4. 檢核數字（任務 4）

| 項目 | 數值 |
|---|---|
| 新錨點 | [121.939084, 24.845304]，在走廊內，距陸地 146 m（校閱海岸），不在陸地內 |
| 與原錨點距離 | 1,288 m |
| 現行正式母體（run04 所用）5 個釋放點的平均位置 | (121.89298, 24.83394)；原錨點到它 6.00 km，**新錨點到它 4.82 km** |
| 以新錨點重建後的 5 個釋放點 | 平均 (121.90557, 24.82677)，距新錨點 **3.96 km**；各點距新錨點 0.79、1.60、2.14、7.88、10.61 km，均 ≤ 12.5 km 核心半徑，且皆在走廊內 |
| 受體候選池（重建後） | 走廊內候選 2,843（舊 2,673），核心內候選 4,612（舊 4,467）；5 點全由走廊抽出，無 core fallback |
| 新受體核心（半徑 12.5 km）外緣到 A 區 flow domain 邊界最短距離 | **14.58 km**（南緣；原錨點 14.34 km） |
| 新 local domain（半徑 20 km）外緣到 flow domain 邊界最短距離 | 7.08 km；完整落在 flow bbox 內（v3 policy 要求 `covers`） |
| 錨點到 flow domain 四邊 | 西 63.8、東 86.2、南 27.2、北 72.7 km |
| `minimum_flow_domain_margin_local_grid_scales: 2` | **仍滿足**：OCM 三角形邊長中位數 30.16 m → 邊距 60.3 m；若以 NWW 錨點處代表網格尺度 993.7 m 計，2 倍＝1.99 km；兩者都遠小於 14.58 km |
| local domain 面積 | 911.4 km²（舊 960.8 km²）；新 polygon 含錨點，舊 polygon 不含 |

貢寮：錨點 (121.9223889, 25.0964444)、`receptor_core_radius_m=12500`、local 半徑 20,000 不變；以新設定完整重建後，貢寮的 local geometry、50 個 arrival 紀錄、20 筆 receptor 紀錄與現行正式母體**逐筆完全相同**；新竹、南灣、連江的 local 與 receptor 也相同。兩站共用 A 區 forcing，但幾何獨立，互不影響。

## 5. 驗證

### 5.1 以真實資料重建 common-input（伺服器）

在 `~/Workspace/lbt-wp51/guishan/` 以本分支程式與「只替換龜山島站段落的 server 範本（`template-guishan-anchor.yaml`）」執行 `horizon-suite-create --formal --backtrack-days 30 60 90`，輸出到 `~/Workspace/lbt-wp51/guishan/suite/`（不動正式母體與 run04）。結果：`status=approved`、`valid=true`、`errors=[]`、`warnings=[]`，耗時 48 分。
`receptor.json` 的 `provenance.priority_receptors_by_site.guishan` 已含 `anchor_lonlat`、`anchor_policy`、`anchor_snap_provenance`；龜山島 50 筆 arrival 的 `metric_location_kind` 皆為 `anchor`。

### 5.2 單元測試

- 新增 `tests/test_guishan_anchor.py`（12 項）：設定與常數一致、錨點在走廊內、舊陸地錨點與缺 `anchor_policy` 或錨點出走廊會被 schema 擋下、核心與 local 圓與 flow domain 的安全距離、兩份設定的龜山島段落一致、合成網格上「走廊內最近 persistent-wet 面中心」（含乾點、走廊外、邊界距離、同距離排序、無候選、全乾）。
- `tests/test_config.py`：legacy fixture 還原龜山島舊錨點並移除新欄位，歷史 canonical hash 測試不變。
- 全部測試（伺服器，154 個測試檔）：新增與設定相關測試全數通過；其餘與基準 commit `2f3d871` 相同的失敗只有兩類（見 6）。把 `ulimit -n` 提高到 65536 後，`test_input_derivation.py`、`test_manifests.py`、`test_horizon_suite.py`、`test_engineering_window.py` 在本分支與基準都全數通過（175 與 13 項）；`test_footprint_figures.py` 34 項（含星號、`anchor_on_sea`）全數通過。

## 6. 疑慮與需要決定的事

1. **錨點緊貼島岸**：嚴格依規則取最近者，離校閱海岸線只有 146 m，離走廊東界（121.94°E）93 m，星號會畫在龜山島西岸邊。若要避免「貼岸」的質疑，可改要求與陸地至少 500 m：第一個符合者是名次 10（121.939148, 24.850326），距原錨點 1.49 km、距陸地 579 m；改規則只需在 `scripts/derive_guishan_anchor.py` 加一項檢核並同步改設定、常數與測試。
2. **龜山島需整批重建**：新錨點讓 local domain、受體候選池與 random stream 的 pool fingerprint 都改變，龜山島 5 個水平受體全部換新（0.79–10.6 km 於新錨點），run04 中龜山島的成果對應舊受體，不能沿用。common-input 的 artifact index、bindings、validation 與 config hash 也都變，貢寮雖然紀錄逐筆相同，但 release config 的 sha256 變了，是否沿用 run04 貢寮成果由上層決定。
3. **新程式不再接受舊母體設定**：current 設計下 `FORMAL_STUDY_SITE_ANCHORS_LON_LAT` 是 exact 核對，加上 `anchor_policy` 必填，所以新程式 `load_config` 舊的 `release-*-full_horizon_from_deposition.yaml`（舊錨點）會報錯。WP-38 出圖腳本直接讀 YAML，不受影響；其他以新程式載入舊母體 release config 的流程需先用新範本重建。
4. **伺服器範本尚未更新**：我沒有改 `~/Workspace/lbt-inputs/formal/template-v3-25h-newsettling.yaml`（唯讀）。與 repo 範本的龜山島段落差異僅此一處；可直接取 `~/Workspace/lbt-wp51/guishan/template-guishan-anchor.yaml`（只有龜山島段落不同）。
5. **既有失敗與本工項無關**：`test_source_code_architecture_map.py` 3 項在基準就失敗（WP-51 B 新增的 `deposition_labels` 模組未登錄於架構地圖，HTML 需重產；本分支另新增 `receptors.rank_anchor_snap_candidates`，重產時一併納入）；`test_native_postprocess.py::test_manifest_contains_provenance_and_matches_output_checksums` 在基準就失敗（`project_root` 為 symlink）。伺服器 shell 預設 `ulimit -n 1024`，跑全部測試會因開檔數耗盡產生連鎖失敗，與本分支無關。
6. **合併時可能衝突的檔案**：`config.py`（常數區與龜山島驗證）、兩份 `configs/*.yaml` 的龜山島段落、`README.md`、`docs/foundation/08_*.md` 的龜山島列、`tests/test_config.py` 的 legacy fixture。
7. `tests/test_geometry.py` 仍用舊錨點座標當純幾何輸入，與設定無關，未更動。
