# 已審模型岸界：有限範圍的首次碰岸停止

## 已完成站點的四圖與來源追溯

`scripts/build_completed_subset_preview.py` 須與已支援第四幾何的同版 `src` 配套，
沿用既有 `load_boundary_geometries`，不複製loader。`collect_subset` 保留情境component
及geometry兩個完整canonical字典精確相等；不能刪除 `native_coast` key通過驗證。
只有該站兩片都COMPLETE、最終output完整checksum／身分驗證通過才產原四張圖，
不以最後checkpoint的active數代替 `particle_table.parquet` 的最終分類。

可選 `geometry.native_coast_manifest.path` 依config所在目錄解析；其中 `source_files`
依岸界manifest所在目錄解析，絕對引用原樣保留。使用既有 `resolve_manifest_path`
規則，不跟隨symlink，不掃forcing、raw或checkpoint歷史。岸界manifest與所有列明
source檔案在loader讀取前加入SHA快照，擷取結束及四图渲染後各核一次。缺檔、
損壞、來源SHA／投影／站點錯綁仍由現有loader拒絕；中途變動拒絕發布manifest。
沒有岸界設定時不新增來源／欄位，保留原三geometry流程。

`render_inputs.json` 與發布 `manifest.json` 的 `native_coast` 保存實際岸界檔路徑、
檔案SHA、canonical、間接來源路徑/SHA、兩component、投影WKT、region/edge指紋。
`model_plan_commit` 是不可變計畫的模型來源；`plot_code` 是執行腳本所屬checkout的
實際commit、tracked dirty旗標、腳本與幾何loader路徑/SHA，不能互相冒稱。
底圖SHA仍在共同來源快照，但 `basemap_role=display_only_not_physical_boundary`：
**模型岸界不等於底圖岸線**。只覆蓋已審兩component，其餘失敗仍待查，dt-halving及
完整科學驗收仍未完成。

南灣／連江的新版本是原100同受體重算替代，**不是額外40顆**。每站圖面與終止分母
固定20；舊A與新B結果分開，不能合併分母。正式計畫50,000情境僅表示母體設計，
不代表已執行50,000顆。圖與README明列此限制，仍採原四種圖式、不增圖型。

以下命令只為審核準備，需主審GO後才在新的乾淨plot checkout執行：

```bash
PYTHONPATH="$PLOT_CODE/src" "$PYTHON" "$PLOT_CODE/scripts/build_completed_subset_preview.py" \
  --run "$NEW_COAST_RUN" --config "$NEW_COAST_CONFIG" --coastline "$DISPLAY_BASEMAP" \
  --project-root "$PLOT_CODE" --current100-site nanwan \
  --shard "$NANWAN_FIRST_SHARD" --shard "$NANWAN_SECOND_SHARD" --output "$NEW_B_NANWAN_PREVIEW"
```

SERVER圖面只寫 `/data/LBT/previews` 下既定原100成果根的唯一
`B-nanwan` 子目錄，不覆寫舊圖；所有TMPDIR/MPLCONFIGDIR/XDG_CACHE_HOME/UV_CACHE_DIR/
NUMBA_CACHE_DIR皆本次NFS scratch，五項threads=1。此處不部署、不重算、不改運算程序。

## 已審兩站40顆操作入口

`scripts/run_native_coast40_subset.py` 只適用本次已審來源／新岸界計畫對，雜湊常數
在腳本 `REVIEWED`；沒有任意站點、worker數、續跑或縮短步數的公開參數。
先沿用固定100選樣核對五站完整母體，再取南灣／連江各兩片，四片內外均須
PLANNED、零保存步數、零checkpoint。兩張完整scenario／seed表的實檔SHA必須與來源及
plan相符，master_seed／component及三舊geometry保持；唯一新增已審native_coast。
完整 `validate_run_inputs(execution_priority=True)` 仍執行，不使用defer或私有略過入口。

```bash
python scripts/run_native_coast40_subset.py \
  --workspace "$NEW_WORKSPACE" --source-workspace "$READONLY_SOURCE" \
  --config "$NEW_CONFIG" --project-root "$CLEAN_CODE" \
  --checkpoint-root "$NEW_CHECKPOINT" --ocm-native-root "$OCM_NATIVE_ROOT" \
  --ocm-reconstruction-root "$OCM_RECONSTRUCTION_ROOT" \
  --scratch-root "$NEW_SCRATCH" --log-root "$NEW_LOGS" \
  --numba-cache-dir "$NEW_NUMBA_CACHE" --storage-gate-evidence "$STORAGE_GATE" \
  --expected-commit "$ACTUAL_OPERATOR_COMMIT"
```

以上預設只核對與寫新session，不啟動粒子；另獲GO才加 `--execute`。所有目標、
checkpoint、scratch、log與cache須在 `/data/LBT`，且即時掛載身分須與已通過的NFS
storage gate一致。呼叫前建立目錄並將 `TMPDIR`、`MPLCONFIGDIR`、`XDG_CACHE_HOME`、
`UV_CACHE_DIR`、`NUMBA_CACHE_DIR` 放在本次scratch；log與Numba cache須為分開的嚴格
子目錄。五項thread環境固定1，場快取固定20／4GiB；此為執行記憶體配置，不是物理變更，
快取常駐上限也不等於RSS。

每個工作區的operator鎖位於工作區父目錄，以固定run名稱產生零長度檔案，不加到
run根白名單、不修改plan或清冊；鎖從核對開始持有到自身子程序收尾結束。這不是
run_gate獨占鎖，避免擋住worker需要的共用鎖；既有assigned shard鎖照常防止同片雙寫。
操作者PID與啟動tick存於session manifest，鎖檔不寫入內容也不刪除重建。

共用原生命週期只接受五站或此次兩站、各兩個唯一shard；完成分母分別為100／40，
兩站完成不等其餘三站。MemAvailable不足30GiB或無法讀取即停止啟動／安全收尾；
SIGINT、SIGTERM或任何非零worker退出，只對本次新建process group使用原300秒
SIGINT→TERM／KILL策略，不依名稱pkill、不碰其他run，不自動重試。

`demo-manifest.json` 分開保存immutable plan的6992模型來源與actual operator來源、
已審綁定、表SHA、各站兩片命令、執行緒／快取實值、40顆／50000母體分母及待驗狀態。
逐站完成仍須有實際原版native completed、批次group及source SHA、fallback=0，保留
rejected／物理QC，不將有失敗的完成片稱為科學通過。dry-run不產生native推進證據，
真資料與dt-halving驗收仍待後續授權；50k保持HOLD。原固定100 main的來源hash與
default／resume契約不變。

本版本只新增南灣 `nanwan-boundary-0` 主缺支援區的原生交界，及連江
`lienchiang-boundary-35` 的112條內孔邊。這是已審的**模型岸界**，不是底圖岸線，
不自動將所有單一三角形擁有的邊、未知網格孔洞或水平品質旗標1改為海岸。
底圖FID417474／328045與交集比例只提供來源佐證，不作分類門檻。

## 靜態產品與來源契約

`scripts/build_reviewed_native_coast.py` 讀取此次C1的兩份固定SHA原始證據，並核對
原生節點、原生邊、全域節點與面索引。只讀已前處理靜態網格及既有幾何JSON，
不讀raw NetCDF、不載入速度月份、不推進粒子、不消耗亂數。

```bash
python scripts/build_reviewed_native_coast.py \
  --components /absolute/c1-components/result.json \
  --c1-root /absolute/c1-edge-sources \
  --output /absolute/new-native-coast-artifact
```

執行環境必須將暫存與快取設在新的NFS目錄、五個執行緒限制為1；SERVER靜態
建置限單一程序、180秒、啟動前MemAvailable至少30GiB。輸出目錄必須不存在。
命令須放在既有有界操作方式中執行，不可拿它切換現行模型。

輸出 `native_coast_manifest.json`（schema 1.0.0）保存兩站區域與停止線，
`geometry-regression.json` 保存七點的相鄰有效原三角形中心至失敗點之幾何首交點、
查詢耗時、來源SHA及 `formal_result=false`。此直線只驗定位，不是假裝完成真實軌跡重算。

產品保存：投影WKT、原生公尺座標、節點local/global索引、原生edge與source face
ID、單一owner與core端點、原grid/domain/core政策及SHA、C1證據SHA、底圖原SHA及FID、
區域與邊canonical指紋。節點公尺座標直接由原來源確認，不使用經緯度往返投影值
當新原始節點。南灣只納入含原5點的主差集區；另一小區不納入。停止線是主區與
原生邊的實際交界，排除domain人工封口、open外框、非core裁切邊；連江保留原112邊。
沒有buffer、nearest、snap、補封口或資料填零。

## 可選設定與不可續接的邊界版本

新設定可明示：

```yaml
geometry:
  native_coast_manifest:
    path: /absolute/new-native-coast-artifact/native_coast_manifest.json
    sha256: "此檔案實際64位小寫SHA256"
```

路徑相對於設定檔；引用必須同時有path與SHA。載入時核對原幾何SHA、來源檔SHA、
站點／flow domain／投影／component身分及線區域指紋。檔案SHA及canonical hash加入
static geometry binding；設定引用本身亦進config hash，因此岸界版本不同必須拒絕
續接舊checkpoint。`status=approved`只表示此兩component的模型岸界範圍已審，
不表示粒子結果已完成科學驗收。

**沒有此設定時**不新增預設欄位、不改原config hash或三份幾何hash、不改既有運算路徑。
現行run/config/plan/checkpoint均不修改。C3尚未核准前不得掛載、切換或重算；新設定也
需要重新建立相容的靜態輸入綁定，不能偽稱舊inventory與新config hash一致。

## 首次接觸與未涵蓋情形

載入時一次建立線段空間索引，每步查segment包圍盒後做精確交集，不逐粒子聯集網格。
岸界候選與既有flow/local事件依步內比例共同排序；首次 `COAST_CONTACT` 在原線交點
停止，同步UTC、深度與age，事件記錄manifest／region／原生邊／source face身分。
事件 `boundary_s_m` 為沿該 `boundary_segment_id` 公尺子段自首端的距離，不是全島累積岸距。
水域到水域的單步穿島亦截停。切線、共點算接觸；步首在線且向水側離開不重報，
向內或沿岸移動在fraction=0停止。非法步首、未知孔洞、乾點、時間缺口不藉此重分類。
原flow開放出口與local事件保留。

RK中間點遇純水平不支援時，只接既有步首速度直線近似定位，不改PI原版插值/RK4、
分組、擴散、dt、seed、快取或停止條件。近似事件仍保存
`boundary_locator=reference_drift_after_rk_stage_invalid` 與
`requires_dt_halving_validation=true`；正式驗收尚須做dt折半敏感性比較。
不允許讓非法步首state直接變成碰岸，也不將乾點／時間／混合品質旗標改成海岸。

未驗項包括：C3隔離真資料重算、H30完整終止結果、dt折半、其餘未審邊界分量與獨立科學驗收。
本次測試或七點定位通過不代表所有數值失敗已解決，50k仍保持HOLD。
