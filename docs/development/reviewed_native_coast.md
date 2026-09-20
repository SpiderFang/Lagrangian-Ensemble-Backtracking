# 已審模型岸界：有限範圍的首次碰岸停止

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
