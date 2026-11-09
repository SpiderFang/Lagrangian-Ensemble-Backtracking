# 沿開放邊界之相對來源權重曲線（依組別分線）

案例：`finite_depth_stokes`

本目錄由 `lbt native-arc-by-group` 產生，只讀取 `native-postprocess` 的輸出，不改動任何輸入。

## 內容

- `figures/<站>/`：含文字版（標題、副標、圖例、座標軸標籤），每張圖同時輸出 PNG 與 SVG。
- `figures-notext/<站>/`：無文字版（不畫標題與副標，只留座標軸、各邊名稱與圖例）。
- `tables/<站>/<分組>_summary.json`：各組粒子數、各邊（南、東、北、西）占比、組間兩兩總變異距離、抽樣雜訊基準（同組隨機對半切；類別另有同釋放點交換類別標籤的配對置換，季節與潮況為沉底事件層級的標籤置換）。
- `tables/<站>/<分組>_curves.csv`：各組曲線格點值（%／km）。
- `manifest.json`：參數、軸範圍、輸入與輸出檢核碼，以及每張表的摘要。
- `captions.md`：每張圖的圖說。

圖種：

- `boundary_arc_weight_by_class_overlay`
- `boundary_arc_weight_by_class_panels`
- `boundary_arc_weight_by_season_overlay`
- `boundary_arc_weight_by_season_panels`
- `boundary_arc_weight_by_tide_overlay`
- `boundary_arc_weight_by_tide_panels`

## 方法摘要

每組只取穿越開放邊界且有弧長位置的粒子，依弧長分成 2000 格，以頻寬 5 km 的一維高斯核平滑，曲線對弧長積分為 100%。閉合的邊界折線首尾環狀接續，未閉合者於端點鏡射。季節與大小潮使用沉底時刻計算的`deposition_season`、`deposition_tide_class`。曲線稱為相對來源權重，不可解讀為絕對來源或因果歸因。

站：gongliao, guishan, hsinchu, lienchiang, nanwan

重建命令：

```
lbt native-arc-by-group --case finite_depth_stokes /home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/postprocess /home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/figures-arc-by-group --dpi 600
```
