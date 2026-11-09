# 圓形來源方位圖（WP-60）

案例：`finite_depth_stokes`。本目錄由 `lbt native-polar-source` 產生，只讀取 `native-postprocess` 的 `departures.parquet`。

- 含文字版圖：本目錄 `<地點>/`；無文字版圖：同層的 `figures-polar-notext/<地點>/`。
- `tables/`：扇區比例長表（含穿越開放邊界與海岸來源分欄）、穿越／海岸來源比例表、可區辨性組對與摘要表。
- `summary.md`：統計摘要與每地點結論；`captions.md`：圖說；`manifest.json`：參數與檢核碼。

重建命令：

```
lbt native-polar-source --case finite_depth_stokes /home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/postprocess /home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/figures-polar /home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/figures-polar-notext --release-config /home/mustlab/Workspace/lbt-inputs/formal/v8-run09/20261019_final-sites_h180_common_m10_v8/release-configs/release-180d-full_horizon_from_deposition.yaml --coastline /home/mustlab/Workspace/lbt-inputs/coastline-taiwan.geojson --bathymetry /home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/isobaths.geojson --dpi 600 --workers 20 --envelope-auto
```
