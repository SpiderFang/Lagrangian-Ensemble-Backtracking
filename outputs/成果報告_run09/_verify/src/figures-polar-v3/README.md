# 圓形來源方位圖：報告版面（WP-65）

案例：`finite_depth_stokes`。只改版面、不改數值：畫布寬 6.27 英寸（＝報告插入寬度），字級 8 pt，圖例在主圖正下方；每地點 4 張圖（十類總覽、十類各自十個沉降速度之多格圖、季節、大小潮），圖檔在同層 `figures-polar-v2-notext/<地點>/`。統計表、summary.md 與 captions.md 沿用 `figures-polar/`（數值相同）。

重建命令：

```
lbt native-polar-source --case finite_depth_stokes /home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/postprocess /home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/figures-polar-v3 /home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/figures-polar-v3-notext --release-config /home/mustlab/Workspace/lbt-inputs/formal/v8-run09/20261019_final-sites_h180_common_m10_v8/release-configs/release-180d-full_horizon_from_deposition.yaml --coastline /home/mustlab/Workspace/lbt-inputs/coastline-taiwan.geojson --bathymetry /home/mustlab/Workspace/lbt-runs/run09-h180-m10-v8/isobaths.geojson --dpi 600 --workers 20 --no-tables --modes none --report-layout --report-axes-style one_decimal
```
