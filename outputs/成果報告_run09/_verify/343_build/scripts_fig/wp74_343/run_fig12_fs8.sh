#!/bin/bash
# WP-74：3-4-3 圖 1–2 以 8 pt 重畫並與前版（v4，.203 ~/Workspace/lbt-wp64-343/make_343_figs_run09_v4.py）逐項比對
# 用法：run_fig12_fs8.sh <輸出資料夾（須不存在）>
W=$HOME/Workspace
S=$W/lbt-inputs/formal/v8-run09/20261019_final-sites_h180_common_m10_v8
cd $W/lbt-repo-wp63 && PYTHONPATH=$W/lbt-repo-wp55-fig/src MPLCONFIGDIR=$W/lbt-runs/mpl-cache $W/lbt-repo/.venv/bin/python scripts/wp74_343/compare_fig12.py \
  --prev-script $W/lbt-wp64-343/make_343_figs_run09_v4.py --new-script scripts/wp74_343/make_343_figs_run09.py --out-dir "$1" \
  --control-png1 $W/lbt-wp64-343/out_v4/343_run09v4_fig1_circle.png --control-png2 $W/lbt-wp64-343/out_v4/343_run09v4_fig2_circle.png -- \
  --coastline $W/lbt-inputs/coastline-taiwan.geojson --domain-json $S/common-input/domain.json --receptor-json $S/common-input/receptor.json \
  --release-config $S/release-configs/release-180d-full_horizon_from_deposition.yaml --radius-km 10 --out-prefix 343_run09v5 --dpi 600
