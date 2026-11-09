#!/bin/bash
W=$HOME/Workspace; D=$W/lbt-wp75/site_order
S=$W/lbt-inputs/formal/v8-run09/20261019_final-sites_h180_common_m10_v8
cd $D
for v in orig new; do
PYTHONPATH=$W/lbt-repo-wp55-fig/src MPLCONFIGDIR=$W/lbt-runs/mpl-cache $W/lbt-repo/.venv/bin/python make_$v.py \
  --coastline $W/lbt-inputs/coastline-taiwan.geojson --domain-json $S/common-input/domain.json --receptor-json $S/common-input/receptor.json \
  --release-config $S/release-configs/release-180d-full_horizon_from_deposition.yaml --radius-km 10 --out-prefix 343_run09v5 --dpi 600 --out-dir $D/$v > $D/$v.log 2>&1
echo $v rc=$?
done
sha256sum orig/*.png new/*.png
