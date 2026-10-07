#!/bin/bash
cd ~/Workspace/lbt-wp51/verify
W=$HOME/Workspace; D=$W/lbt-local-data; B=$W/lbt-runs/run04-h90-m10-v3
S=$(ls -d $W/lbt-inputs/formal/v3-newsettling/*_final-sites_h30-h60-h90_common_m10_v3 | tail -1)
export LBT_NATIVE_CONFIG=$S/release-configs/release-90d-full_horizon_from_deposition.yaml
export OCM_NATIVE_ROOT=${D}/ocm_native NWW_ANALYSIS_ROOT=${D}/nww3_analysis OCM_RECONSTRUCTION_ROOT=${D}/reconstruction
export LBT_CHECKSUM_CACHE_DIR=$W/lbt-runs/checksum-cache LBT_BAYTRACE_ROOT=$W/lbt-runs/baytrace-v4.7.2
export NUMBA_CACHE_DIR=$PWD/numba_bench PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export LBT_NATIVE_PLAN=$B/no_stokes/native-plan.json LBT_NATIVE_IDENTITY_TABLE=$B/inputs/identity_table.npz
export LBT_NATIVE_RESULTS_DIR=$PWD/bench_results; mkdir -p $LBT_NATIVE_RESULTS_DIR
export LBT_NATIVE_SMAGORINSKY_IMPL=numba LBT_NATIVE_BACKTRACK_MODE=full_horizon_from_deposition LBT_NATIVE_FORMAL=1
export PYTHONPATH=$PWD/wp51/src
ulimit -n 65536
~/Workspace/lbt-repo/.venv/bin/python d2_bench_diffusion.py "$@"
