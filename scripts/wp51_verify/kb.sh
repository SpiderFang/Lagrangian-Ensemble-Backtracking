#!/bin/bash
cd ~/Workspace/lbt-wp51/verify
export PYTHONDONTWRITEBYTECODE=1 LBT_BAYTRACE_ROOT=~/Workspace/lbt-runs/baytrace-v4.7.2 NUMBA_CACHE_DIR=$PWD/numba_kb NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=~/Workspace/lbt-repo/.venv/bin/python
for d in base wp51; do echo "--- $d"; PYTHONPATH=$PWD/$d/src $PY b_kernel_bench.py $PWD/$d/tests 20000 150 5 2>&1 | tail -8; done
