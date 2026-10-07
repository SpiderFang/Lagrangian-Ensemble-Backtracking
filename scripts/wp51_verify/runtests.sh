#!/bin/bash
cd ~/Workspace/lbt-wp51/verify/wp51
PY=~/Workspace/lbt-repo/.venv/bin/python
export PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1 LBT_BAYTRACE_ROOT=~/Workspace/lbt-runs/baytrace-v4.7.2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
N=8
mkdir -p ../testlogs
ls tests/test_*.py | sort > ../testlogs/files.txt
for i in $(seq 0 $((N-1))); do
  awk -v n=$N -v i=$i 'NR%n==i' ../testlogs/files.txt > ../testlogs/shard$i.txt
  ( NUMBA_CACHE_DIR=$HOME/Workspace/lbt-wp51/verify/numba$i nohup $PY -m pytest -q -p no:cacheprovider $(cat ../testlogs/shard$i.txt) > ../testlogs/shard$i.log 2>&1; echo "RC=$?" >> ../testlogs/shard$i.log ) &
done
wait
