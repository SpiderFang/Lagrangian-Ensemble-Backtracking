#!/bin/bash
ulimit -n 65536
cd ~/Workspace/lbt-wp51/verify
for d in wp51 base; do
  ( cd $d && [ -d .git ] || (git init -q . && git add -A && git -c user.name=v -c user.email=v@v commit -qm snapshot) )
done
PY=~/Workspace/lbt-repo/.venv/bin/python
for d in wp51 base; do
  cd ~/Workspace/lbt-wp51/verify/$d
  export PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1 LBT_BAYTRACE_ROOT=~/Workspace/lbt-runs/baytrace-v4.7.2 NUMBA_CACHE_DIR=$HOME/Workspace/lbt-wp51/verify/numba_rr_$d OMP_NUM_THREADS=1
  $PY -m pytest -q -p no:cacheprovider $(sed 's/^[A-Z]* //' ../testlogs/failed_wp51.txt | tr '\n' ' ') > ../testlogs/rerun_$d.log 2>&1
  echo RC=$? >> ../testlogs/rerun_$d.log
done
