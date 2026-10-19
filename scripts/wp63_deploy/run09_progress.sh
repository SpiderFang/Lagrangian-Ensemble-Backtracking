#!/bin/bash
# run09 進度查詢（唯讀）。用法：bash ~/Workspace/lbt-runs/tools/run09_progress.sh
W=$HOME/Workspace
B=${LBT_B:-$W/lbt-runs/run09-h180-m10-v8}
CASE=finite_depth_stokes
R=$B/$CASE
PY=$W/lbt-repo/.venv/bin/python
[ -f "$R/native-plan.json" ] || { echo "尚未建立工作計畫：$R/native-plan.json"; exit 0; }
echo "== run09 進度（$(date -u '+%F %T')Z）=="
[ -f "$B/RUN_STATUS.md" ] && sed -n '1,12p' "$B/RUN_STATUS.md"
R="$R" B="$B" "$PY" - <<'PYEOF'
import json, os, glob, time, re
from datetime import datetime, timezone
R, B = os.environ["R"], os.environ["B"]
plan = json.load(open(R + "/native-plan.json"))
total = len(plan["hour_major_members"])
done = failed = 0
elapsed_done = []
for p in glob.glob(R + "/unit-records/*.done.json"):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if d.get("status") == "completed":
        done += 1; elapsed_done.append(float(d.get("elapsed_seconds") or 0))
    else:
        failed += 1
ckpt = R + "/checkpoints"
started = set(os.listdir(ckpt)) if os.path.isdir(ckpt) else set()
finished = {os.path.basename(p)[:-len(".done.json")] for p in glob.glob(R + "/unit-records/*.done.json")}
running = len(started - finished)
print(f"分段：完成 {done}／失敗 {failed}／進行中（有 checkpoint、尚無完成紀錄）{running}／總數 {total}")
start = None
st = B + "/RUN_STATUS.md"
if os.path.isfile(st):
    m = re.search(r"啟動時間（UTC）：(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)Z", open(st, encoding="utf-8").read())
    if m:
        start = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
if start:
    el = (datetime.now(timezone.utc) - start).total_seconds()
    print(f"已過時間：{el/3600:.2f} 小時（{el/86400:.2f} 日）")
    if done:
        rem = total - done
        print(f"剩餘時間（以已完成分段比例線性推估，分段成本不均，僅供參考）：約 {el/done*rem/3600:.1f} 小時")
    else:
        print("剩餘時間：尚無完成的分段，無法推估")
    if elapsed_done:
        print(f"已完成分段的單段耗時：平均 {sum(elapsed_done)/len(elapsed_done)/3600:.2f} 小時，最長 {max(elapsed_done)/3600:.2f} 小時")
for p in glob.glob(R + "/unit-records/*.done.json"):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if d.get("status") != "completed":
        print("失敗分段：", d.get("unit_id"), str(d.get("error"))[:300])
PYEOF
for f in "$R/logs/native-run.log" "$B/logs/run_formal_v8.log"; do
  [ -f "$f" ] || continue
  echo "-- 最近的錯誤訊息（$f）--"
  grep -nE "Traceback|Error|ERROR|失敗|中止|Killed|OOM|MemoryError" "$f" | tail -5 | cut -c1-300
done
echo "-- 磁碟 --"
df -h /home | tail -1
echo "-- 記憶體 --"
free -g | head -2
echo "-- 程序 --"
echo "native-run 程序數：$(pgrep -fc 'lbt native-run' || true)；worker 程序數：$(pgrep -fc 'multiprocessing.spawn' || true)"
