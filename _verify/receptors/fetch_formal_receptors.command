#!/bin/zsh
# 從 .203 讀取正式母體 v3 的受體與到達時刻設定（唯讀，不修改伺服器任何檔案）
DEST="$HOME/Workspace/Lagrangian-Ensemble-Backtracking/_verify/receptors"
BASE='~/Workspace/lbt-inputs/formal/v3-newsettling/20260924_final-sites_h30-h60-h90_common_m10_v3'
HOST='mustlab@140.117.88.203'
CTL="/tmp/lbt203-ctl-$$"
SSHOPT=(-o ControlMaster=auto -o ControlPath="$CTL" -o ControlPersist=120)
mkdir -p "$DEST"
echo "連線 $HOST（密碼只需輸入一次）..."
ssh "${SSHOPT[@]}" "$HOST" "cd $BASE && find . -maxdepth 4 -type f | head -300" > "$DEST/server_listing.txt" 2>&1
ssh "${SSHOPT[@]}" "$HOST" "cd $BASE && find . -maxdepth 4 -type f \( -iname '*receptor*' -o -iname 'arrival*.json' -o -iname '*initial_condition*' \) -size -300M | tar czf - -T -" | tar xzf - -C "$DEST"
ssh -o ControlPath="$CTL" -O exit "$HOST" 2>/dev/null
echo "完成，檔案在 $DEST"
ls -laR "$DEST" | head -60
echo "可以關閉這個視窗了。"
