# Git 部署與資料同步手冊

## 1. 目的與狀態

本手冊定義把既有 SERVER canonical 專案根目錄接入本機 Git 完整歷史的安全流程。`<...>`
都是執行時 placeholder；不得寫入帳密、IP、token、密碼或真實 SERVER 私密絕對路徑。

部署是否已完成、bundle 是否已傳輸、環境是否已重建或研究 run 是否已完成，均以獨立的
部署驗收紀錄為準；本手冊本身不構成完成證據。科學流程仍見[SERVER 執行手冊](06_server_runbook_plan.md)。

同步的必要條件是 SERVER `HEAD` 等於本機核定 commit，且 `git ls-files` 的路徑與內容相同、
tracked working tree 乾淨。GitHub 尚未授權 push 時，不得稱為雲端同步。

## 2. 來源與資料邊界

本機是唯一開發來源；SERVER 只部署核定、乾淨且可由 release record 指認的 commit。SERVER
若有 dirty source、獨有 source、來源不明的同名檔案或 run binding 不一致，必須停止，不可
自動覆蓋。

Git 不同步大型 OCM／NWW 主資料；只依選定清單同步結果、必要 replay 輸入及 manifest／SHA-256。
`data/`、`outputs/`、`server_outputs/`、`server_runs/`、`versions/`、checkpoint、scratch
與 log 留在原位，另行管理，不得刪除、搬移、合併或重建。大型未追蹤資料保持原位；已追蹤
的小型 README 索引仍依核定 commit 同步。不要搬 macOS `.venv`；SERVER 依
`uv.lock` 重建自己的環境。BayTrace 與 PDF／標註附件因體積或授權而另盤點，不能稱為全量同步。

## 3. 更新前與首次接入

任何 SERVER 寫入前，先確認 `<SERVER canonical project root>` 沒有 active worker、queue job、
checkpoint writer 或 publisher；檢查程序樹與 run record，不只看 tmux pane。確認 root、
`root/.git`、runtime 根與備份位置不是未核准 symlink，且容量足夠。

首次傳輸必須是 `refs/heads/main` 可達的完整歷史 bundle，不是在 SERVER 做 initial commit。
`git init` 若必要，只建立 Git metadata；禁止 `git add . && git commit` 偽造新根提交。

建立 bundle 前，先盤點正式 `refs/heads/*`、`refs/tags/*` 與各自 tip；`refs/remotes/origin/*`
只是 remote-tracking refs，`refs/codex/*` 是 app 內部 refs，均不自動納入。若核定來源是
`main`，命令必須明示 `refs/heads/main`：

```bash
LOCAL_ROOT="<本機專案根目錄>"
BUNDLE="<本機暫存目錄>/lbt-main-<核定commit>.bundle"
cd "$LOCAL_ROOT"
git status --short --branch
git for-each-ref --format='%(refname) %(objectname)' refs/heads refs/tags refs/remotes/origin refs/codex
git bundle create "$BUNDLE" refs/heads/main
git bundle verify "$BUNDLE"
git bundle list-heads "$BUNDLE"
shasum -a 256 "$BUNDLE"
```

`refs/heads/main` 會帶出其完整祖先；禁止改成 `git bundle create ... --all`。若發現其他正式
branch/tag，先 inventory、審核範圍，再逐一明示 `refs/heads/<branch>` 或 `refs/tags/<tag>`；
不要偷帶 `refs/codex/*` 或暫存 refs。SERVER 先比對 bundle SHA-256，再通過 verify／list-heads。

接入前，把 incoming tracked tree 將覆蓋的 SERVER 既有檔案備份到
`<SERVER backup root>/<timestamp>/`，保留相對路徑、大小與 SHA-256；獨有 source 或無法備份
的衝突一律停止。只更新核定 tracked source，保留 `data/`、`outputs/`、`server_outputs/`、
`server_runs/`、`versions/` 與 `.venv`。

Git 邊界必須是 canonical root 的 `root/.git`；`provenance.py` 不會把上層 parent checkout
當成本專案 Git。若採 `git clone <bundle>`，完成後必須把 clone 產生的 `origin` 改回
`<核定遠端 Git URL>`，不得讓 origin 留在暫存 bundle：

```bash
cd "<SERVER canonical project root>"
git remote set-url origin "<核定遠端 Git URL>"
git remote -v
```

若無 `origin` 則以核定 URL 新增；未核定前不要寫入 placeholder。嚴禁 root-level destructive
command、`rsync --delete`、`git reset --hard`、force checkout 或無清單遞迴刪除。

## 4. 驗收與未來更新

主審取得 bundle checksum、ref inventory、SERVER `HEAD`、`git status`、tracked-file 清單／
逐檔 SHA-256、`deployment_tree_sha256`、`uv.lock` SHA-256、run provenance、
active-worker 證據與 runtime 保留證據前，狀態只能寫「待驗收」。舊版 snapshot 原樣留存；
舊版 snapshot 保留原 `no_git_pilot` 及當時 commit/dirty，不回填新 Git 資訊。

```bash
cd "<SERVER canonical project root>"
test "$(git rev-parse HEAD)" = "<核定 commit>"
test -z "$(git status --porcelain --untracked-files=all)"
git ls-files > "<證據目錄>/server-tracked-files.txt"
shasum -a 256 uv.lock
uv run --frozen --no-sync lbt-code-provenance --project-root "<SERVER canonical project root>"
```

後續離線更新仍先確認無 active worker、clean source 與無獨有 source，再以明示 refs 建 bundle、
驗證 SHA／heads，並只允許 fast-forward：

```bash
git fetch "<transferred bundle>" refs/heads/main:refs/remotes/import/main
git merge-base --is-ancestor HEAD refs/remotes/import/main
git merge --ff-only refs/remotes/import/main
```

也可在取得明確授權、確認 `origin` 為核定 URL 後 fetch／push；本次不執行 push／fetch。dirty、
非 fast-forward、active run、獨有 source 或 runtime 衝突都要停。此流程只證明 source Git
同步；不動研究資料、不聲稱全量同步，遠端部署完成與否以獨立部署驗收紀錄為準。
