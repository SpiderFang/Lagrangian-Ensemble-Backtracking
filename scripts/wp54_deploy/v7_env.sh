#!/bin/bash
# WP-54：v7（十類沉降速度改為「每類 10 個候選速度、逐粒子均衡抽樣」）正式重算共用的環境與檢查函式。
# 只能被 prep_v7.sh 與 run_formal_v7.sh 以 source 載入；本檔本身不建立任何檔案或目錄。
# 與 v6_env.sh（scripts/wp53_deploy）的差異：
#   * 程式碼 clone 改為 lbt-repo-wp54（含逐粒子沉降速度的 fork 核心、均衡抽樣與輸出欄位）；
#   * 設定範本改為 template-v7-class-velocity.yaml（由 make_template_v7.py 從 v6 範本產生：
#     材質區塊換成 5 物件 x 阻力上下界的 10 個候選速度，physics.settling.velocity_sampling: class_candidates_balanced）；
#   * 母體／輸出位置改為 v7-class-velocity、run08-h90-m10-v7，獨立的 checksum／numba 快取（核心程式碼已改，不共用 v6 的 numba 快取）；
#   * 流場資料根「唯讀共用」lbt-local-data-v6 與 lbt-local-data-v6-full，不複製、不寫入；
#   * 新增磁碟空間檢查（require_disk）與 v7 設定／候選速度檢查（require_v7_config）。
# 其餘（per_member 沉底年齡、M=10、擴散組態、Kz 3e-4、H90 full_horizon）完全不變，擴散組態啟動前核對，不符即中止。
# 不修改 run04～run07 使用的任何檔案。

W=$HOME/Workspace
CODE=${LBT_CODE:-$W/lbt-repo-wp54}
# 路徑可用環境變數覆寫，只供「小規模試跑」把所有輸出導到暫存目錄；正式執行不要設定這些變數。
B=${LBT_B:-$W/lbt-runs/run08-h90-m10-v7}
V7ROOT=${LBT_V7ROOT:-$W/lbt-inputs/formal/v7-class-velocity}
TEMPLATE=${LBT_TEMPLATE:-$W/lbt-inputs/formal/template-v7-class-velocity.yaml}
V6TEMPLATE=$W/lbt-inputs/formal/template-v6-per-member.yaml
RUN07=$W/lbt-runs/run07-h90-m10-v6
# 流場資料根：與 v6 共用（唯讀）。
D=$W/lbt-local-data-v6
# 母體建置專用的 ocm_native 根（結構指紋用；見 prep_v7.sh）。執行期不使用。
APRIME_OCM=$W/lbt-local-data-v6-full/ocm_native
APRIME_SURFACE=$W/lbt-local-data-v6-full/ocm_surface
REQUIRED_COMMIT=ea97469230231a999dd4f81bfaff128850d1e15a
COAST=$W/lbt-inputs/coastline-taiwan.geojson
PY=$W/lbt-repo/.venv/bin/python
LBT=$W/lbt-repo/.venv/bin/lbt
TOOLS=$W/lbt-runs/tools
# 單案例磁碟需求的估計（GB）：run07 實測每案 checkpoints 約 28、results 約 6、後處理＋出圖約 1，
# 加上逐粒子欄位的少量增量；兩案同時跑再加共用輸入與餘裕。可用 LBT_MIN_FREE_GB 覆寫整體下限。
EST_GB_PER_CASE=40
EST_GB_SHARED=5
MIN_FREE_GB=${LBT_MIN_FREE_GB:-100}

export PYTHONPATH=$CODE/src
export PYTHONDONTWRITEBYTECODE=1
export OCM_NATIVE_ROOT=${D}/ocm_native NWW_ANALYSIS_ROOT=${D}/nww3_analysis OCM_RECONSTRUCTION_ROOT=${D}/reconstruction
export LBT_CHECKSUM_CACHE_DIR=${LBT_CHECKSUM_CACHE_DIR_OVERRIDE:-$W/lbt-runs/checksum-cache-v7} LBT_BAYTRACE_ROOT=$W/lbt-runs/baytrace-v4.7.2
export NUMBA_CACHE_DIR=${LBT_NUMBA_CACHE_DIR_OVERRIDE:-$W/lbt-runs/numba-cache-v7}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export LBT_NATIVE_BACKTRACK_MODE=full_horizon_from_deposition LBT_NATIVE_FORMAL=1 LBT_NATIVE_SMAGORINSKY_IMPL=numba
# 擴散組態：明確寫出（與 physics_case 的預設值相同）。Kz 不用環境變數：設定檔是唯一來源。
export LBT_NATIVE_DIFFUSION_NOISE=uniform
export LBT_NATIVE_VERTICAL_DIFFUSION=constant
export LBT_NATIVE_VISSER=0
unset LBT_NATIVE_KZ_CONSTANT_M2PS

log() { echo "[$(date -u "+%F %T")Z] $*"; }
die() { log "中止：$*"; exit 2; }

# 最新的 v7 母體（prep_v7.sh 建立後才存在）。
latest_population() {
  ls -d "$V7ROOT"/*_final-sites_h30-h60-h90_common_m10_v7 2>/dev/null | tail -1
}

# 磁碟檢查：/home 可用空間必須 >= 估計需求與 MIN_FREE_GB 的較大者。$1 = 本次要同時跑的案例數（預設 2）。
require_disk() {
  local ncase=${1:-2}
  local need=$(( EST_GB_SHARED + EST_GB_PER_CASE * ncase ))
  [ "$need" -lt "$MIN_FREE_GB" ] && need=$MIN_FREE_GB
  local avail
  avail=$(df -BG --output=avail "$W" | tail -1 | tr -dc '0-9')
  log "磁碟：$W 所在檔案系統可用 ${avail} GB；本次估計需求 ${need} GB（案例數 $ncase）"
  [ "${avail:-0}" -ge "$need" ] || die "磁碟可用空間 ${avail} GB 小於需求 ${need} GB（可設 LBT_PURGE_CHECKPOINTS=1 並改依序跑案例以降低峰值，或清出空間）"
}

# 程式碼來源檢查：git clone、工作樹乾淨、含可行性修補與逐粒子沉降速度實作、import 路徑正確。
require_clean_code() {
  [ -d "$CODE/.git" ] || die "$CODE 不是 git clone（請用 git bundle 建立可追溯 clone）"
  [ -z "$(git -C "$CODE" status --porcelain)" ] || die "$CODE 工作樹不乾淨（git status 有變更）"
  grep -q "feasibility_repair" "$CODE/src/lagrangian_backtracking/bed_residence.py" \
    || die "$CODE 的 bed_residence.py 不含 per_member 可行性修補（feasibility_repair），版本不對"
  [ -f "$CODE/src/lagrangian_backtracking/native_runner/settling_sampling.py" ] \
    && [ -f "$CODE/src/lagrangian_backtracking/native_runner/settling_kernels.py" ] \
    || die "$CODE 不含逐粒子沉降速度實作（settling_sampling.py／settling_kernels.py），不是 WP-54 版本"
  git -C "$CODE" merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD \
    || die "$CODE 的 HEAD 不含 ${REQUIRED_COMMIT:0:7}（重建 patch 發散流速守門與單小時子步數上限）"
  [ -x "$PY" ] && [ -x "$LBT" ] || die "找不到 $W/lbt-repo/.venv 的 python／lbt"
  "$PY" -c "import sys, lagrangian_backtracking as m; sys.exit(0 if m.__file__.startswith('$CODE/src') else 1)" \
    || die "import lagrangian_backtracking 沒有來自 $CODE/src"
  [ -d "$LBT_BAYTRACE_ROOT" ] || die "找不到 BayTrace 根 $LBT_BAYTRACE_ROOT"
}

# 資料根檢查（唯讀）：reconstruction 根必須是 v2 方法、四區齊全、manifest 雜湊與索引一致，且 ocm_native 與 nww3_analysis 存在。
# 不建立任何檔案。與 v6 完全相同（資料根不變）。
require_v6_data_root() {
  # 完整檢查約需 6～7 分鐘（逐區驗證 patch 雜湊）。只有小規模試跑可設 LBT_SKIP_DATA_ROOT_CHECK=1 略過驗證；正式執行不得設定。
  if [ "${LBT_SKIP_DATA_ROOT_CHECK:-0}" = 1 ]; then log "注意：略過 reconstruction 根的完整驗證（LBT_SKIP_DATA_ROOT_CHECK=1，僅限試跑）"; return 0; fi
  [ -d "$D/ocm_native" ] && [ -d "$D/nww3_analysis" ] && [ -d "$D/reconstruction" ] || die "資料根 $D 缺少 ocm_native／nww3_analysis／reconstruction"
  "$PY" - <<PYEOF || die "reconstruction 根檢查失敗（見上方訊息）"
import json, os, sys
from pathlib import Path
from lagrangian_backtracking.ocm_reconstruction import validate_reconstruction_patch
root = Path(os.environ["OCM_RECONSTRUCTION_ROOT"])
index = json.loads((root / "reconstruction-index.json").read_text(encoding="utf-8"))
domains = sorted(p.name for p in root.iterdir() if p.is_dir())
if len(domains) != 4:
    print("reconstruction 根的 domain 數不是 4：", domains, file=sys.stderr); sys.exit(1)
for d in domains:
    manifest = json.loads((root / d / "reconstruction-manifest.json").read_text(encoding="utf-8"))
    method = json.dumps(manifest, ensure_ascii=False)
    if "ocm_harmonic_residual_bridge_bounded_v2" not in method:
        print(d, "manifest 不含 v2 方法識別字", file=sys.stderr); sys.exit(1)
    validate_reconstruction_patch(root / d)
print("reconstruction 根檢查通過：", domains)
PYEOF
}

# 核對 v7 設定檔（範本或 release config；$1 = 路徑）：
#   * physics.settling.velocity_sampling == class_candidates_balanced；
#   * 十類各有 10 個嚴格為負、遞增幅度的候選速度，且逐一等於程式內建值（scenarios.BASELINE_BEHAVIORS）；
#   * 代表值等於程式內建代表值。任何不符就中止。
require_v7_config() {
  local cfg=$1
  [ -f "$cfg" ] || die "找不到設定檔 $cfg"
  "$PY" - "$cfg" <<'PYEOF' || die "v7 設定檢查失敗（見上方訊息）"
import sys
from lagrangian_backtracking.config import load_config
from lagrangian_backtracking.scenarios import BASELINE_BEHAVIORS, SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED

config = load_config(sys.argv[1], formal_release=False)
settling = config.physics["settling"]
if settling.get("velocity_sampling") != SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED:
    print("velocity_sampling 不是 class_candidates_balanced：", settling.get("velocity_sampling"), file=sys.stderr); sys.exit(1)
by_id = {m["material_id"]: m for m in settling["material_classes"]}
if len(by_id) != 10:
    print("材質類別數不是 10：", len(by_id), file=sys.stderr); sys.exit(1)
for b in BASELINE_BEHAVIORS:
    m = by_id.get(b.material_id)
    if m is None:
        print("設定檔缺少材質", b.material_id, file=sys.stderr); sys.exit(1)
    cands = tuple(float(x) for x in m["settling_velocity_candidates_mps"])
    if len(cands) != 10 or any(c >= 0 for c in cands):
        print(b.material_id, "候選速度不是 10 個嚴格負值：", cands, file=sys.stderr); sys.exit(1)
    if cands != tuple(b.settling_velocity_candidates_mps):
        print(b.material_id, "候選速度與程式內建值不一致", file=sys.stderr); sys.exit(1)
    if float(m["settling_velocity_mps"]) != float(b.settling_velocity_mps):
        print(b.material_id, "代表值與程式內建值不一致", file=sys.stderr); sys.exit(1)
print("v7 設定檢查通過：velocity_sampling=class_candidates_balanced；十類各 10 個候選速度與程式內建值一致")
PYEOF
}

# 解析並核對擴散組態（需先 export LBT_NATIVE_CONFIG）。印出 JSON；與預期不符就中止。
# 預期：uniform／constant／Visser 關／Kz=3.0e-4 且來源為 config（與 v3 以來的實際物理相同）。
resolve_and_check_diffusion() {
  [ -n "${LBT_NATIVE_CONFIG:-}" ] || die "LBT_NATIVE_CONFIG 未設定"
  "$PY" - <<'PYEOF'
import json, os, sys
from lagrangian_backtracking.config import load_config
from lagrangian_backtracking.native_runner.physics_case import config_kz_constant_m2ps, resolve_diffusion_scheme

config = load_config(os.environ["LBT_NATIVE_CONFIG"], formal_release=False)
scheme = resolve_diffusion_scheme(env=os.environ, config_kz_constant_m2ps=config_kz_constant_m2ps(config))
decl = scheme.as_declaration()
print(json.dumps(decl, sort_keys=True, ensure_ascii=False))
expected = {
    "diffusion_noise": "uniform",
    "vertical_diffusion": "constant",
    "kz_constant_m2ps": 0.0003,
    "visser_correction": False,
    "kz_source": "config",
}
bad = {k: (decl.get(k), v) for k, v in expected.items() if decl.get(k) != v}
if bad:
    print("擴散組態與預期不符（實際, 預期）：" + json.dumps(bad, ensure_ascii=False), file=sys.stderr)
    sys.exit(3)
PYEOF
}

# 把部署指紋寫到指定檔（追加）：git commit、擴散組態、設定檔與範本 SHA-256。
record_provenance() {
  local out=$1
  {
    echo "時間（UTC）：$(date -u '+%F %T')Z"
    echo "主機：$(hostname)"
    echo "程式碼：$CODE"
    echo "git commit：$(git -C "$CODE" rev-parse HEAD)"
    echo "git status 變更項數：$(git -C "$CODE" status --porcelain | wc -l)"
    echo "設定檔：${LBT_NATIVE_CONFIG:-（未設定）}"
    [ -n "${LBT_NATIVE_CONFIG:-}" ] && echo "設定檔 SHA-256：$(sha256sum "$(readlink -f "$LBT_NATIVE_CONFIG")" | cut -d' ' -f1)"
    echo "範本 SHA-256：$(sha256sum "$TEMPLATE" | cut -d' ' -f1)"
    echo "沉降速度抽樣：class_candidates_balanced（每站位每類別 10 個候選各 1/10；種子由 master_seed 與站位／類別導出，寫入身分表 settling_sampling 與後處理 manifest）"
    echo "擴散組態：$(resolve_and_check_diffusion)"
    echo "擴散環境變數：NOISE=$LBT_NATIVE_DIFFUSION_NOISE VERTICAL=$LBT_NATIVE_VERTICAL_DIFFUSION VISSER=$LBT_NATIVE_VISSER KZ=${LBT_NATIVE_KZ_CONSTANT_M2PS:-（未設定，由設定檔決定）}"
    echo "---"
  } >> "$out"
}
