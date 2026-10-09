#!/bin/bash
# WP-53：v6（流場資料完整化：全部缺時與供應者重播小時以 ocm_harmonic_residual_bridge_bounded_v2 重建）
# 正式重算共用的環境與檢查函式。只能被 prep_v6.sh 與 run_formal_v6.sh 以 source 載入；本檔本身不建立任何檔案或目錄。
# 與 v5_env.sh 的差異：
#   * 資料根改為 lbt-local-data-v6（ocm_native 與 nww3_analysis 為 v5 的硬連結，reconstruction 為新的 v2 patch 根；
#     patch 以 supersedes_observed 宣告取代供應者重播列，不改寫任何 v5 檔案）；
#   * 母體建置專用的 ocm_native 根（A v6）與 ocm_surface 根（時間軸皆完整）取自 lbt-local-data-v6-full；
#   * 程式碼 clone 改為 lbt-repo-wp53（含 supersedes_observed 載入支援與 v2 重建方法）；
#   * 輸出／母體位置改為 run07-h90-m10-v6、v6-per-member-complete，獨立的 checksum／numba 快取；
#   * 其餘設定（per_member、M=10、擴散組態、Kz 3e-4、十檔沉降速度）完全不變，擴散組態啟動前核對，不符即中止。
# 不修改 run05／run06 使用的任何檔案。

W=$HOME/Workspace
CODE=${LBT_CODE:-$W/lbt-repo-wp53}
B=$W/lbt-runs/run07-h90-m10-v6
V6ROOT=$W/lbt-inputs/formal/v6-per-member-complete
TEMPLATE=$W/lbt-inputs/formal/template-v6-per-member.yaml
D=$W/lbt-local-data-v6
# 母體建置專用的 ocm_native 根（結構指紋用；見 prep_v6.sh）。執行期不使用。
APRIME_OCM=$W/lbt-local-data-v6-full/ocm_native
APRIME_SURFACE=$W/lbt-local-data-v6-full/ocm_surface
REQUIRED_COMMIT=ea97469230231a999dd4f81bfaff128850d1e15a
COAST=$W/lbt-inputs/coastline-taiwan.geojson
PY=$W/lbt-repo/.venv/bin/python
LBT=$W/lbt-repo/.venv/bin/lbt
TOOLS=$W/lbt-runs/tools

export PYTHONPATH=$CODE/src
export PYTHONDONTWRITEBYTECODE=1
export OCM_NATIVE_ROOT=${D}/ocm_native NWW_ANALYSIS_ROOT=${D}/nww3_analysis OCM_RECONSTRUCTION_ROOT=${D}/reconstruction
export LBT_CHECKSUM_CACHE_DIR=$W/lbt-runs/checksum-cache-v6 LBT_BAYTRACE_ROOT=$W/lbt-runs/baytrace-v4.7.2
export NUMBA_CACHE_DIR=$W/lbt-runs/numba-cache-v6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export LBT_NATIVE_BACKTRACK_MODE=full_horizon_from_deposition LBT_NATIVE_FORMAL=1 LBT_NATIVE_SMAGORINSKY_IMPL=numba
# 擴散組態：明確寫出（與 physics_case 的預設值相同）。Kz 不用環境變數：設定檔是唯一來源。
export LBT_NATIVE_DIFFUSION_NOISE=uniform
export LBT_NATIVE_VERTICAL_DIFFUSION=constant
export LBT_NATIVE_VISSER=0
unset LBT_NATIVE_KZ_CONSTANT_M2PS

log() { echo "[$(date -u "+%F %T")Z] $*"; }
die() { log "中止：$*"; exit 2; }

# 最新的 v6 母體（prep_v6.sh 建立後才存在）。
latest_population() {
  ls -d "$V6ROOT"/*_final-sites_h30-h60-h90_common_m10_v6 2>/dev/null | tail -1
}

# 程式碼來源檢查：git clone、工作樹乾淨、含可行性修補、import 路徑正確。
require_clean_code() {
  [ -d "$CODE/.git" ] || die "$CODE 不是 git clone（請用 git bundle 建立可追溯 clone）"
  [ -z "$(git -C "$CODE" status --porcelain)" ] || die "$CODE 工作樹不乾淨（git status 有變更）"
  grep -q "feasibility_repair" "$CODE/src/lagrangian_backtracking/bed_residence.py" \
    || die "$CODE 的 bed_residence.py 不含 per_member 可行性修補（feasibility_repair），版本不對"
  git -C "$CODE" merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD \
    || die "$CODE 的 HEAD 不含 ${REQUIRED_COMMIT:0:7}（重建 patch 發散流速守門與單小時子步數上限）"
  [ -x "$PY" ] && [ -x "$LBT" ] || die "找不到 $W/lbt-repo/.venv 的 python／lbt"
  "$PY" -c "import sys, lagrangian_backtracking as m; sys.exit(0 if m.__file__.startswith('$CODE/src') else 1)" \
    || die "import lagrangian_backtracking 沒有來自 $CODE/src"
}

# 資料根檢查：reconstruction 根必須是 v2 方法、四區齊全、manifest 雜湊與索引一致，且 ocm_native 與 nww3_analysis 存在。
# 不建立任何檔案。方法識別字必須是 ocm_harmonic_residual_bridge_bounded_v2。
require_v6_data_root() {
  [ -d "$D/ocm_native" ] && [ -d "$D/nww3_analysis" ] && [ -d "$D/reconstruction" ] || die "資料根 $D 缺少 ocm_native／nww3_analysis／reconstruction"
  "$PY" - <<PYEOF || die "v6 reconstruction 根檢查失敗（見上方訊息）"
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

# 解析並核對擴散組態（需先 export LBT_NATIVE_CONFIG）。印出 JSON；與預期不符就中止。
# 預期：uniform／constant／Visser 關／Kz=3.0e-4 且來源為 config（與 v3 實際物理相同）。
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
    echo "擴散組態：$(resolve_and_check_diffusion)"
    echo "擴散環境變數：NOISE=$LBT_NATIVE_DIFFUSION_NOISE VERTICAL=$LBT_NATIVE_VERTICAL_DIFFUSION VISSER=$LBT_NATIVE_VISSER KZ=${LBT_NATIVE_KZ_CONSTANT_M2PS:-（未設定，由設定檔決定）}"
    echo "---"
  } >> "$out"
}
