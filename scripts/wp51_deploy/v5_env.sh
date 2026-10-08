#!/bin/bash
# WP-52 階段 B：v5（OCM 補缺後資料根 + v4 同設定）正式重算共用的環境與檢查函式。
# 只能被 prep_v5.sh 與 run_formal_v5.sh 以 `source` 載入；本檔本身不建立任何檔案或目錄。
# 與 v4_env.sh 的差異只有：資料根（lbt-local-data-v5）、程式碼 clone（lbt-repo-wp52，含 3312a0a）、
# 輸出／母體位置（run06-h90-m10-v5、v5-per-member-gapfill）、獨立的 checksum／numba 快取；
# 其餘設定（per_member、M=10、擴散組態、Kz 3e-4）完全不變。不修改 run05 使用的任何檔案。
#
# 與 v3（run_formal_v3.sh）的差異（沿用自 v4_env.sh）：
#   * 程式碼來源改用「乾淨的 git clone」$CODE（預設 ~/Workspace/lbt-repo-wp52，可用 LBT_CODE 覆寫），
#     以 lbt-repo 的 venv 執行但 PYTHONPATH 指到 $CODE/src（沿用 WP-38 部署腳本的做法），並在
#     開始前確認 import 的確來自 $CODE、工作樹乾淨、含 per_member 可行性修補；
#   * 擴散組態明確 export（與預設值相同：uniform／constant／Visser 關），且不設
#     LBT_NATIVE_KZ_CONSTANT_M2PS——Kz 由設定檔 physics.vertical_diffusion.constant_kz_m2ps（3.0e-4）
#     唯一決定；啟動前核對實際解析值，不符預期就中止；
#   * 輸出目錄是新的 run06-h90-m10-v5，不覆蓋 run04／run05。

W=$HOME/Workspace
CODE=${LBT_CODE:-$W/lbt-repo-wp52}
B=$W/lbt-runs/run06-h90-m10-v5
V5ROOT=$W/lbt-inputs/formal/v5-per-member-gapfill
TEMPLATE=$W/lbt-inputs/formal/template-v5-per-member.yaml
D=$W/lbt-local-data-v5
# 母體建置專用的 ocm_native 根（結構指紋用；見 prep_v5.sh）。執行期不使用。
APRIME_OCM=$W/lbt-local-data-v5-full/ocm_native
REQUIRED_COMMIT=3312a0aca61cefedc0e1438a2a4a55661761359e
COAST=$W/lbt-inputs/coastline-taiwan.geojson
PY=$W/lbt-repo/.venv/bin/python
LBT=$W/lbt-repo/.venv/bin/lbt
TOOLS=$W/lbt-runs/tools

export PYTHONPATH=$CODE/src
export PYTHONDONTWRITEBYTECODE=1
export OCM_NATIVE_ROOT=${D}/ocm_native NWW_ANALYSIS_ROOT=${D}/nww3_analysis OCM_RECONSTRUCTION_ROOT=${D}/reconstruction
export LBT_CHECKSUM_CACHE_DIR=$W/lbt-runs/checksum-cache-v5 LBT_BAYTRACE_ROOT=$W/lbt-runs/baytrace-v4.7.2
export NUMBA_CACHE_DIR=$W/lbt-runs/numba-cache-v5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export LBT_NATIVE_BACKTRACK_MODE=full_horizon_from_deposition LBT_NATIVE_FORMAL=1 LBT_NATIVE_SMAGORINSKY_IMPL=numba
# 擴散組態：明確寫出（與 physics_case 的預設值相同）。Kz 不用環境變數：設定檔是唯一來源。
export LBT_NATIVE_DIFFUSION_NOISE=uniform
export LBT_NATIVE_VERTICAL_DIFFUSION=constant
export LBT_NATIVE_VISSER=0
unset LBT_NATIVE_KZ_CONSTANT_M2PS

log() { echo "[$(date -u "+%F %T")Z] $*"; }
die() { log "中止：$*"; exit 2; }

# 最新的 v5 母體（prep_v5.sh 建立後才存在）。
latest_population() {
  ls -d "$V5ROOT"/*_final-sites_h30-h60-h90_common_m10_v5 2>/dev/null | tail -1
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
