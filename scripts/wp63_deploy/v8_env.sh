#!/bin/bash
# WP-63：run09（v8：回溯 180 日；只跑含斯托克斯漂流的 finite_depth_stokes 案例）共用的環境與檢查函式。
# 只能被 prep_v8.sh、run_formal_v8.sh 與 wait_and_run_v8.sh 以 source 載入；本檔本身不建立任何檔案或目錄。
#
# 與 wp54_deploy/v7_env.sh 的差異：
#   * 資料根只使用 lbt-local-data-v6、-v6-full、-v7-full、-v7-surface-fix202511-12 與 NAS（不引用已淘汰的舊資料根）；
#   * 執行期：OCM_NATIVE_ROOT=v7-full/ocm_native、NWW_ANALYSIS_ROOT=v7-full/nww3_analysis（202407 起硬連結 v6 實檔，
#     加補搬的 2024-04～06 與 NWW 2024-01～06）、OCM_RECONSTRUCTION_ROOT=v6/reconstruction；
#   * 母體建置：APRIME_OCM=v6-full/ocm_native（讀標頭與時間軸；沉底整點的 zcor/elev/wetdry 數值也由此讀，
#     v6-full 在 202410 以後是實檔，更早月份是稀疏占位檔，建置程式與 check_run09_data_roots.py 都有 fail-closed 守門）、
#     APRIME_SURFACE=v7-surface-fix202511-12/ocm_surface、nww3_analysis 讀 NAS 的 24 個月版本；
#   * 母體／輸出位置：v8-run09、run09-h180-m10-v8；獨立的 checksum／numba 快取；
#   * 單一案例 finite_depth_stokes，worker 預設 52（見 run_formal_v8.sh 的記憶體估計）。
# 不修改 run04～run08 使用的任何檔案。

W=$HOME/Workspace
CODE=${LBT_CODE:-$W/lbt-repo-wp63}
B=${LBT_B:-$W/lbt-runs/run09-h180-m10-v8}
V8ROOT=${LBT_V8ROOT:-$W/lbt-inputs/formal/v8-run09}
TEMPLATE=${LBT_TEMPLATE:-$W/lbt-inputs/formal/template-v8-run09.yaml}
V7TEMPLATE=$W/lbt-inputs/formal/template-v7-class-velocity.yaml
D6=$W/lbt-local-data-v6
D7=$W/lbt-local-data-v7-full
D=$D7                                       # 等深線／水深格點（grid）讀取用
APRIME_OCM=${LBT_APRIME_OCM:-$W/lbt-local-data-v6-full/ocm_native}
APRIME_SURFACE=$W/lbt-local-data-v7-surface-fix202511-12/ocm_surface
# 母體建置的 NWW 根：預設用本機 v7-full（與執行期同一份；各檔大小與 NAS 逐檔一致）。NAS 讀取實測只有約 10 MB/s，
# 且與 v7-full 的 OCM 搬運共用頻寬，母體建置會拖到數小時。需要改回 NAS 時設 LBT_APRIME_NWW=<NAS 路徑>。
APRIME_NWW=${LBT_APRIME_NWW:-$D7/nww3_analysis}
REQUIRED_COMMIT=ea97469230231a999dd4f81bfaff128850d1e15a
COAST=$W/lbt-inputs/coastline-taiwan.geojson
PY=$W/lbt-repo/.venv/bin/python
LBT=$W/lbt-repo/.venv/bin/lbt
TOOLS=$W/lbt-runs/tools
CASE=finite_depth_stokes
HORIZON_DAYS=180
MAX_AGE_DAYS=90
# 單案例磁碟估計（GB）：run07 實測 H90 每案 checkpoints 約 28、results 約 6；H180 約兩倍，保守取 80。
EST_GB_PER_CASE=80
EST_GB_SHARED=10
MIN_FREE_GB=${LBT_MIN_FREE_GB:-150}

export PYTHONPATH=$CODE/src
export PYTHONDONTWRITEBYTECODE=1
export OCM_NATIVE_ROOT=$D7/ocm_native NWW_ANALYSIS_ROOT=$D7/nww3_analysis OCM_RECONSTRUCTION_ROOT=$D6/reconstruction
export LBT_CHECKSUM_CACHE_DIR=${LBT_CHECKSUM_CACHE_DIR_OVERRIDE:-$W/lbt-runs/checksum-cache-v8} LBT_BAYTRACE_ROOT=$W/lbt-runs/baytrace-v4.7.2
export NUMBA_CACHE_DIR=${LBT_NUMBA_CACHE_DIR_OVERRIDE:-$W/lbt-runs/numba-cache-v8}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export LBT_NATIVE_BACKTRACK_MODE=full_horizon_from_deposition LBT_NATIVE_FORMAL=1 LBT_NATIVE_SMAGORINSKY_IMPL=numba
export LBT_NATIVE_DIFFUSION_NOISE=uniform
export LBT_NATIVE_VERTICAL_DIFFUSION=constant
export LBT_NATIVE_VISSER=0
unset LBT_NATIVE_KZ_CONSTANT_M2PS

log() { echo "[$(date -u "+%F %T")Z] $*"; }
die() { log "中止：$*"; exit 2; }

latest_population() {
  ls -d "$V8ROOT"/*_final-sites_h180_common_m10_v8 2>/dev/null | tail -1
}

require_disk() {
  local ncase=${1:-1}
  local need=$(( EST_GB_SHARED + EST_GB_PER_CASE * ncase ))
  [ "$need" -lt "$MIN_FREE_GB" ] && need=$MIN_FREE_GB
  local avail
  avail=$(df -BG --output=avail "$W" | tail -1 | tr -dc '0-9')
  log "磁碟：$W 所在檔案系統可用 ${avail} GB；本次估計需求 ${need} GB（案例數 $ncase）"
  [ "${avail:-0}" -ge "$need" ] || die "磁碟可用空間 ${avail} GB 小於需求 ${need} GB"
}

require_clean_code() {
  [ -e "$CODE/.git" ] || die "$CODE 不是 git clone"
  [ -z "$(git -C "$CODE" status --porcelain)" ] || die "$CODE 工作樹不乾淨（git status 有變更）"
  [ -f "$CODE/src/lagrangian_backtracking/bad_hour_rule.py" ] && [ -f "$CODE/src/lagrangian_backtracking/seam_jumps.py" ] \
    && [ -f "$CODE/src/lagrangian_backtracking/receptor_stratified.py" ] \
    || die "$CODE 不含 WP-63 新增模組（bad_hour_rule／seam_jumps／receptor_stratified）"
  git -C "$CODE" merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD || die "$CODE 的 HEAD 不含 ${REQUIRED_COMMIT:0:7}"
  [ -x "$PY" ] && [ -x "$LBT" ] || die "找不到 $W/lbt-repo/.venv 的 python／lbt"
  "$PY" -c "import sys, lagrangian_backtracking as m; sys.exit(0 if m.__file__.startswith('$CODE/src') else 1)" \
    || die "import lagrangian_backtracking 沒有來自 $CODE/src"
  [ -d "$LBT_BAYTRACE_ROOT" ] || die "找不到 BayTrace 根 $LBT_BAYTRACE_ROOT"
}

# reconstruction 根檢查（與 v7_env.sh 的 require_v6_data_root 相同的驗證；根改為 $D6/reconstruction）。
require_reconstruction_root() {
  if [ "${LBT_SKIP_DATA_ROOT_CHECK:-0}" = 1 ]; then log "注意：略過 reconstruction 根的完整驗證（LBT_SKIP_DATA_ROOT_CHECK=1，僅限試跑）"; return 0; fi
  [ -d "$D6/reconstruction" ] || die "找不到 $D6/reconstruction"
  "$PY" - <<PYEOF || die "reconstruction 根檢查失敗（見上方訊息）"
import json, os, sys
from pathlib import Path
from lagrangian_backtracking.ocm_reconstruction import validate_reconstruction_patch
root = Path(os.environ["OCM_RECONSTRUCTION_ROOT"])
json.loads((root / "reconstruction-index.json").read_text(encoding="utf-8"))
domains = sorted(p.name for p in root.iterdir() if p.is_dir())
if len(domains) != 4:
    print("reconstruction 根的 domain 數不是 4：", domains, file=sys.stderr); sys.exit(1)
for d in domains:
    manifest = json.loads((root / d / "reconstruction-manifest.json").read_text(encoding="utf-8"))
    if "ocm_harmonic_residual_bridge_bounded_v2" not in json.dumps(manifest, ensure_ascii=False):
        print(d, "manifest 不含 v2 方法識別字", file=sys.stderr); sys.exit(1)
    validate_reconstruction_patch(root / d)
print("reconstruction 根檢查通過：", domains)
PYEOF
}

# 資料根檢查：$1 = roots 的額外參數（例如 --aprime-only）
require_data_roots() {
  "$PY" "$CODE/scripts/wp63_deploy/check_run09_data_roots.py" roots \
    --runtime-ocm "$OCM_NATIVE_ROOT" --runtime-nww "$NWW_ANALYSIS_ROOT" \
    --aprime-ocm "$APRIME_OCM" --aprime-surface "$APRIME_SURFACE" "$@" || die "資料根檢查失敗"
}

# 核對 v8 設定檔：run09 的所有關鍵設定（$1 = 路徑）。
require_v8_config() {
  local cfg=$1
  [ -f "$cfg" ] || die "找不到設定檔 $cfg"
  "$PY" - "$cfg" <<'PYEOF' || die "v8 設定檢查失敗（見上方訊息）"
import sys
from lagrangian_backtracking.config import load_config
from lagrangian_backtracking.scenarios import BASELINE_BEHAVIORS, SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED

c = load_config(sys.argv[1], formal_release=False)
bad = []
if c.inputs.backtrack_support_days != 270: bad.append(f"backtrack_support_days={c.inputs.backtrack_support_days}")
if c.arrival_time_selection.pairs_northeast_utc is not False: bad.append("northeast_pair_utc 未關閉")
if c.arrival_time_selection.seam_jump_exclusion is None: bad.append("缺少 seam_jump_exclusion")
if c.scenarios.horizontal_receptor_selection_policy != "azimuth_distance_equal_area_latin_stratified_v1": bad.append("受體 policy 不是拉丁方格分層")
s = c.scenarios.horizontal_receptor_sampling
if s is None or float(s.radius_m) != 10000.0: bad.append("抽樣半徑不是 10 km")
elif s.azimuth_sectors_for("guishan") != ((0.0, 180.0, 2), (180.0, 360.0, 3)): bad.append("龜山島方位分配不是西 3 東 2")
sites = {x.study_site_id: x for x in c.study_sites}
if tuple(sites["hsinchu"].anchor_lonlat) != (120.45, 24.75): bad.append("新竹錨點被改動")
if tuple(sites["guishan"].anchor_lonlat) != (121.939148, 24.850326): bad.append("龜山島錨點被改動")
if tuple(sites["nanwan"].anchor_lonlat) != (120.770675, 21.92041) or sites["nanwan"].anchor_source != "config_override_v1": bad.append("南灣錨點不是 (120.770675, 21.92041)")
if tuple(sites["gongliao"].anchor_lonlat) != (121.9223889, 25.0964444) or tuple(sites["lienchiang"].anchor_lonlat) != (119.95, 26.2): bad.append("貢寮或連江錨點被改動")
bed = c.scenarios.bed_residence_time
if bed.age_sampling_scope != "per_member" or bed.maximum_age_days != 90: bad.append("沉底年齡設定不是 per_member／90 日")
settling = c.physics["settling"]
if settling.get("velocity_sampling") != SETTLING_VELOCITY_SAMPLING_CLASS_CANDIDATES_BALANCED: bad.append("沉降速度抽樣不是 class_candidates_balanced")
by_id = {m["material_id"]: m for m in settling["material_classes"]}
for b in BASELINE_BEHAVIORS:
    m = by_id.get(b.material_id)
    if m is None or tuple(float(x) for x in m["settling_velocity_candidates_mps"]) != tuple(b.settling_velocity_candidates_mps):
        bad.append(f"{b.material_id} 候選速度與程式內建不一致")
if bad:
    print("v8 設定檢查失敗：", "；".join(bad), file=sys.stderr); sys.exit(1)
print("v8 設定檢查通過：H180（支援窗 270 日）、貢寮／龜山島各自選時刻、接縫排除、拉丁方格 10 km（龜山島西 3 東 2）、南灣錨點覆寫、十類各 10 個候選速度")
PYEOF
}

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
expected = {"diffusion_noise": "uniform", "vertical_diffusion": "constant", "kz_constant_m2ps": 0.0003,
            "visser_correction": False, "kz_source": "config"}
bad = {k: (decl.get(k), v) for k, v in expected.items() if decl.get(k) != v}
if bad:
    print("擴散組態與預期不符（實際, 預期）：" + json.dumps(bad, ensure_ascii=False), file=sys.stderr)
    sys.exit(3)
PYEOF
}

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
    echo "資料根（執行期）：OCM_NATIVE_ROOT=$OCM_NATIVE_ROOT NWW_ANALYSIS_ROOT=$NWW_ANALYSIS_ROOT OCM_RECONSTRUCTION_ROOT=$OCM_RECONSTRUCTION_ROOT"
    echo "資料根（母體建置）：APRIME_OCM=$APRIME_OCM APRIME_SURFACE=$APRIME_SURFACE APRIME_NWW=$APRIME_NWW"
    echo "擴散組態：$(resolve_and_check_diffusion)"
    echo "---"
  } >> "$out"
}
