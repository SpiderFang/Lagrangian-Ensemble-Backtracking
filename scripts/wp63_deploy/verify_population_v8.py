#!/usr/bin/env python3
"""run09 母體核對（唯讀）。任何一項不符就以非零離開碼結束，並把逐項結果寫入 --output JSON。

核對項目（對應協調者要求）：
  1. 每地點 50 個觀測到達時刻（每個觀測 M=10 個成員 -> 500 筆 arrival）；龜山島與貢寮的觀測時刻不同；
  2. 受體：每地點 20 個（5 水平 x 4 垂向），水平位置符合分層紀錄（距離 <= 10 km、方位／距離區間、龜山島西半 3 個、東半 2 個），
     manifest 記錄每地點的方位分界；
  3. 觀測時刻不落在接縫排除清單上；沉底時刻不落在壞時刻／NWW 不可支撐的排除清單上，也不落在母體 OCM 時間軸缺漏的整點上
     （執行期 OCM 時間軸缺漏的整點會由 reconstruction patch 補，僅計數回報）；
  4. 粒子數：身分表每地點 100,000 顆（共 500,000），每個（地點, 類別）的每個候選速度各 1,000 顆；
  5. 輸入數值讀取不碰占位檔：沉底整點所在月份的母體 OCM zcor/elev/wetdry_elem 全是實檔，
     且（母體 provenance）沒有任何沉底整點在 2024-10-01 以前。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check_roots", HERE / "check_run09_data_roots.py")
check_roots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_roots)

SITES = ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang")
HOUR = 3_600_000_000_000


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", required=True)
    ap.add_argument("--identity-table", required=True)
    ap.add_argument("--aprime-ocm", required=True)
    ap.add_argument("--runtime-ocm", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--members", type=int, default=10)
    a = ap.parse_args()
    P = Path(a.population) / "common-input"
    arrival = json.load(open(P / "arrival.json", encoding="utf-8"))
    receptor = json.load(open(P / "receptor.json", encoding="utf-8"))
    checks: list[dict] = []

    def rec(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(("通過 " if ok else "失敗 ") + name + "：" + (json.dumps(detail, ensure_ascii=False, default=str)[:600]))

    # 1. 到達時刻
    obs_by_site, dep_by_site = defaultdict(list), defaultdict(list)
    for r in arrival["records"]:
        obs_by_site[r["study_site_id"]].append(int(r["metadata"]["observation_time_utc_ns"]))
        dep_by_site[r["study_site_id"]].append(int(r["time_utc_ns"]))
    uniq_obs = {s: sorted(set(obs_by_site[s])) for s in SITES}
    rec("每地點 50 個觀測到達時刻、500 筆 arrival",
        all(len(uniq_obs[s]) == 50 and len(obs_by_site[s]) == 50 * a.members for s in SITES),
        {s: (len(uniq_obs[s]), len(obs_by_site[s])) for s in SITES})
    same = set(uniq_obs["gongliao"]) & set(uniq_obs["guishan"])
    rec("龜山島與貢寮的觀測時刻不同（獨立選取）", uniq_obs["gongliao"] != uniq_obs["guishan"],
        {"相同的觀測時刻數": len(same), "貢寮": 50, "龜山島": 50})
    # 3. 排除清單
    seam = find_key(arrival.get("provenance", {}), "seam_jump_exclusion")
    seam_ok, seam_detail = seam is not None, {}
    if seam is not None:
        ex = seam.get("excluded_times_by_site") or {}
        union = set()
        for s in SITES:
            union |= {int(x) for x in (ex.get(s) or {}).get("utc_ns", [])}
        hit = {s: len(set(uniq_obs[s]) & union) for s in SITES}
        seam_ok = all(v == 0 for v in hit.values())
        seam_detail = {"清單大小": len(union), "觀測時刻落入清單": hit, "sha256": seam.get("excluded_times_sha256_all_sites"), "各站排除筆數": {s: (ex.get(s) or {}).get("count") for s in SITES}}
    rec("到達時刻不落在接縫跳動排除清單", seam_ok, seam_detail or "provenance 沒有 seam_jump_exclusion")
    support = find_key(arrival.get("provenance", {}), "deposition_support_conditioning")
    sup_ok, sup_detail = support is not None, {}
    if support is not None:
        for s in SITES:
            ex = {int(x) for x in support["by_site"][s]["excluded_hours_utc_ns"]}
            ext = ((support.get("external_exclusion") or {}).get("by_site") or {}).get(s, {}).get("excluded_hours_utc_ns", [])
            hit = len(set(dep_by_site[s]) & ex)
            sup_detail[s] = {"排除整點": len(ex), "其中壞時刻規則": len(ext), "沉底落入排除": hit}
            sup_ok &= hit == 0
        sup_detail["external_exclusion_policy"] = (support.get("external_exclusion") or {}).get("policy_id")
    rec("沉底時刻不落在壞時刻／不可支撐排除清單", sup_ok and support is not None, sup_detail or "provenance 沒有 deposition_support_conditioning")
    # 沉底整點與母體 OCM 時間軸
    first_dep = min(min(v) for v in dep_by_site.values())
    rec("沉底時刻不早於 2024-10-01（母體 OCM 實檔月份）",
        first_dep >= int(datetime(2024, 10, 1, tzinfo=UTC).timestamp()) * 1_000_000_000,
        {"最早沉底": str(np.datetime64(first_dep, "ns"))})
    dep_months = sorted({f"{datetime.fromtimestamp(t / 1e9, tz=UTC):%Y%m}" for v in dep_by_site.values() for t in v})
    domains = check_roots.DOMAINS
    # 母體 OCM 時間軸缺漏／執行期時間軸缺漏
    miss_aprime, miss_runtime = {}, {}
    all_dep = np.unique(np.concatenate([np.asarray(v, dtype=np.int64) for v in dep_by_site.values()]))
    for d in domains:
        ax_a, ax_r = [], []
        for mon in dep_months:
            for root, ax in ((a.aprime_ocm, ax_a), (a.runtime_ocm, ax_r)):
                f = Path(root) / d / "months" / mon / "time_utc_ns.npy"
                if f.is_file():
                    ax.append(np.load(f))
        ta = np.unique(np.concatenate(ax_a)) if ax_a else np.array([], dtype=np.int64)
        tr = np.unique(np.concatenate(ax_r)) if ax_r else np.array([], dtype=np.int64)
        miss_aprime[d] = int((~np.isin(all_dep, ta)).sum())
        miss_runtime[d] = int((~np.isin(all_dep, tr)).sum())
    rec("沉底時刻都在母體 OCM 時間軸內", all(v == 0 for v in miss_aprime.values()), miss_aprime)
    print("   （回報）沉底時刻不在執行期 OCM 時間軸內的整點數（由 reconstruction patch 補）：", miss_runtime)
    # 5. 占位檔
    errors: list[str] = []
    for d in domains:
        for mon in dep_months:
            check_roots.check_arrays(Path(a.aprime_ocm) / d / "months" / mon, check_roots.OCM_VALUE_ARRAYS, errors, f"{d}/{mon}")
    rec("沉底整點所在月份的母體 OCM 數值陣列都是實檔（未碰占位檔）", not errors, {"月份": dep_months, "錯誤": errors[:5]})
    # 2. 受體
    recs = receptor["records"]
    per_site = Counter(r["study_site_id"] for r in recs)
    rec("受體 100 個、每地點 20 個", len(recs) == 100 and all(per_site[s] == 20 for s in SITES), dict(per_site))
    horiz = defaultdict(dict)
    for r in recs:
        horiz[r["study_site_id"]][r["metadata"]["source_face_global_index"]] = (r["lon"], r["lat"], r["metadata"]["horizontal_selection_draw_order"])
    rec("每地點 5 個水平位置（各 4 個垂向）", all(len(horiz[s]) == 5 for s in SITES) and all(
        Counter(r["metadata"]["source_face_global_index"] for r in recs if r["study_site_id"] == s).most_common(1)[0][1] == 4 for s in SITES),
        {s: len(horiz[s]) for s in SITES})
    by_site = find_key(receptor.get("provenance", {}), "random_horizontal_receptors_by_site") or {}
    strat = {s: (by_site.get(s) or {}).get("stratified_sampling") for s in SITES}
    rec("每地點都有分層抽樣稽核（含方位分界、距離分界、排列）", all(v is not None for v in strat.values()),
        {s: (None if v is None else {"azimuth_edges_deg": [round(x, 2) for x in v["azimuth_edges_deg"]], "distance_edges_m": [round(x) for x in v["distance_edges_m"]], "排列": v["permutation_azimuth_bin_to_distance_bin"], "嘗試": v["permutation_attempts"]}) for s, v in strat.items()})
    if all(v is not None for v in strat.values()):
        rec("受體距錨點都 <= 10 km、方位區間與距離區間各 1 個",
            all(v["radius_m"] == 10000.0 and all(p["distance_m"] <= 10000.0 + 1e-6 for p in v["picks"])
                and sorted(p["azimuth_bin"] for p in v["picks"]) == [1, 2, 3, 4, 5]
                and sorted(p["distance_bin"] for p in v["picks"]) == [1, 2, 3, 4, 5] for v in strat.values()), "ok")
        g = strat["guishan"]
        west = sum(1 for p in g["picks"] if p["azimuth_deg"] >= 180.0)
        rec("龜山島西半（方位 180°–360°）3 個受體、東半 2 個", west == 3 and len(g["picks"]) - west == 2,
            {"西半": west, "azimuth_sectors": g.get("azimuth_sectors"), "azimuth_edges_deg": g["azimuth_edges_deg"]})
        rec("其他地點為隨機起始方位的 5 等分（無扇區分配）", all("azimuth_sectors" not in strat[s] for s in SITES if s != "guishan"), "ok")
    # 4. 粒子
    t = np.load(a.identity_table, allow_pickle=True)
    site = t["study_site_id"].astype(str)
    mat = t["material_id"].astype(str)
    idx = t["settling_candidate_index"].astype(np.int64)
    rec("身分表 500,000 顆、每地點 100,000 顆", site.size == 500000 and all(int((site == s).sum()) == 100000 for s in SITES),
        {s: int((site == s).sum()) for s in SITES})
    ok = True
    for s in SITES:
        for m in sorted(set(mat.tolist())):
            c = np.bincount(idx[(site == s) & (mat == m)], minlength=10)
            ok &= bool(np.all(c == 1000))
    rec("每地點每類別每個候選速度各 1,000 顆", ok, "10 類 x 10 速度 x 5 地點")
    Path(a.output).write_text(json.dumps({"population": str(a.population), "checks": checks}, ensure_ascii=False, indent=1))
    failed = [c["check"] for c in checks if not c["ok"]]
    if failed:
        print("母體核對未通過：", failed, file=sys.stderr)
        return 1
    print(f"母體核對全數通過（{len(checks)} 項）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
