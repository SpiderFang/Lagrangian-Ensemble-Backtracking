#!/usr/bin/env python3
"""WP-53：v6 母體檢查（只讀）。

輸出每站：記錄數、獨立沉底時刻數、因缺時被拒絕的年齡候選數（rejection audit）、gap-unsafe（跨缺口／被缺口終止）記錄數、
沉底標籤 unknown 數（潮汐類別、潮汐相位）、強流／高浪缺值數；另核對每站受體數、十檔沉降速度與 v5 逐字相同、
觀測錨點與 v5 的重疊、初始條件時間來源（ocm_time_origin）計數與缺口清單是否為空。
用法：check_population_v6.py <v6 母體> [<v5 母體> [<v6 reconstruction 根>]]
第三個參數存在時，另統計觀測錨點與沉底時刻落在「重建小時」（patch 時間軸）的記錄數。
"""
import collections, json, sys
from pathlib import Path


def load(root, name):
    return json.loads((Path(root) / "common-input" / name).read_text(encoding="utf-8"))


def main() -> int:
    v6 = sys.argv[1]
    v5 = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    recon_root = sys.argv[3] if len(sys.argv) > 3 else None
    arr = load(v6, "arrival.json")
    gap = load(v6, "ocm_gap_safe_arrival.json")
    rec = load(v6, "receptor.json")["records"]
    mat = load(v6, "material.json")["records"]
    ini = load(v6, "initial_conditions.json")["records"]
    out = {"records_total": len(arr["records"])}
    per = collections.defaultdict(lambda: collections.Counter())
    uniq = collections.defaultdict(set)
    obs = collections.defaultdict(set)
    for r in arr["records"]:
        s = r["study_site_id"]
        m = r["metadata"]
        per[s]["records"] += 1
        uniq[s].add(m["deposition_time_utc_ns"])
        obs[s].add(m["observation_time_utc_ns"])
        per[s]["rejected_age_hours"] += len(m["bed_residence_availability_rejection_audit"].get("rejected_age_hours", []))
        per[s]["rejection_count"] += int(m["bed_residence_availability_rejection_audit"].get("rejection_count", 0))
        if m.get("deposition_tide_class") in (None, "unknown"):
            per[s]["unknown_tide_class"] += 1
        if m.get("deposition_tide_phase") in (None, "unknown"):
            per[s]["unknown_tide_phase"] += 1
        if m.get("deposition_strong_current") is None:
            per[s]["strong_current_missing"] += 1
        if m.get("deposition_high_wave") is None:
            per[s]["high_wave_missing"] += 1
        if not m.get("deposition_wave_available", True):
            per[s]["wave_unavailable"] += 1
    gp = collections.defaultdict(lambda: collections.Counter())
    arr_site = {r["arrival_time_id"]: r["study_site_id"] for r in arr["records"]}
    for g in gap["records"]:
        s = arr_site.get(g["arrival_time_id"], "?")
        for k in ("censoring_required", "censored_by_data_gap", "crossed_gap", "selection_censoring_required", "selection_censored_by_data_gap", "selection_crossed_gap"):
            if g.get(k):
                gp[s][k] += 1
        if g.get("missing_utc"):
            gp[s]["records_with_missing_utc"] += 1
        if g.get("deposition_time_available") is False:
            gp[s]["deposition_unavailable"] += 1
    rc = collections.Counter(x.get("study_site_id") for x in rec)
    iorigin = collections.Counter(x.get("ocm_time_origin") for x in ini)
    out["per_site"] = {s: {**dict(per[s]), "unique_deposition_times": len(uniq[s]), "gap_safe": dict(gp[s]), "receptors": rc[s]} for s in sorted(per)}
    out["initial_conditions_ocm_time_origin"] = dict(iorigin)
    out["initial_conditions_records"] = len(ini)
    out["settling_velocity_mps"] = {x["material_id"]: x["settling_velocity_mps"] for x in mat}
    out["gap_manifest"] = {"status": gap.get("status"), "censoring_required_records": gap["provenance"].get("censoring_required_records"),
                           "selection_censoring_required_records": gap["provenance"].get("selection_censoring_required_records"),
                           "policy": gap.get("policy")}
    if recon_root:
        import numpy as np
        rec_ns = set()
        first = sorted(p.name for p in Path(recon_root).iterdir() if p.is_dir())[0]
        for mdir in (Path(recon_root) / first / "months").iterdir():
            if mdir.name >= "202407" and (mdir / "time_utc_ns.npy").is_file():
                rec_ns.update(int(t) for t in np.load(mdir / "time_utc_ns.npy"))
        out["reconstructed_hours_in_patch"] = len(rec_ns)
        on_obs = collections.Counter(); on_dep = collections.Counter()
        for r in arr["records"]:
            m = r["metadata"]
            on_obs[r["study_site_id"]] += int(m["observation_time_utc_ns"] in rec_ns)
            on_dep[r["study_site_id"]] += int(m["deposition_time_utc_ns"] in rec_ns)
        out["records_with_observation_anchor_on_reconstructed_hour"] = dict(on_obs)
        out["records_with_deposition_hour_on_reconstructed_hour"] = dict(on_dep)
    if v5:
        m5 = load(v5, "material.json")["records"]
        out["settling_equal_to_v5"] = [json.dumps(a, sort_keys=True) for a in mat] == [json.dumps(a, sort_keys=True) for a in m5]
        a5 = load(v5, "arrival.json")["records"]
        o5 = collections.defaultdict(set)
        for r in a5:
            o5[r["study_site_id"]].add(r["metadata"]["observation_time_utc_ns"])
        out["observation_anchor_overlap_with_v5"] = {s: len(obs[s] & o5[s]) for s in sorted(obs)}
        out["observation_anchor_count"] = {s: len(obs[s]) for s in sorted(obs)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
