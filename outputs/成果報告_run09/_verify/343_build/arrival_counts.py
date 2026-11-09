# 由 run09 母體 manifest（arrival.json）計算各地點 50 個到達時間之分類數量（供核對程式與圖內數字比對）
import sys, os, json
sys.path.insert(0, os.path.expanduser("~/Workspace/lbt-repo-wp63/scripts/wp61"))
sys.path.insert(0, os.path.expanduser("~/Workspace/lbt-repo-wp63/src"))
import arrival_check as ac, arrival_figs as af
recs = ac.load_manifest_records()
out = {}
for site in ac.SITES:
    S = ac.load_series(site)
    rows = ac.analyze_site(site, recs[site], S, None)["rows"]
    c = af._ov_counts(rows)
    out[site] = {"total": c["total"], "class": c["class"], "phase": c["phase"], "event": c["event"],
                 "season_class": {f"{k[0]}|{k[1]}": v for k, v in c["season_class"].items()}}
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False)[:600])
