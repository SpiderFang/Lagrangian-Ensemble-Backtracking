# 連江（final_status_code=2，達回溯上限）與南灣（=4，觸底）未穿越邊界者之最終位置距海岸線距離之統計。
# 輸入：coast.py 輸出之 *_dist.csv（海岸線＝報告底圖 coastline-taiwan.geojson；距離為至陸地多邊形之最短距離，位於陸地範圍內者以 0 計）。
# 方位區間：自錨點（figures-polar-v2 manifest anchors_lonlat）起算之大圓初始方位，每 10° 一區（sector=round(az/10)%36），與圓形圖相同。
import pandas as pd, numpy as np, json, os
m=json.load(open(os.path.expanduser("~/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes/figures-polar-v2/manifest.json")))
lon0,lat0=m["anchors_lonlat"]["lienchiang"]
d=pd.read_csv("lienchiang_status2_dist.csv")
p1=np.radians(lat0); p2=np.radians(d.departure_lat_deg.values); dl=np.radians(d.departure_lon_deg.values-lon0)
az=(np.degrees(np.arctan2(np.sin(dl)*np.cos(p2), np.cos(p1)*np.sin(p2)-np.sin(p1)*np.cos(p2)*np.cos(dl)))+360)%360
d["sec"]=(np.round(az/10).astype(int))%36
d.to_csv("lienchiang_status2_dist_sec.csv",index=False)
k=d.dist_km.values
out={}
for nm,mk in (("all",np.ones(len(d),bool)),("sector26_255_265",(d.sec==26).values),("other",(d.sec!=26).values)):
    kk=k[mk]; out[nm]=dict(n=int(mk.sum()),median_km=float(np.median(kk)),p90_km=float(np.percentile(kk,90)),max_km=float(kk.max()),n_le_2km=int((kk<=2).sum()),n_le_5km=int((kk<=5).sum()),frac_le_2km=float((kk<=2).mean()),frac_le_5km=float((kk<=5).mean()),n_inside_land_polygon=int(d.inland.values[mk].sum()))
n=pd.read_csv("nanwan_status4_dist.csv"); out["nanwan_status4"]=dict(n=len(n),dist_km=[float(x) for x in n.dist_km])
json.dump(out,open("coast_distance_summary.json","w"),indent=1,ensure_ascii=False); print(json.dumps(out,indent=1,ensure_ascii=False))
