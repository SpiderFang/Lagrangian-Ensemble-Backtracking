# 交叉驗證：改用經緯度座標之最近點（shapely STRtree + 測地線距離 pyproj.Geod），與 coast.py 之等距方位投影平面距離比對。
import json, os, numpy as np, pandas as pd, pyproj
from shapely.geometry import shape, Point, box
from shapely.strtree import STRtree
from shapely.ops import nearest_points
g=json.load(open(os.path.expanduser("~/Workspace/lbt-inputs/coastline-taiwan.geojson")))
lands=[shape(f["geometry"]) for f in g["features"]]
geod=pyproj.Geod(ellps="WGS84")
for site,fn in (("lienchiang","lienchiang_status2_dist_sec.csv"),("nanwan","nanwan_status4_dist.csv")):
    d=pd.read_csv(fn)
    bb=box(d.departure_lon_deg.min()-0.6,d.departure_lat_deg.min()-0.6,d.departure_lon_deg.max()+0.6,d.departure_lat_deg.max()+0.6)
    ls=[l for l in lands if l.intersects(bb)]
    tree=STRtree(ls)
    dd=[]
    for lo,la in zip(d.departure_lon_deg,d.departure_lat_deg):
        p=Point(lo,la)
        if any(l.contains(p) for l in [ls[i] for i in tree.query(p.buffer(0.001))]): dd.append(0.0); continue
        i=tree.nearest(p); q=nearest_points(ls[i].boundary,p)[0]
        _,_,m=geod.inv(lo,la,q.x,q.y); dd.append(m/1000)
    dd=np.array(dd); diff=np.abs(dd-d.dist_km.values)
    print(site,"n",len(d),"max |diff| km %.4f  mean %.5f  n(diff>0.05km)=%d  inland_agree=%.4f"%(diff.max(),diff.mean(),(diff>0.05).sum(),((dd==0)==d.inland.values).mean()))
    print(" median %.4f p90 %.4f <=2 %.4f <=5 %.4f"%(np.median(dd),np.percentile(dd,90),(dd<=2).mean(),(dd<=5).mean()))
