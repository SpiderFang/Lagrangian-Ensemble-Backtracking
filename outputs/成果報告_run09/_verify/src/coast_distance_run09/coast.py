import os, json, numpy as np, pyarrow.parquet as pq, time
from shapely.geometry import shape, Point, box
from shapely.ops import transform, unary_union
import pyproj
t0=time.time()
R=os.path.expanduser("~/Workspace/lbt-runs/run09-h180-m10-v8/finite_depth_stokes")
OUT=os.path.expanduser("~/Workspace/lbt-runs/run09-h180-m10-v8/_coast_r3")
t=pq.read_table(R+"/postprocess/departures.parquet",columns=["study_site_id","final_status_code","departure_lon_deg","departure_lat_deg","departure_azimuth_sector_index","material_id","particle_id"]).to_pandas()
g=json.load(open(os.path.expanduser("~/Workspace/lbt-inputs/coastline-taiwan.geojson")))
lands=[shape(f["geometry"]) for f in g["features"]]
print("features",len(lands),[l.geom_type for l in lands][:5],flush=True)
res={}
for site,code in (("lienchiang",2),("nanwan",4)):
    d=t[(t.study_site_id==site)&(t.final_status_code==code)].reset_index(drop=True)
    lonc,latc=float(d.departure_lon_deg.mean()),float(d.departure_lat_deg.mean())
    pad=0.6
    bb=box(d.departure_lon_deg.min()-pad,d.departure_lat_deg.min()-pad,d.departure_lon_deg.max()+pad,d.departure_lat_deg.max()+pad)
    part=[l.intersection(bb) for l in lands if l.intersects(bb)]
    part=[p for p in part if not p.is_empty]
    U=unary_union(part)
    proj=pyproj.Transformer.from_crs("EPSG:4326","+proj=aeqd +lat_0=%f +lon_0=%f +units=m"%(latc,lonc),always_xy=True).transform
    Up=transform(proj,U)
    from shapely.prepared import prep
    P=prep(Up)
    print(site,"land parts",len(part),"area km2",Up.area/1e6,flush=True)
    key=d.departure_lon_deg.round(5).astype(str)+"_"+d.departure_lat_deg.round(5).astype(str)
    uniq={}
    for k,lo,la in zip(key,d.departure_lon_deg,d.departure_lat_deg):
        if k in uniq: continue
        x,y=proj(lo,la); p=Point(x,y)
        uniq[k]=(0.0 if P.contains(p) else Up.distance(p), P.contains(p))
    dist=np.array([uniq[k][0] for k in key]); inland=np.array([uniq[k][1] for k in key])
    d["dist_km"]=dist/1000; d["inland"]=inland
    d.to_csv(OUT+"/%s_status%d_dist.csv"%(site,code),index=False)
    print(site,"n",len(d),"unique pos",len(uniq),"time",time.time()-t0,flush=True)
