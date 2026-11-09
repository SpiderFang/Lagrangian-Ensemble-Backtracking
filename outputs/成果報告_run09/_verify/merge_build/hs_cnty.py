import json, collections
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, polygonize, unary_union
d=json.load(open('/home/mustlab/ov_cnty.json'))
polys={}
for e in d['elements']:
    nm=e['tags'].get('name'); ls=[]
    for m in e['members']:
        if m['type']=='way' and m.get('role')=='outer' and 'geometry' in m:
            ls.append(LineString([(p['lon'],p['lat']) for p in m['geometry']]))
    mg=linemerge(ls); pl=list(polygonize(mg))
    polys[nm]=unary_union(pl); print(nm, len(ls), len(pl), polys[nm].area)
a=json.load(open('/home/mustlab/hs_arr.json'))
res=collections.defaultdict(collections.Counter); mx=0
for i,lo,la in a:
    p=Point(lo,la)
    ds={k:v.distance(p) for k,v in polys.items()}
    k=min(ds,key=ds.get)
    res[i+1][k]+=1; mx=max(mx,ds[k])
    if ds[k]>0.01: print('far',i+1,lo,la,ds)
print('max dist deg',mx)
tot=collections.Counter()
for s in sorted(res): print('C0001-%02d'%s, dict(res[s])); tot.update(res[s])
print(tot)
