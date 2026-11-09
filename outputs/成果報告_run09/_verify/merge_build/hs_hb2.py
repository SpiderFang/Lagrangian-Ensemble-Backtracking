import sys,json
sys.path.insert(0,'/home/mustlab/Workspace/lbt-wp75/fullpage'); sys.path.insert(0,'/home/mustlab/Workspace/lbt-wp75/combined_proto_src3')
import numpy as np
import cp_lib as C
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union
S=C.L0.prep_site('hsinchu'); sd=S.sd; cc=S.ccount['all']
j=json.load(open('/home/mustlab/ov_hb_my.json'))
want={'w193191982','n4107702311','w204884668','w204884666','r7397596','n4157596118','n4157596590'}
def tp(lo,la):
    x,y=sd.proj.project(np.array(lo,float),np.array(la,float)); return list(zip(x,y))
for e in j['elements']:
    k=e['type'][0]+str(e['id'])
    if k not in want: continue
    if e['type']=='node': g=Point(*tp([e['lon']],[e['lat']])[0])
    elif e['type']=='way': g=LineString(tp([p['lon'] for p in e['geometry']],[p['lat'] for p in e['geometry']]))
    else:
        ls=[]
        for m in e['members']:
            if 'geometry' in m and len(m['geometry'])>1: ls.append(LineString(tp([p['lon'] for p in m['geometry']],[p['lat'] for p in m['geometry']])))
        g=unary_union(ls)
    best=[]
    for i,s in enumerate(sd.cseg):
        d=s.geom.distance(g)
        best.append((d/1000,s.seg_id,int(cc[i])))
    best.sort()
    near=[(round(d,2),sid,c) for d,sid,c in best[:3]]
    big=[(round(d,2),sid,c) for d,sid,c in best if c>=100][:1]
    print(k,e['tags'].get('name'),'nearest3',near,'nearest>=100',big)
