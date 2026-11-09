# 共用資料載入：由 _verify/src 之 postprocess 副本、facts_remote_run09.json 與 WP-55 summary.json（native-arc-by-group 於 run09 產出）取得 run09 數字
import os, json, csv, math, re
import pandas as pd
ROOT=os.path.expanduser('~/mnt/Lagrangian-Ensemble-Backtracking/outputs/成果報告_run09')
SRC=ROOT+'/_verify/src'
CASE='finite_depth_stokes'
PP=SRC+'/postprocess/'+CASE
ARC=SRC+'/figures-arc-by-group'
FIGDIR='figures/'+CASE+'/無文字'          # md 內之圖檔相對路徑（相對 ROOT）
SITES=[('gongliao','貢寮'),('guishan','龜山島'),('hsinchu','新竹外海'),('nanwan','南灣'),('lienchiang','連江')]
SITE_ZH=dict(SITES)
SEC_ZH={'N':'北','NNE':'北北東','NE':'東北','ENE':'東北東','E':'東','ESE':'東南東','SE':'東南','SSE':'南南東','S':'南','SSW':'南南西','SW':'西南','WSW':'西南西','W':'西','WNW':'西北西','NW':'西北','NNW':'北北西'}
# 類別（依類別代表沉降速度由慢到快，與圖面排序相同）
CLASSES=[('oca_nonrecyclable_flexible_sheet','不可回收塑膠'),('oca_wastepaper_folded_fiber','廢紙'),('oca_styrofoam_porous_fragment','保麗龍'),('oca_fishinggear_open_mesh_bundle','廢漁網漁具'),('oca_pet_waterfilled_bottle','寶特瓶'),('oca_wood_waterlogged_elongated','竹木'),('oca_other_recyclable_irregular_fragment','可回收塑膠'),('oca_aluminum_crushed_cylinder','鋁罐'),('oca_steel_rigid_cylinder','鐵罐'),('oca_glass_bottle_or_fragment','玻璃瓶')]
CLASS_ZH=dict(CLASSES)
EDGES=[('south','南'),('east','東'),('north','北'),('west','西')]
EDGE_ZH=dict(EDGES)
HORIZON_DAYS=180
def csvp(name): return pd.read_csv(f'{PP}/{name}.csv')
FA=json.load(open(SRC+'/facts_remote_run09.json'))
def arc_summary(site,g):
    return json.load(open(f'{ARC}/tables/{site}/{g}_summary.json'))
def arc_curves(site,g):
    rows=list(csv.DictReader(open(f'{ARC}/tables/{site}/{g}_curves.csv',encoding='utf-8')))
    cols=[c for c in rows[0] if c!='x_km']
    return [float(r['x_km']) for r in rows],{c.split('__')[1]:[float(r[c]) for r in rows] for c in cols}
def sampling_radius_km():
    m=json.load(open(SRC+'/figures-notext/manifest.json'))
    rs={s:m['footprint_figures']['receptors']['by_site'][s]['release_geometry']['sampling_range']['radius_m'] for s,_ in SITES}
    assert len(set(rs.values()))==1,rs
    return list(rs.values())[0]/1000.0
def three_sig(v): return f'{v:#.3g}'.rstrip('.')
def pct(x,d=3): return f'{x*100:.{d}f}%'
def i0(x): return f'{x:,.0f}'
