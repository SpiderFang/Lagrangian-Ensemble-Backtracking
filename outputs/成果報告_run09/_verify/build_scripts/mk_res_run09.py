import sys, os, json, math, csv, re
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from d09 import *
from loc import to_loc
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
OUT=ROOT
# ------------------------------------------------------------------ 百分比格式：一律取小數二位（四捨五入）；無法以二位小數表達者（實際不為 0 卻捨入為 0.00%，或實際不為 100 卻捨入為 100.00%）保留三位
KEEP3=[]
def q(v,nd): return str(Decimal(f'{v:.10f}').quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
def ppp(v):
    """v 為百分數（例如 58.35）。"""
    s=q(v,2)
    if (v>1e-9 and Decimal(s)==0) or (v<100-1e-9 and Decimal(s)==100):
        s=q(v,3); assert Decimal(s)!=0 and Decimal(s)!=100,(v,s); KEEP3.append(s+'%')
    return s+'%'
def fixsp(t): return re.sub(r'(\d(?:,\d{3})* (?:km²|km|m/s|m|h|d))(?=[一-鿿])',r'\1 ',t)
L=[]; A=L.append
figno=[0]
def cap_f(text):
    figno[0]+=1; return f'圖{figno[0]} {text}\n', figno[0]
def img(site,stem,w):
    return f'![]({FIGDIR}/{site}/{stem}.png){{width={w}in}}\n'
NR='oca_nonrecyclable_flexible_sheet'
HZ=HORIZON_DAYS
FW=6.27   # 所有圖之顯示寬度（吋）：文字區全寬 9026 twips = 6.268 吋

# ================================================================ 方法 B（抵達海岸即視為找到來源）版：資料來源為 _verify/src_coastB（.203 postprocess-coastB 等之副本）
SRCB=ROOT+'/_verify/src_coastB'; CT=SRCB+'/coastB-tables'; PPB=SRCB+'/postprocess-coastB'; PT=SRCB+'/figures-polar-v3-ax2-coastB/tables'
def _rd(p): return list(csv.DictReader(open(p,encoding='utf-8-sig',newline='')))
FA=json.load(open(CT+'/facts_remote_run09B.json'))          # 覆寫 d09 載入之舊版 facts
assert FA['stop_status_codes']==[1,8,4] and FA['n_total']==500000
OUTC={r['site']:r for r in _rd(CT+'/outcome_by_site.csv')}
GEO={r['site']:r for r in _rd(CT+'/coast_geography_by_site.csv')}
POLY=_rd(CT+'/coast_geography_polygons.csv')
REPB=json.load(open(CT+'/rep_traj_coastB.json'))

# ================================================================ 標題與引言
# 章標題、引言第一段與小標「逐地點成果」為使用者於 Word 手改之版本（逐字保留，見 _verify/src/user_edits_20261021.txt、user_edits_20261022.txt）。
_ue=open(OUT+'/_verify/src/user_edits_20261021.txt',encoding='utf-8').read().split('\n')
assert len(_ue)==3 and _ue[0]=='Lagrangian系集逆向溯源執行成果' and _ue[2]=='逐地點成果' and _ue[1].startswith('　　本節呈現Lagrangian系集逆向溯源含斯托克斯漂流案例之執行成果。情境矩陣由')
INTRO_SUBS=[('之執行成果。情境矩陣','之執行成果，目的在於推估各研究地點海底廢棄物之來源方向及其機率。情境矩陣'),
            ('10 個沉降速度類別','10 類代表物件'),
            ('20 個受體','20 個受體（5 個水平位置 × 4 個深度）'),
            ('50 個觀測時刻','50 個到達時間'),
            ('與 10 個系集成員交叉而成；每個研究地點','與 10 個系集成員交叉而成，每顆虛擬質點之沉降速度自所屬類別之 10 個沉降速度中抽取；每個研究地點')]
_intro=_ue[1]
for a,b in INTRO_SUBS:
    assert _intro.count(a)==1,a; _intro=_intro.replace(a,b)
GUIDE=('　　各研究地點依序以七張圖說明：水平軌跡總覽呈現 24 條代表虛擬質點自受體回溯之軌跡，仍在追蹤之虛擬質點比例圖呈現全部虛擬質點於回溯期間內仍在追蹤之比例，'
       '條件式來源足跡呈現虛擬質點於回溯期間停留之位置，四張圓形來源方位圖依序呈現十類、同類十個沉降速度、季節與大小潮之來源方向機率；綜合說明與整體結論置於章末。')
_ue2=open(OUT+'/_verify/src/user_edits_20261022.txt',encoding='utf-8').read().split('\n')
assert len(_ue2)==3 and _ue2[0]=='Lagrangian系集逆向溯源分析成果' and _ue2[2]==_ue[2]=='逐地點成果'
assert _ue[0].replace('執行成果','分析成果')==_ue2[0] and _intro.count('執行成果')==1 and _intro.replace('之執行成果，','之分析成果，')==_ue2[1]
_intro=_ue2[1]
A('# '+_ue2[0]+'\n')
A(_intro+'\n')
A(GUIDE+'\n')
A('#- '+_ue2[2]+'\n')

bf=pd.read_csv(PPB+'/boundary_flux_by_site.csv').set_index('study_site_id')
fs=pd.read_csv(PPB+'/footprint_summary_by_site.csv').set_index('study_site_id')   # 停留時數密度為 1 km × 1 km 規則公尺網格；方法 B 之停留時數於抵達海岸處截斷
assert (fs['grid_cell_km']==1.0).all()
cnt={s:dict(total=int(bf.loc[s,'total_particle_count']),cross=int(bf.loc[s,'count_crossed']),coast=int(bf.loc[s,'count_status_8']),maxb=int(bf.loc[s,'count_status_2']),dep=int(bf.loc[s,'count_status_4'])) for s,_ in SITES}
assert sum(int(bf.loc[s,f'count_status_{k}']) for s,_ in SITES for k in (2,3,5,6,7))==0     # 無達回溯上限；無其他結局
for s,_ in SITES:
    v=cnt[s]; o=OUTC[s]; st=FA['status'][s]
    assert v['total']==100000 and v['cross']+v['coast']+v['dep']==100000
    assert (v['cross'],v['coast'],v['dep'])==(int(o['crossed']),int(o['coast']),int(o['touch_bottom']))==(st.get('1',0),st.get('8',0),st.get('4',0))
ndep=sum(v['dep'] for v in cnt.values())
assert ndep==len(FA['dep'])==1 and FA['dep'][0]['site']=='nanwan' and FA['dep'][0]['material']==NR and cnt['nanwan']['dep']==1
_d=FA['dep'][0]; assert abs(_d['dur_h']-5.0)<1e-9 and abs(abs(_d['v'])-0.000897)<1e-12
_dfm=json.load(open(SRCB+'/figures-domain-fraction-coastB/manifest.json')); assert _dfm['max_days']==HZ and _dfm['marks_days']==[90.0,float(HZ)]
for s,_ in SITES: assert (fs.loc[s,'depth_threshold_m']==200)

# ---------------------------------------------------------------- 圓形來源方位圖統計（方法 B 之 figures-polar-v3-ax2-coastB/tables）
POL_TERM={r['site']:r for r in _rd(PT+'/termination_by_group.csv') if r['grouping']=='site_all' and r['group_id']=='all'}
POL_SUM=_rd(PT+'/distinguishability_summary.csv')
POL_PAIRS=_rd(PT+'/distinguishability_pairs.csv')
assert set(POL_TERM)=={s_ for s_,_ in SITES}
SEASONS=[('DJF','冬季'),('MAM','春季'),('JJA','夏季'),('SON','秋季')]
SEC=_rd(PT+'/sector_shares_long.csv')
def sec_counts(site,grouping,gid=None):
    rows=[r for r in SEC if r['site']==site and r['grouping']==grouping and (gid is None or r['group_id']==gid)]
    c=[0]*36
    for r in rows: c[int(r['sector_index'])]+=int(r['count'])
    return c
for s,_ in SITES:
    rows=[r for r in POL_SUM if r['site']==s]
    VERD={k:[r['verdict'] for r in rows if r['kind']==k] for k in ('class','season','tide','velocity_within_class')}
    assert VERD['class']==['within_noise'] and VERD['tide']==['within_noise'] and VERD['season']==['over_noise']
    assert VERD['velocity_within_class']==['within_noise']*10
_SP={}
for r in POL_PAIRS:
    if r['kind']=='season': _SP[(r['site'],r['group_a'],r['group_b'])]=(r['over_p99']=='True')
def season_over(s,a,b): return _SP[(s,a,b)] if (s,a,b) in _SP else _SP[(s,b,a)]
_SQ=[g for g,_n in SEASONS]
SIM={s:{(a,b) for i,a in enumerate(_SQ) for b in _SQ[i+1:] if not season_over(s,a,b)} for s,_ in SITES}
for s,_ in SITES: assert sum(1 for k in _SP if k[0]==s)==6
# 主要來源方向之取法（方法 B 起）：方向機率高於平均值（1/36＝2.78%，即計數 × 36 > 總數）之相鄰方向區間（環狀）合併為一個範圍，範圍機率為各區間機率合計；
# 取合計最大者為最主要來源方向，第二大者一律一併列出（若有）；方位名稱依範圍中心度數所在之 8 方位命名（中心距方位分界不超過 2.5° 者標示相鄰兩方位）。
def top_ranges(c):
    tot=sum(c); on=[v*36>tot for v in c]
    assert not all(on) and any(on)
    st=on.index(False); out=[]; cur=None
    for k in range(1,37):
        j=(st+k)%36
        if on[j]:
            if cur is None: cur=[j,1]
            else: cur[1]+=1
        else:
            if cur: out.append(tuple(cur)); cur=None
    if cur: out.append(tuple(cur))
    rs=[(a,ln,sum(c[(a+k)%36] for k in range(ln))) for a,ln in out]
    rs.sort(key=lambda r:(-r[2],r[0]))
    return rs[:2],tot
N8=['北','東北','東','東南','南','西南','西','西北']
def rname(a,ln):
    c=(a*10+(ln-1)*5)%360
    for k in range(8):
        d=abs(2*c-(45+90*k)); d=min(d,720-d)
        if d<=5: return N8[k]+'至'+N8[(k+1)%8]+'方'
    return N8[((2*c+45)%720)//90]+'方'
def rn(r): return rname(r[0],r[1])
def rpp(r,tot): return ppp(r[2]/tot*100)
MAIN={}
for s,_ in SITES:
    c_all=sec_counts(s,'class'); assert sum(c_all)==100000
    assert sec_counts(s,'season')==c_all and sec_counts(s,'tide')==c_all and sec_counts(s,'class_velocity')==c_all
    MAIN[s]={'all':top_ranges(c_all)}
    assert len(MAIN[s]['all'][0])==2
    for g,_n in SEASONS: MAIN[s][g]=top_ranges(sec_counts(s,'season',g))
    for g_,(rs_,t_) in MAIN[s].items(): assert len({rn(r) for r in rs_})==len(rs_),(s,g_,[rn(r) for r in rs_])
    t=POL_TERM[s]; assert int(t['n_total'])==100000 and int(t['n_crossed'])==cnt[s]['cross'] and int(t['n_status_8'])==cnt[s]['coast'] and int(t['n_status_4'])==cnt[s]['dep']

# 海岸來源之位置（底圖陸地多邊形分類；占全部虛擬質點之比例）
def pc(n): return ppp(n*100/100000)
def hamilton(parts,total):
    """最大餘數法：parts 為顆數（母數 100,000），以 0.01% 為單位；各分項加總恰等於已公布總計（total 之兩位小數四捨五入值，ROUND_HALF_UP）。
    餘數相同者，顆數多者先得。回傳各分項之百分比字串。"""
    tu=int((Decimal(total)*100/Decimal(100000)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)*100)   # 總計（0.01% 單位）
    fl=[p//10 for p in parts]; rem=[p%10 for p in parts]       # 顆數 /10 ＝ 0.01% 單位（餘數以十分之一計）
    k=tu-sum(fl); assert 0<=k<=len(parts),(parts,total,k)
    order=sorted(range(len(parts)),key=lambda i:(-rem[i],-parts[i]))
    for i in order[:k]: fl[i]+=1
    for p,f in zip(parts,fl): assert abs(Decimal(f)/100-Decimal(p)/1000)<=Decimal('0.01')+Decimal('1e-12'),(p,f)
    assert sum(fl)==tu
    return [f'{Decimal(f)/100:.2f}%' for f in fl]
def _g(s,k): return int(GEO[s][k])
for s,_ in SITES: assert _g(s,'n_coast')==cnt[s]['coast']
def poly_cnt(s,pid): return sum(int(r['count']) for r in POLY if r['site']==s and r['polygon']==pid)
def coast_sent(s):
    """各圓形圖（十類圖）旁之海岸來源位置說明（一句，數字取自底圖陸地多邊形分類表）。"""
    sn=dict(SITES)[s]; n=_g(s,'n_coast'); big=_g(s,'n_on_large_polygon')
    if s=='gongliao':
        gs=poly_cnt(s,'96:0'); oth=n-big-gs; assert big==24496 and gs==883 and oth==1219
        a,b,c=hamilton([big,gs,oth],n)
        return f'{sn}之海岸來源（{pc(n)}）中，臺灣本島占 {a}，龜山島占 {b}，其他島嶼占 {c}。'
    if s=='guishan':
        gs=poly_cnt(s,'96:0'); oth=n-big-gs; assert gs==9565 and big==11955 and oth==127
        a,b,c=hamilton([big,gs,oth],n)
        return f'{sn}之海岸來源（{pc(n)}）中，臺灣本島占 {a}，龜山島占 {b}，其他島嶼占 {c}。'
    if s=='hsinchu':
        assert big==n==2234
        return f'{sn}之海岸來源（{pc(n)}）皆位於臺灣本島。'
    if s=='nanwan':
        xl=poly_cnt(s,'30:0'); ly=poly_cnt(s,'6:0'); oth=n-big-xl-ly; assert big==29942 and xl==578 and ly==534 and oth==31
        a,b,c,d=hamilton([big,xl,ly,oth],n)
        return f'{sn}之海岸來源（{pc(n)}）中，臺灣本島占 {a}，小琉球占 {b}，蘭嶼占 {c}，其他島嶼占 {d}。'
    if s=='lienchiang':
        mf=_g(s,'n_fujian_mainland'); fi=_g(s,'n_fujian_coastal_islands'); mt=_g(s,'n_matsu_islands')
        assert mf+fi+mt==n and (mf,fi,mt)==(15821,13858,14265) and big==mf
        a,b=hamilton([mf+fi,mt],n)
        return f'{sn}之海岸來源（{pc(n)}）中，福建沿岸（含近岸島嶼）占 {a}，馬祖各島占 {b}。'
    raise KeyError(s)
LIEN_FJ,LIEN_MT=hamilton([_g('lienchiang','n_fujian_mainland')+_g('lienchiang','n_fujian_coastal_islands'),_g('lienchiang','n_matsu_islands')],_g('lienchiang','n_coast'))

def season_list(s):
    parts=[]
    for g,zn in SEASONS:
        rs,tot=MAIN[s][g]
        t=f'{zn}為{rn(rs[0])}，機率為 {rpp(rs[0],tot)}'
        if len(rs)>1: t+=f'，另有{rn(rs[1])} {rpp(rs[1],tot)}'
        parts.append(t)
    return '；'.join(parts)+'。'
def season_names(s):
    parts=[]
    for g,zn in SEASONS:
        rs,_t=MAIN[s][g]
        parts.append(f'{zn}為{rn(rs[0])}')
    return '，'.join(parts)
def seas_clauses(s):
    """季節間比較（逐地點）：相近／差異明顯依 distinguishability_pairs.csv 之兩兩比較（over_p99）。"""
    allpairs=[(a,b) for i,a in enumerate(_SQ) for b in _SQ[i+1:]]
    sim=SIM[s]
    if not sim:
        assert all(season_over(s,a,b) for a,b in allpairs); return '四季之來源方向分布兩兩差異明顯'
    if sim=={('DJF','MAM'),('JJA','SON')}:
        assert all(season_over(s,a,b) for a,b in allpairs if (a,b) not in sim); return '冬、春兩季之來源方向分布相近，夏、秋兩季相近，兩組之間差異明顯'
    if sim=={('MAM','SON')}:
        assert all(season_over(s,a,b) for a,b in allpairs if (a,b) not in sim); return '春、秋兩季之來源方向分布相近，其餘季節兩兩之間差異明顯'
    raise KeyError((s,sim))

# 圖序：1 水平軌跡總覽、2 仍在追蹤比例、3 條件式來源足跡、4–7 圓形圖
FIGSTEMS=[('horizontal_overview','代表虛擬質點水平軌跡總覽'),('domain_fraction_vs_backtrack_time','回溯期間仍在追蹤之虛擬質點比例'),('source_footprint_residence','條件式來源足跡（停留時數密度）')]
assert REPB['trajectory_sample_seed']==0 and REPB['size_per_site']==24
for s_,_ in SITES:
    R_=REPB['sites'][s_]; assert R_['n']==24 and set(R_['status_counts'])<={'1','8'} and sum(R_['status_counts'].values())==24 and R_['status_counts'].get('8',0)>=1
OBJ_CLASS='十類之曲線相近，物件種類對來源方向之影響不明顯。'
OBJ_VEL='同類十個沉降速度之曲線相近，沉降速度對來源方向之影響不明顯。'
OBJ_TIDE='大小潮兩組之曲線相近，大小潮對來源方向之影響不明顯。'
H50={s:fs.loc[s,'hdr50_area_km2'] for s,_ in SITES}
_o=sorted(H50,key=H50.get); SMALLEST,LARGEST=_o[0],_o[-1]
HDR_LOC={'gongliao':'50% 高密度區間沿臺灣本島東北部海岸外側呈南北向帶狀，通過受體並向南北兩側延伸。',
         'guishan':'50% 高密度區間位於受體周圍之近岸海域，西側緊鄰臺灣本島東北部海岸，向北延伸至受體北側，向南延伸至流場範圍邊界。',
         'hsinchu':'50% 高密度區間沿臺灣本島西部海岸外側呈西南至東北向帶狀，通過受體並向東北與西南兩側延伸。',
         'nanwan':'50% 高密度區間自受體沿恆春半島西側向西北延伸，另有一段沿其東側向東北延伸。',
         'lienchiang':'50% 高密度區間自受體向東北延伸，另有一獨立區塊緊鄰福建沿岸。'}
assert SMALLEST=='guishan' and LARGEST=='nanwan'
for si,(s,sn) in enumerate(SITES):
    v=cnt[s]
    FIRST=(si==0)
    A(f'## {sn}\n')
    # ---- 圖 1：水平軌跡總覽
    n1=figno[0]+1; sc=REPB['sites'][s]['status_counts']; k1,k8=sc.get('1',0),sc.get('8',0)
    if FIRST: g1=f'圖 {n1} 為隨機抽取之 24 條代表虛擬質點回溯軌跡，線色依起始深度由淺至深。'
    else: g1=f'圖 {n1} 為{sn}隨機抽取之 24 條代表虛擬質點回溯軌跡。'
    A('　　'+g1+'\n'); c,_=cap_f(f'{sn}{FIGSTEMS[0][1]}'); A(img(s,FIGSTEMS[0][0],FW)); A(c)
    # ---- 圖 2：仍在追蹤之虛擬質點比例（全部 100,000 顆之統計）
    n_=figno[0]+1; nfrac=n_
    I=FA['inside'][s]; assert I['n']==100000
    pts=[]
    for th,lab in ((0.5,'50%'),(0.1,'10%'),(0.01,'1%')):
        d=I[f'first_le_{th}']; assert d is not None
        pts.append(f'回溯第 {d:.1f} 天降至 {lab} 以下' if not pts else f'第 {d:.1f} 天降至 {lab} 以下')
    zero=I['first_zero_day']; t='、'.join(pts)
    if zero is not None: t+=f'，至第 {zero:.1f} 天降至 0%'
    v90,v180=ppp(I['at90']*100),ppp(I[f'at{HZ}']*100)
    assert zero is not None and zero<=90 and v90==v180==ppp(0) and I['n_not_stopped']==0
    tail='。'
    _pc=ppp(v["cross"]/v["total"]*100); _cc=ppp(v["coast"]/v["total"]*100)
    if FIRST:
        h=(f'圖 {n_} 為全部 {v["total"]:,} 顆虛擬質點於回溯期間內仍在追蹤之比例，垂直虛線標示 90 天與 {HZ} 天。虛擬質點穿越流場範圍邊界或抵達海岸時停止追蹤；'
           f'{sn}之虛擬質點穿越流場範圍邊界者占 {_pc}，稱為穿越比例，抵達海岸者占 {_cc}，稱為海岸來源比例。')
    else:
        h=f'圖 {n_} 為全部 {v["total"]:,} 顆虛擬質點於回溯期間內仍在追蹤之比例。{sn}之穿越比例為 {_pc}，海岸來源比例為 {_cc}'
        if v['dep']>0: h+=f'，其餘 {v["dep"]} 顆（{ppp(v["dep"]/v["total"]*100)}）於回溯途中觸底而停止追蹤'
        h+='。'
    A('　　'+h+f'{sn}仍在追蹤之比例於'+t+tail+'\n')
    c,_=cap_f(f'{sn}{FIGSTEMS[1][1]}'); A(img(s,FIGSTEMS[1][0],FW)); A(c)
    # ---- 圖 3：條件式來源足跡
    n2=figno[0]+1
    h50,h75,h90=(fs.loc[s,k] for k in ('hdr50_area_km2','hdr75_area_km2','hdr90_area_km2'))
    shv=fs.loc[s,"hours_fraction_depth_below_threshold"]*100; shallow=ppp(shv)
    if shv>=99.9999: shelf='皆屬大陸棚與近岸海域'
    elif shv>=50: shelf='多屬大陸棚與近岸海域'
    else: shelf='多屬水深 200 m 以深之外海'
    if s==SMALLEST: rk='，為五個研究地點中最小，停留時數最集中於受體附近'
    elif s==LARGEST: rk='，為五個研究地點中最大，停留時數分布於較廣之海域'
    else: rk=''
    if FIRST:
        g2=(f'圖 {n2} 之底色為停留時數密度：虛擬質點每在流場範圍內漂流 1 小時，即在其所在之 1 km × 1 km 網格計 1 小時停留時數，各網格之停留時數占該研究地點停留時數總量之比例，除以網格面積即為停留時數密度；虛擬質點抵達海岸後不再計入。'
            f'高密度區間係將各網格依停留時數密度由高至低排序，依序累加其停留時數，累加達 50%、75%、90% 時所涵蓋之網格範圍，即包含該比例停留時數之最小面積區域；50% 高密度區間為包含一半停留時數之範圍，面積愈小，表示停留時數愈集中於受體附近，面積愈大，表示分布於較廣之海域。'
            f'{sn}之 50%、75%、90% 高密度區間面積分別為 {i0(h50)}、{i0(h75)} 與 {i0(h90)} km²；停留時數之 {shallow} 位於水深淺於 200 m 處，{shelf}。'+HDR_LOC[s])
    else:
        g2=(f'圖 {n2} 中，{sn}之 50%、75%、90% 高密度區間面積分別為 {i0(h50)}、{i0(h75)} 與 {i0(h90)} km²{rk}；'
            f'停留時數之 {shallow} 位於水深淺於 200 m 處，{shelf}。'+HDR_LOC[s])
    A('　　'+g2+'\n'); c,_=cap_f(f'{sn}{FIGSTEMS[2][1]}'); A(img(s,FIGSTEMS[2][0],FW)); A(c)
    # ---- 圖 4–7：圓形來源方位圖
    _n=figno[0]+1
    rs,tot=MAIN[s]['all']
    main_s=f'{sn}全年（十類合併）之主要來源方向為{rn(rs[0])}，機率為 {rpp(rs[0],tot)}，其次為{rn(rs[1])} {rpp(rs[1],tot)}。'
    if FIRST:
        A(f'　　圖 {_n} 至圖 {_n+3} 為{sn}之圓形來源方位圖，依序呈現十類、同類十個沉降速度、季節與大小潮之來源方向機率。'
          '來源方向為虛擬質點所在位置自研究地點錨點起算之方位，每 10° 分為一個方向區間，共 36 個，方位自正北順時針量測。統計全部虛擬質點之來源方向落在各方向區間之機率，連成封閉曲線。'
          '穿越流場範圍邊界之虛擬質點，以其穿越位置之方向計入；抵達海岸者，以其抵達位置之方向計入，稱為海岸來源；回溯途中觸底而停止追蹤者，以其最終位置之方向計入，稱為觸底。因此各曲線 36 個方向區間之方向機率合計為 100%。'
          '主要來源方向之取法為：方向機率高於平均值 2.78%（即 1/36）之相鄰方向區間合併為一個範圍，範圍機率為各方向區間機率之合計，取機率最大者為最主要來源方向，並列出第二大者。'
          '海岸來源各段之比例皆為占全部虛擬質點之比例。\n')
        pre=f'圖 {_n} 為十類之來源方向曲線。'
    else:
        pre=f'圖 {_n} 至圖 {_n+3} 為{sn}之圓形來源方位圖，依序為十類、同類十個沉降速度、季節與大小潮；其中圖 {_n} 為十類之來源方向曲線。'
    A('　　'+pre+main_s+coast_sent(s)+OBJ_CLASS+'\n')
    _items=[('polar_source_v2_by_class',f'{sn}來源方向機率（十類）'),('polar_source_v2_velocity_panels',f'{sn}來源方向機率（十類各十個沉降速度）'),('polar_source_v2_by_season',f'{sn}來源方向機率（依沉底時刻之季節）'),('polar_source_v2_by_tide',f'{sn}來源方向機率（依大小潮）')]
    _c,_=cap_f(_items[0][1]); A(img(s,_items[0][0],FW)); A(_c)
    A(f'　　圖 {_n+1} 以十類各一格，呈現同類十個沉降速度之曲線'+(f'（各格依圖 {_n} 圖例之類別順序排列）' if FIRST else '')+'；'+OBJ_VEL+'\n')
    _c,_=cap_f(_items[1][1]); A(img(s,_items[1][0],FW)); A(_c)
    sea=f'圖 {_n+2} 為依沉底時刻劃分之{sn}四季曲線。'
    if FIRST:
        sea+=('沉底時刻為廢棄物沉降至海床之時刻，即虛擬質點回溯起點所對應之時刻，為所屬到達時間往前推一個 0–90 天之沉底年齡；'
              '季節依沉底時刻所在月份劃分，冬季為 12–2 月、春季為 3–5 月、夏季為 6–8 月、秋季為 9–11 月。')
    sea+='各季節之主要來源方向及其機率為：'+season_list(s)+'季節間之差異：'+seas_clauses(s)+'。'
    A('　　'+sea+'\n')
    _c,_=cap_f(_items[2][1]); A(img(s,_items[2][0],FW)); A(_c)
    A(f'　　圖 {_n+3} 為依'+('沉底時刻之潮位資料劃分為大潮與小潮之曲線。' if FIRST else '大小潮劃分之曲線。')+OBJ_TIDE+'\n')
    _c,_=cap_f(_items[3][1]); A(img(s,_items[3][0],FW)); A(_c)
assert figno[0]==35

# ---------------------------------------------------------------- 章末
A('#- 綜合說明與結論\n')
_h50=[H50[s] for s,_ in SITES]; _sh=[fs.loc[s,"hours_fraction_depth_below_threshold"]*100 for s,_ in SITES]
assert SIM['gongliao']=={('DJF','MAM'),('JJA','SON')}==SIM['guishan'] and SIM['nanwan']=={('MAM','SON')} and all(SIM[s]==set() for s in ('hsinchu','lienchiang'))
A('　　各研究地點十類之間、同類十個沉降速度之間與大小潮之間，來源方向機率相近，物件種類、沉降速度與大小潮對來源方向之影響不明顯。'
  '季節之影響則因研究地點而異：貢寮與龜山島冬、春季相近，夏、秋季相近，兩組之間差異明顯；南灣春、秋季相近，其餘季節兩兩之間差異明顯；新竹外海與連江四個季節兩兩之間差異皆明顯。\n')
assert fs.loc['hsinchu','hours_fraction_depth_below_threshold']==1.0==fs.loc['lienchiang','hours_fraction_depth_below_threshold'] and _sh[3]<50
A(f'　　足跡範圍方面，各研究地點 50% 高密度區間面積為 {i0(min(_h50))}–{i0(max(_h50))} km²，停留時數位於水深淺於 200 m 處之比例為 {ppp(min(_sh))}–{ppp(max(_sh))}；'
  f'新竹外海與連江之停留時數全部位於水深淺於 200 m 處，南灣之停留時數則多數位於水深 200 m 以深之外海。\n')
_cr=[cnt[s]['cross']/1000 for s,_ in SITES]; _co=[cnt[s]['coast']/1000 for s,_ in SITES]
assert max(range(5),key=lambda i:_cr[i])==2 and min(range(5),key=lambda i:_cr[i])==4
assert all(FA['inside'][s]['first_zero_day'] is not None for s,_ in SITES)
_zd=max(FA['inside'][s]['first_zero_day'] for s,_ in SITES)
A(f'　　回溯期間內，各研究地點之虛擬質點皆於 {_zd:.1f} 天內停止追蹤（穿越流場範圍邊界或抵達海岸，南灣另有 {ndep} 顆觸底），無回溯達 {HZ} 天者。'
  f'穿越比例為 {ppp(min(_cr))}–{ppp(max(_cr))}（新竹外海最高，連江最低），海岸來源比例為 {ppp(min(_co))}–{ppp(max(_co))}（新竹外海最低，連江最高）。\n')
A('#- 整體結論\n')
for i,(s,sn) in enumerate(SITES,1):
    rs,tot=MAIN[s]['all']
    b=f'{sn}：全年主要來源方向為{rn(rs[0])}，機率為 {rpp(rs[0],tot)}，其次為{rn(rs[1])} {rpp(rs[1],tot)}'
    b+='；各季節主要來源方向：'+season_names(s)+'；'+seas_clauses(s)+'。'
    b+=f'海岸來源占 {ppp(cnt[s]["coast"]/1000)}'
    if s=='lienchiang': b+=f'，其中福建沿岸（含近岸島嶼）占 {LIEN_FJ}，馬祖各島占 {LIEN_MT}'
    b+='。'
    if s=='nanwan': b+=f'另有 {cnt[s]["dep"]} 顆（{ppp(cnt[s]["dep"]/1000)}）於回溯途中觸底而停止追蹤。'
    A(f'{i}、{b}\n')
_m1=[MAIN[s]['all'][0][0][2]/MAIN[s]['all'][1]*100 for s,_ in SITES]
A(f'6、綜合五個研究地點：各研究地點最主要來源方向之機率為 {ppp(min(_m1))}–{ppp(max(_m1))}，海岸來源比例為 {ppp(min(_co))}–{ppp(max(_co))}；物件種類與大小潮之影響不明顯，季節之影響因研究地點而異。\n')

_PROT=('#- 逐地點成果','#- 本地點結論')
_lines='\n'.join(L).split('\n')
_txt='\n'.join(l if l in _PROT else to_loc(l).replace('本研究地點結論','本地點結論') for l in _lines)
open(OUT+'/執行成果_run09_含斯托克斯漂流.md','w',encoding='utf-8').write(fixsp(_txt))
json.dump({'tables':0,'figs':figno[0]},open(OUT+'/_verify/res_counts.json','w'))
json.dump(sorted(set(KEEP3)),open(OUT+'/_verify/res_keep3.json','w'),ensure_ascii=False)
print('figs',figno[0],'keep3',sorted(set(KEEP3)))
