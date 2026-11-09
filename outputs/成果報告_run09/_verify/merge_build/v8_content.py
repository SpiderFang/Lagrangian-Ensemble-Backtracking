# -*- coding: utf-8 -*-
"""v8 成果段文字：由 segprob_v3 之 manifest 與表產生（數字不手打；子分項沿用母版）。"""
import re, csv, math
from v8data import *

def sp(s):
    s = re.sub(r'([一-鿿])([0-9A-Za-z])', r'\1 \2', s)
    s = re.sub(r'([0-9A-Za-z%°])([一-鿿])', r'\1 \2', s)
    return s
PC = lambda v: '%.2f%%' % v
def lst(a):
    a = list(a)
    return a[0] if len(a) == 1 else ('、'.join(a[:-1]) + '與' + a[-1])

KEYS = ['traj', 'still', 'fp', 'all_b', 'all_c', 'cls_b', 'cls_c', 'sea_b', 'sea_c', 'tid_b', 'tid_c']
MF = {'all_b': 'boundary_all', 'all_c': 'coast_all', 'cls_b': 'boundary_class', 'cls_c': 'coast_class', 'sea_b': 'boundary_season', 'sea_c': 'coast_season', 'tid_b': 'boundary_tide', 'tid_c': 'coast_tide'}
BASE = {'gongliao': 8, 'guishan': 19, 'hsinchu': 30, 'nanwan': 41, 'lienchiang': 52}
def fno(s, k): return BASE[s] + KEYS.index(k)
OLDB = {'gongliao': 8, 'guishan': 15, 'hsinchu': 22, 'nanwan': 29, 'lienchiang': 36}   # 舊軌跡圖號；舊 still=+1, fp=+2, 圓形圖=+3..+6

GT = {}
for r in csv.DictReader(open(B + 'tables/group_totals.csv', encoding='utf-8-sig')):
    GT[(r['site'], r['group'])] = r
TOT = {s: (float(GT[(s, 'all')]['crossed_pct']), float(GT[(s, 'all')]['coast_pct'])) for s in SITES}
assert [TOT[s] for s in SITES] == [(73.40, 26.60), (78.35, 21.65), (97.77, 2.23), (68.91, 31.09), (56.06, 43.94)]
SUB = {   # 母版之海岸來源子分項（維持不變）
    'gongliao': '臺灣本島占 24.50%，龜山島占 0.88%，其他島嶼占 1.22%',
    'guishan': '臺灣本島占 11.96%，龜山島占 9.57%，其餘為其他島嶼',
    'hsinchu': None,
    'nanwan': '臺灣本島占 29.94%，小琉球占 0.58%，蘭嶼占 0.54%，其他島嶼占 0.03%',
    'lienchiang': '福建沿岸（含近岸島嶼）占 29.68%，馬祖各島占 14.26%'}

# ------------------------------------------------------------ 邊界描述
def bside(d):
    return d.split('（')[0] if re.match(r'^[東西]?[南北]?角|^[東西南北]{2}角', d) else d.split('約')[0]
def is_corner(d): return '角' in d.split('約')[0].split('（')[0] and d[2] == '角'
def bshort(s, ids):
    d = bdesc(s, ids)
    return d.split('（')[0] if is_corner(d) else d
def bside_name(s, ids):
    d = bdesc(s, ids)
    return d.split('（')[0] if is_corner(d) else d.split('約')[0]

def all_boundary(s):
    a = m['figures'][s]['boundary_all']['annotations']['stretches']
    mains = sorted([(bdesc(s, x['segments']), float(x['shown_pct'])) for x in a if x['style'] == 'main'], key=lambda x: -x[1])
    gaps = [(bshort(s, x['segments']), float(x['shown_pct'])) for x in a if x['style'] == 'gap']
    return mains, gaps

def bpanel(s, fig):
    """每格：標題、穿越％、[(位置描述, 機率)]"""
    out = []
    for p in fig_panels(s, fig):
        items = [(bdesc(s, b['segments']), v) for (mk, v), b in zip(p['items'].items(), p['brackets'])]
        out.append((p['head'], p['total'], items, p['rest']))
    return out

# ------------------------------------------------------------ 海岸描述
def ring_of(st): return sorted(set(i.split('-')[0] for i in st['segments']))
RINGNAME = {'C0377': '馬祖北竿', 'C0379': '馬祖南竿', 'C0023': '龜山島沿岸'}
def rng(d): return '約 %.2f–%.2f°N、%.2f–%.2f°E 段' % (d['lat'][0], d['lat'][1], d['lon'][0], d['lon'][1])
def cfall(s, st):
    """備援描述：地名（明確者）或類別加經緯度範圍。"""
    d = cdesc(s, st); rg = ring_of(st)
    if rg[0] in RINGNAME: return RINGNAME[rg[0]]
    if d['cls'] == '福建沿岸':
        return ('福建沿岸近岸島嶼' if d['island'] else '福建沿岸') + rng(d)
    if d['cls'] == '馬祖各島': return '馬祖各島' + rng(d)
    return '臺灣本島海岸' + rng(d)
ALLLAB = {   # 全年海岸括號之描述（人工核定之相對位置）
    ('gongliao', '16.90'): '研究地點西北側臺灣本島海岸' + '約 25.07–25.15°N、121.81–121.92°E 段',
    ('gongliao', '6.10'): '研究地點東南側臺灣本島海岸' + '約 24.98–25.05°N、121.93–122.00°E 段',
    ('guishan', '9.57'): '龜山島沿岸',
    ('guishan', '6.29'): '研究地點西南側臺灣本島海岸' + '約 24.60–24.80°N、121.82–121.88°E 段',
    ('nanwan', '27.23'): '研究地點周邊恆春半島南端海岸' + '約 21.90–22.03°N、120.69–120.86°E 段',
    ('lienchiang', '6.26'): '福建沿岸約 25.94–26.04°N、119.62–119.71°E 段',
    ('lienchiang', '7.99'): '馬祖北竿',
    ('lienchiang', '4.56'): '馬祖南竿'}
def coast_all_st(s):
    f = m['figures'][s]['coast_all']
    a = list(f['annotations']['stretches'])
    if f.get('inset'): a += f['inset']['annotations']['stretches']
    return a
ALLSET = {}
for s in SITES:
    ALLSET[s] = [(set(st['segments']), ALLLAB[(s, st['pct'])], float(st['pct'])) for st in coast_all_st(s)]
def clabel(s, st):
    """多格海岸括號：與全年括號之線段集合重疊 ≥70% 者沿用全年描述，否則用備援描述。"""
    ss = set(st['segments'])
    for A, lab, p in ALLSET[s]:
        if len(ss & A) / len(ss | A) >= 0.7: return lab
    return cfall(s, st)
def cpanel(s, fig):
    out = []
    for p in fig_panels(s, fig):
        items = [(clabel(s, b), v) for (mk, v), b in zip(p['items'].items(), p['brackets'])]
        out.append((p['head'], p['total'], items))
    return out

# ------------------------------------------------------------ 位置詞（季節、大小潮、十類段落用；不寫經緯度）
EXT = {}
for _s in SITES:
    for _sd in 'NSEW':
        _k = ('start_lon', 'end_lon') if _sd in 'NS' else ('start_lat', 'end_lat')
        _v = [float(r[x]) for r in seg[_s].values() if r['side'] == _sd for x in _k]
        EXT[(_s, _sd)] = (min(_v), max(_v))
def bpos(s, ids):
    """邊界括號之位置詞：角、整段邊界，或邊界之西／中／東段（南北向邊界為南／中／北段）。"""
    p = pieces(s, ids)
    big = [x for x in p if x[2] - x[1] > 0.12]
    if len(big) == 1 and len(p) > 1: p = big
    if len(p) == 2: return CORNER[frozenset(p[0][0] + p[1][0])]
    assert len(p) == 1, (s, ids, p)
    sd, lo, hi = p[0]; a, b = EXT[(s, sd)]
    w = (hi - lo) / (b - a); mid = ((lo + hi) / 2 - a) / (b - a)
    if w >= 0.6: return SIDE[sd] + '邊界'
    words = ('西段', '中段', '東段') if sd in 'NS' else ('南段', '中段', '北段')
    return SIDE[sd] + '邊界' + words[0 if mid < 0.3 else (2 if mid > 0.7 else 1)]
def _cpt(s, i):
    r = cseg[s][i]; return (float(r['start_lon']), float(r['start_lat'])), (float(r['end_lon']), float(r['end_lat']))
def _km(s, p):
    ax, ay = ANCHOR[s]; return ((p[0] - ax) * 111.32 * math.cos(math.radians(ay)), (p[1] - ay) * 110.57)
def _pdist(s, i):
    a, b = _cpt(s, i); A = _km(s, a); B = _km(s, b); dx, dy = B[0] - A[0], B[1] - A[1]; L = dx * dx + dy * dy
    t = max(0.0, min(1.0, -(A[0] * dx + A[1] * dy) / L)) if L else 0.0
    return math.hypot(A[0] + t * dx, A[1] + t * dy)
def two_sided(s, st):
    """海岸括號跨研究地點兩側：(a) 含同一海岸環上距研究地點最近之海岸段，且該段不在括號之端點（括號兩端分居其兩側）；
    (b) 括號兩端點相對研究地點之方位夾角 ≥ 150°。僅適用於距研究地點 10 km 內之海岸（遠處之海岸不稱『周邊』）。"""
    ids = sorted(st['segments']); ring = ids[0].split('-')[0]
    near = min([i for i in cseg[s] if i.startswith(ring + '-')], key=lambda i: _pdist(s, i))
    if min(_pdist(s, i) for i in ids) > 10.0: return False
    a_ = near in ids and 0 < ids.index(near) < len(ids) - 1
    e1, e2 = _cpt(s, ids[0])[0], _cpt(s, ids[-1])[1]
    b1 = math.degrees(math.atan2(*_km(s, e1))); b2 = math.degrees(math.atan2(*_km(s, e2)))
    return a_ or abs((b1 - b2 + 180) % 360 - 180) >= 150
OCT = re.compile(r'^研究地點(?:東北|東南|西北|西南|[東西南北])側(臺灣本島)?海岸$')
def cpos(s, st):
    """海岸括號之位置詞：沿用全年之相對位置，否則以研究地點為基準之方位；不寫經緯度。
    跨研究地點兩側之括號不稱方位，改稱『研究地點周邊臺灣本島海岸』。"""
    lab = clabel(s, st).split('約')[0]
    if lab == '臺灣本島海岸': lab = '研究地點%s側臺灣本島海岸' % cdesc(s, st)['dir']
    mo = OCT.match(lab)
    if mo and two_sided(s, st): lab = '研究地點周邊' + ('臺灣本島海岸' if mo.group(1) else '海岸')
    return lab
def alllab(s, st):
    """全年海岸括號之描述（含方位者亦套用跨兩側規則）。"""
    lab = ALLLAB[(s, st['pct'])]; head = lab.split('約')[0]; mo = OCT.match(head)
    if mo and two_sided(s, st): return lab.replace(head, '研究地點周邊' + ('臺灣本島海岸' if mo.group(1) else '海岸'))
    return lab
LABGT = {(r['site'], r['label']): r for r in GT.values()}

# ------------------------------------------------------------ 句型
def conc(items, lead='集中於'):
    """[(位置, 機率)] → 『集中於 A（x），其次為 B（y）與 C（z）』"""
    def one(it):
        d, p = it
        return (d[:-1] + '，' + PC(p) + '）') if d.endswith('）') else '%s（%s）' % (d, PC(p))
    t = lead + one(items[0])
    if len(items) > 1: t += '，其次為' + lst([one(i) for i in items[1:]])
    return t

def b_top(s, fig):
    """[(格名, 穿越比例, 第一名位置詞, 第一名機率)]（只取第一名）。"""
    out = []
    for p in fig_panels(s, fig):
        if p['brackets']: out.append((p['head'].split('（')[0], p['total'], bpos(s, p['brackets'][0]['segments']), list(p['items'].values())[0]))
        else: out.append((p['head'].split('（')[0], p['total'], None, None))
    return out
def sea_b_clause(s):
    return '；'.join(('%s穿越比例為 %s，集中於%s（%s）' % (nm, PC(tot), pos, PC(v))) if pos else ('%s穿越比例為 %s' % (nm, PC(tot))) for nm, tot, pos, v in b_top(s, 'boundary_season'))
def tid_b_clause(s):
    return '；'.join(('%s穿越比例為 %s，集中於%s（%s）' % (nm, PC(tot), pos, PC(v))) if pos else ('%s穿越比例為 %s' % (nm, PC(tot))) for nm, tot, pos, v in b_top(s, 'boundary_tide'))
def c_clause(s, fig):
    """每格只寫海岸來源比例與第一名海岸段；零值：恰為 0 顆者寫『無海岸來源』，不為 0 而低於 0.005% 者寫『<0.01%』。"""
    cl = []; run = []
    def flush():
        if run:
            names = list(run)
            if all(x.endswith('季') for x in names): nm = '、'.join([x[:-1] for x in names[:-1]] + [names[-1]])
            else: nm = '、'.join(names)
            cl.append('%s無海岸來源' % nm); run.clear()
    for p in fig_panels(s, fig):
        nm = p['head'].split('（')[0]; tot = p['total']
        r = LABGT[(s, p['head'])]; n_c = int(r['coast_n'])
        if n_c == 0: run.append(nm); continue
        flush()
        if tot == 0: cl.append('%s海岸來源比例為 <0.01%%' % nm); continue   # 顆數不為 0 而四捨五入為 0.00%（< 0.005%）
        if p['brackets']: cl.append('%s海岸來源比例為 %s，集中於%s（%s）' % (nm, PC(tot), cpos(s, p['brackets'][0]), PC(list(p['items'].values())[0])))
        else: cl.append('%s海岸來源比例為 %s，分散於各海岸段，無機率集中之連續海岸段' % (nm, PC(tot)))
    flush()
    return '；'.join(cl)

def cls_range(s, kind):
    key = 'boundary_class' if kind == 'b' else 'coast_class'
    v = []
    for t in m['figures'][s][key]['panel_titles']:
        v.append(float(re.search(r'([\d.]+)%', t.split('\n')[1]).group(1)))
    return min(v), max(v)

SEA_VERDICT = {'gongliao': '四季之分段分佈兩兩差異明顯，季節對來源之影響明顯。',
               'guishan': '冬、春兩季之分段分佈相近，其餘季節兩兩之間差異明顯。',
               'hsinchu': '四季之分段分佈兩兩差異明顯，季節對來源之影響明顯。',
               'nanwan': '四季之分段分佈兩兩差異明顯，季節對來源之影響明顯。',
               'lienchiang': '四季之分段分佈兩兩差異明顯，季節對來源之影響明顯。'}

SEA_DEF = ('沉底時刻為廢棄物沉降至海床之時刻，亦即虛擬質點回溯之起點；由於廢棄物實際何時沉底未知，沉底時刻由所屬到達時間往前推一段隨機抽取之沉底年齡決定，'
           '每個系集成員各自抽取；季節依沉底時刻所在月份劃分，冬季為 12–2 月、春季為 3–5 月、夏季為 6–8 月、秋季為 9–11 月。')

# ------------------------------------------------------------ 各地點之導引段（八張來源分段機率圖）
def guides(s):
    n = lambda k: fno(s, k); S = SN[s]; t_b, t_c = TOT[s]
    out = {}
    mains, gaps = all_boundary(s)
    mm = [(('流場範圍' + d) if i == 0 else d, p) for i, (d, p) in enumerate(mains)]
    r = '穿越主要' + conc(mm)
    if gaps: r += '；其餘連續非零邊界段為' + lst(['%s（%s）' % (d, PC(p)) for d, p in gaps])
    out['all_b'] = ('圖 %d 為%s全年（十類合併）之邊界穿越來源分段機率圖，穿越比例為 %s。%s。' % (n('all_b'), S, PC(t_b), r))
    # 海岸全年
    ca = coast_all_st(s)
    t = '圖 %d 為%s全年（十類合併）之海岸來源分段機率圖。%s之海岸來源（%s）' % (n('all_c'), S, S, PC(t_c))
    t += ('皆位於臺灣本島。' if SUB[s] is None else '中，%s。' % SUB[s])
    if ca:
        its = sorted([(alllab(s, st), float(st['pct'])) for st in ca], key=lambda x: -x[1])
        if s == 'guishan':   # 龜山島之值已於上句寫出（9.57%，即龜山島沿岸括號值），此處不重複
            assert its[0][0] == '龜山島沿岸' and PC(its[0][1]) == '9.57%'
            t += '海岸來源主要集中於龜山島沿岸，其次為' + lst(['%s（%s）' % (d, PC(p_)) for d, p_ in its[1:]]) + '。'
        else:
            t += '海岸來源主要' + conc(its) + '。'
    else:
        t += '各海岸段之機率分散於臺灣本島海岸，無機率集中之連續海岸段。'
    out['all_c'] = t
    lo, hi = cls_range(s, 'b')
    out['cls_b'] = ('圖 %d 以十類各一格，呈現%s各類之邊界穿越來源分段機率；各類穿越比例介於 %s 至 %s，各格邊界穿越位置之分佈相近。' % (n('cls_b'), S, PC(lo), PC(hi)))
    lo, hi = cls_range(s, 'c')
    out['cls_c'] = ('圖 %d 以十類各一格，呈現%s各類之海岸來源分段機率；各類海岸來源比例介於 %s 至 %s。依圖 %d 與圖 %d 之分段分佈（邊界與海岸合計），十類之間相近，物件種類對來源之影響不明顯；同類十個沉降速度之間亦相近，沉降速度對來源之影響不明顯。'
                    % (n('cls_c'), S, PC(lo), PC(hi), n('cls_b'), n('cls_c')))
    pre = (SEA_DEF if s == 'gongliao' else '')
    out['sea_b'] = '圖 %d 為依沉底時刻劃分之%s四季邊界穿越來源分段機率圖。%s各季節之穿越比例及穿越最集中之邊界段為：%s。' % (n('sea_b'), S, pre, sea_b_clause(s))
    out['sea_c'] = '圖 %d 為依沉底時刻劃分之%s四季海岸來源分段機率圖。各季節之海岸來源比例及海岸來源最集中之海岸段為：%s。依圖 %d 與圖 %d 之分段分佈（邊界與海岸合計），%s' % (n('sea_c'), S, c_clause(s, 'coast_season'), n('sea_b'), n('sea_c'), SEA_VERDICT[s])
    head = '依沉底時刻之潮位資料劃分為大潮與小潮' if s == 'gongliao' else '依大小潮劃分'
    out['tid_b'] = '圖 %d 為%s之%s邊界穿越來源分段機率圖。大潮與小潮之穿越比例及穿越最集中之邊界段為：%s。' % (n('tid_b'), head, S, tid_b_clause(s))
    out['tid_c'] = '圖 %d 為%s之%s海岸來源分段機率圖。大潮與小潮之海岸來源比例及海岸來源最集中之海岸段為：%s。依圖 %d 與圖 %d 之分段分佈（邊界與海岸合計），大小潮兩組相近，大小潮對來源之影響不明顯。' % (n('tid_c'), head, S, c_clause(s, 'coast_tide'), n('tid_b'), n('tid_c'))
    return {k: sp(v) for k, v in out.items()}

# ------------------------------------------------------------ 圖說
BCAP = '邊界每 10 km 一段，色階為各段占全部虛擬質點之比例'
CCAP = '海岸每 5 km 一段，僅計流場範圍內海岸，色階為各段占全部虛擬質點之比例'
def captions(s):
    S = SN[s]; c = {}
    c['all_b'] = '%s邊界穿越來源分段機率（全年，十類合併）；%s，黑色括號標示機率集中之連續邊界段，灰色括號標示其餘連續非零邊界段' % (S, BCAP)
    c['all_c'] = '%s海岸來源分段機率（全年，十類合併）；%s，引線旁數值為機率集中之海岸段之機率%s' % (S, CCAP, '，插圖為馬祖列島之放大' if s == 'lienchiang' else '')
    c['cls_b'] = '%s邊界穿越來源分段機率（十類）；%s，各格標題為該類之穿越比例' % (S, BCAP)
    c['cls_c'] = '%s海岸來源分段機率（十類）；%s，各格標題為該類之海岸來源比例' % (S, CCAP)
    c['sea_b'] = '%s邊界穿越來源分段機率（依沉底時刻之季節）；%s，括號標示機率集中之連續邊界段，圓圈數字依機率由大至小編號，其後數值為該段機率，「其餘」為未編號邊界段之機率合計' % (S, BCAP)
    c['sea_c'] = '%s海岸來源分段機率（依沉底時刻之季節）；%s，圓圈數字依機率由大至小標示機率集中之海岸段，其後數值為該段機率' % (S, CCAP)
    c['tid_b'] = '%s邊界穿越來源分段機率（依大小潮）；%s，括號標示機率集中之連續邊界段，圓圈數字依機率由大至小編號，其後數值為該段機率，「其餘」為未編號邊界段之機率合計' % (S, BCAP)
    c['tid_c'] = '%s海岸來源分段機率（依大小潮）；%s，圓圈數字依機率由大至小標示機率集中之海岸段，其後數值為該段機率' % (S, CCAP)
    return {k: v for k, v in c.items()}

# ------------------------------------------------------------ 方法段、33、34、綜合說明、整體結論
def method_par():
    n = lambda k: fno('gongliao', k)
    t = ('圖 %d 至圖 %d 為貢寮之來源分段機率圖，依序呈現全年、十類、季節與大小潮之邊界穿越來源與海岸來源，每項各以邊界與海岸兩張圖呈現。'
         '來源分段之取法為：流場範圍邊界每 10 km 為一段，海岸每 5 km 為一段（僅計流場範圍內海岸）；穿越流場範圍邊界之虛擬質點，以其穿越位置歸入所在之邊界段；'
         '抵達海岸之虛擬質點，以其首次接觸海岸之位置歸入所在之海岸段，稱為海岸來源；回溯途中觸底而停止追蹤者，另計為觸底。'
         '每顆虛擬質點只歸入邊界段、海岸段與觸底三者之一，且只計入一次，故邊界穿越與海岸來源互不重疊，邊界各段、海岸各段與觸底之機率合計為 100%%。'
         '各段機率為占全部虛擬質點之比例，圖中以色階表示；黑色括號標示機率集中之連續邊界段，灰色括號標示其餘連續非零邊界段，海岸機率集中之各段以引線標示其機率。'
         % (n('all_b'), n('tid_c')))
    return sp(t)

P33 = sp('本節呈現Lagrangian系集逆向溯源含斯托克斯漂流案例之分析成果，目的在於推估各研究地點海底廢棄物之來源及其機率。情境設計如（一）所述，另以 10 個系集成員交叉，每顆虛擬質點之沉降速度自所屬類別之 10 個沉降速度中抽取；每個研究地點 100,000 顆、五個研究地點合計 500,000 顆虛擬質點。').replace('現 Lagrangian 系', '現Lagrangian系')
P34 = sp('各研究地點依序以十一張圖說明：水平軌跡總覽呈現 24 條代表虛擬質點自受體回溯之軌跡，仍在追蹤之虛擬質點比例圖呈現全部虛擬質點於回溯期間內仍在追蹤之比例，條件式來源足跡呈現虛擬質點於回溯期間停留之位置，八張來源分段機率圖依序呈現全年、十類、季節與大小潮之邊界穿越來源與海岸來源（每項各以邊界與海岸兩張圖呈現）；綜合說明與整體結論置於最後。')
P148 = sp('各研究地點十類之間、同類十個沉降速度之間與大小潮之間，分段分佈（邊界與海岸合計）相近，物件種類、沉降速度與大小潮對來源之影響不明顯。季節之影響明顯：貢寮、新竹外海、南灣與連江四季兩兩之間差異明顯；龜山島僅冬、春兩季相近，其餘季節兩兩之間差異明顯。')

def conclusion(s):
    S = SN[s]; t_b, t_c = TOT[s]
    mains, gaps = all_boundary(s)
    one = lambda d, p: '%s（%s）' % (d, PC(p))
    t = '%s：全年穿越比例為 %s，穿越主要' % (S, PC(t_b)) + conc(mains)
    # 季節最集中邊界段
    P = fig_panels(s, 'boundary_season')
    grp = {}   # 季節依最集中邊界段所在邊界（或角）分組，依首次出現次序；不寫季節之百分比
    for nm, p in zip(['冬季', '春季', '夏季', '秋季'], P):
        grp.setdefault(bside_name(s, p['brackets'][0]['segments']), []).append(nm[0])
    segs = []
    for i, (sd, ns) in enumerate(grp.items()):
        segs.append('%s季%s集中於%s' % ('、'.join(ns), '穿越' if i == 0 else '', sd))
    if len(grp) == 1: segs = ['四季穿越皆集中於%s' % list(grp)[0]]
    ca = coast_all_st(s)
    t += '；海岸來源占 %s' % PC(t_c)
    if s == 'lienchiang': t += '，其中福建沿岸（含近岸島嶼）占 29.68%，馬祖各島占 14.26%'
    if ca:
        its = sorted([(alllab(s, st).split('約')[0], float(st['pct'])) for st in ca], key=lambda x: -x[1])
        t += '，主要' + conc(its)
    else:
        t += '，皆位於臺灣本島，分散於各海岸段'
    t += '；' + '，'.join(segs) + '；' + SEA_VERDICT[s]
    if s == 'nanwan': t += '另有 1 顆（0.001%）於回溯途中觸底而停止追蹤。'
    return sp(t)

def conclusion6():
    mx = [all_boundary(s)[0][0][1] for s in SITES]
    return sp('綜合五個研究地點：全年穿越最集中邊界段之機率為 %.2f%%–%.2f%%，穿越比例為 56.06%%–97.77%%，海岸來源比例為 2.23%%–43.94%%；物件種類、沉降速度與大小潮之影響不明顯，季節之影響明顯（僅龜山島冬、春兩季相近）。' % (min(mx), max(mx)))

if __name__ == '__main__':
    print(method_par()); print()
    for s in SITES:
        print('=====', SN[s])
        g = guides(s)
        for k in KEYS[3:]: print(k, g[k]); print()
        for k, v in captions(s).items(): print('CAP', k, v)
        print('CONC', conclusion(s)); print()
    print(conclusion6())
