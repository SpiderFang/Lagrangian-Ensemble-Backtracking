# -*- coding: utf-8 -*-
import sys, copy, json, math, re, os
sys.path.insert(0, '/tmp/w'); sys.path.insert(0, '/tmp/w/build')
from lxml import etree
from lib import q, ns, ptext, lin, show, load, body_items
from edit import *
import v7data as v

SRC_XML = sys.argv[1]      # Task 1 後的 document.xml
OUT_XML = sys.argv[2]
ST = json.load(open('/tmp/w/run08/stats_run08.json'))['n50']
ALU = json.load(open('/tmp/w/run08/alu_example_run08.json'))

rows, summ = v.load()
tree = load(SRC_XML)
I = body_items(tree)          # 以原索引取得元素參照


def P(i):
    return I[i]


# ---------------------------------------------------------------- 類別順序（依代表值由小到大）
ORDER = sorted(range(1, 11), key=lambda c: summ[c]['rep'])
print('order by rep:', [v.CLASS_SHORT[c] for c in ORDER], file=sys.stderr)

# ================================================================ 正文
# (1) 說明段落
replace_in_p(P(4), '十種速度均為沉降速度，不含上升情境。其終端沉降速度如下（垂直座標向上為正，故十項均為負值）：',
             '各類速度均為沉降速度，不含上升情境。每類由 5 種代表物件、各代入阻力係數之上下界，求得 10 個終端沉降速度；'
             '每顆虛擬質點隨機取用其所屬類別 10 個速度中的一個，10 個速度各占該類虛擬質點的十分之一。'
             '下表列各類 10 個速度之範圍，以及 10 個速度之幾何平均（代表值）；代表值僅供摘要，模擬並非全類只用此單一速度。'
             '垂直座標向上為正，故各速度均為負值：')
replace_in_p(P(6), '十個速度由本研究以受力平衡求解', '各類 10 個速度由本研究以受力平衡求解')

# (2) 速度表
SHAPES = {
 4: ('不可回收塑膠', '進水之塑膠袋薄膜、塑膠袋、鋁箔複合包材片、複合包材厚片或塑膠帆布片'),
 3: ('廢紙', '吸水之薄紙片、濕紙片、濕紙板碎片、厚紙板或紙箱板'),
 1: ('保麗龍', '吸水或生物附著之發泡板、碎塊、箱體殘片、板狀殘片或箱體側板'),
 5: ('廢漁網漁具', '單根網線、單絲網線、網線截段、細繩或繩索'),
 6: ('寶特瓶', '瓶身碎片、完整進水瓶身（600 mL、1250 mL）或壓扁、重壓扁之瓶身'),
 2: ('竹木', '水浸飽和之細竹枝、小木條、中型木條、大型木段或大型竹木段'),
 7: ('可回收塑膠', '硬質塑膠小碎片、硬質塑膠容器碎片、塑膠盒板片、厚塑膠板、橡膠塊或塑膠籃側板'),
 8: ('鋁罐', '罐身碎片、完整進水罐（330 mL、500 mL）或半壓扁、重壓扁之罐身'),
 9: ('鐵罐', '罐身碎片、小型或完整進水罐，或半壓扁、重壓扁之罐身'),
 10: ('玻璃瓶', '完整進水瓶（330 mL 小瓶與較大者）、等效球徑 10、20 mm 之緻密碎片或瓶身曲殼薄片'),
}
tb = P(5)
trs = tb.findall(q('w:tr'))
hdr, body_rows = trs[0], trs[1:]
old_class_of_row = {}
# 原列順序即舊代表值順序：4,1,3,5,2,6,7,8,9,10
OLD_ORDER = [4, 1, 3, 5, 2, 6, 7, 8, 9, 10]
for cid, tr in zip(OLD_ORDER, body_rows):
    old_class_of_row[cid] = tr
assert ''.join(x.text for x in old_class_of_row[4].findall('.//' + q('w:t'))).startswith('不可回收')

# 新增一欄（複製最後一欄）、重設欄寬
GRID = [1100, 1650, 2757, 1650, 1300]
grid = tb.find(q('w:tblGrid'))
for gc in list(grid):
    grid.remove(gc)
for w_ in GRID:
    gc = etree.SubElement(grid, q('w:gridCol')); gc.set(q('w:w'), str(w_))


def fix_cells(tr):
    tcs = tr.findall(q('w:tc'))
    if len(tcs) == 4:
        new = copy.deepcopy(tcs[3])
        tcs[3].addnext(new)
        tcs = tr.findall(q('w:tc'))
    for tc, w_ in zip(tcs, GRID):
        tc.find(q('w:tcPr')).find(q('w:tcW')).set(q('w:w'), str(w_))
    return tcs


def set_run_text_at(p, text):
    """段落沿用第一個 run 的格式，其餘 run 移除"""
    rs = p.findall(q('w:r'))
    for r in rs[1:]:
        p.remove(r)
    set_rtext(rs[0], text)


def set_speed_cell(tc, lo, hi=None):
    """速度格：'−' 以 MS Gothic 的 run 呈現（沿用原格式）；範圍以換行分成兩行"""
    p = tc.find(q('w:p'))
    rs = p.findall(q('w:r'))
    r_minus, r_num = rs[0], rs[1]
    for r in rs:
        p.remove(r)
    def add(minus_r, num_r, num):
        a = copy.deepcopy(minus_r); b = copy.deepcopy(num_r)
        set_rtext(b, num)
        p.append(a); p.append(b)
    add(r_minus, r_num, lo)
    if hi is not None:
        br = etree.SubElement(p, q('w:r'))
        rp = copy.deepcopy(r_num.find(q('w:rPr')))
        br.append(rp)
        etree.SubElement(br, q('w:br'))
        r_to = copy.deepcopy(r_num); set_rtext(r_to, '至 ')
        p.append(r_to)
        add(r_minus, r_num, hi)


for tr in [hdr] + body_rows:
    fix_cells(tr)
# 表頭
htcs = hdr.findall(q('w:tc'))
hp = htcs[3].find(q('w:p'))
rs = hp.findall(q('w:r'))
set_rtext(rs[0], '10 個速度之範圍'); set_rtext(rs[1], ' (m/s)')
hp5 = htcs[4].find(q('w:p'))
rs = hp5.findall(q('w:r'))
set_rtext(rs[0], '代表值'); set_rtext(rs[1], ' (m/s)')

for tr in body_rows:
    tb.remove(tr)
for cid in ORDER:
    tr = old_class_of_row[cid]
    tcs = tr.findall(q('w:tc'))
    # 代表形狀與狀態
    name, shape_txt = SHAPES[cid]
    set_cell_text(tcs[2], shape_txt)
    s = summ[cid]
    # 範圍（兩行）：新欄在 tcs[3]；其內容為舊速度格複製品
    set_speed_cell(tcs[3], v.sig3(s['wmin']), v.sig3(s['wmax']))
    set_speed_cell(tcs[4], v.sig3(s['rep']))
    tb.append(tr)
CLASS_MAIN_NAME = {cid: SHAPES[cid][0] for cid in SHAPES}

# ================================================================ 附錄
# 導言
replace_in_p(P(42), '本附錄說明正文十個終端沉降速度之求得方式', '本附錄說明正文各類 10 個終端沉降速度之求得方式')
replace_in_p(P(42), '每類海廢設定三種代表物件，各代入阻力係數之上下界，共得 6 個解，取其幾何平均為採用值。結果見第一節',
             '每類海廢設定 5 種代表物件，各代入阻力係數之上下界，共得 10 個解；模擬時每顆虛擬質點取用其所屬類別 10 個解中的一個，'
             '10 個解各占十分之一，其幾何平均為各類之代表值，供摘要之用。結果見第一節')

# 第一節 結果表
tb = P(44)
trs = tb.findall(q('w:tr'))
hdr1, body1 = trs[0], trs[1:]
replace_in_p(hdr1.findall(q('w:tc'))[2].find(q('w:p')), '採用值', '代表值')
rowmap = {}
for tr in body1:
    nm = ''.join(x.text or '' for x in tr.findall(q('w:tc'))[0].iter(q('w:t')))
    rowmap[nm] = tr
CLS_ROW_KEY = {1: '保麗龍', 2: '竹木', 3: '廢紙', 4: '不可回收塑膠（軟質包材）', 5: '廢漁網漁具', 6: '寶特瓶',
               7: '可回收塑膠（硬質塑膠、橡膠）', 8: '鋁罐', 9: '鐵罐', 10: '玻璃瓶'}
for tr in body1:
    tb.remove(tr)
for cid in ORDER:
    tr = rowmap[CLS_ROW_KEY[cid]]
    tcs = tr.findall(q('w:tc'))
    s = summ[cid]
    set_cell_text(tcs[1], v.rnd(s['wmin'], 5))
    set_cell_text(tcs[2], v.sig3(s['rep']))
    set_cell_text(tcs[3], v.rnd(s['wmax'], 4))
    tb.append(tr)

# 第一節 說明段
p46 = P(46)
replace_in_p(p46, '採用值為該類 6 個解（3 種代表物件 × 阻力係數上下界）之幾何平均，取三位有效數字。同類 6 個解可相差一個數量級以上，'
                   '幾何平均不受最大值主導，較算術平均更能代表其中間值。6 個解等權；寶特瓶、鋁罐、鐵罐之兩種壓扁狀態速度相同（補充二），'
                   '等同壓扁狀態占 2/3 權重，其影響見第八節。表中',
             '代表值為該類 10 個解（5 種代表物件 × 阻力係數上下界）之幾何平均，取三位有效數字，僅供各類摘要與比較；'
             '模擬時每顆虛擬質點隨機取用其所屬類別 10 個解中的一個，10 個解各占該類虛擬質點的十分之一，並非全類只用代表值。'
             '同類 10 個解中，保麗龍、竹木、廢紙、不可回收塑膠、廢漁網漁具與可回收塑膠六類之最大值為最小值的 10 倍以上，'
             '幾何平均不受最大值主導，較算術平均更能代表其中間值。寶特瓶、鋁罐、鐵罐之兩種壓扁狀態速度相同（補充二），'
             '故這三類各有 8 個相異速度，其中壓扁狀態之速度各出現兩次，合計占 4/10，其影響見第八節。表中')
replace_in_p(p46, '下界與上界為該類 6 個解之最小值與最大值。', '下界與上界為該類 10 個解之最小值與最大值。')

# 第三節
replace_in_p(P(60), '對採用值之影響見第八節', '對代表值之影響見第八節')

# 第五節（一）
p76 = P(76)
replace_in_p(p76, '各類海廢設定三種代表物件，共 30 組。片狀與柱狀物件之最長邊取 1、5、10、15、20、30 與 50 cm',
             '各類海廢設定 5 種代表物件，共 50 組。片狀與柱狀物件之最長邊取 1、2.5、5、10、15、20、25、30 與 50 cm')
replace_in_p(p76, '以及等效球徑 2 cm 之玻璃碎片', '以及等效球徑 1 cm 與 2 cm 之玻璃碎片')
n_slab, n_in, n_lt = ST['stats']['n_slab'], ST['stats']['n_slab_25_2500'], ST['stats']['n_slab_lt25']
assert (n_slab, n_in, n_lt) == (30, 22, 8)
replace_in_p(p76, '本案片狀物件之長寬多在此範圍內。',
             '本案 %d 個薄板物件之投影面積為 %g–%g cm²，其中 %d 個介於 25–2500 cm²，其餘 %d 個小於 25 cm²。' % (
                 n_slab, ST['stats']['slab_area_min'], ST['stats']['slab_area_max'], n_in, n_lt))
replace_in_p(p76, '得 60 個解：', '得 100 個解：')

# 第五節 大表
tb = P(77)
trs = tb.findall(q('w:tr'))
hdr5 = trs[0]
T_FIRST, T_MID, T_LAST = trs[1], trs[2], trs[3]
T_VENV = trs[16].findall(q('w:tc'))[3].find(q('w:p'))
T_DEQ = trs[29].findall(q('w:tc'))[3].find(q('w:p'))
for tr in trs[1:]:
    tb.remove(tr)
byclass = {}
for r in rows:
    byclass.setdefault(r['cid'], []).append(r)
for cid in range(1, 11):
    rs_ = byclass[cid]
    assert len(rs_) == 5
    for k, r in enumerate(rs_):
        tmpl = T_FIRST if k == 0 else (T_LAST if k == 4 else T_MID)
        tr = copy.deepcopy(tmpl)
        tcs = tr.findall(q('w:tc'))
        if k == 0:
            set_cell_text(tcs[0], CLASS_LABEL := v.CLASS_LABEL[cid])
        set_cell_text(tcs[1], r['name'])
        set_cell_text(tcs[2], r['geo'])
        if isinstance(r['size'], tuple):
            kind, val = r['size']
            tmplp = T_VENV if kind == 'Venv' else T_DEQ
            newp = copy.deepcopy(tmplp)
            unit = 'cm³' if kind == 'Venv' else 'mm'
            rr = newp.findall(q('w:r'))[-1]
            set_rtext(rr, ' = %d %s' % (val, unit))
            if k == 4:   # 組內最後一列不設 keepNext
                for kn in newp.iter(q('w:keepNext')):
                    kn.getparent().remove(kn)
            oldp = tcs[3].find(q('w:p'))
            tcs[3].replace(oldp, newp)
        else:
            set_cell_text(tcs[3], r['size'])
        set_cell_text(tcs[4], v.sci_u(r['V']))
        set_cell_text(tcs[5], v.sci_u(r['A']))
        set_cell_text(tcs[6], '%d' % int(round(r['rho'] + 1e-9)))
        set_cell_text(tcs[7], '%.1f–%.1f' % (r['cdlo'], r['cdhi']))
        set_cell_text(tcs[8], '%s–%s' % (v.rnd(r['wlo'], 4), v.rnd(r['whi'], 4)))
        tb.append(tr)

# 第五節 表後說明
p78 = P(78)
replace_in_p(p78, '30 組代表物件之雷諾數介於', '50 組代表物件之雷諾數介於')
n1 = ST['stats']['n_sol_lt1e3']; o1 = ST['stats']['n_obj_lt1e3']
smin, smax = ST['stats']['share_lt1e3_min'], ST['stats']['share_lt1e3_max']
replace_in_p(p78, '。尺寸最小之代表物件雷諾數低於', '。雷諾數低於')
replace_in_p(p78, '，已超出教科書值之適用範圍（', '之解有 %d 個，來自 %d 組尺寸較小或極薄之代表物件，已超出教科書值之適用範圍（' % (n1, o1))
replace_in_p(p78, '第一項約占阻力係數四分之一至三分之二，用以', '第一項約占阻力係數 %d%%–%d%%，用以' % (round(smin), round(smax)))

# 參數來源表
tb = P(80)
cells = [tc for tr in tb.findall(q('w:tr')) for tc in tr.findall(q('w:tc'))]
def cell_ps(tc): return tc.findall(q('w:p'))
std_ps = cell_ps(cells[4])      # 標準物性
spec_ps = cell_ps(cells[7])     # 可查證規格
set_ps = cell_ps(cells[10])     # 本案設定
replace_in_p(find_p(std_ps, 'PET'), '用於寶特瓶殼體；', '用於寶特瓶殼體及瓶身碎片；')
replace_in_p(find_p(std_ps, '鋁'), '用於鋁罐殼體；', '用於鋁罐殼體及罐身碎片；')
replace_in_p(find_p(std_ps, '鋼'), '用於鐵罐殼體（鍍錫鋼板）；', '用於鐵罐殼體（鍍錫鋼板）及罐身碎片；')
replace_in_p(find_p(std_ps, '尼龍'), '用於廢漁網漁具之網線截段；', '用於廢漁網漁具之單絲網線與網線截段；')
sp = set_ps[0]
replace_in_p(sp, '薄壁容器之殼重與外包絡體積（330 mL 鋁罐殼重 13 g、外包絡體積 390 cm³；600 mL 寶特瓶殼重 25 g、外包絡體積 620 cm³；'
                 '鐵罐殼重 55 g、外包絡體積 425 cm³；玻璃瓶殼重 500 g、外包絡體積 700 cm³）；吸水或生物附著後之有效密度（保麗龍 1026/1060/1100、'
                 '竹木 1030/1090/1150、廢紙 1050/1120/1200、不可回收塑膠之塑膠袋薄膜 1030 與鋁箔複合包材 1200、廢漁網漁具之單根網線 1050 與繩索 1200 kg/m³）',
             '薄壁容器之殼重與外包絡體積（330 mL 鋁罐殼重 13 g、外包絡體積 390 cm³；500 mL 鋁罐殼重 17 g、外包絡體積 590 cm³；'
             '600 mL 寶特瓶殼重 25 g、外包絡體積 620 cm³；1250 mL 寶特瓶殼重 40 g、外包絡體積 1300 cm³；'
             '鐵罐殼重 55 g、外包絡體積 425 cm³；小鐵罐殼重 30 g、外包絡體積 220 cm³；'
             '玻璃瓶殼重 500 g、外包絡體積 700 cm³；330 mL 玻璃瓶殼重 200 g、外包絡體積 400 cm³）；'
             '容器碎片之厚度（寶特瓶 0.3 mm、鋁罐 0.1 mm、鐵罐 0.2 mm、玻璃瓶身 3 mm）；吸水或生物附著後之有效密度（保麗龍 1026/1040/1060/1100、'
             '竹木 1030/1060/1090/1150、廢紙 1050/1080/1120/1150/1200、不可回收塑膠之塑膠袋薄膜與塑膠袋 1030、鋁箔複合包材與複合包材厚片 1200、'
             '廢漁網漁具之單根網線 1050、細繩與繩索 1200 kg/m³）')


# ================================================================ 第六節 計算實例：鋁罐
from decimal import Decimal, ROUND_HALF_UP
OBJ = ALU['objects']            # 5 種代表物件（碎片, 330, 半壓扁, 重壓扁, 500）
SOL = ALU['solutions']
LABEL = ['罐身碎片', '完整罐（330 mL）', '半壓扁', '重壓扁', '完整罐（500 mL）']
SHAPE_TXT = ['薄板', '鈍體', '薄板', '薄板', '鈍體']
assert [o['name'] for o in OBJ] == LABEL


def sig(x, n):
    return '{:.{}g}'.format(x, n)

replace_in_p(P(83), '逐步演算一類採用值之完整過程', '逐步演算一類代表值之完整過程')

# --- 輸入段落 P84：沿用原 OMML，重組內容
p84 = P(84)
oms = p84.findall(q('m:oMath'))
M_msolid, M_rhosolid, M_Venv, M_V330, M_V150, M_V48, M_rhow, M_nu, M_g, M_cd = [copy.deepcopy(x) for x in oms]
rpr_body = copy.deepcopy(p84.find(q('w:r')).find(q('w:rPr')))
fac = MathFactory(sz=None, szcs='28')


def sci_m(mant, exp):
    return fac.omath(fac.sci(mant, exp))


def build_p(p, parts):
    for ch in list(p):
        if ch.tag != q('w:pPr'):
            p.remove(ch)
    for x in parts:
        if isinstance(x, str):
            p.append(text_run(x, rpr_body))
        else:
            p.append(x)

def mm(m_el, old, new):
    m_el = copy.deepcopy(m_el)
    for t_ in m_el.iter(q('m:t')):
        if t_.text == old:
            t_.text = new
    return m_el

sp = ' '
parts84 = [
 '輸入：鋁密度', sp, M_rhosolid, ' kg/m³', '；330 mL 罐殼重', sp, M_msolid, ' kg', '、500 mL 罐殼重', sp,
 mm(M_msolid, '=0.013', '=0.017'), ' kg', '（半壓扁與重壓扁由 330 mL 罐壓扁而得，殼重同為 0.013 kg）',
 '；5 種代表物件之外包絡體積', sp, M_Venv, '：330 mL 完整罐', sp, M_V330, ' m³', '（390 cm³）',
 '，500 mL 完整罐', sp, sci_m('5.90', -4), ' m³', '（590 cm³）',
 '，半壓扁 100×60×25 mm＝', M_V150, ' m³', '，重壓扁 100×60×8 mm＝', M_V48, ' m³',
 '，罐身碎片 50×30×0.1 mm＝', sci_m('1.50', -7), ' m³', '（實心薄片，無空腔，密度直接取鋁密度）',
 '；流體常數', sp, M_rhow, ' kg/m³', '、', M_nu, ' m²/s', '、', M_g, ' m/s²',
 '；形態常數', sp, M_cd, '：兩種完整罐為鈍體 0.6–1.2，半壓扁、重壓扁與罐身碎片為薄板 1.1–2.0（第三節）。',
]
build_p(p84, parts84)

replace_in_p(P(86), '以完整罐為例', '以 330 mL 完整罐為例')
replace_in_p(P(88), '半壓扁與重壓扁同法，結果如下表：', '其餘代表物件同法（罐身碎片為實心薄片，密度直接取鋁密度），結果如下表：')

# --- 表 89：有效密度
tb = P(89)
trs = tb.findall(q('w:tr'))
tmpl = trs[1]
for tr in trs[1:]:
    tb.remove(tr)


def set_sci_cell(tc, mant, exp):
    ts = list(tc.iter(q('m:t')))
    assert len(ts) == 3 and ts[0].text.endswith('×'), [t_.text for t_ in ts]
    ts[0].text = '%s×' % mant
    ts[2].text = str(exp)


def sci_str(x, nd):
    m, e = v.sci(x, nd)
    return m, e

for k, o in enumerate(OBJ):
    tr = copy.deepcopy(tmpl)
    tcs = tr.findall(q('w:tc'))
    set_cell_text(tcs[0], LABEL[k])
    set_sci_cell(tcs[1], *sci_str(o['V'], 3))
    if o['Vwater'] is None:
        # V_water 無空腔：以文字「—」取代
        p_ = tcs[2].find(q('w:p'))
        for ch in list(p_):
            if ch.tag != q('w:pPr'):
                p_.remove(ch)
        rp20 = copy.deepcopy(tcs[3].find(q('w:p')).find(q('w:r')).find(q('w:rPr')))
        p_.append(text_run('—', rp20))
        set_cell_text(tcs[3], '2700（實心）')
    else:
        set_sci_cell(tcs[2], *sci_str(o['Vwater'], 3))
        set_cell_text(tcs[3], v.rnd(o['rho'], 1))
    tb.append(tr)

# --- 幾何
replace_in_p(P(91), '完整罐身，依第四節（二）', '330 mL 完整罐身，依第四節（二）')
replace_in_p(P(93), '半壓扁與重壓扁為壓扁後之薄板', '半壓扁與重壓扁為壓扁後之薄板（以半壓扁為例）')
# A9 表(94) 之後新增一段：碎片與 500 mL 罐
o_frag, o_500 = OBJ[0], OBJ[4]
new_p = copy.deepcopy(P(98))
parts_new = [
 '罐身碎片亦為薄板：', fac.omath(fac.sub('A', 'proj'), fac.r('=0.050×0.030=1.50×'), fac.sup('10', '-3')), ' m²',
 '，', fac.omath(fac.sub('d', 'A'), fac.r('=%s' % v.rnd(o_frag['dA'], 5))), ' m',
 '；500 mL 完整罐同 330 mL 完整罐之算法：', fac.omath(fac.sub('D', 'eq'), fac.r('=%s' % v.rnd(o_500['dA'], 4))), ' m',
 '，', fac.omath(fac.sub('A', 'proj'), fac.r('=%s×' % sci_str(o_500['A'], 3)[0]), fac.sup('10', str(sci_str(o_500['A'], 3)[1]))), ' m²。',
]
build_p(new_p, parts_new)
P(94).addnext(new_p)

# --- 表 97：R 與 N
tb = P(97)
trs = tb.findall(q('w:tr'))
tmpl = trs[1]
for tr in trs[1:]:
    tb.remove(tr)
for k, o in enumerate(OBJ):
    tr = copy.deepcopy(tmpl)
    tcs = tr.findall(q('w:tc'))
    set_cell_text(tcs[0], LABEL[k])
    set_cell_text(tcs[1], sig(o['R'], 4))
    set_cell_text(tcs[2], sig(o['N'], 4))
    tb.append(tr)

# --- 步驟四 標示 330 mL
replace_in_p(P(99), '以完整罐、', '以 330 mL 完整罐、')
replace_in_p(P(102), '完整罐、', '330 mL 完整罐、')

# --- 步驟五：十個解
replace_in_p(P(106), '六個解', '十個解')
replace_in_p(P(107), '六個解（3 種代表物件', '十個解（5 種代表物件')
tb = P(108)
trs = tb.findall(q('w:tr'))
tmpl = trs[1]
for tr in trs[1:]:
    tb.remove(tr)
by_obj = {}
for s in SOL:
    by_obj.setdefault(s['obj'], []).append(s)
for k, o in enumerate(OBJ):
    for s in sorted(by_obj[o['name']], key=lambda x: x['cd_inf']):
        tr = copy.deepcopy(tmpl)
        tcs = tr.findall(q('w:tc'))
        set_cell_text(tcs[0], LABEL[k])
        set_cell_text(tcs[1], SHAPE_TXT[k])
        set_cell_text(tcs[2], '%.1f' % s['cd_inf'])
        set_cell_text(tcs[3], '%d' % int(round(s['Re'])))
        set_cell_text(tcs[4], '%.4f' % s['Cd'])
        set_cell_text(tcs[5], '%.5f' % s['w'])
        tb.append(tr)
wmin, wmax = ALU['w_min'], ALU['w_max']
replace_in_p(P(109), '六個解之範圍 0.10787–0.17528 m/s', '十個解之範圍 %s–%s m/s' % (v.rnd(wmin, 5), v.rnd(wmax, 5)))
replace_in_p(P(109), '下界與上界。', '下界與上界；半壓扁與重壓扁之解逐位相同，故十個解中有兩個數值各出現兩次（相異者 8 個）。')

# --- 步驟六：幾何平均 (A12)
cell = P(111).findall('.//' + q('w:tc'))[0]
om = cell.find('.//' + q('m:oMath'))
# 取得各 w（依表列順序，5 位小數）
wlist = []
for k, o in enumerate(OBJ):
    for s in sorted(by_obj[o['name']], key=lambda x: x['cd_inf']):
        wlist.append(s['w'])
w5 = [v.rnd(w, 5) for w in wlist]
ts = list(om.iter(q('m:t')))
for t_ in ts:
    if t_.text == '6':
        t_.text = '10'
    elif t_.text == '1/6':
        t_.text = '1/10'
inner = [m_ for m_ in om.iter(q('m:m'))][1]
mrs = inner.findall(q('m:mr'))
# 三列：4 + 3 + 3 個
lines = ['×'.join(w5[0:4]), '×' + '×'.join(w5[4:7]), '×' + '×'.join(w5[7:10])]
for k_ in range(len(mrs), 3):
    inner.append(copy.deepcopy(mrs[-1]))
mrs = inner.findall(q('m:mr'))
for mr, ln in zip(mrs, lines):
    list(mr.iter(q('m:t')))[0].text = ln
for t_ in ts:
    if t_.text is not None and t_.text.startswith('=0.1321'):
        t_.text = '=%s ' % v.rnd(ALU['geo_mean'], 4)
assert any((t_.text or '').startswith('=0.1077') for t_ in om.iter(q('m:t')))

p112 = P(112)
replace_in_p(p112, ' 0.132 m/s', ' %s m/s' % v.sig3(ALU['geo_mean']), count=1)
replace_in_p(p112, '鋁罐之採用值為', '鋁罐之代表值為')
replace_in_p(p112, ' −0.132 m/s', ' −%s m/s' % v.sig3(ALU['geo_mean']), count=1)

# ================================================================ 第七節
tb = P(114)
trs = tb.findall(q('w:tr'))
al = summ[8]; fe = summ[9]
def set_two_line(tc, line1, line2):
    set_cell_text(tc, line1)
    p_ = tc.find(q('w:p'))
    r0 = p_.find(q('w:r'))
    br = etree.Element(q('w:r')); br.append(copy.deepcopy(r0.find(q('w:rPr')))); etree.SubElement(br, q('w:br'))
    r2 = copy.deepcopy(r0); set_rtext(r2, line2)
    r0.addnext(br); br.addnext(r2)
set_two_line(trs[1].findall(q('w:tc'))[1], v.sig3(al['rep']), '（%s–%s）' % (v.sig3(al['wmin']), v.sig3(al['wmax'])))
set_two_line(trs[2].findall(q('w:tc'))[1], v.sig3(fe['rep']), '（%s–%s）' % (v.sig3(fe['wmin']), v.sig3(fe['wmax'])))
set_cell_text(trs[3].findall(q('w:tc'))[1], '%s / %s / %s' % (v.sig3(summ[1]['rep']), v.sig3(summ[2]['rep']), v.sig3(summ[3]['rep'])))
replace_in_p(trs[2].findall(q('w:tc'))[3].find(q('w:p')), '本案罐體等效直徑約為瓶蓋十倍', '本案完整罐體等效球徑約為瓶蓋之 7–9 倍')
replace_in_p(P(115), '（單位均為 m/s。）', '（單位均為 m/s；本案推導值為代表值，括號內為 10 個解之範圍。）')
p116 = P(116)
replace_in_p(p116, '與本案罐體之尺寸', '與本案完整罐體之尺寸')
replace_in_p(p116, '鋁罐各代表物件之導出範圍', '鋁罐 10 個解之範圍')
replace_in_p(p116, ' 0.108–0.175 m/s ', ' %s–%s m/s ' % (v.sig3(al['wmin']), v.sig3(al['wmax'])))
replace_in_p(p116, '順序亦與推導結果一致', '順序亦與兩類之代表值（鐵罐 %s、鋁罐 %s m/s）一致' % (v.sig3(fe['rep']), v.sig3(al['rep'])))

# ================================================================ 第八節
replace_in_p(P(118), '以下各項檢驗僅改變單一設定，其餘條件不變：', '以下各項檢驗以第五節 50 組代表物件重算，僅改變單一設定，其餘條件不變：')
VAR = ST['sens']['variants']


def f1(x):
    return str(Decimal(repr(abs(x))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))

def sgn(x):
    return ('+' if x > 0 else '−') + f1(x)

def rng(pcts, lo_to_hi_by_abs=True):
    xs = [p_ for p_ in pcts if abs(p_) > 0.05]
    return min(xs), max(xs)

tb = P(119)
trs = tb.findall(q('w:tr'))
def cellp(r, c): return trs[r].findall(q('w:tc'))[c].find(q('w:p'))
replace_in_p(cellp(0, 1), '十類採用值變化', '十類代表值變化')
# r1 教科書
a, b = VAR['textbook']['pct_min'], VAR['textbook']['pct_max']
replace_in_p(cellp(1, 1), '+1.1%', sgn(a) + '%'); replace_in_p(cellp(1, 1), '+2.9%', sgn(b) + '%')
# r2 去除第一項
a, b = VAR['no_first_term']['pct_min'], VAR['no_first_term']['pct_max']
replace_in_p(cellp(2, 1), '+6.1%', sgn(a) + '%'); replace_in_p(cellp(2, 1), '+26.3%', sgn(b) + '%')
# r3 圓柱 d_n
pc = VAR['cyl_dn']['pct']
replace_in_p(cellp(3, 1), '−1.9%', sgn(pc['2']) + '%'); replace_in_p(cellp(3, 1), '−7.7%', sgn(pc['5']) + '%')
assert VAR['cyl_dn']['order_same'] is False
set_cell_text(trs[3].findall(q('w:tc'))[2], '保麗龍與廢漁網漁具互換')
# r4 壓扁代表物件重複只計一次
set_cell_text(trs[4].findall(q('w:tc'))[0], '寶特瓶、鋁罐、鐵罐中速度相同之兩個壓扁代表物件只計一次（8 個相異速度各占八分之一）')
pc = VAR['dedupe']['pct']
set_cell_text(trs[4].findall(q('w:tc'))[1], '寶特瓶 %s%%、鋁罐 %s%%、鐵罐 %s%%，其餘不變' % (sgn(pc['6']), sgn(pc['8']), sgn(pc['9'])))
assert VAR['dedupe']['order_same'] is True
# r5 黏滯係數
n10, n28 = VAR['nu10']['pct'], VAR['nu28']['pct']
lo10, hi10 = max(n10.values()), min(n10.values())     # 負值：小幅在前
lo28, hi28 = min(n28.values()), max(n28.values())
cell5 = cellp(5, 1)
replace_in_p(cell5, '−0.5%', sgn(lo10) + '%'); replace_in_p(cell5, '−2.4%', sgn(hi10) + '%')
replace_in_p(cell5, '+0.3%', sgn(lo28) + '%'); replace_in_p(cell5, '+1.5%', sgn(hi28) + '%')
for k_ in ('textbook', 'no_first_term', 'dedupe', 'nu10', 'nu28'):
    assert VAR[k_]['order_same'] is True, k_

# P120 加註
p120 = P(120)
last = p120.findall(q('w:r'))[-1]
base = ST['sens']['base']
gap = (base['5'] / base['1'] - 1) * 100
nr = copy.deepcopy(last)
set_rtext(nr, '圓柱改以等體積球徑計算 Re 時，互換排序之保麗龍與廢漁網漁具，其代表值原僅相差 %s%%（%s 與 %s m/s）。' % (
    f1(gap), v.sig3(summ[1]['rep']), v.sig3(summ[5]['rep'])))
last.addnext(nr)

# ================================================================ 補充一
replace_in_p(P(122), '全部 30 組代表物件', '全部 50 組代表物件')
assert ST['stats']['n_iter_max'] < 100

tree.write(OUT_XML, xml_declaration=True, encoding='UTF-8', standalone=True)
print('written', OUT_XML)
