
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
