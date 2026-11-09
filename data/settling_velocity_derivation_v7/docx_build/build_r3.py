# -*- coding: utf-8 -*-
"""r3：以使用者最新的 3-4-3_run08.docx 為底，新增附錄「推導流程」並做一致性修正。
用法：python3 build_r3.py <in document.xml> <out document.xml>
"""
import sys, copy, re, math
sys.path.insert(0, '/tmp/w'); sys.path.insert(0, '/tmp/w/build')
from lxml import etree
from lib import q, ptext, lin, load, body_items
from edit import *
sys.path.insert(0, '/tmp/w/v7')
import v7data as v

SRC, OUT = sys.argv[1], sys.argv[2]
tree = load(SRC)
I = body_items(tree)


def P(i):
    return I[i]


def strip_proof(p):
    for e in list(p):
        if e.tag == q('w:proofErr'):
            p.remove(e)


def clear_runs(p):
    for e in list(p):
        if e.tag != q('w:pPr'):
            p.remove(e)


# ------------------------------------------------------------ 行內數學式工具
MF = MathFactory(None, '28')
FONT = lambda: None


def ctrl_rpr():
    c = etree.Element(q('m:ctrlPr'))
    rp = etree.SubElement(c, q('w:rPr'))
    f = etree.SubElement(rp, q('w:rFonts')); f.set(q('w:ascii'), 'Cambria Math'); f.set(q('w:hAnsi'), 'Cambria Math')
    return c


def bar(base_text):
    b = etree.Element(q('m:bar'))
    pr = etree.SubElement(b, q('m:barPr'))
    pos = etree.SubElement(pr, q('m:pos')); pos.set(q('m:val'), 'top')
    pr.append(ctrl_rpr())
    e = etree.SubElement(b, q('m:e')); e.append(MF.r(base_text))
    return b


def dlm(*parts):
    d = etree.Element(q('m:d'))
    pr = etree.SubElement(d, q('m:dPr')); pr.append(ctrl_rpr())
    e = etree.SubElement(d, q('m:e'))
    for p_ in parts:
        e.append(p_)
    return d


def supx(base_el, sup_text):
    s = etree.Element(q('m:sSup'))
    pr = etree.SubElement(s, q('m:sSupPr')); pr.append(ctrl_rpr())
    e = etree.SubElement(s, q('m:e')); e.append(base_el)
    b = etree.SubElement(s, q('m:sup')); b.append(MF.r(sup_text))
    return s


TOKEN = re.compile(r'(?P<base>Re|10|[A-Za-zρνω∞\*])(?:_\{(?P<sub>[^}]*)\}|\^\{(?P<sup>[^}]*)\})?')


def M(spec):
    """簡易標記 -> m:oMath。X_{sub}、X^{sup}；其餘字元一律當作一般文字。"""
    parts, buf, pos = [], '', 0
    while pos < len(spec):
        m = TOKEN.match(spec, pos)
        if m and (m.group('sub') is not None or m.group('sup') is not None):
            if buf:
                parts.append(MF.r(buf)); buf = ''
            if m.group('sub') is not None:
                s = m.group('sub')
                parts.append(MF.sub(m.group('base'), s, subplain=len(s) > 1 and s.isalpha()))
            else:
                parts.append(MF.sup(m.group('base'), m.group('sup')))
            pos = m.end()
        else:
            buf += spec[pos]; pos += 1
    if buf:
        parts.append(MF.r(buf))
    return MF.omath(*parts)


RPR = lambda bold=False: rpr_from(szcs='28', bold=bold)


def mkpara(ppr_src, parts, ind=None, spacing=None, keep_next=False):
    p = etree.Element(q('w:p'))
    ppr = copy.deepcopy(ppr_src.find(q('w:pPr')))
    for k in ppr.findall(q('w:keepNext')):
        ppr.remove(k)
    if keep_next:
        ppr.insert(0, etree.Element(q('w:keepNext')))
    if spacing:
        sp = ppr.find(q('w:spacing'))
        for k, val in spacing.items():
            sp.set(q('w:' + k), val)
    if ind:
        i = etree.Element(q('w:ind'))
        for k, val in ind.items():
            i.set(q('w:' + k), val)
        jc = ppr.find(q('w:jc'))
        jc.addprevious(i)
    p.append(ppr)
    for x in parts:
        if isinstance(x, str):
            p.append(text_run(x, RPR()))
        elif isinstance(x, tuple) and x[0] == 'b':
            p.append(text_run(x[1], RPR(True)))
        else:
            p.append(x)
    return p


# ------------------------------------------------------------ 先取用既有數學式（避免之後被改動）
om122 = list(P(122).iter(q('m:oMath')))
OM_N = copy.deepcopy(om122[0])        # N = 2VgR/A_proj
OM_RE = copy.deepcopy(om122[2])       # Re = w d_A/ν
OM_WSTAR = copy.deepcopy(om122[4])    # w* = √(N/C_d)
OM_UPD = copy.deepcopy(om122[5])      # w ← 0.5w + 0.5w*
om53 = list(P(53).iter(q('m:oMath')))
OM_RHOW = copy.deepcopy(om53[0])      # ρw = 1025
OM_NU = copy.deepcopy(om53[1])        # ν = 1.05×10^-6
OM_G = copy.deepcopy(om53[4])         # g = 9.81


def fresh(o):
    return copy.deepcopy(o)


# ================================================================ 附錄說明段（I[41]）→ 一句概述
p41 = P(41)
clear_runs(p41)
for s in ('本附錄說明正文各類', ' 10 ', '個終端沉降速度之求得方式：依受力平衡，由 5 種代表物件之體積、投影面積、密度與阻力係數計算終端沉降速度，'
          '阻力係數之形狀項取流體力學教科書之標準值（Çengel & Cimbala, 2018），計算流程見下節「推導流程」；結果見第一節，完整計算實例見第六節。'):
    p41.append(text_run(s, RPR()))

# ================================================================ 新增「推導流程」
head = copy.deepcopy(P(42)); strip_proof(head)
clear_runs(head)
head.append(text_run('推導流程', RPR()))
NORMAL = P(41)
STEPH = P(84)
ITEM = dict(spacing=dict(before='0', after='60'), ind=dict(left='700', hanging='700'))


def item(num, *parts):
    return mkpara(NORMAL, [num + '　'] + list(parts), **ITEM)


new = []
new.append(head)
new.append(mkpara(NORMAL, [('b', '輸入：'), '各類 5 種代表物件之幾何尺寸（薄板 ', M('L'), '、', M('W'), '、', M('t'), '；圓柱 ', M('d'), '、', M('L'),
                           '；鈍體 ', M('V_{env}'), ' 或 ', M('D_{eq}'), '）、密度 ', M('ρ_{p}'), '（薄壁中空容器則為殼重 ', M('m_{solid}'),
                           '、材質密度 ', M('ρ_{solid}'), ' 與外包絡體積 ', M('V_{env}'), '），以及形態之 ', M('C_{d,∞}'),
                           ' 上下界（第三節）；海水密度 ', fresh(OM_RHOW), ' kg/m³、運動黏滯係數 ', fresh(OM_NU), ' m²/s、重力加速度 ', fresh(OM_G), ' m/s²。']))
new.append(mkpara(NORMAL, [('b', '輸出：'), '各類 10 個終端沉降速度 ', M('w_{1}'), '…', M('w_{10}'), '，與其代表值 ', MF.omath(bar('w')), '。']))

new.append(mkpara(STEPH, [('b', '步驟 1　逐一處理 10 類中每一類（'), MF.omath(MF.r('c')), ('b', '）之 5 種代表物件（'), MF.omath(MF.r('j')), ('b', '）：')], keep_next=True))
new.append(item('1.1', '若為薄壁中空容器，依式(A5) 由 ', M('m_{solid}'), '、', M('ρ_{solid}'), '、', M('V_{env}'), ' 求有效密度 ', M('ρ_{p}'),
                '；否則 ', M('ρ_{p}'), ' 取第五節表列值。'))
new.append(item('1.2', '依第四節之幾何模型計算 ', M('V'), '、', M('A_{proj}'), '、', M('d_{A}'), '。'))
new.append(item('1.3', '計算淹沒比重 ', M('R=ρ_{p}/ρ_{w}-1'), '。若 ', M('R≤0'), '，物件不下沉，', M('w=0'), '；本案 50 組皆 ', M('R>0'), '。'))
new.append(item('1.4', '計算 ', fresh(OM_N), '。'))
new.append(item('1.5', M('C_{d,∞}'), ' 分別取下界與上界，各執行一次步驟 2。'))

new.append(mkpara(STEPH, [('b', '步驟 2　迭代求終端沉降速度（式(A10)）：')], keep_next=True))
new.append(item('2.1', '設初值 ', MF.omath(MF.r('w←1×'), MF.sup('10', '-4')), ' m/s。'))
new.append(item('2.2', '計算 ', fresh(OM_RE), '；',
                MF.omath(MF.sub('C', 'd'), MF.r('=24/Re×(1+0.15'), MF.r(' '), MF.sup('Re', '0.687'), MF.r(')+'), MF.sub('C', 'd,∞')),
                '（式(A3)）；', fresh(OM_WSTAR), '。'))
new.append(item('2.3', '若 ', MF.omath(MF.r('|'), MF.sup('w', '*'), MF.r('-w|<'), MF.sup('10', '-12'), MF.sup('w', '*')), '，輸出 ', M('w^{*}'),
                '；否則令 ', fresh(OM_UPD), '，回到 2.2。上限 500 次。'))

new.append(mkpara(NORMAL, [('b', '步驟 3　彙整：'), '每類得 10 個解。最小值與最大值即第一節之下界與上界；代表值 ',
                           MF.omath(bar('w'), MF.r('='), supx(dlm(MF.sub('w', '1'), MF.r('⋅'), MF.sub('w', '2'), MF.r('⋅…⋅'), MF.sub('w', '10')), '1/10')),
                           '（即 10 個解之幾何平均），取三位有效數字。']))
new.append(mkpara(NORMAL, [('b', '步驟 4　用於模擬：'), '每顆虛擬質點自其所屬類別的 10 個解中取用一個，10 個解各占該類虛擬質點的十分之一。垂直座標向上為正，故取 ',
                           MF.omath(MF.r('-w')), '。']))
ref = p41
for e in new:
    ref.addnext(e)
    ref = e

# ================================================================ 第六節：步驟名稱對應
p82 = P(82); strip_proof(p82)
replace_in_p(p82, '步驟一省略。', '步驟一省略。各步驟標題後括號註明所對應之推導流程步驟。')
for idx, old, newt in ((84, '　有效密度', '　有效密度（推導流程 1.1）'),
                       (89, '　幾何', '　幾何（推導流程 1.2）'),
                       (95, '　淹沒比重與迭代常數', '　淹沒比重與迭代常數（推導流程 1.3、1.4）'),
                       (106, '　十個解', '　十個解（推導流程 1.5）'),
                       (110, '　幾何平均', '　幾何平均（推導流程步驟 3）')):
    strip_proof(P(idx)); replace_in_p(P(idx), old, newt)
strip_proof(P(99)); replace_in_p(P(99), '（以', '（推導流程步驟 2；以')
strip_proof(P(100)); replace_in_p(P(100), '迭代式（補充一）', '迭代式（推導流程步驟 2）')
strip_proof(P(52)); replace_in_p(P(52), '須同時求解（補充一）', '須同時求解（推導流程步驟 2）')

# ================================================================ 補充一（一）
p122 = P(122)
clear_runs(p122)
p122.append(text_run('（一）數值解法：迭代程序見推導流程步驟 2。全部 50 組代表物件均在數十次迭代內收斂。', RPR()))

# ================================================================ B1：第一節結果表 3 位有效數字
rows, summ = v.load()
CN = {1: '保麗龍', 2: '竹木', 3: '廢紙', 4: '不可回收塑膠', 5: '廢漁網漁具', 6: '寶特瓶', 7: '可回收塑膠', 8: '鋁罐', 9: '鐵罐', 10: '玻璃瓶'}
tb1 = P(43)
trs = list(tb1.iter(q('w:tr')))[1:]
assert len(trs) == 10
for tr in trs:
    tcs = tr.findall(q('w:tc'))
    name = ''.join(ptext(p) for p in tcs[0].iter(q('w:p')))
    cid = [k for k, nm in CN.items() if name.startswith(nm)][0]
    set_cell_text(tcs[1], v.sig3(summ[cid]['wmin']))
    set_cell_text(tcs[3], v.sig3(summ[cid]['wmax']))
    assert v.sig3(summ[cid]['rep']) == ptext(tcs[2]).strip(), (cid, ptext(tcs[2]))

p45 = P(45); strip_proof(p45)
replace_in_p(p45, '，取三位有效數字，僅供各類摘要與比較；模擬時', '，僅供各類摘要與比較，表中數值均取三位有效數字。模擬時')
p109 = P(109)
strip_proof(p109)
replace_in_p(p109, '，即第一節之', '，取三位有效數字即第一節之')

# ================================================================ B2：正文說明
p3 = P(3); strip_proof(p3)
replace_in_p(p3, '；代表值僅供摘要，模擬並非全類只用此單一速度。',
             '。模擬時每顆虛擬質點自所屬類別 10 個速度中取用一個，各速度各占該類虛擬質點的十分之一；代表值僅供摘要。')

# ================================================================ B3：正文速度表兩處
tb4 = P(4)
cnt = 0
for tr in tb4.iter(q('w:tr')):
    for tc in tr.findall(q('w:tc')):
        for p in tc.iter(q('w:p')):
            t_ = ptext(p)
            if '小型或完整進水罐' in t_:
                strip_proof(p); replace_in_p(p, '小型或完整進水罐，或', '完整進水罐（小罐、一般罐），或'); cnt += 1
            if '330 mL 小瓶與較大者' in t_:
                strip_proof(p); replace_in_p(p, '小瓶與較大者', '小瓶、一般瓶'); cnt += 1
assert cnt == 2, cnt

tree.write(OUT, xml_declaration=True, encoding='UTF-8', standalone=True)
print('written', OUT)
