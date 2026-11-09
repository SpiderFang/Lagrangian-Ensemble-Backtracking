# -*- coding: utf-8 -*-
"""r4：套用使用者同意之 10 項建議（地點撞名、計畫書句註記、第二節精簡、第四節（二）算式、第一節兩段、
第八節排序欄、完整進水罐/瓶、半壓扁、虛擬質點、本研究）。
用法：python3 build_r4.py <in document.xml> <out document.xml>
"""
import sys, copy, re
sys.path.insert(0, '/tmp/w'); sys.path.insert(0, '/tmp/w/build')
from lxml import etree
from lib import q, ptext, lin, load, body_items
from edit import *

SRC, OUT = sys.argv[1], sys.argv[2]
tree = load(SRC)
root = tree.getroot()
I = body_items(tree)
log = {}


def strip_proof(p):
    for e in list(p):
        if e.tag == q('w:proofErr'):
            p.remove(e)


def nhits(p, old):
    n = 0
    for g in groups(p):
        txt = ''.join(rtext(r) for r in g)
        n += txt.count(old)
    return n


def global_replace(old, new, key, skip=lambda p: False):
    """全文（含表格）取代純文字；回傳取代處數。"""
    total = 0
    for p in list(root.iter(q('w:p'))):
        if skip(p):
            continue
        if old not in ptext(p):
            continue
        strip_proof(p)
        n = nhits(p, old)
        if n:
            replace_in_p(p, old, new, count=n)
            total += n
    log[key] = log.get(key, 0) + total
    return total


def para_text(p):
    return ptext(p)


# ================================================================ 10. 本案 → 本研究（先做，後面的新句不含「本案」）
global_replace('本案', '本研究', '10 本案→本研究')

# ================================================================ 7. 完整罐 → 完整進水罐
global_replace('完整罐', '完整進水罐', '7 完整罐→完整進水罐')
# 正文速度表「完整進水瓶身（600 mL、1250 mL）」→「完整進水瓶（…）」；半壓扁
global_replace('完整進水瓶身', '完整進水瓶', '7 完整進水瓶身→完整進水瓶')

# ================================================================ 8. 壓扁瓶身 → 半壓扁瓶身；正文表「壓扁、重壓扁之瓶身」
n = 0
for p in list(root.iter(q('w:p'))):
    if ptext(p).strip() == '壓扁瓶身':
        strip_proof(p); replace_in_p(p, '壓扁瓶身', '半壓扁瓶身'); n += 1
assert n == 1, n
log['8 壓扁瓶身→半壓扁瓶身（大表）'] = n
global_replace('或壓扁、重壓扁之瓶身', '或半壓扁、重壓扁之瓶身', '8 正文表寶特瓶描述')

# ================================================================ 9. 粒子 → 虛擬質點
global_replace('粒子', '虛擬質點', '9 粒子→虛擬質點')

# ================================================================ 1. 研究地點
global_replace('各地點', '各研究地點', '1 各地點→各研究地點（內文）')
global_replace('該地點關注海域', '該研究地點關注海域', '1 該地點→該研究地點')
# 表頭「地點」
n = 0
for p in list(root.iter(q('w:p'))):
    if ptext(p).strip() == '地點':
        strip_proof(p); replace_in_p(p, '地點', '研究地點'); n += 1
assert n == 1, n
log['1 表頭 地點→研究地點'] = n
# 圖片替代文字
n = 0
for el in root.iter():
    d = el.get('descr')
    if d and '各地點' in d:
        el.set('descr', d.replace('各地點', '各研究地點')); n += 1
assert n == 2, n
log['1 圖片替代文字 各地點→各研究地點'] = n

# ================================================================ 2. 計畫書原句後加註
p2 = I[2]
assert '假設' in ptext(p2) and '上升速度' in ptext(p2)
strip_proof(p2)
replace_in_p(p2, '上升速度，以涵蓋', '上升速度（10 類，各類 10 個速度），以涵蓋')
log['2 計畫書原句加註'] = 1

# ================================================================ 3. 第二節「反之…」
p67 = None
for p in root.iter(q('w:p')):
    if '反之' in ptext(p) and '直接反求' in ptext(p):
        p67 = p
assert p67 is not None
oms = list(p67.iter(q('m:oMath')))
def omtext(o):
    return ''.join(t.text or '' for t in o.iter(q('m:t')))
OM_W = [copy.deepcopy(o) for o in oms if omtext(o) == 'w'][-1]
OM_CD = [copy.deepcopy(o) for o in oms if omtext(o) == 'Cd'][-1]
# 取得「反之」所在 run：其前一個 run 結尾為「（推導流程步驟 2）。」
strip_proof(p67)
cut = None
for ch in list(p67):
    if ch.tag == q('w:r') and '反之' in ''.join(t.text or '' for t in ch.iter(q('w:t'))):
        cut = ch
        break
assert cut is not None
# 刪除自 cut 起之所有子元素
seq = list(p67)
for ch in seq[seq.index(cut):]:
    p67.remove(ch)
rp = copy.deepcopy(seq[1].find(q('w:rPr'))) if seq[1].tag == q('w:r') else None
def tr(s):
    return text_run(s, rp)
for x in (tr('若已有同尺度之實測 '), copy.deepcopy(OM_W), tr('，可由式(A1)反求 '), copy.deepcopy(OM_CD),
          tr('；本研究十類物件均無此類實測，故由教科書取得 '), copy.deepcopy(OM_CD), tr(' 再求 '), copy.deepcopy(OM_W), tr('。')):
    p67.append(x)
log['3 第二節「反之」改一句'] = 1

# ================================================================ 4. 第四節（二）
p84 = None
for p in root.iter(q('w:p')):
    if ptext(p).startswith('（二）完整瓶罐以鈍體近似'):
        p84 = p
assert p84 is not None
strip_proof(p84)
replace_in_p(p84, '介於端面迎流與側面迎流之間（', '介於端面與側面迎流面積之間（')
# 以簡單式取代原長算式
from edit import MathFactory
MF = MathFactory(None, '28')
old_m = [o for o in p84.iter(q('m:oMath')) if omtext(o).startswith('L/d')]
assert len(old_m) == 1
new_m = MF.omath(MF.r('L/d>1.09'))
old_m[0].getparent().replace(old_m[0], new_m)
log['4 第四節（二）長徑比算式'] = 1

# ================================================================ 5. 第一節表後長段拆兩段
p60 = None
for p in root.iter(q('w:p')):
    if ptext(p).startswith('代表值為該類 10 個解'):
        p60 = p
assert p60 is not None
oms = list(p60.iter(q('m:oMath')))
OM_W2 = copy.deepcopy([o for o in oms if omtext(o) == 'w'][0])
rp60 = copy.deepcopy([r for r in p60.findall(q('w:r'))][0].find(q('w:rPr')))
p60b = copy.deepcopy(p60)
for p_ in (p60, p60b):
    for e in list(p_):
        if e.tag != q('w:pPr'):
            p_.remove(e)
def tr60(s):
    return text_run(s, rp60)
for x in (tr60('代表值為該類 10 個解（5 種代表物件 × 阻力係數上下界）之幾何平均，僅供各類摘要與比較，表中數值均取三位有效數字，其中 '),
          copy.deepcopy(OM_W2),
          tr60(' 下界與上界為該類 10 個解之最小值與最大值。模擬時每顆虛擬質點隨機取用其所屬類別 10 個解中的一個，10 個解各占該類虛擬質點的十分之一，並非全類只用代表值。')):
    p60.append(x)
p60b.append(tr60('同類 10 個解中，保麗龍、竹木、廢紙、不可回收塑膠、廢漁網漁具與可回收塑膠六類之最大值為最小值的 10 倍以上。'
                 '寶特瓶、鋁罐、鐵罐之半壓扁與重壓扁兩種代表物件速度相同（補充二（二）），故這三類各有 8 個相異速度，其影響見第八節。'))
p60.addnext(p60b)
log['5 第一節表後拆兩段'] = 1

# ================================================================ 6. 第八節排序欄
n = 0
for tc in root.iter(q('w:tc')):
    if ptext(tc.find(q('w:p'))).strip() == '保麗龍與廢漁網漁具互換':
        set_cell_text(tc, '保麗龍與廢漁網漁具互換（兩者代表值僅差 4.7%：0.0482、0.0504 m/s），其餘八類不變'); n += 1
assert n == 1, n
n = 0
for p in list(root.iter(q('w:p'))):
    if '圓柱改以等體積球徑計算' in ptext(p) and '互換排序之保麗龍與廢漁網漁具' in ptext(p):
        strip_proof(p)
        replace_in_p(p, '圓柱改以等體積球徑計算 Re 時，互換排序之保麗龍與廢漁網漁具，其代表值原僅相差 4.7%（0.0482 與 0.0504 m/s）。', '')
        n += 1
assert n == 1, n
log['6 第八節排序欄與表後段落'] = 1

tree.write(OUT, xml_declaration=True, encoding='UTF-8', standalone=True)
for k, v in log.items():
    print(k, v)
print('written', OUT)
