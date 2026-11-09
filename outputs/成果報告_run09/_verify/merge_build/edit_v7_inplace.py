#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合併版 v7：就地修改（只動 word/document.xml；其餘 zip 成員逐位元組保留）。
(1) 沉降速度一律不寫負號：表 1 十列去負號；附錄刪「垂直座標向上為正，故各解取負值。」與表 A1 註「；垂直座標向上為正時取負值」。
(2) 附錄沉降速度（表 A1、A4「w 範圍」欄、A9「w」欄、A10、A11 基準上界、A12 兩速度欄與正文引用）改科學記號：a.bc×10⁻ⁿ，
    指數為 Word 上標（vertAlign=superscript，U+2212 負號），有效位數照原值；Decimal 逐一驗證。迭代過程中間值（表 A8、(A11) 驗算式、數學式內）不改。
(3) 表 1 欄寬調整（tblGrid／tcW）；其他表視預覽需要調整。
用法：python3 edit_v7_inplace.py <合併版docx>   前提：sha256 須為 32a3ca06…（v6f），否則停止。"""
import sys, re, zipfile, hashlib, os, copy
from decimal import Decimal
from lxml import etree
PRE = '32a3ca06f15a7bc2'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
q = lambda t: '{%s}%s' % (W, t)
XMLSP = '{http://www.w3.org/XML/1998/namespace}space'
path = sys.argv[1]
raw = open(path, 'rb').read()
assert hashlib.sha256(raw).hexdigest().startswith(PRE), '合併版 sha 與預期不同，停止'
z = zipfile.ZipFile(path)
xml = z.read('word/document.xml')
i0 = xml.index(b'<w:document'); head = xml[:i0]
root = etree.fromstring(xml); body = root.find(q('body')); kids = list(body)

# ---------------------------------------------------------------- 欄寬（tblGrid／tcW，總寬不變）
WIDTHS = {   # 區塊序 → 新欄寬（twips）；None 表示不改
    5: [1100, 2050, 3407, 1900],      # 表 1：材質欄 −150、速度欄 +150（總寬 8457 不變）
    275: [1494, 2300, 2673, 1990],    # 表 A10：文獻實測值欄 −807、本計畫範圍欄 +807（總寬 8457 不變）
}

def ptxt(p): return ''.join(t.text or '' for t in p.iter(q('t')))
def run_text(r): return ''.join(t.text or '' for t in r.findall(q('t')))
def set_text(r, s):
    ts = r.findall(q('t')); assert len(ts) == 1, 'run 含多個 w:t'
    ts[0].text = s
    if s != s.strip() or '  ' in s: ts[0].set(XMLSP, 'preserve')
    elif XMLSP in ts[0].attrib: del ts[0].attrib[XMLSP]
def simple_run(r):   # 僅 rPr 與一個 w:t
    return [c.tag for c in r if c.tag not in (q('rPr'), q('lastRenderedPageBreak'))] == [q('t')]
AFTER = [q(x) for x in ('rtl', 'cs', 'em', 'lang', 'eastAsianLayout', 'specVanish', 'oMath')]
def add_sup(rpr):
    if rpr is None: rpr = etree.Element(q('rPr'))
    for v in rpr.findall(q('vertAlign')): rpr.remove(v)
    va = etree.Element(q('vertAlign')); va.set(q('val'), 'superscript')
    for n, c in enumerate(rpr):
        if c.tag in AFTER: rpr.insert(n, va); break
    else: rpr.append(va)
    return rpr
def sci(s):
    d = s.replace('.', '').lstrip('0'); v = Decimal(s); e = v.adjusted()
    m = d[0] + ('.' + d[1:] if len(d) > 1 else '')
    assert Decimal(m + 'E' + str(e)) == v and len(m.replace('.', '')) == len(d), (s, m, e)
    return m, e

def strip_lrpb(r):
    for x in r.findall(q('lastRenderedPageBreak')): r.remove(x)
    return r
def convert_par(p, tokre, allowed=None):
    """段落內符合 tokre 之數字全部改科學記號；回傳換算清單 [(舊, 新字串)]。跨 run 亦處理（沿用第一個受影響 run 之 rPr）。"""
    done = []
    def getruns(): return [r for r in p.findall(q('r')) if r.findall(q('t'))]
    runs = getruns()
    for r in runs: assert simple_run(r), ptxt(p)[:40]
    text = ''.join(run_text(r) for r in runs)
    toks = list(tokre.finditer(text))
    if allowed is not None: toks = [m for m in toks if m.group(0) in allowed]
    for m in reversed(toks):
        runs = getruns(); s, e = m.start(), m.end(); pos = 0; seg = []
        for r in runs:
            L = len(run_text(r)); seg.append((r, pos, pos + L)); pos += L
        ov = [(r, a, b) for r, a, b in seg if a < e and b > s]
        r1, a1, _ = ov[0]; rk, ak, _ = ov[-1]
        pre = run_text(r1)[:s - a1]; suf = run_text(rk)[e - ak:]
        mant, ex = sci(m.group(0))
        mr = strip_lrpb(copy.deepcopy(r1)); set_text(mr, mant + '×10')
        sr = strip_lrpb(copy.deepcopy(r1)); set_text(sr, '−' + str(abs(ex)))
        rp = sr.find(q('rPr'))
        if rp is None: rp = etree.Element(q('rPr')); sr.insert(0, rp)
        add_sup(rp)
        new = [mr, sr]; remove = []
        if suf:
            if rk is r1: sfr = strip_lrpb(copy.deepcopy(r1)); set_text(sfr, suf); new.append(sfr)
            else: set_text(rk, suf)
        for r, _, _ in ov:
            if r is r1:
                if pre: set_text(r1, pre)
                else: remove.append(r)
            elif r is rk and suf: pass
            else: remove.append(r)
        par = r1.getparent(); at = par.index(r1) + (1 if pre else 0)
        if rk is not r1 and suf: at = par.index(rk)   # 新 run 放在 rk（其後餘字）之前
        for n, el in enumerate(new[:2]): par.insert(at + n, el)
        if rk is r1 and suf: par.insert(at + 2, new[2])
        for r in remove: par.remove(r)
        done.append((m.group(0), '%s×10^%d' % (mant, ex)))
    return list(reversed(done))

def replace_par(p, old, new):
    runs = [r for r in p.findall(q('r')) if r.findall(q('t'))]
    for r in runs: assert simple_run(r)
    text = ''.join(run_text(r) for r in runs); assert text.count(old) == 1, old
    s = text.index(old); e = s + len(old); pos = 0
    for r in runs:
        L = len(run_text(r)); a, b = pos, pos + L; pos = b
        if not (a < e and b > s): continue
        t = run_text(r); lo = max(s, a) - a; hi = min(e, b) - a
        ins = new if a <= s < b or (a == s) else ''
        # 新字放第一個受影響之 run
        first = (s >= a)
        t2 = t[:lo] + (new if first else '') + t[hi:]
        if t2: set_text(r, t2)
        else: r.getparent().remove(r)

def cells(tbl, rows=None):
    for ri, tr in enumerate(tbl.findall(q('tr'))):
        for ci, tc in enumerate(tr.findall(q('tc'))): yield ri, ci, tc

TOK = re.compile(r'(?<![\d.])0\.\d+(?![\d%])')
log = {}
def expect(name, got, n):
    log[name] = got; assert len(got) == n, (name, len(got), n)

# ---- 0. 錨點檢查（區塊序與文字）
assert ptxt(kids[194]).startswith('步驟 3　用於模擬') and ptxt(kids[198]).startswith('（單位均為 m/s；垂直座標向上為正時取負值。）')
assert kids[5].tag == q('tbl') and kids[197].tag == q('tbl') and kids[234].tag == q('tbl') and kids[271].tag == q('tbl')
assert kids[275].tag == q('tbl') and kids[281].tag == q('tbl') and kids[287].tag == q('tbl')
assert ptxt(kids[272]).startswith('十個解之範圍 0.03671') and ptxt(kids[277]).startswith('表 A10 之實測對象') and ptxt(kids[292]).startswith('（一）薄板之沉速')

# ---- 1. 符號慣例句
replace_par(kids[194], '垂直座標向上為正，故各解取負值。', '')
replace_par(kids[198], '；垂直座標向上為正時取負值', '')
assert ptxt(kids[198]) == '（單位均為 m/s。）', ptxt(kids[198])
assert ptxt(kids[194]).endswith('不含斯托克斯漂流案例與含斯托克斯漂流案例相同。'), ptxt(kids[194])[-30:]

# ---- 2. 表 1 去負號（第 4 欄資料列之前導負號 run；指數之負號為上標 run，保留）
n_minus = 0
for ri, ci, tc in cells(kids[5]):
    if ri == 0 or ci != 3: continue
    for r in list(tc.iter(q('r'))):
        rp = r.find(q('rPr'))
        if run_text(r) == '−' and (rp is None or rp.find(q('vertAlign')) is None):
            r.getparent().remove(r); n_minus += 1
assert n_minus == 20, n_minus
log['T1_去負號'] = n_minus

# ---- 3. 附錄科學記號
def conv_table(idx, cols, n, allowed=None, skip_header=True):
    out = []
    for ri, ci, tc in cells(kids[idx]):
        if (skip_header and ri == 0) or (cols is not None and ci not in cols): continue
        for p in tc.findall(q('p')): out += convert_par(p, TOK, allowed)
    expect('表@%d' % idx, out, n)
conv_table(197, {1, 2}, 20)        # 表 A1
conv_table(234, {8}, 100)          # 表 A4 w 範圍
conv_table(271, {5}, 10)           # 表 A9 w
conv_table(275, {1, 2}, 14)        # 表 A10
conv_table(281, None, 4, allowed={'0.187', '0.175'})   # 表 A11 基準上界（百分比與 Cd 範圍不動）
conv_table(287, {2, 3}, 10)        # 表 A12
for idx, n, al in ((272, 2, {'0.03671', '0.17548'}), (277, 8, {'0.0367', '0.175', '0.115', '0.120', '0.107', '0.427', '0.176', '0.177'}), (292, 1, {'0.1501'})):
    expect('段@%d' % idx, convert_par(kids[idx], TOK, al), n)

# ---- 4. 欄寬
for idx, ws in WIDTHS.items():
    if ws is None: continue
    t = kids[idx]; g = t.find(q('tblGrid')).findall(q('gridCol')); assert len(g) == len(ws)
    assert sum(ws) == sum(int(x.get(q('w'))) for x in g), (idx, sum(ws), sum(int(x.get(q('w'))) for x in g))
    for x, w in zip(g, ws): x.set(q('w'), str(w))
    for tr in t.findall(q('tr')):
        for tc, w in zip(tr.findall(q('tc')), ws): tc.find(q('tcPr')).find(q('tcW')).set(q('w'), str(w))

# ---- 5. 表 1 資料列加 cantSplit（列不跨頁；原「寶特瓶」列於頁末被拆成兩頁，（PET）孤行）
n_cs = 0
for ri, tr in enumerate(kids[5].findall(q('tr'))):
    if ri == 0: continue
    assert tr.find(q('trPr')) is None
    trpr = etree.Element(q('trPr')); etree.SubElement(trpr, q('cantSplit'))
    tr.insert(tr.index(tr.find(q('tc'))), trpr); n_cs += 1
assert n_cs == 10; log['表1_cantSplit'] = n_cs

for k, v in log.items():
    if isinstance(v, list): print(k, len(v), v[:3])
    else: print(k, v)
newxml = head + etree.tostring(root, encoding='UTF-8')
tmp = path + '.partial'
with zipfile.ZipFile(tmp, 'w') as zo:
    for info in z.infolist():
        zi = zipfile.ZipInfo(info.filename, info.date_time)
        zi.compress_type = info.compress_type; zi.external_attr = info.external_attr; zi.create_system = info.create_system
        zo.writestr(zi, newxml if info.filename == 'word/document.xml' else z.read(info.filename))
os.replace(tmp, path)
print('OK', hashlib.sha256(open(path, 'rb').read()).hexdigest(), os.path.getsize(path))
