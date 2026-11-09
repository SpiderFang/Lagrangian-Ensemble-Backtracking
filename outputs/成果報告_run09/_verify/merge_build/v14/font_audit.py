#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""字型稽核：逐 run 解析 ascii／hAnsi／cs 實際字型（docDefaults → 段落樣式鏈 → 表格樣式 → 字元樣式 → run 直接格式；theme 屬性解到 theme1.xml）。
用法：font_audit.py <docx> [--json out.json]"""
import sys, re, zipfile, json, collections
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
q = lambda t: '{%s}%s' % (W, t)
LATIN = re.compile(r'[A-Za-z0-9]')
def load(docx):
    z = zipfile.ZipFile(docx)
    st = etree.fromstring(z.read('word/styles.xml'))
    th = etree.fromstring(z.read('word/theme/theme1.xml'))
    minor = th.find('.//{%s}minorFont/{%s}latin' % (A, A)).get('typeface'); major = th.find('.//{%s}majorFont/{%s}latin' % (A, A)).get('typeface')
    styles = {}
    default_p = default_c = default_t = None
    for s in st.findall(q('style')):
        sid = s.get(q('styleId')); typ = s.get(q('type'))
        b = s.find(q('basedOn'))
        rf = s.find(q('rPr')).find(q('rFonts')) if s.find(q('rPr')) is not None else None
        styles[sid] = dict(type=typ, based=b.get(q('val')) if b is not None else None, rf=rf, name=(s.find(q('name')).get(q('val')) if s.find(q('name')) is not None else sid))
        if s.get(q('default')) == '1':
            if typ == 'paragraph': default_p = sid
            if typ == 'character': default_c = sid
            if typ == 'table': default_t = sid
    dd = st.find(q('docDefaults')); ddrf = dd.find('.//' + q('rFonts')) if dd is not None else None
    return z, styles, ddrf, minor, major, default_p
def chain(styles, sid):
    out = []
    seen = set()
    while sid and sid in styles and sid not in seen:
        seen.add(sid); out.append(sid); sid = styles[sid]['based']
    return list(reversed(out))   # 由基底到衍生
def resolve(layers, minor, major):
    """layers：依序之 rFonts 元素（可為 None）。回傳 dict(ascii,hAnsi,cs,eastAsia) 之 (字型, 來源)"""
    res = {}
    for kind in ('ascii', 'hAnsi', 'cs', 'eastAsia'):
        val = None; src = None
        for tag, rf in layers:
            if rf is None: continue
            th = rf.get(q(kind + 'Theme')); ex = rf.get(q(kind))
            if th:
                f = {'minorHAnsi': minor, 'majorHAnsi': major, 'minorAscii': minor, 'majorAscii': major, 'minorBidi': minor, 'majorBidi': major}.get(th, th)
                val, src = f, tag + ':' + kind + 'Theme=' + th
            elif ex is not None:
                val, src = ex, tag + ':' + kind
        res[kind] = (val, src)
    return res
def audit(docx, parts=None):
    z, styles, ddrf, minor, major, default_p = load(docx)
    rows = []
    names = parts or ['word/document.xml', 'word/footer1.xml', 'word/footnotes.xml', 'word/endnotes.xml'] + [n for n in z.namelist() if re.match(r'word/(header|footer)\d+\.xml$', n) and n != 'word/footer1.xml']
    for part in names:
        if part not in z.namelist(): continue
        root = etree.fromstring(z.read(part))
        for r in root.iter(q('r')):
            ts = [t.text or '' for t in r.iter(q('t'))]
            txt = ''.join(ts)
            if not LATIN.search(txt): continue
            if any(a.tag.startswith('{%s}' % M) for a in r.iterancestors()): continue   # OMML 內
            p = next((a for a in r.iterancestors() if a.tag == q('p')), None)
            ppr = p.find(q('pPr')) if p is not None else None
            psid = (ppr.find(q('pStyle')).get(q('val')) if ppr is not None and ppr.find(q('pStyle')) is not None else default_p)
            rpr = r.find(q('rPr'))
            rsid = rpr.find(q('rStyle')).get(q('val')) if rpr is not None and rpr.find(q('rStyle')) is not None else None
            tsid = None
            tbl = next((a for a in r.iterancestors() if a.tag == q('tbl')), None)
            if tbl is not None and tbl.find(q('tblPr')) is not None and tbl.find(q('tblPr')).find(q('tblStyle')) is not None:
                tsid = tbl.find(q('tblPr')).find(q('tblStyle')).get(q('val'))
            layers = [('docDefaults', ddrf)]
            if tsid: layers += [('tbl:' + s_, styles[s_]['rf']) for s_ in chain(styles, tsid)]
            layers += [('p:' + s_, styles[s_]['rf']) for s_ in chain(styles, psid)]
            if rsid: layers += [('c:' + s_, styles[s_]['rf']) for s_ in chain(styles, rsid)]
            layers += [('run', rpr.find(q('rFonts')) if rpr is not None else None)]
            rs = resolve(layers, minor, major)
            nlat = len(LATIN.findall(txt))
            rows.append(dict(part=part, text=txt[:30], nlat=nlat, pstyle=psid, rstyle=rsid, intbl=tbl is not None,
                             ascii=rs['ascii'][0], hAnsi=rs['hAnsi'][0], cs=rs['cs'][0], ascii_src=rs['ascii'][1], hAnsi_src=rs['hAnsi'][1], cs_src=rs['cs'][1]))
    return rows
if __name__ == '__main__':
    rows = audit(sys.argv[1])
    print('含半形英數之 run 數', len(rows), '；英數字元數', sum(r['nlat'] for r in rows))
    bad = [r for r in rows if r['ascii'] != 'Times New Roman' or r['hAnsi'] != 'Times New Roman']
    print('ascii/hAnsi 非 TNR 之 run 數', len(bad), '；字元數', sum(r['nlat'] for r in bad))
    c = collections.Counter((r['ascii'], r['hAnsi'], r['ascii_src'], r['hAnsi_src']) for r in bad)
    for k, v in c.most_common(30): print(v, k)
    cc = collections.Counter(r['cs'] for r in rows); print('cs 解析：', dict(cc))
    cp = collections.Counter((r['pstyle'], r['ascii']) for r in bad); print('非TNR 依段落樣式：', cp.most_common(15))
    ct = collections.Counter(('表格內' if r['intbl'] else '表外', r['part']) for r in bad); print(ct)
    for r in bad[:8]: print(r)
    if '--json' in sys.argv: json.dump(rows, open(sys.argv[sys.argv.index('--json') + 1], 'w'), ensure_ascii=False, indent=1)
