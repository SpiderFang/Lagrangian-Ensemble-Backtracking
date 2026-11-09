#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""依 LibreOffice 轉出之 PDF（pdfdump.py 之 json）與 docx，列出每張圖之導引段起訖頁、圖頁、圖說頁與狀態；另列表格。
用法：layout_table.py <docx> <dump.json> [out.json]"""
import sys, re, json, zipfile
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
q = lambda t: '{%s}%s' % (W, t)
def tx(p): return ''.join(x.text or '' for x in p.iter(q('t')))
def flat(s): return re.sub(r'[\s ]+', '', s)
def analyse(docx, dump):
    B = list(etree.fromstring(zipfile.ZipFile(docx).read('word/document.xml')).find(q('body')))
    D = json.load(open(dump, encoding='utf-8'))
    P = [flat(t) for t in D['pages']]
    imgs = [i for i in D['imgs'] if i[1] == 'image']
    figs = [i for i, p in enumerate(B) if p.tag == q('p') and any(True for _ in p.iter(q('drawing')))]
    assert len(figs) == len(imgs), (len(figs), len(imgs))
    figpage = {k + 1: imgs[k][0] for k in range(len(figs))}
    # 段落起訖頁（依文件序逐段向後找）
    cur = 1; loc = {}
    for i, p in enumerate(B):
        if p.tag == q('tbl'):
            t = flat(tx(p.find(q('tr')))) if p.find(q('tr')) is not None else ''
            if len(t) >= 6:
                for pg in range(cur, min(cur + 6, D['n']) + 1):
                    if t[:8] in P[pg - 1]: loc[i] = (pg, pg); cur = pg; break
            continue
        t = flat(tx(p))
        if len(t) < 8: continue
        a, b = t[:10], t[-10:]
        if re.match(r'^[圖表]A?\d+', tx(p)): b = t[-2:]   # 圖說／表題：末幾字，才能偵測最後一行單獨落到下一頁
        s = None
        for pg in range(cur, min(cur + 6, D['n']) + 1):
            if a in P[pg - 1]: s = pg; break
        if s is None: continue
        e = s
        for pg in range(s, min(s + 4, D['n']) + 1):
            if b in P[pg - 1]: e = pg; break
        loc[i] = (s, e); cur = e
    rows = []
    for k, bi in enumerate(figs, 1):
        cap = bi + 1
        if k <= 2: g = figs[0] - 1
        elif k <= 7: g = figs[2] - 1
        else: g = bi - 1
        gs, ge = loc.get(g, (None, None)); cp = loc.get(cap, (None, None))[0]; ce = loc.get(cap, (None, None))[1]; fp = figpage[k]
        shared = 3 <= k <= 7 or k <= 2
        st = []
        if k in (1, 2):
            # 圖 1、2：導引段（首次提及處）在表 2 之前，不可能與圖同頁，改以「圖頁是否緊接表 2 與圖說」判斷
            pass
        if cp != fp: st.append('圖說不同頁')
        elif ce != fp: st.append('圖說末行落到下一頁')
        if ge is not None and ge < fp and not (shared and k not in (1, 3)): st.append('導引段早於圖')
        status = 'NG：' + '、'.join(st) if st else ('跨頁開頭' if (gs is not None and gs < fp and ge == fp) else 'OK')
        rows.append(dict(fig=k, guide_blk=g, g_start=gs, g_end=ge, fig_p=fp, cap_p=cp, cap_end_p=ce, status=status, shared=shared))
    tbls = []
    def cells(tr): return [flat(tx(c)) for c in tr.findall(q('tc'))]
    def find_page(cs, p0):
        cs = [c for c in cs if len(c) >= 2]
        if not cs: return None
        best, bp = 0, None
        for pg in range(p0, min(p0 + 4, D['n']) + 1):
            sc = sum(1 for c in cs if c[:4] in P[pg - 1] or c[-4:] in P[pg - 1])
            if sc > best: best, bp = sc, pg
        return bp if best * 2 >= len(cs) else None
    for i, p in enumerate(B):
        if p.tag == q('tbl') and i > 0 and re.match(r'^表A?\d+ ', tx(B[i - 1])):
            capp = loc.get(i - 1, (None, None))[0]
            trs = p.findall(q('tr'))
            sp = find_page(cells(trs[0]), capp) if capp else None
            ep = find_page(cells(trs[-1]), sp or capp) if capp else None
            st = 'OK' if (capp is not None and sp == capp and ep is not None) else ('NG：表首列與表題不同頁' if sp != capp else '待查')
            if sp is not None and ep is not None and ep > sp: st += '（表跨頁 %d–%d）' % (sp, ep)
            tbls.append(dict(name=tx(B[i - 1])[:12], cap_p=capp, head_p=sp, last_p=ep, rows=len(trs), status=st))
    return rows, tbls, D['n']
if __name__ == '__main__':
    rows, tbls, n = analyse(sys.argv[1], sys.argv[2])
    print('pages', n)
    for r in rows: print(r)
    for t in tbls: print(t)
    if len(sys.argv) > 3: json.dump(dict(pages=n, figs=rows, tables=tbls), open(sys.argv[3], 'w'), ensure_ascii=False, indent=1)
