#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合併版 v13 候選：(1) 移除條件式來源足跡（停留時數密度）；(2) 來源分段機率圖 8 張／地點 → 4 張合併圖（邊界與海岸同一張地圖）；
(3) 8 句修正後之海岸說明插入各地點全年合併圖導引段；全文圖號重編為 37 張。
基準：現行主檔 v12（sha256 43435cf179185e896b34092a76164bd739f5a74776f630f12b15454e37bd71a7，357 區塊），不被修改。
用法：edit_v13_inplace.py <主檔> <figs目錄> <輸出docx> <log.json>"""
import sys, os, re, zipfile, hashlib, json, copy, struct, unicodedata
from lxml import etree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v13_texts as T

SHA = '43435cf179185e896b34092a76164bd739f5a74776f630f12b15454e37bd71a7'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
PIC = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
XML = 'http://www.w3.org/XML/1998/namespace'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
q = lambda t: '{%s}%s' % (W, t)
FIG1_SCALE = float(os.environ.get('FIG1_SCALE', '0.90'))
MERGE = os.environ.get('MERGE', '1') == '1'
LIM = (16838 - 2880) / 20.0 - 17.0
BW = (11906 - 2880 - 991) / 20.0
CW = (11906 - 2880) / 20.0
CX = 5733288
# v12 文稿之季節邊界段機率係以分段四捨五入後再加總，與計數精確值差 0.01；改依計數精確值（與圖一致）
# v13a 曾將季節 ① 值更正為 36.61/39.56/19.97/41.65（計數精確值）；v13b 起季節導引段改為定性敘述、不列數值，故不再需要

def nl(text, wpt):
    ln, w = 1, 0.0
    for ch in text:
        cw = 14.0 if unicodedata.east_asian_width(ch) in ('W', 'F') else 7.0
        if w + cw > wpt: ln += 1; w = cw
        else: w += cw
    return ln
def fits(text, fig_cy, cap_text):
    tot = nl(text, BW) * 22 + 18 + 12 + fig_cy / 12700.0 + (2 + nl(cap_text, CW) * 27 + 8)
    return tot <= LIM, tot
def ptx(p): return ''.join(t.text or '' for t in p.iter(q('t')))

def set_text(p, text):
    """段落內容改為單一 run（沿用第一個 run 之 rPr）"""
    runs = p.findall(q('r')); rpr = None
    for r in runs:
        if r.find(q('rPr')) is not None: rpr = copy.deepcopy(r.find(q('rPr'))); break
    for r in runs: p.remove(r)
    r = etree.SubElement(p, q('r'))
    if rpr is not None: r.append(rpr)
    t = etree.SubElement(r, q('t')); t.text = text; t.set('{%s}space' % XML, 'preserve')
def strip_ids(p):
    for a in ('{%s}paraId' % W14, '{%s}textId' % W14):
        if a in p.attrib: del p.attrib[a]
def png_size(path):
    d = open(path, 'rb').read(32); assert d[:8] == b'\x89PNG\r\n\x1a\n'
    return struct.unpack('>II', d[16:24])

def patch_styles(data, log):
    """v14r5：英、數字一律 Times New Roman。「List Paragraph」(afe) 及其連結字元樣式 (aff) 之 rFonts 由 Calibri 改為 TNR（ascii／hAnsi／cs；eastAsia 不動）。
    合併段內原區塊 9 之 run 無直接字型，隨段落樣式 afe 解析為 Calibri；docDefaults 已為 TNR。"""
    s = data.decode('utf-8')
    for sid, old, new in (
        ('afe', '<w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:szCs w:val="22"/></w:rPr>', '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:szCs w:val="22"/></w:rPr>'),
        ('aff', None, None)):
        i = s.index('w:styleId="%s"' % sid); j = s.index('</w:style>', i)
        seg = s[i:j]
        if sid == 'afe':
            assert seg.count(old) == 1, seg
            s = s[:i] + seg.replace(old, new) + s[j:]
        else:
            m = re.search(r'<w:rFonts w:ascii="Calibri" w:eastAsia="標楷體" w:hAnsi="Calibri"/>', seg)
            assert m, seg
            s = s[:i] + seg.replace(m.group(0), '<w:rFonts w:ascii="Times New Roman" w:eastAsia="標楷體" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>') + s[j:]
    log['notes'].append(['styles.xml：樣式 afe（List Paragraph）與 aff（其連結字元樣式）之 rFonts Calibri → Times New Roman（ascii／hAnsi／cs）；eastAsia 不動；其他樣式未改'])
    return s.encode('utf-8')


def main(src, figdir, dst, logp):
    raw = open(src, 'rb').read(); assert hashlib.sha256(raw).hexdigest() == SHA, '主檔 sha 與預期不同，停止'
    zin = zipfile.ZipFile(src)
    root = etree.fromstring(zin.read('word/document.xml')); body = root.find(q('body')); kids = list(body)
    assert len(kids) == 357
    rels_root = etree.fromstring(zin.read('word/_rels/document.xml.rels'))
    log = dict(base_sha=SHA, removed=[], fig_map={}, keep=[], notes=[])
    # ---------------- 前提斷言
    for key, name, n0, head, trk, trc, foot, G in T.SITES:
        assert ptx(kids[head]) == name, (head, ptx(kids[head]))
        assert ptx(kids[trk]).startswith('圖 %d 為' % (n0 if key == 'guishan' else {'gongliao': 19, 'hsinchu': 30, 'nanwan': 41, 'lienchiang': 52}[key]))
        assert re.search(r'條件式來源足跡', ptx(kids[foot + 2])), ptx(kids[foot + 2])
        assert ptx(kids[G + 2]).startswith('圖') and '全年，十類合併' in ptx(kids[G + 2])
    assert ptx(kids[209]).startswith('足跡範圍方面') and ptx(kids[34]).startswith('各研究地點依序以十一張圖')
    assert ptx(kids[46]).startswith('圖 11 至圖 18 為龜山島之來源分段機率圖')
    # ---------------- 取舊導引段文字
    old = {}
    for key, name, n0, head, trk, trc, foot, G in T.SITES:
        o = {}
        for nm, off in (('all', 0), ('class', 6), ('season', 12), ('tide', 18)):
            o['B' + nm] = ptx(kids[G + off]); o['C' + nm] = ptx(kids[G + off + 3])
            o['capB' + nm] = ptx(kids[G + off + 2]); o['capC' + nm] = ptx(kids[G + off + 5])
        old[key] = o
    # ---------------- 模板
    tg, tf, tc = kids[47], kids[48], kids[49]
    # ---------------- 移除與替換之區塊
    remove = set()
    old_rids = set()
    for key, name, n0, head, trk, trc, foot, G in T.SITES:
        for i in list(range(foot, foot + 3)) + list(range(G, G + 24)): remove.add(i)
    remove.add(209)
    for i in remove:
        for blip in kids[i].iter('{%s}blip' % A): old_rids.add(blip.get('{%s}embed' % R))
    assert len(old_rids) == 5 * 9, len(old_rids)   # 足跡 1 + 來源分段 8
    # ---------------- 新圖號
    newn = {}
    for key, name, n0, *_ in T.SITES:
        newn[key] = dict(traj=n0, track=n0 + 1, all=n0 + 2, **{'class': n0 + 3}, season=n0 + 4, tide=n0 + 5)
    oldn = {'guishan': (8, 9), 'gongliao': (19, 20), 'hsinchu': (30, 31), 'nanwan': (41, 42), 'lienchiang': (52, 53)}
    # ---------------- rels 與 docPr id
    ids_used = [int(e.get('id')) for e in body.iter('{%s}docPr' % WP)]
    next_dp = max(ids_used) + 1
    rid_nums = [int(re.sub(r'\D', '', r.get('Id'))) for r in rels_root]
    next_rid = max(rid_nums) + 1
    new_media = {}
    def add_image(n):
        nonlocal next_rid
        fn = 'fig%d.png' % n; path = os.path.join(figdir, fn)
        data = open(path, 'rb').read(); w, h = png_size(path)
        mname = 'media/v13_fig%d.png' % n
        rid = 'rId%d' % next_rid; next_rid += 1
        rel = etree.SubElement(rels_root, '{%s}Relationship' % REL)
        rel.set('Id', rid); rel.set('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image'); rel.set('Target', mname)
        new_media['word/' + mname] = data
        return rid, int(round(CX * h / w)), fn
    def make_fig(n, capheading):
        nonlocal next_dp
        f = copy.deepcopy(tf); strip_ids(f)
        rid, cy, fn = add_image(n)
        inl = next(f.iter('{%s}inline' % WP))
        for a in list(inl.attrib):
            if 'anchorId' in a or 'editId' in a: del inl.attrib[a]
        inl.find('{%s}extent' % WP).set('cx', str(CX)); inl.find('{%s}extent' % WP).set('cy', str(cy))
        dp = inl.find('{%s}docPr' % WP); dp.set('id', str(next_dp)); dp.set('name', fn); dp.set('descr', capheading)
        cn = next(inl.iter('{%s}cNvPr' % PIC)); cn.set('id', str(next_dp)); cn.set('name', fn); next_dp += 1
        next(inl.iter('{%s}blip' % A)).set('{%s}embed' % R, rid)
        ext = next(inl.iter('{%s}ext' % A)); ext.set('cx', str(CX)); ext.set('cy', str(cy))
        return f, cy
    def make_pair(n, gtext, ctext, force=False):
        g = copy.deepcopy(tg); strip_ids(g); set_text(g, gtext)
        c = copy.deepcopy(tc); strip_ids(c); set_text(c, ctext)
        head = re.match(r'^(圖\d+ .+?)；', ctext)
        f, cy = make_fig(n, head.group(1) if head else ctext[:40])
        ok, tot = fits(gtext, cy, ctext)
        ppr = g.find(q('pPr')); kn = ppr.find(q('keepNext'))
        if not ok and kn is not None and not force: ppr.remove(kn)   # force：季節圖導引段（v13b 已縮短）保留 keepNext，放不下時由排版引擎自行斷開
        log['keep'].append(dict(fig=n, keepNext=ok, est_pt=round(tot, 1)))
        return g, f, c
    # ---------------- 保留區塊之改寫
    # 區塊 34
    s34 = ptx(kids[34])
    assert '八張來源分段機率圖依序呈現全年、十類、季節與大小潮之邊界穿越來源與海岸來源（每項各以邊界與海岸兩張圖呈現）' in s34
    n34 = s34.replace('各研究地點依序以十一張圖說明', '各研究地點依序以六張圖說明')
    n34 = n34.replace('條件式來源足跡呈現虛擬質點於回溯期間停留之位置，', '')
    n34 = n34.replace('八張來源分段機率圖依序呈現全年、十類、季節與大小潮之邊界穿越來源與海岸來源（每項各以邊界與海岸兩張圖呈現）',
                      '四張來源分段機率圖依序呈現全年、十類、季節與大小潮之邊界穿越來源與海岸來源（每張圖以同一張地圖同時呈現邊界與海岸）')
    assert '足跡' not in n34 and n34 != s34
    set_text(kids[34], n34); log['notes'].append(['區塊34', s34, n34])
    # 區塊 46（方法段）
    s46 = ptx(kids[46])
    a = '圖 11 至圖 18 為龜山島之來源分段機率圖，依序呈現全年、十類、季節與大小潮之邊界穿越來源與海岸來源，每項各以邊界與海岸兩張圖呈現。'
    assert s46.startswith(a)
    b = '來源分段機率圖中，%s。' % T.PHRASE
    n46 = b + s46[len(a):]
    e = '圖中以色階表示；黑色括號標示機率集中之連續邊界段，灰色括號標示其餘連續非零邊界段，海岸機率集中之各段以引線標示其機率。'
    assert n46.endswith(e)
    n46 = n46[:-len(e)] + '圖中以色階表示；全年圖以黑色括號標示機率集中之連續邊界段，灰色括號標示其餘連續非零邊界段，海岸機率集中之各段以引線標示其機率；十類、季節與大小潮圖各格改以圓圈數字①②依機率由大至小標示機率集中之邊界段，ⓐⓑ依機率由大至小標示機率集中之海岸段，格內列各編號段之機率。'
    assert '龜山島' not in n46 and not re.search(r'圖 ?\d+', n46), n46
    set_text(kids[46], n46); log['notes'].append(['區塊46（v14：移至區塊 34 之後，改為一般性用語）', s46, n46])
    # 軌跡、仍在追蹤比例（改圖號）
    renum = {}
    for key, name, n0, head, trk, trc, foot, G in T.SITES:
        o1, o2 = oldn[key]; n1, n2 = n0, n0 + 1
        renum[key] = {o1: n1, o2: n2}
        # 軌跡導引段
        g = ptx(kids[trk]); g2 = re.sub(r'^圖 %d ' % o1, '圖 %d ' % n1, g)
        g2 = re.sub(r'與圖 \d+(?: 及圖 \d+)? 之主要來源一致', '與圖 %d 之主要來源一致' % newn[key]['all'], g2)
        assert g2 != g and '與圖 %d 之主要來源一致' % newn[key]['all'] in g2, g
        set_text(kids[trk], g2)
        c = ptx(kids[trk + 2]); set_text(kids[trk + 2], re.sub(r'^圖%d' % o1, '圖%d' % n1, c))
        g = ptx(kids[trc]); set_text(kids[trc], re.sub(r'^圖 %d ' % o2, '圖 %d ' % n2, g))
        c = ptx(kids[trc + 2]); set_text(kids[trc + 2], re.sub(r'^圖%d' % o2, '圖%d' % n2, c))
        for (dg, ii, nn) in ((kids[trk + 1], o1, n1), (kids[trc + 1], o2, n2)):
            for e_ in dg.iter('{%s}docPr' % WP, '{%s}cNvPr' % PIC):
                for at in ('name', 'descr'):
                    v = e_.get(at)
                    if v is None: continue
                    v2 = re.sub(r'^fig%d\.' % ii, 'fig%d.' % nn, v); v2 = re.sub(r'^圖%d' % ii, '圖%d' % nn, v2)
                    if v2 != v: e_.set(at, v2)
    # ---------------- 組裝新區塊序列
    new_blocks = []
    made = {}
    for key, name, n0, head, trk, trc, foot, G in T.SITES:
        N = newn[key]
        made[key] = T.compose(key, name, N, old[key])
    for i, k in enumerate(kids):
        if i in remove:
            for key, name, n0, head, trk, trc, foot, G in T.SITES:
                if i == foot - 0 and False: pass
            # 在各地點足跡三段之位置不放新圖；新圖放在原來源分段起點 G
            for key, name, n0, head, trk, trc, foot, G in T.SITES:
                if i == G:
                    for kd in T.KINDS:
                        gtext, ctext = made[key][kd]
                        g, f, c = make_pair(newn[key][kd], gtext, ctext)
                        new_blocks += [g, f, c]
                        log['fig_map'][newn[key][kd]] = '%s %s' % (name, kd)
            continue
        if i == 46: continue   # v14：方法段移至（二）說明之後
        if 14 <= i <= 17: continue   # v14r2：圖 1、圖 2（含圖說）改緊接首次提及處（區塊 9）之後
        if MERGE and i == 9: continue   # v14r4：區塊 9 併入區塊 8（見下）
        new_blocks.append(k)
        if MERGE and i == 8:
            # v14r4：區塊 8（2、到達地點…）與區塊 9（共由「5 個水平位置…」…）合為一段，保留區塊 8 之 pPr；接縫處插入「這 20 種到達地點」run（rPr 取自區塊 9 之「5 」run）
            k9 = kids[9]
            runs9 = k9.findall(q('r'))
            assert ''.join(t_.text or '' for t_ in runs9[0].iter(q('t'))) == '共', 'block 9 first run'
            r5 = next(r_ for r_ in runs9 if ''.join(t_.text or '' for t_ in r_.iter(q('t'))) == '5 ')
            if os.environ.get('INSERT', '1') == '1':
                nr = etree.Element(q('r')); nr.append(copy.deepcopy(r5.find(q('rPr'))))
                nt = etree.SubElement(nr, q('t')); nt.text = '這 20 種到達地點'
                k.append(nr)
            for ch in list(k9):
                if ch.tag == q('pPr'): continue
                k.append(ch)
            ppr9 = k.find(q('pPr'))
            for tag in (() if os.environ.get('KEEPNEXT', '1') == '0' else (('keepLines', 'keepNext') if os.environ.get('KEEPLINES', '1') == '1' else ('keepNext',))):
                if ppr9.find(q(tag)) is None:
                    el = etree.Element(q(tag)); pos = 1 if ppr9.find(q('pStyle')) is not None else 0
                    ppr9.insert(pos, el)
            sp14 = kids[14].find(q('pPr')).find(q('spacing')); assert sp14.get(q('before')) == '240'; sp14.set(q('before'), os.environ.get('B10_BEFORE', '240'))
            sp15 = kids[15].find(q('pPr')).find(q('spacing')); assert sp15.get(q('before')) == '40'; sp15.set(q('before'), os.environ.get('C1_BEFORE', '40'))
            if FIG1_SCALE != 1.0:
                for e_ in kids[14].iter('{%s}extent' % WP, '{%s}ext' % A_NS):
                    if e_.get('cx') is not None and e_.get('cy') is not None:
                        e_.set('cx', str(int(round(int(e_.get('cx')) * FIG1_SCALE)))); e_.set('cy', str(int(round(int(e_.get('cy')) * FIG1_SCALE))))
            new_blocks.extend(kids[14:18])
            log['notes'].append(['區塊 8、9 合為一段（保留區塊 8 之 pPr，加 keepNext＋keepLines；接縫插入「這 20 種到達地點」）', '〔區塊 8 末〕…任意懸浮深度。／〔區塊 9 首〕共由「5 個水平位置 × 每個位置 4 個深度抽樣」構成。', '…任意懸浮深度。這 20 種到達地點共由「5 個水平位置 × 每個位置 4 個深度抽樣」構成。'])
            log['notes'].append(['圖 1、圖 2 置於合併段之後；圖 1 縮放 %s；圖 1 段段前 %s、圖 1 圖說段前 %s（twips）' % (FIG1_SCALE, sp14.get(q('before')), sp15.get(q('before')))])
        if (not MERGE) and i == 9:
            ppr9 = k.find(q('pPr'))
            for tag in ('keepLines', 'keepNext'):   # 導引段不跨頁、與下一段（圖 1）同頁
                if ppr9.find(q(tag)) is None:
                    el = etree.Element(q(tag)); pos = 1 if ppr9.find(q('pStyle')) is not None else 0
                    ppr9.insert(pos, el)
            # v14r3：補主語。「共由「5 個水平位置…」→「20 種到達地點共由「5 個水平位置…」（新增一個 run，rPr 取自同段「5 」之 run，不指定字型，與段內其他半形數字同）
            runs9 = [r_ for r_ in k.findall(q('r'))]
            assert ''.join(t_.text or '' for t_ in runs9[0].iter(q('t'))) == '共', 'block 9 first run'
            r5 = next(r_ for r_ in runs9 if ''.join(t_.text or '' for t_ in r_.iter(q('t'))) == '5 ')
            nr = etree.Element(q('r')); nr.append(copy.deepcopy(r5.find(q('rPr'))))
            nt = etree.SubElement(nr, q('t')); nt.text = '20 種到達地點'; nt.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            runs9[0].addprevious(nr)
            log['notes'].append(['區塊 9（圖 1／2 導引段）', '共由「5 個水平位置 × 每個位置 4 個深度抽樣」構成。', '20 種到達地點共由「5 個水平位置 × 每個位置 4 個深度抽樣」構成。'])
            # v14r3：補主語後導引段多一行（末行僅「隨機抽取。」），圖 1 圖說第二行落到下頁；只調段落間距與圖 1 縮放，不動文字
            sp9 = ppr9.find(q('spacing')); assert sp9.get(q('after')) == '120'; sp9.set(q('after'), os.environ.get('G9_AFTER', '0'))
            sp14 = kids[14].find(q('pPr')).find(q('spacing')); assert sp14.get(q('before')) == '240'; sp14.set(q('before'), os.environ.get('B10_BEFORE', '120'))
            sp15 = kids[15].find(q('pPr')).find(q('spacing')); assert sp15.get(q('before')) == '40'; sp15.set(q('before'), os.environ.get('C1_BEFORE', '40'))
            log['notes'].append(['區塊 9 段後 120→%s；圖 1 段段前 240→%s；圖 1 圖說段前 40→%s（twips，僅格式，文字不變）' % (sp9.get(q('after')), sp14.get(q('before')), sp15.get(q('before')))])
            # 圖 1 縮小 10%（等比），使圖與兩行圖說能與導引段同頁
            for e_ in kids[14].iter('{%s}extent' % WP, '{%s}ext' % A_NS):
                if e_.get('cx') is not None and e_.get('cy') is not None:
                    e_.set('cx', str(int(round(int(e_.get('cx')) * FIG1_SCALE)))); e_.set('cy', str(int(round(int(e_.get('cy')) * FIG1_SCALE))))
            new_blocks.extend(kids[14:18])
            log['notes'].append(['圖 1、圖 2 改置於區塊 9（首次提及處）之後，區塊 9 加 keepLines＋keepNext；文字不變'])
        if i == 34: new_blocks.append(kids[46])
    # 移除舊 body 子元素並寫入
    for k in kids: body.remove(k)
    for k in new_blocks: body.append(k)
    # ---------------- rels 與 media
    rm_targets = set()
    for rel in list(rels_root):
        if rel.get('Id') in old_rids:
            rm_targets.add(rel.get('Target')); rels_root.remove(rel)
    still = set(r.get('Target') for r in rels_root)
    for t in rm_targets: assert t not in still, t
    xml_out = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    xs = xml_out.decode('utf-8')
    for rid in old_rids: assert ('"%s"' % rid) not in xs, rid
    rels_out = etree.tostring(rels_root, xml_declaration=True, encoding='UTF-8', standalone=True)
    zout = zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED)
    for zi in zin.infolist():
        if zi.filename.startswith('word/') and zi.filename[len('word/'):] in rm_targets: log['removed'].append(zi.filename); continue
        data = zin.read(zi.filename)
        if zi.filename == 'word/document.xml': data = xml_out
        elif zi.filename == 'word/_rels/document.xml.rels': data = rels_out
        elif zi.filename == 'word/styles.xml' and os.environ.get('FONTFIX', '1') == '1': data = patch_styles(data, log)
        ni = zipfile.ZipInfo(zi.filename, date_time=zi.date_time); ni.compress_type = zi.compress_type; ni.external_attr = zi.external_attr
        zout.writestr(ni, data)
    for k_, d_ in new_media.items():
        ni = zipfile.ZipInfo(k_, date_time=(2026, 10, 30, 0, 0, 0)); ni.compress_type = zipfile.ZIP_STORED; zout.writestr(ni, d_)
    zout.close()
    # 全文殘留之圖號掃描（保留區塊中是否仍有 ≥8 之舊圖號）
    log['n_blocks'] = len(new_blocks)
    json.dump(log, open(logp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('ok', dst, 'blocks', len(new_blocks), 'removed media', len(log['removed']), 'sha', hashlib.sha256(open(dst, 'rb').read()).hexdigest())

if __name__ == '__main__':
    main(*sys.argv[1:5])
