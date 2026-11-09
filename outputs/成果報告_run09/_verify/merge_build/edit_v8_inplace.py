#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合併版 v8 候選：以來源分段機率圖取代圓形來源方位圖，並改寫相關文字（就地修改 word/document.xml 與關係；其餘 zip 成員逐位元組保留）。
用法：python3 edit_v8_inplace.py <合併版docx(主檔)> <輸出docx>
前提：主檔 sha256 須為 45f8d015…（v7 加使用者手改），否則停止。主檔不被修改。
文字內容由 v8_content.py（數字取自 segprob_v3 之 manifest 與表）產生；圖檔取自 segprob_v3/<site>/*.png。"""
import sys, os, re, zipfile, hashlib, copy, json, unicodedata
from lxml import etree
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v8_content as C
from v8data import B as SEGDIR, SITES, SN

PRE = '45f8d015914eeb6d'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
WP14 = 'http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
PIC = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
XMLSP = '{http://www.w3.org/XML/1998/namespace}space'
q = lambda t: '{%s}%s' % (W, t)
q14 = lambda t: '{%s}%s' % (W14, t)

src, dst = sys.argv[1], sys.argv[2]
raw = open(src, 'rb').read()
assert hashlib.sha256(raw).hexdigest().startswith(PRE), '主檔 sha 與預期不同，停止'
zin = zipfile.ZipFile(src)
xml = zin.read('word/document.xml')
root = etree.fromstring(xml); body = root.find(q('body')); kids = list(body)
assert len(kids) == 299

def ptxt(p): return ''.join(t.text or '' for t in p.iter(q('t')))
def runs_of(p): return [r for r in p.findall(q('r'))]
def rtxt(r): return ''.join(t.text or '' for t in r.findall(q('t')))

# ------------------------------------------------------------ 區塊定位（主檔區塊序）與錨點檢查
TRAJ = {'gongliao': 37, 'guishan': 60, 'hsinchu': 82, 'nanwan': 104, 'lienchiang': 126}
POLAR = {'gongliao': (46, 58), 'guishan': (69, 80), 'hsinchu': (91, 102), 'nanwan': (113, 124), 'lienchiang': (135, 146)}   # 含頭尾
OLDB = C.OLDB                     # 舊軌跡圖號
NEWB = C.BASE
for s in SITES:
    t0 = TRAJ[s]
    assert ptxt(kids[t0]).startswith('圖 %d 為' % OLDB[s]), (s, ptxt(kids[t0])[:20])
    for j in range(3):
        assert re.match(r'^圖%d ' % (OLDB[s] + j), ptxt(kids[t0 + 3 * j + 2])), (s, j, ptxt(kids[t0 + 3 * j + 2]))
    a, b = POLAR[s]
    assert re.match(r'^圖 %d ' % (OLDB[s] + 3), ptxt(kids[a])), (s, ptxt(kids[a])[:20])
    assert re.match(r'^圖%d ' % (OLDB[s] + 6), ptxt(kids[b])) and ptxt(kids[b]).endswith('（依大小潮）'), (s, ptxt(kids[b]))
    assert b - a + 1 == (13 if s == 'gongliao' else 12), s
assert ptxt(kids[33]).startswith('本節呈現') and ptxt(kids[34]).startswith('各研究地點依序以七張圖說明')
assert ptxt(kids[148]).startswith('各研究地點十類之間') and ptxt(kids[152]).startswith('（1）貢寮') and ptxt(kids[157]).startswith('（6）綜合')
assert ptxt(kids[147]).startswith('綜合說明與結論') and ptxt(kids[151]).startswith('整體結論')

# ------------------------------------------------------------ 工具
used_pid = {e.get(q14('paraId')) for e in root.iter() if e.get(q14('paraId'))}
_ctr = [0x3B000000]
def new_pid():
    while True:
        _ctr[0] += 0x1357
        v = '%08X' % _ctr[0]
        if v not in used_pid: used_pid.add(v); return v
used_anchor = set()
def new_anchor():
    while True:
        _ctr[0] += 0x2461
        v = '%08X' % (_ctr[0] ^ 0x1A2B3C4D & 0x7FFFFFFF)
        if v not in used_anchor and v not in used_pid: used_anchor.add(v); return v
for e in root.iter():
    for k in ('{%s}anchorId' % WP14, '{%s}editId' % WP14):
        if e.get(k): used_anchor.add(e.get(k))
used_dp = [int(e.get('id')) for e in root.iter('{%s}docPr' % WP)]
_dp = [max(used_dp)]
def new_dp(): _dp[0] += 1; return _dp[0]

NUMTOK = re.compile(r'(\s?\d[\d,.]*(?:%|°[EN]?)?\s?)')
def mkrun(text, rpr):
    r = etree.Element(q('r'))
    if rpr is not None: r.append(copy.deepcopy(rpr))
    t = etree.SubElement(r, q('t')); t.text = text
    if text != text.strip() or '  ' in text: t.set(XMLSP, 'preserve')
    return r
def split_runs(text, rpr, number_runs=True):
    parts = NUMTOK.split(text) if number_runs else [text]
    return [mkrun(p, rpr) for p in parts if p != '']
def set_par_text(p, text, rpr, keep_first=0):
    """保留 pPr 與前 keep_first 個 run，其後內容改為 text（以 rpr 之副本切 run）。"""
    kept = 0; rs = []
    for ch in list(p):
        if ch.tag == q('pPr'): continue
        if ch.tag == q('r') and kept < keep_first:
            kept += 1; rs.append(ch); continue
        p.remove(ch)
    for r in split_runs(text, rpr): p.append(r)
def clean_lrpb(p):
    for x in list(p.iter(q('lastRenderedPageBreak'))): x.getparent().remove(x)

def set_keep(p, keepnext, keeplines):
    ppr = p.find(q('pPr'))
    for t in ('keepNext', 'keepLines'):
        for x in ppr.findall(q(t)): ppr.remove(x)
    at = 1 if ppr.find(q('pStyle')) is not None else 0
    if keeplines: ppr.insert(at, etree.Element(q('keepLines')))
    if keepnext: ppr.insert(at, etree.Element(q('keepNext')))

# ------------------------------------------------------------ 樣板（取自主檔）
T_GUIDE = kids[47]      # 圖前導引段（keepLines）
T_METHOD = kids[46]     # 一般段（無 keep）
T_IMG = kids[48]
T_CAP = kids[49]
RPR_G = copy.deepcopy(runs_of(T_GUIDE)[0].find(q('rPr')))
RPR_C = copy.deepcopy(runs_of(T_CAP)[0].find(q('rPr')))

def new_par(tmpl):
    p = copy.deepcopy(tmpl)
    for a in list(p.attrib):
        if a == q14('paraId'): p.set(a, new_pid())
    for ch in list(p):
        if ch.tag != q('pPr'): p.remove(ch)
    return p

# ------------------------------------------------------------ 圖檔與關係
rels_x = etree.fromstring(zin.read('word/_rels/document.xml.rels'))
rel_by_id = {r.get('Id'): r for r in rels_x}
ct = zin.read('[Content_Types].xml').decode('utf-8')
assert 'Extension="png"' in ct and 'Extension="jpg"' in ct
rid_num = max(int(i[3:]) for i in rel_by_id)
media_new = []      # (zip 路徑, bytes)
next_img = [43]
FIGDEF = [('all_b', 'boundary_all'), ('all_c', 'coast_all'), ('cls_b', 'boundary_class'), ('cls_c', 'coast_class'),
          ('sea_b', 'boundary_season'), ('sea_c', 'coast_season'), ('tid_b', 'boundary_tide'), ('tid_c', 'coast_tide')]
import io
def add_media(path):
    global rid_num
    b = open(path, 'rb').read()
    w, h = Image.open(io.BytesIO(b)).size
    assert w == 3762, (path, w)
    n = next_img[0]; next_img[0] += 1
    rid_num += 1; rid = 'rId%d' % rid_num
    rel = etree.SubElement(rels_x, '{http://schemas.openxmlformats.org/package/2006/relationships}Relationship')
    rel.set('Id', rid); rel.set('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image'); rel.set('Target', 'media/image%d.png' % n)
    media_new.append(('word/media/image%d.png' % n, b))
    return rid, w, h, 'image%d.png' % n

def fig_par(site, key, mf, n, title):
    p = copy.deepcopy(T_IMG)
    p.set(q14('paraId'), new_pid())
    for x in list(p.iter(q('lastRenderedPageBreak'))): x.getparent().remove(x)
    rid, w, h, fname = add_media(os.path.join(SEGDIR, site, mf + '.png'))
    cx = 5733288; cy = round(h / 600.0 * 914400)
    inl = next(p.iter('{%s}inline' % WP))
    inl.set('{%s}anchorId' % WP14, new_anchor()); inl.set('{%s}editId' % WP14, new_anchor())
    ext = inl.find('{%s}extent' % WP); ext.set('cx', str(cx)); ext.set('cy', str(cy))
    dp = inl.find('{%s}docPr' % WP); did = str(new_dp()); dp.set('id', did); dp.set('name', 'fig%d.png' % n); dp.set('descr', '圖%d %s' % (n, title))
    cn = next(inl.iter('{%s}cNvPr' % PIC)); cn.set('id', did); cn.set('name', 'fig%d.png' % n)
    next(inl.iter('{%s}blip' % A)).set('{%s}embed' % R, rid)
    for e in inl.iter('{%s}ext' % A): e.set('cx', str(cx)); e.set('cy', str(cy))
    return p, fname, h

def cap_par(n, caption):
    p = new_par(T_CAP)
    p.append(mkrun('圖', RPR_C)); p.append(mkrun('%d ' % n, RPR_C)); p.append(mkrun(C.sp(caption), RPR_C))
    set_keep(p, False, True)   # v8c：新增圖說加 keepLines，避免圖說跨頁（主檔既有 42 個圖說皆無 keepLines；表格圖說有）
    return p

def guide_par(text, keepnext):
    p = new_par(T_GUIDE)
    for r in split_runs(text, RPR_G): p.append(r)
    set_keep(p, keepnext, True)
    return p

# ------------------------------------------------------------ 版面估算（與 check_merged_run09.py K4–K6 同式）
LIM = (16838 - 2880) / 20.0 - 17.0
def nl(text, wpt):
    ln, w = 1, 0.0
    for ch in text:
        cw = 14.0 if unicodedata.east_asian_width(ch) in ('W', 'F') else 7.0
        if w + cw > wpt: ln += 1; w = cw
        else: w += cw
    return ln
BW = (11906 - 2880 - 991) / 20.0; CW = (11906 - 2880) / 20.0
def fits(chain_texts, text, fig_cy, cap_text):
    tot = sum(nl(t, BW) * 22 + 18 for t in chain_texts) + nl(text, BW) * 22 + 18 + 12 + fig_cy / 12700.0 + (2 + nl(cap_text, CW) * 27 + 8)
    return tot <= LIM, tot

# ------------------------------------------------------------ 1. 組裝新區塊
changes = []     # 變更紀錄（供變更清單與檢查使用）
newblocks = {}   # 起始主檔區塊序 → 新區塊串列
removed = set()
fig_map = []     # (site, key, 新圖號, 圖檔, 圖高 px, 圖說)
for s in SITES:
    a, b = POLAR[s]
    G = C.guides(s); CAP = C.captions(s)
    blocks = []
    if s == 'gongliao':
        mp = new_par(T_METHOD)
        for r in split_runs(C.method_par(), RPR_G): mp.append(r)
        blocks.append(mp)
        changes.append(dict(kind='new', site=s, role='method', old='（圓形來源方位圖之方法段，主檔區塊 46）', new=C.method_par()))
    for key, mf in FIGDEF:
        n = C.fno(s, key)
        title = C.sp(CAP[key]).split('；')[0]
        ip, fname, h = fig_par(s, key, mf, n, title)
        cap_text = '圖%d %s' % (n, C.sp(CAP[key]))
        fit, tot = fits([], G[key], round(h / 600.0 * 914400), cap_text)
        blocks += [guide_par(G[key], fit), ip, cap_par(n, CAP[key])]
        fig_map.append(dict(site=s, key=key, n=n, media=fname, h_px=h, caption=C.sp(CAP[key]), guide=G[key], fit=fit, tot=round(tot, 1)))
        changes.append(dict(kind='new', site=s, role='guide:' + key, new=G[key], fig=n))
        changes.append(dict(kind='new', site=s, role='caption:' + key, new='圖%d %s' % (n, C.sp(CAP[key])), fig=n))
    newblocks[a] = blocks
    for i in range(a, b + 1):
        removed.add(i)
        changes.append(dict(kind='removed', site=s, old_block=i, old=ptxt(kids[i]) if '[IMG]' not in ptxt(kids[i]) and kids[i].tag == q('p') and not list(kids[i].iter('{%s}inline' % WP)) else '（圖：' + next(kids[i].iter('{%s}docPr' % WP)).get('descr') + '）'))

# 被移除圖之媒體與關係
drop_media = []
for i in sorted(removed):
    for inl in kids[i].iter('{%s}inline' % WP):
        rid = next(inl.iter('{%s}blip' % A)).get('{%s}embed' % R)
        rel = rel_by_id[rid]; drop_media.append('word/' + rel.get('Target')); rels_x.remove(rel)
assert len(drop_media) == 20 and len(set(drop_media)) == 20

# ------------------------------------------------------------ 2. 保留圖之重編號與前導段改寫
renum = []   # (舊圖號, 新圖號)
for s in SITES:
    t0 = TRAJ[s]
    for j in range(3):
        old_n, new_n = OLDB[s] + j, NEWB[s] + j
        g, im, cp = kids[t0 + 3 * j], kids[t0 + 3 * j + 1], kids[t0 + 3 * j + 2]
        if old_n != new_n:
            # 前導段：首個 " N " run
            rs = runs_of(g); k = next(i for i, r in enumerate(rs) if rtxt(r) == ' %d ' % old_n and rtxt(rs[i - 1]) == '圖')
            rs[k].find(q('t')).text = ' %d ' % new_n
            # 圖與圖說
            dp = next(im.iter('{%s}docPr' % WP)); assert dp.get('name') == 'fig%d.jpg' % old_n
            dp.set('name', 'fig%d.jpg' % new_n); dp.set('descr', dp.get('descr').replace('圖%d ' % old_n, '圖%d ' % new_n, 1))
            cn = next(im.iter('{%s}cNvPr' % PIC)); cn.set('name', 'fig%d.jpg' % new_n)
            crs = runs_of(cp); assert rtxt(crs[0]) == '圖' and rtxt(crs[1]) == '%d ' % old_n
            crs[1].find(q('t')).text = '%d ' % new_n
            changes.append(dict(kind='renumber', site=s, old_fig=old_n, new_fig=new_n, old_block=t0 + 3 * j))
        renum.append((old_n, new_n))
    # 軌跡導引段結尾：與圖 N 之主要來源方向一致 → 與圖 A 及圖 B 之主要來源一致（新竹外海僅邊界圖）
    g = kids[t0]; old_text = ptxt(g)
    rs = runs_of(g)
    assert rtxt(rs[-1]) == '之主要來源方向一致。' and re.match(r'^ \d+ $', rtxt(rs[-2])) and rtxt(rs[-3]).endswith('與圖'), ptxt(g)
    a_b, a_c = C.fno(s, 'all_b'), C.fno(s, 'all_c')
    rs[-2].find(q('t')).text = ' %d ' % a_b
    rs[-1].find(q('t')).text = '之主要來源一致。'
    if s != 'hsinchu':
        r1 = copy.deepcopy(rs[-1]); r1.find(q('t')).text = '及圖'
        r2 = copy.deepcopy(rs[-2]); r2.find(q('t')).text = ' %d ' % a_c
        i = list(g).index(rs[-2]); g.insert(i + 1, r1); g.insert(i + 2, r2)
    changes.append(dict(kind='rewrite', site=s, role='traj_tail', old_block=t0, old=old_text, new=ptxt(g)))

# ------------------------------------------------------------ 3. 文字段改寫（33、34、148、152–157）
def rewrite(i, text, keep_first=0, role=''):
    p = kids[i]; old = ptxt(p)
    rs = runs_of(p)
    rpr = rs[keep_first].find(q('rPr')) if len(rs) > keep_first else rs[0].find(q('rPr'))
    rpr = copy.deepcopy(rpr)
    set_par_text(p, text, rpr, keep_first)
    clean_lrpb(p)
    changes.append(dict(kind='rewrite', role=role, old_block=i, old=old, new=ptxt(p)))
rewrite(33, C.P33, role='p33')
rewrite(34, C.P34, role='p34')
rewrite(148, C.P148, role='p148_summary')
for k, s in enumerate(SITES):
    # 保留「（」「n」兩個 run；其後以「）…」改寫
    rewrite(152 + k, '）' + C.conclusion(s), keep_first=2, role='conclusion_%d' % (k + 1))
rewrite(157, '）' + C.conclusion6(), keep_first=2, role='conclusion_6')

# ------------------------------------------------------------ 4. 重排本文
newkids = []
for i, e in enumerate(kids):
    if i in newblocks: newkids += newblocks[i]
    if i in removed: continue
    newkids.append(e)
for e in list(body): body.remove(e)
for e in newkids: body.append(e)
assert len(newkids) == 299 - len(removed) + sum(len(v) for v in newblocks.values())

# 軌跡／still／足跡導引段：以新排版重算 keepNext（與主檔比對，不一致者回報）
flips = []
def intro_fit(g, im, cp, chain):
    cy = int(next(im.iter('{%s}extent' % WP)).get('cy'))
    return fits([ptxt(x) for x in chain], ptxt(g), cy, ptxt(cp))
idx = {id(e): n for n, e in enumerate(newkids)}
for s in SITES:
    t0 = TRAJ[s]
    for j in range(3):
        g, im, cp = kids[t0 + 3 * j], kids[t0 + 3 * j + 1], kids[t0 + 3 * j + 2]
        chain = []
        if j == 0:
            chain = [kids[t0 - 1]] + ([kids[t0 - 2]] if s == 'gongliao' else [])
        f, tot = intro_fit(g, im, cp, chain)
        has = g.find(q('pPr')).find(q('keepNext')) is not None
        if f != has: flips.append((s, j, has, f, round(tot, 1)))
json.dump(dict(flips=flips), open(os.path.join(HERE, 'v8_keep_flips.json'), 'w'))

# ------------------------------------------------------------ 5. 寫出
xml_out = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
rels_out = etree.tostring(rels_x, xml_declaration=True, encoding='UTF-8', standalone=True)
zout = zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED)
for zi in zin.infolist():
    if zi.filename in drop_media: continue
    data = zin.read(zi.filename)
    if zi.filename == 'word/document.xml': data = xml_out
    elif zi.filename == 'word/_rels/document.xml.rels': data = rels_out
    ni = zipfile.ZipInfo(zi.filename, date_time=zi.date_time); ni.compress_type = zi.compress_type; ni.external_attr = zi.external_attr
    zout.writestr(ni, data)
for path, b in media_new:
    ni = zipfile.ZipInfo(path, date_time=(2026, 10, 28, 0, 0, 0)); ni.compress_type = zipfile.ZIP_STORED
    zout.writestr(ni, b)
zout.close()

json.dump(dict(fig_map=fig_map, renum=renum, changes=changes, dropped_media=drop_media, flips=flips, n_blocks=len(newkids)),
          open(os.path.join(HERE, 'v8_build_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('ok', dst, len(newkids), 'flips', flips)
