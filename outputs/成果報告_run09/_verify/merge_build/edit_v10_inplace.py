#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合併版 v10 候選：以「現行主檔」（v9 升格後使用者再存，sha 20d83708…）為基準，就地修改 word/document.xml（其餘 zip 成員逐位元組保留，圖檔另計）。
(1) 40 個來源分段機率圖圖說改為不寫位置、不寫「數值框」之新句（v10_caption_draft.md；圖 38、40 依實際圈號；圖 15、48 註記未編號括號段；全年邊界圖加括號旁數值句）；
(2) 依新圖說重算 40 張圖之導引段 keepNext（同 v8／v9 估算式）；
(3) （選用）--curves：以選項 B 之 5 張「仍在追蹤比例」圖（圖 9、20、31、42、53）取代原圖（改為 png，寬 6.27 in 不縮放）。
用法：edit_v10_inplace.py <現行主檔> <caps_v10.json> <輸出docx> <v10_build_log.json> [--curves <目錄>]
前提：主檔 sha256 須以 20d8370880154b53 起首且區塊數 359，否則停止。主檔不被修改。"""
import sys, os, re, zipfile, hashlib, json, unicodedata, struct
from lxml import etree

PRE = '20d8370880154b53'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PR = 'http://schemas.openxmlformats.org/package/2006/relationships'
XML = 'http://www.w3.org/XML/1998/namespace'
q = lambda t: '{%s}%s' % (W, t)
SITES = ['gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang']
CURVE_FIG = {'gongliao': 9, 'guishan': 20, 'hsinchu': 31, 'nanwan': 42, 'lienchiang': 53}
SENT_E = '各類 10 個速度之跨距較大，表 1 以科學記號列其範圍；模擬時 10 個速度各自直接使用，不取平均值或對數平均。'   # 使用者核定：加在表 1 後第一段（區塊 6）之末
BLK6_OLD = '各類 10 個速度係以受力平衡求解，完整推導過程、代表物件之尺寸與輸入參數、實測量級對照與敏感度分析詳見附錄。'
LIM = (16838 - 2880) / 20.0 - 17.0
BW = (11906 - 2880 - 991) / 20.0
CW = (11906 - 2880) / 20.0


def nl(text, wpt):
    ln, w = 1, 0.0
    for ch in text:
        cw = 14.0 if unicodedata.east_asian_width(ch) in ('W', 'F') else 7.0
        if w + cw > wpt:
            ln += 1; w = cw
        else:
            w += cw
    return ln


def fits(text, fig_cy, cap_text):
    tot = nl(text, BW) * 22 + 18 + 12 + fig_cy / 12700.0 + (2 + nl(cap_text, CW) * 27 + 8)
    return tot <= LIM, tot


SITE_ZH = {'gongliao': '貢寮', 'guishan': '龜山島', 'hsinchu': '新竹外海', 'nanwan': '南灣', 'lienchiang': '連江'}
SLOW = {'gongliao', 'guishan', 'nanwan'}   # 不可回收塑膠明顯略慢者（由資料判定：與其餘九類最大差距 2–6 個百分點）
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'


def para_replace(p, old, new):
    """於段落（直屬 w:r 之 w:t 串接文字）中以 new 取代唯一出現之 old；可跨 run，新文字落在 old 首字所在 run（沿用其格式）。"""
    ts = [t for r in p.findall(q('r')) for t in r.findall(q('t'))]
    s = ''.join(t.text or '' for t in ts); assert s.count(old) == 1, (old, s.count(old))
    a = s.index(old); b = a + len(old); pos = 0; first = True
    for t in ts:
        tx = t.text or ''; lo, hi = pos, pos + len(tx); pos = hi
        if hi <= a or lo >= b: continue
        o0 = max(a, lo) - lo; o1 = min(b, hi) - lo
        t.text = (tx[:o0] + new + tx[o1:]) if first else (tx[:o0] + tx[o1:]); first = False
        t.set(XML_SPACE, 'preserve')
    return ''.join(t.text or '' for t in ts)


def main(src, capj, dst, logp, curves=None):
    raw = open(src, 'rb').read()
    assert hashlib.sha256(raw).hexdigest().startswith(PRE), '主檔 sha 與預期（20d83708…）不同，停止'
    zin = zipfile.ZipFile(src)
    root = etree.fromstring(zin.read('word/document.xml')); body = root.find(q('body')); kids = list(body)
    assert len(kids) == 359, len(kids)
    rels_root = etree.fromstring(zin.read('word/_rels/document.xml.rels'))
    rels = {r.get('Id'): r.get('Target') for r in rels_root}
    caps = json.load(open(capj, encoding='utf-8'))
    # (4) 表 1 後第一段（區塊 6）末加一句（使用者核定）；就地加在最後一個 run 之文字後，格式不動
    ptx0 = lambda p: ''.join(t.text or '' for t in p.iter(q('t')))
    assert ptx0(kids[6]) == BLK6_OLD, ptx0(kids[6])
    lastr = [r for r in kids[6].findall(q('r')) if r.find(q('t')) is not None][-1]
    lt = lastr.find(q('t')); lt.text = lt.text + SENT_E
    assert ptx0(kids[6]) == BLK6_OLD + SENT_E
    block6 = dict(index=6, old=BLK6_OLD, added=SENT_E, new=BLK6_OLD + SENT_E)
    ptxt = lambda p: ''.join(t.text or '' for t in p.iter(q('t')))
    imgpar = {}
    for i, p in enumerate(kids):
        if p.tag != q('p'): continue
        for inl in p.iter('{%s}inline' % WP):
            m = re.match(r'^fig(\d+)\.(?:png|jpg)$', inl.find('{%s}docPr' % WP).get('name'))
            if m: imgpar[int(m.group(1))] = i
    flips = []; changes = []; warn = []
    for ns, rec in sorted(caps.items(), key=lambda kv: int(kv[0])):
        n = int(ns); i = imgpar[n]; g, ip, cp = kids[i - 1], kids[i], kids[i + 1]
        assert ptxt(cp) == '圖%d %s' % (n, rec['old']), (n, ptxt(cp)[:50])
        if rec['new'] != rec['old']:
            runs = cp.findall(q('r')); assert len(runs) >= 3
            assert ptxt(runs[0]) == '圖' and ptxt(runs[1]) == '%d ' % n, (n, ptxt(runs[0]), ptxt(runs[1]))
            rp = [etree.tostring(r.find(q('rPr'))) for r in runs]; assert len(set(rp)) == 1, (n, '圖說 run 格式不一致')
            t = runs[2].find(q('t')); t.text = rec['new']; t.set('{%s}space' % XML, 'preserve')
            for r in runs[3:]: cp.remove(r)
            assert ptxt(cp) == '圖%d %s' % (n, rec['new'])
            changes.append(dict(kind='v10_caption', fig=n, old='圖%d %s' % (n, rec['old']), new='圖%d %s' % (n, rec['new'])))
        ext = next(ip.iter('{%s}inline' % WP)).find('{%s}extent' % WP); cy = int(ext.get('cy'))
        ppr = g.find(q('pPr')); kn = ppr.find(q('keepNext'))
        f_old, _ = fits(ptxt(g), cy, '圖%d %s' % (n, rec['old']))
        if f_old != (kn is not None): warn.append((n, '主檔導引段 keepNext 與 v9 估算不同', kn is not None, f_old))
        fit, tot = fits(ptxt(g), cy, '圖%d %s' % (n, rec['new']))
        if fit != (kn is not None):
            flips.append(dict(fig=n, old=kn is not None, new=fit, tot=round(tot, 1)))
            if kn is not None: ppr.remove(kn)
            else: ppr.insert(1 if ppr.find(q('pStyle')) is not None else 0, etree.Element(q('keepNext')))
    media_new = {}; rels_change = {}; curve_log = []; curve_text = []
    if curves:
        for s, n in CURVE_FIG.items():
            i = imgpar[n]; inl = next(kids[i].iter('{%s}inline' % WP)); ext = inl.find('{%s}extent' % WP)
            png = open(os.path.join(curves, s, 'domain_fraction_vs_backtrack_time.png'), 'rb').read()
            assert png[:8] == b'\x89PNG\r\n\x1a\n'
            w_, h_ = struct.unpack('>II', png[16:24]); assert w_ == 3762, w_
            cy = round(h_ / 600.0 * 914400)
            assert ext.get('cx') == '5733288'
            old_cy = int(ext.get('cy'))
            ext.set('cy', str(cy))
            for e in inl.iter('{%s}ext' % A): e.set('cx', '5733288'); e.set('cy', str(cy))
            blip = next(inl.iter('{%s}blip' % A)); rid = blip.get('{%s}embed' % R)
            oldm = 'word/' + rels[rid]; newm = 'word/media/curve_B_fig%d.png' % n
            media_new[newm] = png; rels_change[rid] = 'media/curve_B_fig%d.png' % n
            # ---- 文字（使用者核定）：導引段、圖說、區塊 34
            ig, icap = kids[i - 1], kids[i + 1]; zh = SITE_ZH[s]
            g_old, c_old = ptxt(ig), ptxt(icap)
            para_replace(ig, '圖 %d 為全部 100,000 顆虛擬質點於回溯期間內仍在追蹤之比例。' % n, '圖 %d 為十類虛擬質點（每類 10,000 顆）各自於回溯期間內仍在追蹤之比例。' % n)
            para_replace(ig, '%s仍在追蹤之比例於回溯第' % zh, '全部 100,000 顆虛擬質點合計，%s仍在追蹤之比例於回溯第' % zh)
            tail = '十類曲線相近，其中不可回收塑膠略慢。' if s in SLOW else '十類曲線相近。'
            assert g_old.endswith('降至 0%。')
            para_replace(ig, '降至 0%。', '降至 0%。' + tail)
            para_replace(icap, '虛擬質點比例', '虛擬質點比例（十類）；小圖為回溯前 20 天之放大')
            g_new, c_new = ptxt(ig), ptxt(icap)
            assert g_new == g_old.replace('圖 %d 為全部 100,000 顆虛擬質點於回溯期間內仍在追蹤之比例。' % n, '圖 %d 為十類虛擬質點（每類 10,000 顆）各自於回溯期間內仍在追蹤之比例。' % n).replace('%s仍在追蹤之比例於回溯第' % zh, '全部 100,000 顆虛擬質點合計，%s仍在追蹤之比例於回溯第' % zh) + tail
            assert c_new == c_old + '（十類）；小圖為回溯前 20 天之放大'
            fit_c, tot_c = fits(g_new, cy, c_new); kn_c = ig.find(q('pPr')).find(q('keepNext')) is not None
            if fit_c != kn_c:
                flips.append(dict(fig=n, old=kn_c, new=fit_c, tot=round(tot_c, 1), kind='curve_guide'))
                pp = ig.find(q('pPr'))
                if kn_c: pp.remove(pp.find(q('keepNext')))
                else: pp.insert(1 if pp.find(q('pStyle')) is not None else 0, etree.Element(q('keepNext')))
            curve_text.append(dict(fig=n, site=s, guide_old=g_old, guide_new=g_new, caption_old=c_old, caption_new=c_new, guide_keepNext=fit_c, tot=round(tot_c, 1)))
            curve_log.append(dict(site=s, fig=n, rid=rid, old_member=oldm, new_member=newm, h_px=h_, old_cy=old_cy, new_cy=cy, png_sha256=hashlib.sha256(png).hexdigest()))
    if curves:
        b34_old = ptxt(kids[34])
        para_replace(kids[34], '呈現全部虛擬質點於回溯期間內仍在追蹤之比例', '呈現十類虛擬質點各自於回溯期間內仍在追蹤之比例')
        block34 = dict(index=34, old=b34_old, new=ptxt(kids[34]))
    else:
        block34 = None
    for r in rels_root:
        if r.get('Id') in rels_change: r.set('Target', rels_change[r.get('Id')])
    xml_out = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    rels_out = etree.tostring(rels_root, xml_declaration=True, encoding='UTF-8', standalone=True)
    drop = {c['old_member'] for c in curve_log}
    zout = zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED)
    for zi in zin.infolist():
        if zi.filename in drop: continue
        data = zin.read(zi.filename)
        if zi.filename == 'word/document.xml': data = xml_out
        elif zi.filename == 'word/_rels/document.xml.rels': data = rels_out
        ni = zipfile.ZipInfo(zi.filename, date_time=zi.date_time); ni.compress_type = zi.compress_type; ni.external_attr = zi.external_attr
        zout.writestr(ni, data)
    for m, d in media_new.items():
        ni = zipfile.ZipInfo(m, date_time=(2026, 10, 29, 0, 0, 0)); ni.compress_type = zipfile.ZIP_STORED; zout.writestr(ni, d)
    zout.close()
    log = dict(base_sha_prefix=PRE, n_blocks=359, n_captions_changed=len(changes), changes=changes, keep_flips=flips, warnings=warn, curves=curve_log, block6=block6, curve_text=curve_text, block34=block34)
    json.dump(log, open(logp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('ok', dst, 'captions', len(changes), 'flips', flips, 'warn', warn, 'curves', len(curve_log))


if __name__ == '__main__':
    a = sys.argv[1:]
    cur = None
    if '--curves' in a:
        k = a.index('--curves'); cur = a[k + 1]; del a[k:k + 2]
    main(a[0], a[1], a[2], a[3], cur)
