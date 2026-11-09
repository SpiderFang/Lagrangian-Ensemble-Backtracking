#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合併版 v9 候選：以新版面（單行標題、圖內數值框、十類 4×3、季節 2×2、大小潮 1×2）取代 v8 之 30 張多格來源分段機率圖，
並改寫該 30 個圖說中指稱「各格標題」數值之文字（就地修改 word/document.xml；其餘 zip 成員逐位元組保留）。
用法：python3 edit_v9_inplace.py <主檔docx(v8，sha 74d288ed…)> <final 根目錄（內含 <site>/<kind>_<name>.png 與 manifest.json）> <輸出docx> [v8_build_log.json] [v9_build_log.json]
前提：主檔 sha256 須以 74d288ed1dc16b94 起首且區塊數 359，否則停止。主檔不被修改。
替換之圖檔沿用原 zip 成員名（word/media/imageN.png）、原圖號與 docPr；寬 6.27 in（5733288 EMU，3762 px／600 dpi，不縮放），高依像素等比。
全年（all_b、all_c）之 10 張圖與其餘全部內容不動。"""
import sys, os, re, zipfile, hashlib, json, unicodedata, struct
from lxml import etree

PRE = '74d288ed1dc16b94'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
q = lambda t: '{%s}%s' % (W, t)
SITES = ['gongliao', 'guishan', 'hsinchu', 'nanwan', 'lienchiang']
BASE = {'gongliao': 8, 'guishan': 19, 'hsinchu': 30, 'nanwan': 41, 'lienchiang': 52}
KEYS8 = ['all_b', 'all_c', 'cls_b', 'cls_c', 'sea_b', 'sea_c', 'tid_b', 'tid_c']
MF = {'all_b': 'boundary_all', 'all_c': 'coast_all', 'cls_b': 'boundary_class', 'cls_c': 'coast_class', 'sea_b': 'boundary_season',
      'sea_c': 'coast_season', 'tid_b': 'boundary_tide', 'tid_c': 'coast_tide'}
CHANGED = ['cls_b', 'cls_c', 'sea_b', 'sea_c', 'tid_b', 'tid_c']
POSW = {'UR': '右上', 'UL': '左上', 'LR': '右下', 'LL': '左下'}
fno = lambda s, k: BASE[s] + 3 + KEYS8.index(k)


def sp(s):   # 同 v8_content.sp：中文與英數之間加空白
    s = re.sub(r'([一-鿿])([0-9A-Za-z])', r'\1 \2', s)
    return re.sub(r'([0-9A-Za-z%°])([一-鿿])', r'\1 \2', s)


def new_caption(old, key, pos):
    """舊（v8）圖說 → 新圖說。只改指稱各格標題數值之句尾，並於十類邊界補述括號與圈號；必須恰好命中一次。"""
    def sub(o, n):
        assert old.count(o) == 1, (key, o)
        return old.replace(o, n)
    p = POSW[pos]
    if key == 'cls_b':
        return sp(sub('，各格標題為該類之穿越比例', '，括號標示機率集中之連續邊界段，圓圈數字依機率由大至小編號，各格%s數值框列出該類之穿越比例、各括號段之機率（圈號）與其餘，「其餘」為未編號邊界段之機率合計' % p))
    if key == 'cls_c':
        return sp(sub('，各格標題為該類之海岸來源比例', '，各格%s數值框列出該類之海岸來源比例' % p))
    if key in ('sea_b', 'tid_b'):
        return sp(sub('，其後數值為該段機率，「其餘」為未編號邊界段之機率合計', '，各格%s數值框列出穿越比例、各括號段之機率（圈號）與其餘，「其餘」為未編號邊界段之機率合計' % p))
    if key in ('sea_c', 'tid_c'):
        return sp(sub('，其後數值為該段機率', '，各格%s數值框列出海岸來源比例及各圈號海岸段之機率' % p))
    raise KeyError(key)


# ---- 版面估算（與 edit_v8_inplace.py、check_merged_run09.py K4–K6 同式）
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


def fits(chain_texts, text, fig_cy, cap_text):
    tot = sum(nl(t, BW) * 22 + 18 for t in chain_texts) + nl(text, BW) * 22 + 18 + 12 + fig_cy / 12700.0 + (2 + nl(cap_text, CW) * 27 + 8)
    return tot <= LIM, tot


def main(src, final_dir, dst, v8log_p, v9log_p):
    raw = open(src, 'rb').read()
    assert hashlib.sha256(raw).hexdigest().startswith(PRE), '主檔 sha 與預期（74d288ed…）不同，停止'
    zin = zipfile.ZipFile(src)
    root = etree.fromstring(zin.read('word/document.xml')); body = root.find(q('body')); kids = list(body)
    assert len(kids) == 359, len(kids)
    rels = {r.get('Id'): r.get('Target') for r in etree.fromstring(zin.read('word/_rels/document.xml.rels'))}
    L8 = json.load(open(v8log_p, encoding='utf-8'))
    fm8 = {(f['site'], f['key']): f for f in L8['fig_map']}
    assert len(fm8) == 40
    ptxt = lambda p: ''.join(t.text or '' for t in p.iter(q('t')))

    # 圖段定位
    imgpar = {}
    for i, p in enumerate(kids):
        if p.tag != q('p'): continue
        for inl in p.iter('{%s}inline' % WP):
            nm = inl.find('{%s}docPr' % WP).get('name')
            m = re.match(r'^fig(\d+)\.png$', nm)
            if m: imgpar[int(m.group(1))] = i

    media_new = {}; fig9 = []; flips = []; changes9 = []
    for s in SITES:
        mf = json.load(open(os.path.join(final_dir, s, 'manifest.json'), encoding='utf-8'))
        for k in KEYS8:
            n = fno(s, k); i = imgpar[n]; f8 = fm8[(s, k)]
            g, ip, cp = kids[i - 1], kids[i], kids[i + 1]
            assert ptxt(cp) == '圖%d %s' % (n, f8['caption']), (s, k, ptxt(cp)[:40])
            assert ptxt(g) == f8['guide'], (s, k, ptxt(g)[:40])
            assert (g.find(q('pPr')).find(q('keepNext')) is not None) == f8['fit'], (s, k, '主檔導引段 keepNext 與 v8 紀錄不同')
            inl = next(ip.iter('{%s}inline' % WP)); ext = inl.find('{%s}extent' % WP)
            rid = next(inl.iter('{%s}blip' % A)).get('{%s}embed' % R); member = 'word/' + rels[rid]
            old_b = zin.read(member)
            assert old_b[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', old_b[16:24])[0] == 3762
            entry = dict(site=s, key=k, n=n, media=member, caption=f8['caption'], guide=f8['guide'], fit=f8['fit'], tot=f8['tot'], h_px=f8['h_px'], changed=False)
            if k in CHANGED:
                fn = MF[k]
                png = open(os.path.join(final_dir, s, fn + '.png'), 'rb').read()
                w_, h_ = struct.unpack('>II', png[16:24])
                assert png[:8] == b'\x89PNG\r\n\x1a\n' and w_ == 3762, (s, k, w_)
                cy = round(h_ / 600.0 * 914400)
                assert ext.get('cx') == '5733288'
                ext.set('cy', str(cy))
                for e in inl.iter('{%s}ext' % A): e.set('cx', '5733288'); e.set('cy', str(cy))
                media_new[member] = png
                rec = mf[fn]; pos = rec['box_corner']
                newcap = new_caption(f8['caption'], k, pos)
                runs = cp.findall(q('r')); assert len(runs) == 3 and ''.join(t.text or '' for t in runs[2].iter(q('t'))) == f8['caption']
                runs[2].find(q('t')).text = newcap
                capfull = '圖%d %s' % (n, newcap)
                fit, tot = fits([], f8['guide'], cy, capfull)
                ppr = g.find(q('pPr')); kn = ppr.find(q('keepNext'))
                if fit != (kn is not None):
                    flips.append(dict(site=s, key=k, n=n, old=kn is not None, new=fit, tot=round(tot, 1)))
                    if kn is not None: ppr.remove(kn)
                    else:
                        at = 1 if ppr.find(q('pStyle')) is not None else 0
                        ppr.insert(at, etree.Element(q('keepNext')))
                entry.update(changed=True, caption=newcap, fit=fit, tot=round(tot, 1), h_px=h_, cy=cy, box_corner=pos, png_sha256=hashlib.sha256(png).hexdigest(),
                             old_caption=f8['caption'], old_h_px=f8['h_px'], box_pad=rec.get('box_pad'), box_compact=rec.get('box_compact'))
                changes9.append(dict(kind='v9_caption', site=s, role='caption:' + k, fig=n, old='圖%d %s' % (n, f8['caption']), new=capfull))
            fig9.append(entry)

    # 建置紀錄（綱要同 v8_build_log.json；圖說以 v9 為準）
    L9 = dict(L8)
    L9['fig_map'] = fig9
    ch = []
    capnew = {(e['site'], e['key']): e['caption'] for e in fig9}
    for c in L8['changes']:
        c = dict(c)
        if c.get('kind') == 'new' and c.get('role', '').startswith('caption:'):
            k = c['role'].split(':')[1]
            if k in CHANGED: c['new'] = '圖%d %s' % (c['fig'], capnew[(c['site'], k)])
        ch.append(c)
    L9['changes'] = ch
    L9['v9'] = dict(base_sha_prefix=PRE, n_images_replaced=len(media_new), changes=changes9, keep_flips=flips)
    L9['flips'] = flips
    # 輸出
    xml_out = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    zout = zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED)
    for zi in zin.infolist():
        data = zin.read(zi.filename)
        if zi.filename == 'word/document.xml': data = xml_out
        elif zi.filename in media_new: data = media_new[zi.filename]
        ni = zipfile.ZipInfo(zi.filename, date_time=zi.date_time); ni.compress_type = zi.compress_type; ni.external_attr = zi.external_attr
        zout.writestr(ni, data)
    zout.close()
    json.dump(L9, open(v9log_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('ok', dst, 'images', len(media_new), 'keep flips', flips)


if __name__ == '__main__':
    a = sys.argv
    here = os.path.dirname(os.path.abspath(__file__))
    main(a[1], a[2], a[3], a[4] if len(a) > 4 else os.path.join(here, 'v8_build_log.json'), a[5] if len(a) > 5 else os.path.join(here, 'v9_build_log.json'))
