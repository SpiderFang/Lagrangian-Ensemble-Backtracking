# -*- coding: utf-8 -*-
"""2026-10-22 小修（v10）：圖 1（word/media/image1.png）換成 figs_new_v3（經緯度範圍 119–123°E、20–27°N，依期中報告圖 2-17）。
先由 document.xml 與 rels 確認 image1.png 即圖 1（第一個 drawing、rId8、圖題「圖1 …」）；其餘 zip 成員逐位元組相同、順序與 ZipInfo 不變。
若新圖像素長寬比與舊圖不同，才改該 drawing 之 wp:extent／a:ext 的 cy（cx 不變）；本次畫布像素相同，故不改。
用法：python3 edit_run09_343_v10.py <src.docx> <dst.docx>"""
import sys, os, re, struct, zipfile, hashlib
SRC, DST = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(HERE, 'figs_new_v3', '343_run09v3_fig1_circle.png')
zin = zipfile.ZipFile(SRC)
doc = zin.read('word/document.xml').decode('utf8')
rels = zin.read('word/_rels/document.xml.rels').decode('utf8')
rid = re.search(r'<a:blip r:embed="(rId\d+)"', doc).group(1)
tgt = re.search(r'Id="%s"[^>]*?Target="(media/[^"]+)"|Target="(media/[^"]+)"[^>]*?Id="%s"' % (rid, rid), rels)
tgt = 'word/' + (tgt.group(1) or tgt.group(2))
assert tgt == 'word/media/image1.png', tgt
paras = re.findall(r'<w:p[ >].*?</w:p>', doc, flags=re.S)
txt = lambda p: ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))
ip = [i for i, p in enumerate(paras) if rid in p and 'w:drawing' in p]
assert len(ip) == 1 and txt(paras[ip[0] + 2]).startswith('圖1 ') or txt(paras[ip[0] + 1]).startswith('圖1 '), 'image1 不是圖 1'
new = open(NEW, 'rb').read()
old = zin.read(tgt)
ow, oh = struct.unpack('>II', old[16:24]); nw, nh = struct.unpack('>II', new[16:24])
print('old px', ow, oh, 'new px', nw, nh)
newdoc = None
if ow * nh != oh * nw:  # 長寬比改變才改 extent，維持寬度
    m = re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"/>', doc)
    cx = int(m.group(1)); cy = round(cx * nh / nw)
    newdoc = doc.replace('cx="%d" cy="%s"' % (cx, m.group(2)), 'cx="%d" cy="%d"' % (cx, cy))
    assert newdoc.count('cy="%d"' % cy) >= 2
    print('extent changed', m.group(2), '->', cy)
repl = {tgt: new}
if newdoc is not None:
    repl['word/document.xml'] = newdoc.encode('utf8')
with zipfile.ZipFile(DST, 'w') as zout:
    for zi in zin.infolist():
        data = repl.get(zi.filename, zin.read(zi.filename))
        ni = zipfile.ZipInfo(zi.filename, zi.date_time)
        ni.compress_type = zi.compress_type; ni.external_attr = zi.external_attr
        ni.create_system = zi.create_system; ni.comment = zi.comment; ni.extra = zi.extra
        zout.writestr(ni, data)
zo = zipfile.ZipFile(DST)
assert zo.testzip() is None and zo.namelist() == zin.namelist()
for n in zin.namelist():
    assert (zo.read(n) == zin.read(n)) == (n not in repl), n
print('ok', hashlib.sha256(open(DST, 'rb').read()).hexdigest())
