# -*- coding: utf-8 -*-
"""2026-10-22 小修（v11）：圖 1（word/media/image1.png）換成 figs_new_v4（經緯度範圍 119–123°E、20–27°N，依期中報告圖 2-17）。
先由 document.xml 與 rels 確認 image1.png 即圖 1（第一個 drawing、rId8、圖題「圖1 …」）；其餘 zip 成員逐位元組相同、順序與 ZipInfo 不變。
該 drawing 之 wp:extent 與 a:ext 一併改為新圖 600 dpi 原尺寸（EMU＝像素數×1524，字級不被縮放），並檢查圖寬不超過頁面文字寬（A4、左右邊界 1440 twips）。
用法：python3 edit_run09_343_v11.py <src.docx> <dst.docx>"""
import sys, os, re, struct, zipfile, hashlib
SRC, DST = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(HERE, 'figs_new_v4', '343_run09v4_fig1_circle.png')
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
m = re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"/>', doc)
ocx, ocy = int(m.group(1)), int(m.group(2))
cx, cy = nw * 1524, nh * 1524  # 600 dpi：914400 EMU/in ÷ 600 = 1524 EMU/px
assert doc.count('cx="%d" cy="%d"' % (ocx, ocy)) == 2  # wp:extent 與 a:ext 各一
newdoc = doc.replace('cx="%d" cy="%d"' % (ocx, ocy), 'cx="%d" cy="%d"' % (cx, cy))
assert newdoc.count('cx="%d" cy="%d"' % (cx, cy)) == 2
sect = re.search(r'<w:pgSz w:w="(\d+)" w:h="(\d+)"/><w:pgMar w:top="(\d+)" w:right="(\d+)" w:bottom="(\d+)" w:left="(\d+)"', doc)
pw, ph, mt, mr, mb, ml = map(int, sect.groups())
textw_in, texth_in = (pw - ml - mr) / 1440, (ph - mt - mb) / 1440
print('extent', ocx, ocy, '->', cx, cy, '= %.4f x %.4f in' % (cx / 914400, cy / 914400), '; page text area %.3f x %.3f in' % (textw_in, texth_in))
assert cx / 914400 <= textw_in and cy / 914400 < texth_in
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
