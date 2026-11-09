# -*- coding: utf-8 -*-
"""2026-10-22（v13，WP-74）：圖 1–圖 7 圖內文字統一為 8 pt、Noto Sans CJK TC（TRAPs 規格）。
只替換 word/media/image1–7.png（圖 1、圖 2＝figs_new_v5；圖 3–7＝figs_arrival_zh_v7）；新舊圖像素尺寸相同，
故 word/document.xml 之顯示尺寸（wp:extent／a:ext）不改。其餘 zip 成員逐位元組相同、順序與 ZipInfo 不變。
用法：python3 edit_run09_343_v13.py <src.docx> <dst.docx>"""
import sys, os, zipfile, hashlib, struct, re
SRC, DST = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
V5, V7 = os.path.join(HERE, 'figs_new_v5'), os.path.join(HERE, 'figs_arrival_zh_v7')
MAP = {'word/media/image1.png': os.path.join(V5, '343_run09v5_fig1_circle.png'), 'word/media/image2.png': os.path.join(V5, '343_run09v5_fig2_circle.png')}
for n, s in (('image3', 'gongliao'), ('image4', 'guishan'), ('image5', 'hsinchu'), ('image6', 'nanwan'), ('image7', 'lienchiang')):
    MAP[f'word/media/{n}.png'] = os.path.join(V7, s, f'wp61_{s}_overview_S_notext.png')
def px(b): return struct.unpack('>II', b[16:24])
zin = zipfile.ZipFile(SRC)
new = {n: open(p, 'rb').read() for n, p in MAP.items()}
for n in MAP: assert px(new[n]) == px(zin.read(n)), (n, px(new[n]), px(zin.read(n)))   # 像素尺寸相同 → 顯示尺寸不必改
with zipfile.ZipFile(DST, 'w') as zout:
    for zi in zin.infolist():
        data = new.get(zi.filename, zin.read(zi.filename))
        ni = zipfile.ZipInfo(zi.filename, zi.date_time)
        ni.compress_type = zi.compress_type; ni.external_attr = zi.external_attr
        ni.create_system = zi.create_system; ni.comment = zi.comment; ni.extra = zi.extra
        zout.writestr(ni, data)
zo = zipfile.ZipFile(DST)
assert zo.testzip() is None and zo.namelist() == zin.namelist()
for n in zin.namelist():
    assert (zo.read(n) == zin.read(n)) == (n not in MAP), n
print('ok', hashlib.sha256(open(DST, 'rb').read()).hexdigest(), os.path.getsize(DST))
