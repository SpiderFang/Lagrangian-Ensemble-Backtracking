# -*- coding: utf-8 -*-
"""2026-10-22 小修（v9）：圖 3–圖 7（word/media/image3–7.png）換成 figs_arrival_zh_v6（.203 重繪，圖例對齊主圖左緣）。
其餘 zip 成員逐位元組相同、順序與 ZipInfo 不變。用法：python3 edit_run09_343_v9.py <src.docx> <dst.docx>"""
import sys, os, zipfile, hashlib
SRC, DST = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.join(HERE, 'figs_arrival_zh_v6')
MAP = {'word/media/image3.png': 'gongliao', 'word/media/image4.png': 'guishan', 'word/media/image5.png': 'hsinchu',
       'word/media/image6.png': 'nanwan', 'word/media/image7.png': 'lienchiang'}
zin = zipfile.ZipFile(SRC)
new = {n: open(os.path.join(V6, s, 'wp61_%s_overview_S_notext.png' % s), 'rb').read() for n, s in MAP.items()}
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
print('ok', hashlib.sha256(open(DST, 'rb').read()).hexdigest())
