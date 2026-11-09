# -*- coding: utf-8 -*-
"""2026-10-21 第三輪：用語依服務建議書（工項3 建議書第 29 頁：「到達時間（Arrival times）」）統一，「各到達時刻」改「各到達時間」。只改 word/document.xml。
用法：python3 edit_run09_343_v5.py <src.docx> <dst.docx>"""
import sys, os, zipfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lxml import etree
from lib import q, ptext
from edit import replace_in_p
SRC, DST = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(SRC)
root = etree.fromstring(zin.read('word/document.xml'))
ps = [k for k in root.find(q('w:body')) if k.tag == q('w:p')]
g = [p for p in ps if '各到達時刻的潮位' in ptext(p)]
assert len(g) == 1
replace_in_p(g[0], '各到達時刻的潮位', '各到達時間的潮位')
assert not any('到達時刻' in ptext(p) for p in ps)
out = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
zo = zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data = out if it.filename == 'word/document.xml' else zin.read(it.filename)
    ni = zipfile.ZipInfo(it.filename, it.date_time); ni.compress_type = it.compress_type
    ni.external_attr = it.external_attr; ni.create_system = it.create_system
    zo.writestr(ni, data)
zo.close()
print('ok')
