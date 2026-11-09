# -*- coding: utf-8 -*-
"""2026-10-21 小修：「4、溯源停止邊界」段，於「回溯期間上限為 180 天。」前加一句。只改 word/document.xml。
用法：python3 edit_run09_343_v4.py <src.docx> <dst.docx>"""
import sys, os, zipfile
sys.path.insert(0, '/tmp/w3')
from lxml import etree
from lib import q, ptext
from edit import replace_in_p
SRC, DST = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(SRC)
root = etree.fromstring(zin.read('word/document.xml'))
ps = [k for k in root.find(q('w:body')) if k.tag == q('w:p')]
g = [p for p in ps if ptext(p).startswith('4、溯源停止邊界')]
assert len(g) == 1
for e in g[0].findall(q('w:proofErr')):
    g[0].remove(e)
replace_in_p(g[0], '回溯期間上限為 180 天。', '本研究之關注區域即圖 1 之四區流場範圍，虛擬質點離開流場範圍邊界即停止追蹤。回溯期間上限為 180 天。')
out = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
zo = zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data = out if it.filename == 'word/document.xml' else zin.read(it.filename)
    ni = zipfile.ZipInfo(it.filename, it.date_time); ni.compress_type = it.compress_type
    ni.external_attr = it.external_attr; ni.create_system = it.create_system
    zo.writestr(ni, data)
zo.close()
print('ok')
