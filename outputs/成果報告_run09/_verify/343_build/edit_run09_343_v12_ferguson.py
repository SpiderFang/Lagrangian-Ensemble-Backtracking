# -*- coding: utf-8 -*-
import sys, zipfile
sys.path.insert(0,'/mnt/attach/outputs/docx_build')
from lxml import etree
from lib import *
from edit import rtext, set_rtext
SRC='/tmp/w15/cur.docx'; DST='/tmp/w15/new.docx'
zin=zipfile.ZipFile(SRC)
raw=zin.read('word/document.xml')
root=etree.fromstring(raw)
ps=[p for p in root.iter(q('w:p')) if ptext(p).startswith('（一）計算式：阻力係數') and '此形式見於' in ptext(p)]
assert len(ps)==1; p=ps[0]
before=ptext(p)
runs=[r for r in p.findall(q('w:r'))]
def sub(old,new):
    hit=[r for r in runs if old in rtext(r)]
    assert len(hit)==1,(old,len(hit))
    set_rtext(hit[0],rtext(hit[0]).replace(old,new))
sub('，此形式見於天然顆粒之','；天然顆粒之')
sub('降速度公式，其','降速度公式即以黏滯與高雷諾數兩極限相加，其')
sub('）。本研究參照此形式：','）。本研究參照其兩極限相加之概念，黏滯項改採 Schiller–Naumann 式：')
after=ptext(p)
print(before); print(after)
out=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
zo=zipfile.ZipFile(DST,'w',zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data=zin.read(it.filename)
    if it.filename=='word/document.xml': data=out
    ni=zipfile.ZipInfo(it.filename,it.date_time); ni.compress_type=it.compress_type; ni.external_attr=it.external_attr; ni.create_system=it.create_system
    zo.writestr(ni,data)
zo.close()
