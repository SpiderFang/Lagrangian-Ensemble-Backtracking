# -*- coding: utf-8 -*-
import sys, zipfile
sys.path.insert(0,'/mnt/attach/outputs/docx_build'); sys.path.insert(0,'/tmp/w')
from lxml import etree
from lib import *
from edit import replace_in_p, set_cell_text
SRC='/tmp/w11/cur.docx'; DST='/tmp/w11/new.docx'
zin=zipfile.ZipFile(SRC)
root=etree.fromstring(zin.read('word/document.xml'))
body=root.find(q('w:body')); kids=list(body)
assert ptext(kids[11]).startswith('表2')
t=kids[12]; r=t.findall(q('w:tr'))[2]; tcs=r.findall(q('w:tc'))
exp=['龜山島','121.95','24.84','121.82–122.08','24.72–24.96']
assert [''.join(x.text or '' for x in c.iter(q('w:t'))) for c in tcs]==exp
for c,v in zip(tcs,['龜山島西側','121.94','24.85','121.81–122.07','24.73–24.97']): set_cell_text(c,v)
cap=[p for p in kids if p.tag==q('w:p') and ptext(p).startswith('圖2 ')]; assert len(cap)==1
for e in cap[0].findall(q('w:proofErr')): cap[0].remove(e)
replace_in_p(cap[0],'龜山島、','龜山島西側、'); print(ptext(cap[0]))
out=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
U='/tmp/w11/fig/'
rep={'word/document.xml':out,'word/media/image1.png':open(U+'n1.png','rb').read(),'word/media/image2.png':open(U+'n2.png','rb').read()}
zout=zipfile.ZipFile(DST,'w',zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data=rep.get(it.filename, zin.read(it.filename))
    ni=zipfile.ZipInfo(it.filename,it.date_time); ni.compress_type=it.compress_type; ni.external_attr=it.external_attr; ni.create_system=it.create_system
    zout.writestr(ni,data)
zout.close()
