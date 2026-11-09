# -*- coding: utf-8 -*-
# 第12輪：接收點抽樣範圍改為半徑 12.5 km 之圓：換圖1/圖2、受體段落寫入半徑、表2 刪除抽樣範圍兩欄、圖說改「虛線圓」
import sys, zipfile, hashlib
sys.path.insert(0,'/mnt/attach/outputs/docx_build')
from lxml import etree
from lib import *
from edit import replace_in_p, set_rtext, rtext
SRC='/tmp/w14/cur.docx'; DST='/tmp/w14/new.docx'
F1='/tmp/w14/343_fig1_circle.png'; F2='/tmp/w14/343_fig2_circle.png'
zin=zipfile.ZipFile(SRC)
root=etree.fromstring(zin.read('word/document.xml'))
body=root.find(q('w:body'))
ps=[k for k in body if k.tag==q('w:p')]
def nop(p):
    for e in p.findall(q('w:proofErr')): p.remove(e)
def findp(prefix):
    r=[p for p in ps if ptext(p).startswith(prefix)]; assert len(r)==1,(prefix,len(r)); return r[0]
def findp_in(sub):
    r=[p for p in ps if sub in ptext(p)]; assert len(r)==1,(sub,len(r)); return r[0]
# 2. 受體段落
p=findp_in('各研究地點以錨點為中心，使受體集中'); nop(p)
replace_in_p(p,'各研究地點以錨點為中心，使受體','抽樣範圍為以各研究地點錨點為圓心、半徑 12.5 km 之圓，使受體')
# 3. 表2 導句、標題、欄
pl=findp('五個研究地點之錨點座標與抽樣範圍見表'); nop(pl)
replace_in_p(pl,'錨點座標與抽樣範圍見表','錨點座標見表')
pc=findp('表2 五個研究地點之錨點座標與抽樣範圍'); nop(pc)
replace_in_p(pc,'錨點座標與抽樣範圍','錨點座標')
tb=[k for k in body if k.tag==q('w:tbl') and ''.join(x.text or '' for x in k.iter(q('w:t'))).startswith('研究地點經度')]
assert len(tb)==1; t=tb[0]
grid=t.find(q('w:tblGrid')); gc=grid.findall(q('w:gridCol')); assert len(gc)==5
for g in gc[3:]: grid.remove(g)
W3=[2800,2829,2828]
for g,w in zip(grid.findall(q('w:gridCol')),W3): g.set(q('w:w'),str(w))
for tr in t.findall(q('w:tr')):
    tcs=tr.findall(q('w:tc')); assert len(tcs)==5
    for c in tcs[3:]: tr.remove(c)
    for c,w in zip(tr.findall(q('w:tc')),W3): c.find(q('w:tcPr')).find(q('w:tcW')).set(q('w:w'),str(w))
# 4. 圖說與 alt
p1=findp('圖1 四區流場範圍'); nop(p1); replace_in_p(p1,'框旁字母','圓旁字母')
p2=findp('圖2 各研究地點受體抽樣範圍'); nop(p2); replace_in_p(p2,'受體抽樣範圍（虛線框）：','受體抽樣範圍（虛線圓）與 5 個受體水平位置（空心圓）：')
n=0
for d in root.iter('{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr'):
    s=d.get('descr') or ''
    if '框旁字母' in s: d.set('descr',s.replace('框旁字母','圓旁字母')); n+=1
    if '受體抽樣範圍（虛線框）：' in s: d.set('descr',s.replace('受體抽樣範圍（虛線框）：','受體抽樣範圍（虛線圓）與 5 個受體水平位置（空心圓）：')); n+=1
assert n==2
out=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
zo=zipfile.ZipFile(DST,'w',zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data=zin.read(it.filename)
    if it.filename=='word/document.xml': data=out
    elif it.filename=='word/media/image1.png': data=open(F1,'rb').read()
    elif it.filename=='word/media/image2.png': data=open(F2,'rb').read()
    ni=zipfile.ZipInfo(it.filename,it.date_time); ni.compress_type=it.compress_type; ni.external_attr=it.external_attr; ni.create_system=it.create_system
    zo.writestr(ni,data)
zo.close()
print('ok')
