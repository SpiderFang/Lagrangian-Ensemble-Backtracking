# -*- coding: utf-8 -*-
import sys, zipfile, copy
sys.path.insert(0,'/mnt/attach/outputs/docx_build'); sys.path.insert(0,'/tmp/w')
from lxml import etree
from lib import *
from edit import replace_in_p, set_cell_text, text_run
import pk
SRC='/tmp/w12/cur.docx'; DST='/tmp/w12/new.docx'
zin=zipfile.ZipFile(SRC)
root=etree.fromstring(zin.read('word/document.xml'))
body=root.find(q('w:body')); kids=list(body)
ps=[k for k in kids if k.tag==q('w:p')]
def nop(p):
    for e in p.findall(q('w:proofErr')): p.remove(e)
def findp(prefix):
    r=[p for p in ps if ptext(p).startswith(prefix)]; assert len(r)==1,(prefix,len(r)); return r[0]
def drop_col(tbl, idx, widths):
    grid=tbl.find(q('w:tblGrid')); gcs=grid.findall(q('w:gridCol')); assert len(gcs)==len(widths)+1
    grid.remove(gcs[idx])
    for g,w in zip(grid.findall(q('w:gridCol')),widths): g.set(q('w:w'),str(w))
    for tr in tbl.findall(q('w:tr')):
        tcs=tr.findall(q('w:tc')); tr.remove(tcs[idx])
        for tc,w in zip(tr.findall(q('w:tc')),widths): tc.find(q('w:tcPr')).find(q('w:tcW')).set(q('w:w'),str(w))
# ---- 1. 龜山島西側 -> 龜山島
cap2=findp('圖2 '); nop(cap2); replace_in_p(cap2,'龜山島西側、','龜山島、')
cap_t2=findp('表2 '); t2=kids[kids.index(cap_t2)+1]; assert t2.tag==q('w:tbl')
tc=t2.findall(q('w:tr'))[2].findall(q('w:tc'))[0]
assert ''.join(x.text or '' for x in tc.iter(q('w:t')))=='龜山島西側'; set_cell_text(tc,'龜山島')
# ---- 2. 代表值
p3=findp('十種項目對應'); nop(p3)
replace_in_p(p3,'表 1 列各類 10 個速度之範圍，以及 10 個速度之幾何平均（代表值）。','表 1 列各類 10 個速度之範圍。')
cap_t1=findp('表1 '); t1=kids[kids.index(cap_t1)+1]; assert t1.tag==q('w:tbl')
assert ''.join(x.text or '' for x in t1.findall(q('w:tr'))[0].findall(q('w:tc'))[4].iter(q('w:t'))).startswith('代表值')
drop_col(t1,4,[1100,2200,3407,1750])
capA1=findp('表A1 '); nop(capA1); replace_in_p(capA1,'下界、代表值與上界','下界與上界')
tA1=kids[kids.index(capA1)+1]; assert tA1.tag==q('w:tbl')
assert ''.join(x.text or '' for x in tA1.findall(q('w:tr'))[0].findall(q('w:tc'))[2].iter(q('w:t')))=='代表值'
drop_col(tA1,2,[3600,2429,2428])
# 表 A1 後說明段（原含代表值說明與「隨機取用」）
p62=findp('代表值為該類 10 個解'); 
for ch in list(p62):
    if ch.tag!=q('w:pPr'): p62.remove(ch)
rp=etree.Element(q('w:rPr')); sz=etree.SubElement(rp,q('w:szCs')); sz.set(q('w:val'),'28')
p62.append(text_run('表 A1 中數值均取三位有效數字，下界與上界為該類 10 個解之最小值與最大值。各解於模擬中之分配方式見推導流程步驟 4。',rp))
# 步驟 3 / 輸出 / 步驟六 保留；步驟六末句不再與表1、表 A1 對照
p138=findp('取三位有效數字為 0.108 m/s'); nop(p138); replace_in_p(p138,'，與表 A1 及表 1 一致','')
# ---- 3. 分配說明
# 附錄步驟 4
p4=findp('步驟 4'); 
runs=[ch for ch in p4 if ch.tag!=q('w:pPr')]
for ch in runs[3:]: p4.remove(ch)
assert ''.join(x.text or '' for x in runs[2].iter(q('w:t'))).startswith('　用於模擬')
p4.append(text_run('各研究地點每類之 10,000 顆虛擬質點，10 個解各分配給其中 1,000 顆，分配由固定隨機種子之隨機排列決定，不含斯托克斯漂流案例與含斯托克斯漂流案例相同。垂直座標向上為正，故各解取負值。',rp))
# 正文：表1 後句之後新增一段
pa=findp('各類 10 個速度由本研究以受力平衡求解')
newp=copy.deepcopy(pa)
for ch in list(newp):
    if ch.tag!=q('w:pPr'): newp.remove(ch)
txt='模擬時，各研究地點每類各有 10,000 顆虛擬質點，該類 10 個速度各分配給其中 1,000 顆。虛擬質點與速度之對應，以固定隨機種子之隨機排列決定，不含斯托克斯漂流案例與含斯托克斯漂流案例採用相同之分配。寶特瓶、鋁罐與鐵罐之半壓扁與重壓扁速度相同，故這三類之 10 個速度中有兩個數值各出現兩次，各有 8 個相異速度。'
newp.append(text_run(txt,rp))
pa.addnext(newp)
print(ptext(newp))
out=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
pk.rewrite(SRC,DST,out)
