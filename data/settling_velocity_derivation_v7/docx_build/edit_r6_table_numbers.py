# -*- coding: utf-8 -*-
import sys, copy, zipfile
sys.path.insert(0,'/mnt/attach/outputs/docx_build')
sys.path.insert(0,'/tmp/w')
from lxml import etree
from lib import *
import edit
from edit import replace_in_p
SRC='/tmp/w8/cur.docx'; DST='/tmp/w8/new.docx'
zin=zipfile.ZipFile(SRC)
root=etree.fromstring(zin.read('word/document.xml'))
body=root.find(q('w:body'))
kids=list(body)
def strip_proof(p):
    for e in p.findall(q('w:proofErr')): p.remove(e)
def cap(label,title):
    p=etree.Element(q('w:p'))
    ppr=etree.SubElement(p,q('w:pPr'))
    etree.SubElement(ppr,q('w:keepNext'))
    sp=etree.SubElement(ppr,q('w:spacing'))
    for k,v in (('before','160'),('after','60'),('line','360'),('lineRule','auto')): sp.set(q('w:'+k),v)
    jc=etree.SubElement(ppr,q('w:jc')); jc.set(q('w:val'),'center')
    r=etree.SubElement(p,q('w:r'))
    rp=etree.SubElement(r,q('w:rPr'))
    c=etree.SubElement(rp,q('w:color')); c.set(q('w:val'),'000000')
    s=etree.SubElement(rp,q('w:szCs')); s.set(q('w:val'),'28')
    t=etree.SubElement(r,q('w:t')); t.text='%s %s'%(label,title)
    return p
# (index of table in body, label, title)
tabs=[(4,'表1','十類海廢之代表形狀與終端沉降速度'),
 (11,'表2','五個研究地點之錨點座標與抽樣範圍'),
 (58,'表A1','十類海廢終端沉降速度之下界、代表值與上界'),
 (77,'表A2','阻力係數形狀項上下界與教科書同類形狀之比較'),
 (81,'表A3','三種幾何模型之適用情境與計算式'),
 (92,'表A4','各類代表物件之輸入參數與沉降速度範圍'),
 (95,'表A5','輸入參數之來源分級'),
 (104,'表A6','鋁罐各代表物件之有效密度'),
 (113,'表A7','鋁罐各代表物件之淹沒比重與迭代常數'),
 (119,'表A8','330 mL 完整進水罐之迭代過程'),
 (124,'表A9','鋁罐 5 種代表物件之十個終端沉降速度'),
 (130,'表A10','本研究推導值與文獻實測值之量級對照'),
 (135,'表A11','敏感度分析之檢驗項目與結果'),
 (140,'表A12','石英砂球體之迭代解與 Clift–Gauvin 式解之比較')]
for i,_,_ in tabs: assert kids[i].tag==q('w:tbl'), i
# 引用句修改（以 kids 原索引定位）
def edit_p(i, old, new, proof=True):
    p=kids[i]; assert p.tag==q('w:p')
    if proof: strip_proof(p)
    replace_in_p(p, old, new)
edit_p(3,'下表列各類 10 個速度之範圍','表 1 列各類 10 個速度之範圍')
edit_p(3,'故各速度均為負值：','故各速度均為負值。')
edit_p(10,'五個研究地點與其錨點座標為：','五個研究地點之錨點座標見表 2。')
edit_p(46,'取第五節表列值','取表 A4 所列值')
edit_p(60,'表中數值均取三位有效數字','表 A1 中數值均取三位有效數字')
edit_p(80,'（下表）','（表 A3）')
edit_p(90,'本研究以下表之有效密度','本研究以表 A4 之有效密度')
edit_p(93,'表中體積','表 A4 中體積')
edit_p(93,'（見（二）參數來源表）','（見表 A5）')
edit_p(94,'如下表：','如表 A5 所示。')
edit_p(98,'輸入取自第五節之表','輸入取自表 A4')
edit_p(103,'結果如下表：','結果如表 A6 所示。')
edit_p(118,'之前幾次迭代如下：','之前幾次迭代如表 A8 所示。')
edit_p(128,'與第一節及正文表列一致','與表 A1 及表 1 一致')
edit_p(141,'此表檢核者','表 A12 檢核者')
for i,l,t in tabs:
    kids[i].addprevious(cap(l,t))
out=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
import pk
pk.rewrite(SRC,DST,out)
print('ok')
