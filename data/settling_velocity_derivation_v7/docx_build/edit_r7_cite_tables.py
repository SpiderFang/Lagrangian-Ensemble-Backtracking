# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'/mnt/attach/outputs/docx_build'); sys.path.insert(0,'/tmp/w')
from lxml import etree
from lib import *
from edit import replace_in_p
import pk, zipfile
SRC='/tmp/w9/cur.docx'; DST='/tmp/w9/new.docx'
zin=zipfile.ZipFile(SRC)
root=etree.fromstring(zin.read('word/document.xml'))
body=root.find(q('w:body'))
ps=[k for k in body if k.tag==q('w:p')]
def find(prefix):
    r=[p for p in ps if ptext(p).startswith(prefix)]
    assert len(r)==1,(prefix,len(r)); return r[0]
def edit(prefix,old,new):
    p=find(prefix)
    for e in p.findall(q('w:proofErr')): p.remove(e)
    before=ptext(p); replace_in_p(p,old,new); print(before[-40:],'->',ptext(p)[-60:] if prefix!='上述實測對象' else ptext(p)[:30])
edit('（二）形狀項上下界：','形狀項上下界：','形狀項上下界：與教科書同類形狀之比較見表 A2。')
edit('淹沒比重','：','，各代表物件之值如表 A7。')
edit('十個解（5 種代表物件','上下界）：','上下界）見表 A9。')
edit('上述實測對象','上述實測對象','表 A10 之實測對象')
edit('以下各項檢驗以第五節','其餘條件不變：','其餘條件不變，各項檢驗結果見表 A11。')
out=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
pk.rewrite(SRC,DST,out)
