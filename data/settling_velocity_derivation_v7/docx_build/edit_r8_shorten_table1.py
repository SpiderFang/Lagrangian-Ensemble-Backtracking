# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'/mnt/attach/outputs/docx_build'); sys.path.insert(0,'/tmp/w')
from lxml import etree
from lib import *
from edit import replace_in_p, set_cell_text
import pk, zipfile
SRC='/tmp/w10/cur.docx'; DST='/tmp/w10/new.docx'
zin=zipfile.ZipFile(SRC)
root=etree.fromstring(zin.read('word/document.xml'))
body=root.find(q('w:body'))
kids=list(body)
t=kids[5]; assert t.tag==q('w:tbl') and ptext(kids[4]).startswith('表1 ')
new={
 '不可回收塑膠':'薄膜至帆布片之軟質包材',
 '廢紙':'吸水紙片至紙箱板',
 '保麗龍':'吸水或附著之發泡板至箱體側板',
 '廢漁網漁具':'網線至繩索',
 '寶特瓶':'瓶身碎片、完整進水瓶、半壓扁至重壓扁瓶身',
 '竹木':'細竹枝至大型竹木段（水浸飽和）',
 '可回收塑膠':'硬質塑膠碎片至塑膠籃側板、橡膠塊',
 '鋁罐':'罐身碎片、完整進水罐、半壓扁至重壓扁罐身',
 '鐵罐':'罐身碎片、完整進水罐、半壓扁至重壓扁罐身',
 '玻璃瓶':'完整進水瓶、緻密碎片、瓶身曲殼薄片',
}
rows=t.findall(q('w:tr'))
seen=set()
for r in rows[1:]:
    tcs=r.findall(q('w:tc'))
    name=''.join(x.text or '' for x in tcs[0].iter(q('w:t')))
    for p in tcs[2].findall(q('w:p')):
        for e in p.findall(q('w:proofErr')): p.remove(e)
    set_cell_text(tcs[2], new[name]); seen.add(name)
assert seen==set(new)
# 表後句
ps=[k for k in body if k.tag==q('w:p')]
cand=[p for p in ps if ptext(p).startswith('各類') and '速度由本研究以受力平衡求解' in ptext(p)]
assert len(cand)==1
p=cand[0]
for e in p.findall(q('w:proofErr')): p.remove(e)
replace_in_p(p,'代表物件輸入參數','代表物件之尺寸與輸入參數（表 A4）')
print(ptext(p))
out=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
pk.rewrite(SRC,DST,out)
