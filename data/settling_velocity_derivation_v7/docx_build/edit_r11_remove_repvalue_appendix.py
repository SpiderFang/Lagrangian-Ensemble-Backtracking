# -*- coding: utf-8 -*-
import sys, zipfile, copy, json, re
sys.path.insert(0,'/mnt/attach/outputs/docx_build'); sys.path.insert(0,'/tmp/w')
from lxml import etree
from lib import *
from edit import replace_in_p, set_cell_text, text_run, set_rtext, rtext
import pk
SRC='/tmp/w13/cur.docx'; DST='/tmp/w13/new.docx'
S=json.load(open('/tmp/w13/s/range_sens_run08.json'))
zin=zipfile.ZipFile(SRC)
root=etree.fromstring(zin.read('word/document.xml'))
body=root.find(q('w:body')); kids=list(body)
ps=[k for k in kids if k.tag==q('w:p')]
def nop(p):
    for e in p.findall(q('w:proofErr')): p.remove(e)
def findp(prefix):
    r=[p for p in ps if ptext(p).startswith(prefix)]; assert len(r)==1,(prefix,len(r)); return p if False else r[0]
def rp28():
    rp=etree.Element(q('w:rPr')); e=etree.SubElement(rp,q('w:szCs')); e.set(q('w:val'),'28'); return rp
def f3(x): return '{:#.3g}'.format(x)
def pf(x):
    if abs(x)<0.05: return '0.0%'
    return ('%+.1f%%'%x).replace('-','−')
# ---- 1. 輸出、步驟 3 刪除、步驟 4 -> 3
p=findp('輸出：各類 10 個終端沉降速度'); nop(p)
replace_in_p(p,'，與其代表值','')
# 該段尾端仍有 w 數學式與句號：移除代表值之 oMath，改以句號收尾
oms=p.findall(q('m:oMath')); assert len(oms)==3, len(oms)   # w1、w10、w
p.remove(oms[2])
rr=[r for r in p.findall(q('w:r'))]
if rtext(rr[-2]).strip()=='' : p.remove(rr[-2])
last=[r for r in p.findall(q('w:r'))][-1]; print('輸出尾 run:',repr(rtext(last)))
p3=findp('步驟 3'); assert '彙整' in ptext(p3); body.remove(p3)
p4=findp('步驟 4'); runs=[c for c in p4 if c.tag==q('w:r')]
set_rtext(runs[1],' 3')
for ch in [c for c in p4 if c.tag!=q('w:pPr')][3:]: p4.remove(ch)
p4.append(text_run('各類 10 個解之最小值與最大值即第一節之下界與上界。各研究地點每類之 10,000 顆虛擬質點，10 個解各分配給其中 1,000 顆，分配由固定隨機種子之隨機排列決定，不含斯托克斯漂流案例與含斯托克斯漂流案例相同。垂直座標向上為正，故各解取負值。',rp28()))
pa1=findp('表 A1 中數值均取三位有效數字'); nop(pa1); replace_in_p(pa1,'推導流程步驟 4','推導流程步驟 3')
p77=findp('第一項除黏滯阻力外'); nop(p77); replace_in_p(p77,'對代表值之影響見第八節','對各類速度範圍之影響見第八節')
p104=findp('以鋁罐為例，逐步演算'); nop(p104); replace_in_p(p104,'逐步演算一類代表值之完整過程','逐步演算一類 10 個解之完整過程')
# ---- 步驟六與式(A12)
t6=findp('步驟六'); i6=list(body).index(t6)
eq=list(body)[i6+1]; assert eq.tag==q('w:tbl') and '(A12)' in ''.join(x.text or '' for x in eq.iter(q('w:t')))
t6b=list(body)[i6+2]; assert ptext(t6b).startswith('取三位有效數字為 0.108')
for e in (t6,eq,t6b): body.remove(e)
# ---- 式號：(A13)->(A12)，(A14)->(A13)
def renum(el,old,new):
    n=0
    for t in el.iter(q('w:t')):
        if t.text and old in t.text:
            t.text=t.text.replace(old,new); n+=1
    return n
cnt={}
for k in list(body):
    for old,new in (('(A13)','(A12)'),('(A14)','(A13)')):
        c=renum(k,old,new)
        if c: cnt[(old,new)]=cnt.get((old,new),0)+c
print('式號更動 w:t 數',cnt)
# ---- 表 A10
def table_after(cap_prefix):
    cp=findp(cap_prefix); t=body[list(body).index(cp)+1]; assert t.tag==q('w:tbl'); return t
tA10=table_after('表A10'); rows=tA10.findall(q('w:tr'))
B=S['base']
rng=lambda c:'%s–%s'%(f3(B[str(c)]['lo']),f3(B[str(c)]['hi']))
def cell(r,i): return rows[r].findall(q('w:tc'))[i]
assert ''.join(x.text or '' for x in cell(0,1).iter(q('w:t')))=='本研究推導值'
set_cell_text(cell(0,1),'本研究 10 個解之範圍')
assert ''.join(x.text or '' for x in cell(1,0).iter(q('w:t')))=='鋁罐'
set_cell_text(cell(1,1),rng(8)); set_cell_text(cell(1,3),'實測值落於推導範圍內（物件與實驗條件不同，見下文）')
assert ''.join(x.text or '' for x in cell(2,0).iter(q('w:t')))=='鐵罐'
set_cell_text(cell(2,1),rng(9)); set_cell_text(cell(2,3),'實測值落於推導範圍內；本研究完整進水罐體等效球徑約為瓶蓋之 7–9 倍')
assert ''.join(x.text or '' for x in cell(3,0).iter(q('w:t')))=='保麗龍、竹木、廢紙'
set_cell_text(cell(3,1),' / '.join(rng(c) for c in (1,2,3)))
pn=findp('（單位均為 m/s；本研究推導值為代表值'); nop(pn)
replace_in_p(pn,'本研究推導值為代表值，括號內為 10 個解之範圍','本研究欄為該類 10 個解之最小值至最大值')
p143=findp('表 A10 之實測對象'); nop(p143)
replace_in_p(p143,'數值接近僅表示量級相符','實測值落於範圍內僅表示量級相符')
replace_in_p(p143,'數值上涵蓋鋁質瓶蓋實測值，且鐵質瓶蓋快於鋁質瓶蓋之順序亦與兩類之代表值（鐵罐 0.257、鋁罐 0.108 m/s）一致。',
  '涵蓋鋁質瓶蓋實測值 0.115–0.120 m/s，鐵罐 %s m/s 涵蓋鐵質瓶蓋實測值 0.176–0.177 m/s；鐵罐之下界與上界均高於鋁罐，與鐵質瓶蓋快於鋁質瓶蓋之順序一致。'%rng(9))
print(ptext(p143))
# ---- 表 A11
tA11=table_after('表A11'); rows=tA11.findall(q('w:tr'))
V=S['variants']
assert ''.join(x.text or '' for x in cell(0,1).iter(q('w:t')))=='十類代表值變化'
set_cell_text(cell(0,1),'十類下界與上界之變化'); set_cell_text(cell(0,2),'十類下界、上界排序')
def rngtxt(k): v=V[k]; return '下界 %s 至 %s，上界 %s 至 %s'%(pf(v['pct_lo_min']),pf(v['pct_lo_max']),pf(v['pct_hi_min']),pf(v['pct_hi_max']))
swap='下界排序不變；上界排序中保麗龍與鋁罐互換（基準上界為保麗龍 %s、鋁罐 %s m/s）'%(f3(B['1']['hi']),f3(B['8']['hi']))
assert not V['textbook']['order_lo_same']==False and V['textbook']['order_hi_same']==False and V['no_first_term']['order_hi_same']==False
assert all(V[k]['order_lo_same'] for k in V) and all(V[k]['order_hi_same'] for k in ('cyl_dn','dedupe','nu10','nu28'))
set_cell_text(cell(1,1),rngtxt('textbook')); set_cell_text(cell(1,2),swap)
set_cell_text(cell(2,1),rngtxt('no_first_term')); set_cell_text(cell(2,2),swap)
c=V['cyl_dn']; assert c['changed_lo']==[2,5] and c['changed_hi']==[2,5]
set_cell_text(cell(3,1),'竹木下界 %s、上界 %s；廢漁網漁具下界 %s、上界 %s，其餘不變'%(pf(c['pct_lo']['2']),pf(c['pct_hi']['2']),pf(c['pct_lo']['5']),pf(c['pct_hi']['5'])))
set_cell_text(cell(3,2),'下界、上界排序均不變')
assert ''.join(x.text or '' for x in cell(4,0).iter(q('w:t'))).startswith('寶特瓶、鋁罐、鐵罐中速度相同')
assert V['dedupe']['changed_lo']==[] and V['dedupe']['changed_hi']==[]
set_cell_text(cell(4,1),'下界與上界均不變（重複解不影響最小值與最大值）'); set_cell_text(cell(4,2),'下界、上界排序均不變')
a,b=V['nu10'],V['nu28']
set_cell_text(cell(5,1),'10 °C：下界 %s 至 %s，上界 %s 至 %s；28 °C：下界 %s 至 %s，上界 %s 至 %s'%(pf(a['pct_lo_min']),pf(a['pct_lo_max']),pf(a['pct_hi_min']),pf(a['pct_hi_max']),pf(b['pct_lo_min']),pf(b['pct_lo_max']),pf(b['pct_hi_min']),pf(b['pct_hi_max'])))
set_cell_text(cell(5,2),'下界、上界排序均不變')
out=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
pk.rewrite(SRC,DST,out)
