# -*- coding: utf-8 -*-
"""由 v7 derive.py（import）取得 50 組代表物件與 100 個解，並整理成報告用的列。"""
import csv, math, os, sys
V7 = os.environ.get("V7_DIR", "/tmp/w/v7")
sys.path.insert(0, V7)
import derive as D
from decimal import Decimal, ROUND_HALF_UP

# (class_id, csv 情境名) -> (文稿代表物件名稱, 幾何標籤, 尺寸字串或 ('Venv', cm3)/('Deq', mm))
DOCNAME = {
 (1,'小塊發泡板'):('小塊發泡板','薄板','10×10×5 mm'),
 (1,'碎塊'):('碎塊','薄板','25×25×8 mm'),
 (1,'中型保麗龍箱體殘片'):('箱體殘片','薄板','50×50×10 mm'),
 (1,'大型浮具/保麗龍板'):('板狀殘片','薄板','100×100×20 mm'),
 (1,'箱體側板'):('箱體側板','薄板','200×200×30 mm'),
 (2,'細竹枝/小木條'):('細竹枝','圓柱','φ5×50 mm'),
 (2,'小木條'):('小木條','圓柱','φ10×100 mm'),
 (2,'中型漂流木條'):('中型木條','圓柱','φ15×150 mm'),
 (2,'大型漂流木段'):('大型木段','圓柱','φ30×300 mm'),
 (2,'大型竹木段'):('大型竹木段','圓柱','φ50×500 mm'),
 (3,'薄紙片'):('薄紙片','薄板','10×10×0.3 mm'),
 (3,'濕紙片'):('濕紙片','薄板','25×25×1 mm'),
 (3,'濕紙板碎片'):('濕紙板碎片','薄板','50×50×3 mm'),
 (3,'飽和厚紙板'):('飽和厚紙板','薄板','100×100×10 mm'),
 (3,'紙箱板'):('紙箱板','薄板','200×200×5 mm'),
 (4,'極薄雜項薄膜'):('塑膠袋薄膜','薄板','50×50×0.03 mm'),
 (4,'塑膠袋'):('塑膠袋','薄板','100×100×0.06 mm'),
 (4,'中厚雜項複合片'):('鋁箔複合包材','薄板','150×150×0.5 mm'),
 (4,'複合包材厚片'):('複合包材厚片','薄板','200×200×1 mm'),
 (4,'厚實雜項廢片'):('塑膠帆布片','薄板','300×300×2 mm'),
 (5,'單根網線'):('單根網線','圓柱','φ0.5×50 mm'),
 (5,'單絲網線'):('單絲網線','圓柱','φ1×100 mm'),
 (5,'網片段'):('網線截段','圓柱','φ2×200 mm'),
 (5,'細繩'):('細繩','圓柱','φ8×300 mm'),
 (5,'繩索'):('繩索','圓柱','φ20×500 mm'),
 (6,'瓶身碎片'):('瓶身碎片','薄板','50×50×0.3 mm'),
 (6,'600 mL 完整進水瓶身'):('600 mL 完整進水瓶','鈍體',('Venv',620)),
 (6,'壓扁瓶身'):('壓扁瓶身','薄板','250×70×20 mm'),
 (6,'重壓扁瓶身'):('重壓扁瓶身','薄板','250×70×8 mm'),
 (6,'1250 mL 完整進水瓶'):('1250 mL 完整進水瓶','鈍體',('Venv',1300)),
 (7,'硬質碎片'):('硬質塑膠小碎片','薄板','25×25×2 mm'),
 (7,'薄片可回收料'):('硬質塑膠容器碎片','薄板','50×50×3 mm'),
 (7,'中厚可回收料'):('塑膠盒板片','薄板','100×100×10 mm'),
 (7,'厚實可回收料'):('厚塑膠板或橡膠塊','薄板','200×200×20 mm'),
 (7,'塑膠籃側板'):('塑膠籃側板','薄板','300×300×5 mm'),
 (8,'罐身碎片'):('罐身碎片','薄板','50×30×0.1 mm'),
 (8,'完整進水鋁罐'):('330 mL 完整進水罐','鈍體',('Venv',390)),
 (8,'半壓扁鋁罐'):('半壓扁','薄板','100×60×25 mm'),
 (8,'重壓扁鋁罐'):('重壓扁','薄板','100×60×8 mm'),
 (8,'500 mL 完整進水罐'):('500 mL 完整進水罐','鈍體',('Venv',590)),
 (9,'罐身碎片'):('罐身碎片','薄板','50×30×0.2 mm'),
 (9,'小鐵罐完整進水'):('小型完整進水罐','鈍體',('Venv',220)),
 (9,'完整進水鐵罐'):('完整進水罐','鈍體',('Venv',425)),
 (9,'半壓扁鐵罐'):('半壓扁','薄板','100×75×30 mm'),
 (9,'重壓扁鐵罐'):('重壓扁','薄板','100×75×12 mm'),
 (10,'小碎片'):('小碎片','鈍體',('Deq',10)),
 (10,'緻密碎片（等效球徑 20 mm）'):('緻密碎片','鈍體',('Deq',20)),
 (10,'瓶身碎片（薄板）'):('瓶身曲殼薄片','薄板','50×30×3 mm'),
 (10,'330 mL 小瓶完整進水'):('330 mL 完整進水瓶','鈍體',('Venv',400)),
 (10,'完整玻璃瓶（進水）'):('完整進水瓶','鈍體',('Venv',700)),
}

CLASS_LABEL = {1:'保麗龍',2:'竹木',3:'廢紙',4:'不可回收塑膠（軟質包材）',5:'廢漁網漁具',6:'寶特瓶',
               7:'可回收塑膠（硬質塑膠、橡膠）',8:'鋁罐',9:'鐵罐',10:'玻璃瓶'}
CLASS_SHORT = {1:'保麗龍',2:'竹木',3:'廢紙',4:'不可回收塑膠',5:'廢漁網漁具',6:'寶特瓶',
               7:'可回收塑膠',8:'鋁罐',9:'鐵罐',10:'玻璃瓶'}


def gmean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def load():
    rows = []
    for cls in D.CLASSES:
        for sc in cls['scenarios']:
            shape = sc.get('shape', cls['shape'])
            cdlo, cdhi = sc.get('cd_inf', D.CD_INF[shape])
            V, A, dA = sc['geom']
            rho = sc['rho_p']
            whi, Re_hi, Cd_hi = D.solve(V, A, dA, rho, cdlo)   # cd 下界 -> 速度上界
            wlo, Re_lo, Cd_lo = D.solve(V, A, dA, rho, cdhi)
            key = (cls['class_id'], sc['scenario'])
            dn, geo, size = DOCNAME[key]
            rows.append(dict(cid=cls['class_id'], csv_name=sc['scenario'], name=dn, geo=geo, size=size,
                             V=V, A=A, dA=dA, rho=rho, cdlo=cdlo, cdhi=cdhi, wlo=wlo, whi=whi,
                             Re_lo=Re_lo, Re_hi=Re_hi, Cd_lo=Cd_lo, Cd_hi=Cd_hi, shell=sc.get('shell')))
    summ = {}
    for cid in range(1, 11):
        ws = [w for r in rows if r['cid'] == cid for w in (r['wlo'], r['whi'])]
        summ[cid] = dict(wmin=min(ws), wmax=max(ws), rep=gmean(ws), ws=sorted(ws))
    return rows, summ


def sci(x, nd=2):
    """科學記號（尾數 nd 位小數，四捨五入取半向上）-> (mantissa_str, exp_int)"""
    s = '%.6e' % x
    m, e = s.split('e')
    m = Decimal(m).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    e = int(e)
    if m >= 10:
        m = (m / 10).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP); e += 1
    return str(m), e

SUP = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
def sci_u(x, nd=2):
    m, e = sci(x, nd)
    return '%s×10%s' % (m, str(e).translate(SUP))


def rnd(x, nd):
    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def sig3(x):
    return '{:#.3g}'.format(x)
