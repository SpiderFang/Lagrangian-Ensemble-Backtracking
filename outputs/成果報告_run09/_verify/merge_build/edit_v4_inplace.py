#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合併版 v4：就地修改（合併版為主檔，不再由 merge_run09.py 重新合併）。
依 343_formal_review.md 三組建議，只改 word/document.xml 之文字；其餘 zip 成員逐位元組保留。
作法：以「段落純文字」定位（run 被 Word 切碎，故跨 run 比對），被替換之字改在第一個受影響之 run 內
（沿用該 run 之格式 rPr），其後受影響之 run 刪去被涵蓋之字；run 變空則整個移除。rsid、paraId、pPr 不動。
用法：python3 edit_v4_inplace.py <合併版docx>
前提：目前 sha256 須為 f07de702…（v3），否則停止。"""
import sys, re, zipfile, hashlib, os, html
from xml.sax.saxutils import escape

PRE = 'f07de7022391a176ff9c2efd'   # 前 24 碼
path = sys.argv[1]
raw = open(path, 'rb').read()
assert hashlib.sha256(raw).hexdigest().startswith(PRE), '合併版 sha 與預期不同，停止（可能又被手改）'
z = zipfile.ZipFile(path)
doc = z.read('word/document.xml').decode('utf-8')

TRE = re.compile(r'(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)')
PRE_ = re.compile(r'<w:p [^>]*>.*?</w:p>', re.S)

# 修改清單：(段落定位用之純文字片段[全文須唯一]，[(舊, 新), …])；舊字串須在該段純文字中恰出現一次
EDITS = [
    # ---- 第一組：易被追問之細節 ----
    ('共由「5 個水平位置', [
        ('候選面須為有效之三角形網格且非陸地，並於全部 50 個到達時間均維持濕潤（不因退潮而乾出）；符合條件者再依',
         '受體位置限於全部到達時間皆有流場與波浪資料之海域；符合條件者再依'),                         # #6
        ('抽中後另須通過流場與波浪資料之逐時空間支援檢查，未通過者剔除，並於同一區塊內依隨機順序遞補。', ''),   # #6
        ('垂向位置抽樣', '深度抽樣'),                                                                 # #5
    ]),
    ('50 個到達時間由 48 個分層時刻', [
        ('挑選時排除流場資料於日間銜接處水位不連續之時刻。', ''),                                      # #8
        ('三種潮相代理」共', '三種潮相」共'),                                                           # #10
        ('三種潮相代理分別為', '三種潮相以潮位資料判定，分別為'),                                        # #10
    ]),
    ('以下列出各研究地點之到達時間驗證圖', [
        ('同一分層格中相隔僅數小時之兩個時刻在全年圖上會相互重疊。', ''),                                # #11
    ]),
    # ---- 第二組：用語統一（#4 海底廢棄物、#14 本研究）----
    ('1、沉降動力參數', [('附著生物之海廢在水體中', '附著生物之海底廢棄物在水體中')]),
    ('表1 十類海廢', [('十類海廢之代表形狀', '十類海底廢棄物之代表形狀')]),
    ('2、到達地點（Receptor locations）', [('針對底床上的海底廢，', '針對底床上的海底廢棄物，')]),
    ('4、溯源停止邊界', [
        ('藉此精確鎖定海廢移入', '藉此精確鎖定海底廢棄物移入'),
        ('離開本計畫所設定', '離開本研究所設定'),
    ]),
    ('附錄　十類海廢終端沉降速度之推導', [('十類海廢終端', '十類海底廢棄物終端')]),
    ('表A1 十類海廢終端沉降速度之下界與上界', [('十類海廢終端', '十類海底廢棄物終端')]),
    ('各類海廢設定 5 種代表物件', [('各類海廢設定', '各類海底廢棄物設定')]),
]


def ptxt(p):
    return ''.join(html.unescape(m.group(2)) for m in TRE.finditer(p))


def run_bounds(p, tpos):
    a = max(p.rfind('<w:r ', 0, tpos), p.rfind('<w:r>', 0, tpos))
    b = p.index('</w:r>', tpos) + 6
    assert a >= 0
    return a, b


def apply_one(p, old, new):
    segs = [(m.start(), m.end(), m.group(1), html.unescape(m.group(2)), m.group(3)) for m in TRE.finditer(p)]
    plain = ''.join(s[3] for s in segs)
    assert plain.count(old) == 1, ('舊字串不唯一或不存在', old, plain.count(old))
    i0 = plain.index(old); i1 = i0 + len(old)
    out = []; off = 0; first = True
    for (s, e, op, tx, cl) in segs:
        a, b = off, off + len(tx); off = b
        if b <= i0 or a >= i1: continue
        lo = max(i0, a) - a; hi = min(i1, b) - a
        if first: nt = tx[:lo] + new + tx[hi:]; first = False
        else: nt = tx[:lo] + tx[hi:]
        out.append((s, e, op, nt, cl))
    for (s, e, op, nt, cl) in reversed(out):
        if nt == '':
            ra, rb = run_bounds(p, s)
            run = p[ra:rb]
            body = re.sub(r'<w:rPr>.*?</w:rPr>', '', run[run.index('>') + 1:-6], flags=re.S)
            if body == p[s:e]: p = p[:ra] + p[rb:]       # 該 run 除 rPr 外只有這個 <w:t>：整個移除
            else: p = p[:s] + p[e:]                      # 另有 lastRenderedPageBreak 等：只移除 <w:t>
        else:
            if (nt != nt.strip()) and 'xml:space' not in op: op = op.replace('<w:t', '<w:t xml:space="preserve"', 1)
            p = p[:s] + op + escape(nt) + cl + p[e:]
    return p


for anchor, reps in EDITS:
    hit = [(m.start(), m.end(), m.group(0)) for m in PRE_.finditer(doc) if anchor in ptxt(m.group(0))]
    assert len(hit) == 1, ('段落定位不唯一', anchor, len(hit))
    a, b, p = hit[0]; n0 = len(ptxt(p))
    for old, new in reps: p = apply_one(p, old, new)
    doc = doc[:a] + p + doc[b:]
    print('EDIT', anchor[:16], n0, '->', len(ptxt(p)))

tmp = path + '.partial'
with zipfile.ZipFile(tmp, 'w') as zo:
    for info in z.infolist():
        zi = zipfile.ZipInfo(info.filename, info.date_time)
        zi.compress_type = info.compress_type; zi.external_attr = info.external_attr; zi.create_system = info.create_system
        zo.writestr(zi, doc.encode('utf-8') if info.filename == 'word/document.xml' else z.read(info.filename))
os.replace(tmp, path)
print('OK', hashlib.sha256(open(path, 'rb').read()).hexdigest(), os.path.getsize(path))
