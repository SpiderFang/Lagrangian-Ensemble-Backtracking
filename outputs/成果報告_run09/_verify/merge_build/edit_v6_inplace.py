#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合併版 v6：就地修改（合併版為主檔，不再由 merge_run09.py 重新合併）。
依 343_formal_review.md 圖 13 導引段沉底時刻定義改寫，只改 word/document.xml 之文字；其餘 zip 成員逐位元組保留。
作法：以「段落純文字」定位（run 被 Word 切碎，故跨 run 比對），被替換之字改在第一個受影響之 run 內
（沿用該 run 之格式 rPr），其後受影響之 run 刪去被涵蓋之字；run 變空則整個移除。rsid、paraId、pPr 不動。
用法：python3 edit_v4_inplace.py <合併版docx>
前提：目前 sha256 須為 3ec0aaca…（v5f），否則停止。"""
import sys, re, zipfile, hashlib, os, html
from xml.sax.saxutils import escape

PRE = '3ec0aaca8aa5349e59733cb3'   # 前 24 碼
path = sys.argv[1]
raw = open(path, 'rb').read()
assert hashlib.sha256(raw).hexdigest().startswith(PRE), '合併版 sha 與預期不同，停止（可能又被手改）'
z = zipfile.ZipFile(path)
doc = z.read('word/document.xml').decode('utf-8')

TRE = re.compile(r'(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)')
PRE_ = re.compile(r'<w:p [^>]*>.*?</w:p>', re.S)

# 修改清單：(段落定位用之純文字片段[全文須唯一]，[(舊, 新), …])；舊字串須在該段純文字中恰出現一次
# (定位方式, 定位字串, [(舊, 新), …], 縮排替換 (舊 ind, 新 ind) 或 None)
# 定位方式：'eq' 段落純文字全等；'start' 段落純文字以該字串開頭（須唯一）
IND_L3 = ('<w:ind w:left="980" w:hangingChars="150" w:hanging="420"/>', '<w:ind w:leftChars="200" w:left="560"/>')     # 逐地點成果／綜合說明與結論／整體結論：起於 560（同原編號位置），無懸掛
IND_L4 = ('<w:ind w:left="1260" w:hangingChars="250" w:hanging="700"/>', '<w:ind w:leftChars="354" w:left="991"/>')       # 五個研究地點：與內文左緣（991）對齊
EDITS = [
    # ---- 一、（二）內小標不編號 ----
    ('eq', '1、逐地點成果：', [('1、', '')], IND_L3),
    ('eq', '2、綜合說明與結論：', [('2、', '')], IND_L3),
    ('eq', '3、整體結論：', [('3、', '')], IND_L3),
    ('eq', '（1）貢寮：', [('（1）', '')], IND_L4),
    ('eq', '（2）龜山島：', [('（2）', '')], IND_L4),
    ('eq', '（3）新竹外海：', [('（3）', '')], IND_L4),
    ('eq', '（4）南灣：', [('（4）', '')], IND_L4),
    ('eq', '（5）連江：', [('（5）', '')], IND_L4),
    # ---- 二、水平軌跡總覽導引段補結果句（依現行圖檔；各地點分別描述）----
    ('start', '圖 8 為隨機抽取之 24 條', [('線色依起始深度由淺至深。', '線色依起始深度由淺至深。軌跡多向南延伸並穿越流場範圍南界，其次向北延伸；少數抵達海岸，與圖 11 之主要來源方向一致。')], None),
    ('start', '圖 15 為龜山島隨機抽取之 24 條', [('回溯軌跡。', '回溯軌跡。軌跡向南至東南延伸者居多，其次向北延伸；部分抵達海岸，與圖 18 之主要來源方向一致。')], None),
    ('start', '圖 22 為新竹外海隨機抽取之 24 條', [('回溯軌跡。', '回溯軌跡。軌跡向西南至西方與東北方延伸，穿越流場範圍西界者較多；僅極少數抵達海岸，與圖 25 之主要來源方向一致。')], None),
    ('start', '圖 29 為南灣隨機抽取之 24 條', [('回溯軌跡。', '回溯軌跡。軌跡多向西北延伸並穿越流場範圍西界，少數向東延伸；部分抵達海岸，與圖 32 之主要來源方向一致。')], None),
    ('start', '圖 36 為連江隨機抽取之 24 條', [('回溯軌跡。', '回溯軌跡。軌跡多向東北延伸並穿越流場範圍北界，另有向西南延伸者；多條抵達福建沿岸與馬祖各島，與圖 39 之主要來源方向一致。')], None),
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


for mode, anchor, reps, ind in EDITS:
    if mode == 'eq': hit = [(m.start(), m.end(), m.group(0)) for m in PRE_.finditer(doc) if ptxt(m.group(0)) == anchor]
    else: hit = [(m.start(), m.end(), m.group(0)) for m in PRE_.finditer(doc) if ptxt(m.group(0)).startswith(anchor)]
    assert len(hit) == 1, ('段落定位不唯一', anchor, len(hit))
    a, b, p = hit[0]; n0 = len(ptxt(p))
    for old, new in reps: p = apply_one(p, old, new)
    if ind is not None:
        assert p.count(ind[0]) == 1, ('縮排字串', anchor)
        p = p.replace(ind[0], ind[1])
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
