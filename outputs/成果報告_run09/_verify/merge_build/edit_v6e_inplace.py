#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合併版 v6e：就地修改（合併版為主檔，不再由 merge_run09.py 重新合併）。
依 343_formal_review.md 圖 13 導引段沉底時刻定義改寫，只改 word/document.xml 之文字；其餘 zip 成員逐位元組保留。
作法：以「段落純文字」定位（run 被 Word 切碎，故跨 run 比對），被替換之字改在第一個受影響之 run 內
（沿用該 run 之格式 rPr），其後受影響之 run 刪去被涵蓋之字；run 變空則整個移除。rsid、paraId、pPr 不動。
用法：python3 edit_v4_inplace.py <合併版docx>
前提：目前 sha256 須為 f1acf955…（v6c），否則停止。"""
import sys, re, zipfile, hashlib, os, html
from xml.sax.saxutils import escape

PRE = 'c9a3edb779c6e3d85ed8566d'   # 前 24 碼
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
EDITS = [
    ('start', '各類 10 個速度由本計畫以受力平衡求解', [('各類 10 個速度由本計畫以受力平衡求解，', '各類 10 個速度係以受力平衡求解，')], None),
    ('start', '（一）計算式：阻力係數', [('本計畫參照其兩極限相加之概念', '參照其兩極限相加之概念')], None),
    ('start', '下界取與沉降姿態最接近之固定物體值', [('本計畫薄板依教科書', '本計畫採用之薄板，依教科書')], None),
    ('start', '（一）代表物件輸入：保麗龍', [('尚無實測，本計畫以表 A4', '尚無實測，故以表 A4')], None),
    ('start', '回溯期間內，各研究地點之虛擬質點皆於 33.4 天內停止追蹤', [('，無回溯達 180 天者。', '。')], None),
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
