#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合併版 v6c：就地修改（合併版為主檔，不再由 merge_run09.py 重新合併）。
依 343_formal_review.md 圖 13 導引段沉底時刻定義改寫，只改 word/document.xml 之文字；其餘 zip 成員逐位元組保留。
作法：以「段落純文字」定位（run 被 Word 切碎，故跨 run 比對），被替換之字改在第一個受影響之 run 內
（沿用該 run 之格式 rPr），其後受影響之 run 刪去被涵蓋之字；run 變空則整個移除。rsid、paraId、pPr 不動。
用法：python3 edit_v4_inplace.py <合併版docx>
前提：目前 sha256 須為 a7da3702…（使用者於 v6b 後手改存檔之版本），否則停止。"""
import sys, re, zipfile, hashlib, os, html
from xml.sax.saxutils import escape

PRE = 'a7da37026b0de67b41ebf956'   # 前 24 碼
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
# 全文「本研究」→「本計畫」：逐字對位替換（研→計、究→畫），各字沿用原所在 run 之格式；跨 run 亦可
OLD, NEW = '本研究', '本計畫'


def ptxt(p):
    return ''.join(html.unescape(m.group(2)) for m in TRE.finditer(p))


def run_bounds(p, tpos):
    a = max(p.rfind('<w:r ', 0, tpos), p.rfind('<w:r>', 0, tpos))
    b = p.index('</w:r>', tpos) + 6
    assert a >= 0
    return a, b


def apply_same(p, old, new, k):
    """將段落純文字中第 k 次出現之 old 換成同長度 new，逐字對位；每字留在原 <w:t>（沿用原 run 格式）。"""
    assert len(old) == len(new)
    segs = [(m.start(), m.end(), m.group(1), html.unescape(m.group(2)), m.group(3)) for m in TRE.finditer(p)]
    plain = ''.join(x[3] for x in segs)
    i0 = -1
    for _ in range(k + 1): i0 = plain.index(old, i0 + 1)
    pos = {i0 + j: new[j] for j in range(len(old))}
    off = 0; edits = []
    for (s_, e_, op, tx, cl) in segs:
        t2 = ''.join(pos.get(off + j, ch) for j, ch in enumerate(tx)); off += len(tx)
        if t2 != tx: edits.append((s_, e_, op, t2, cl))
    for (s_, e_, op, t2, cl) in reversed(edits):
        p = p[:s_] + op + escape(t2) + cl + p[e_:]
    return p


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


log = []
out = []; last = 0; total = 0
for m in PRE_.finditer(doc):
    p = m.group(0); t = ptxt(p)
    n = t.count(OLD)
    if n:
        for _k in range(n): p = apply_same(p, OLD, NEW, 0)
        assert ptxt(p) == t.replace(OLD, NEW)
        log.append((len(out), t[max(0, t.index(OLD) - 15):t.index(OLD) + 25]))
        total += n
    out.append(doc[last:m.start()]); out.append(p); last = m.end()
out.append(doc[last:]); doc = ''.join(out)
assert total == 21, total
assert OLD not in ''.join(ptxt(m.group(0)) for m in PRE_.finditer(doc))
print('REPLACED', total, 'in', len(log), 'paragraphs')

tmp = path + '.partial'
with zipfile.ZipFile(tmp, 'w') as zo:
    for info in z.infolist():
        zi = zipfile.ZipInfo(info.filename, info.date_time)
        zi.compress_type = info.compress_type; zi.external_attr = info.external_attr; zi.create_system = info.create_system
        zo.writestr(zi, doc.encode('utf-8') if info.filename == 'word/document.xml' else z.read(info.filename))
os.replace(tmp, path)
print('OK', hashlib.sha256(open(path, 'rb').read()).hexdigest(), os.path.getsize(path))
