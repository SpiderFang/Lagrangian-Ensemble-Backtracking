#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合併版 v6f：表 1「終端沉降速度範圍 (m/s)」欄 10 列改科學記號（就地修改，只改 word/document.xml 之表 1 第 4 欄）。
格式：−8.97×10⁻⁴ 至 −1.08×10⁻¹；有效位數照原值；指數為 Word 上標（vertAlign=superscript），指數之負號沿用 U+2212；
「至」前手動換行保留。換算後以 Decimal 逐一驗證與原值相等。
用法：python3 edit_v6f_inplace.py <合併版docx>    前提：sha256 須為 1524c8b76d49accc…（使用者於 v6e 後 Word 重存之版本），否則停止。"""
import sys, re, zipfile, hashlib, os
from decimal import Decimal
PRE = '1524c8b76d49accc'
path = sys.argv[1]
raw = open(path, 'rb').read()
assert hashlib.sha256(raw).hexdigest().startswith(PRE), '合併版 sha 與預期不同，停止'
z = zipfile.ZipFile(path)
doc = z.read('word/document.xml').decode('utf-8')

def sci(s):
    """'0.000897' -> ('8.97', -4)；有效位數照原字串（去前導 0 後之數字個數）"""
    d = s.replace('.', '').lstrip('0'); n = len(d)
    v = Decimal(s); e = v.adjusted()
    m = d[0] + ('.' + d[1:] if n > 1 else '')
    assert Decimal(m + 'E' + str(e)) == v, (s, m, e)
    assert len(m.replace('.', '')) == len(s.replace('.', '').lstrip('0')), s
    return m, e
RP = '<w:rPr><w:szCs w:val="28"/></w:rPr>'
MG = '<w:rPr><w:rFonts w:eastAsia="MS Gothic"/><w:szCs w:val="28"/></w:rPr>'
SUPM = '<w:rPr><w:rFonts w:eastAsia="MS Gothic"/><w:szCs w:val="28"/><w:vertAlign w:val="superscript"/></w:rPr>'
SUP = '<w:rPr><w:szCs w:val="28"/><w:vertAlign w:val="superscript"/></w:rPr>'
def newruns(s):
    m, e = sci(s)
    out = '<w:r>%s<w:t>%s×10</w:t></w:r>' % (RP, m)
    ex = str(abs(e))
    if e < 0: out += '<w:r>%s<w:t>−</w:t></w:r>' % SUPM
    out += '<w:r>%s<w:t>%s</w:t></w:r>' % (SUP, ex)
    return out

ts = doc.index('<w:tbl>'); te = doc.index('</w:tbl>', ts) + len('</w:tbl>')
tbl = doc[ts:te]
rows = list(re.finditer(r'<w:tr[ >].*?</w:tr>', tbl, re.S)); assert len(rows) == 11
log = []; out = tbl; shift = 0
for r in rows[1:]:
    rx = r.group(0)
    cells = list(re.finditer(r'<w:tc>.*?</w:tc>', rx, re.S)); assert len(cells) == 4
    c = cells[3]; cx = c.group(0)
    pat = re.compile(r'<w:r>' + re.escape(RP) + r'<w:t>(\d\.\d+)</w:t></w:r>')
    nums = pat.findall(cx); assert len(nums) == 2, nums
    old_txt = '−%s 至 −%s' % tuple(nums)
    cx2 = pat.sub(lambda m: newruns(m.group(1)), cx)
    new_txt = '−%s 至 −%s' % tuple('%s×10^%d' % sci(n) for n in nums)
    log.append((old_txt, new_txt))
    rx2 = rx[:c.start()] + cx2 + rx[c.end():]
    out = out[:r.start() + shift] + rx2 + out[r.end() + shift:]; shift += len(rx2) - len(rx)
for a, b in log: print(a, '→', b)
doc = doc[:ts] + out + doc[te:]
tmp = path + '.partial'
with zipfile.ZipFile(tmp, 'w') as zo:
    for info in z.infolist():
        zi = zipfile.ZipInfo(info.filename, info.date_time)
        zi.compress_type = info.compress_type; zi.external_attr = info.external_attr; zi.create_system = info.create_system
        zo.writestr(zi, doc.encode('utf-8') if info.filename == 'word/document.xml' else z.read(info.filename))
os.replace(tmp, path)
print('OK', hashlib.sha256(open(path, 'rb').read()).hexdigest(), os.path.getsize(path))
